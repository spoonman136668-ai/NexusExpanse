"""Opt-in causal, locally coupled phase mixer for raw byte streams."""
import torch
from torch import nn
import torch.nn.functional as F

from nexus_expanse.substrate import CausalGatedMonarchFFN, rank_safe_blocks


@torch.jit.script
def _advance(z: torch.Tensor, field: torch.Tensor, omega: torch.Tensor, coupling: torch.Tensor):
    # Dot with the tangent produces the Kuramoto sine phase difference.
    tangent = torch.stack((-z[...,1],z[...,0]),dim=-1)
    neighbors = .5*(torch.roll(z,1,-2)+torch.roll(z,-1,-2))
    angular = omega + ((field + coupling.unsqueeze(-1)*neighbors)*tangent).sum(-1)
    displacement = z + .25*angular.unsqueeze(-1)*tangent
    return displacement / torch.linalg.vector_norm(displacement,dim=-1,keepdim=True)


@torch.jit.script
def _scan(z: torch.Tensor, field: torch.Tensor, omega: torch.Tensor, coupling: torch.Tensor):
    phases = torch.jit.annotate(list[torch.Tensor], [])
    for i in range(field.size(1)):
        z = _advance(z,field[:,i],omega[:,i],coupling[:,i])
        phases.append(z)
    return torch.stack(phases,dim=1)


class KuramotoOscillatorMixer(nn.Module):
    """One first-order tangent step on a nearest-neighbor ring of S1 phases."""
    def __init__(self, dim):
        super().__init__()
        if dim%2 or dim<4:
            raise ValueError('an even dimension >=4 is required')
        self.dim=dim
        self.drive=nn.Linear(dim,2*dim)
        self.field=nn.Linear(dim,dim)
        self.readout=nn.Linear(dim,dim)

    def initial(self, reference, batch):
        z=reference.new_zeros(batch,self.dim//2,2)
        z[...,0]=1
        return z

    def inputs(self,x):
        field,rates=self.drive(x).chunk(2,-1)
        omega,coupling=rates.chunk(2,-1)
        return (self.field(F.silu(field)).unflatten(-1,(self.dim//2,2)),
                omega.tanh(),coupling.sigmoid())

    def step(self,x,z):
        z=_advance(z,*self.inputs(x))
        return self.readout(z.flatten(-2)),z

    def forward(self,x):
        z=self.initial(x,x.size(0))
        if self.training:
            phases=_scan(z,*self.inputs(x))
            return self.readout(phases.flatten(-2))
        # Full-sequence reference is layer-major; streaming keeps only the last phases.
        output=[]
        for row in x.unbind(1):
            y,z=self.step(row,z)
            output.append(y)
        return torch.stack(output,1)


class PhaseLayer(nn.Module):
    def __init__(self, dim, ffn_mult, n_blocks, target_skip_rate):
        super().__init__()
        self.attn_norm=nn.LayerNorm(dim)
        self.mixer=KuramotoOscillatorMixer(dim)
        self.ffn_norm=nn.LayerNorm(dim)
        self.ffn=CausalGatedMonarchFFN(dim,dim*ffn_mult,n_blocks,target_skip_rate)

    def step(self,x,z):
        delta,z=self.mixer.step(self.attn_norm(x),z)
        x=x+delta
        delta,_,_=self.ffn(self.ffn_norm(x))
        return x+delta,z

    def sequence(self,x):
        x=x+self.mixer(self.attn_norm(x))
        h=self.ffn_norm(x)
        if self.training:
            delta,scores,deep=self.ffn(h)
        else:
            rows=[self.ffn(row) for row in h.unbind(1)]
            delta,scores,deep=(torch.stack(items,1) for items in zip(*rows))
        return x+delta,x,scores,deep


class KuramotoMonarchByteCore(nn.Module):
    def __init__(self,dim=64,n_layers=2,ffn_mult=2,max_seq_len=256,target_skip_rate=.5):
        super().__init__()
        self.dim=dim
        self.max_seq_len=max_seq_len  # Patch reader interface; recurrence is unbounded.
        self.emb=nn.Embedding(256,dim)
        self.byte_drive=nn.Embedding(256,dim)
        blocks=rank_safe_blocks(dim,dim*ffn_mult)
        self.layers=nn.ModuleList(PhaseLayer(dim,ffn_mult,blocks,target_skip_rate) for _ in range(n_layers))
        self.out_norm=nn.LayerNorm(dim)

    def forward_neural(self,tokens,targets=None):
        x=self.emb(tokens)+self.byte_drive(tokens)
        aux=x.new_zeros(())
        skips=[]
        for layer in self.layers:
            x,pre,scores,deep=layer.sequence(x)
            skips.append((~deep).float().mean().item())
            if targets is not None:
                with torch.no_grad():
                    logits=F.linear(self.out_norm(pre),self.emb.weight)
                    nll=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').view_as(targets)
                aux=aux+F.mse_loss(scores,nll)
        h=self.out_norm(x)
        logits=(F.linear(h,self.emb.weight) if self.training else
                torch.stack([F.linear(row,self.emb.weight) for row in h.unbind(1)],1))
        return logits,h,aux/len(self.layers),skips

    @torch.inference_mode()
    def calibrate_gates(self,batches,max_batches=16):
        was_training=self.training
        self.eval()
        try:
            inputs=[self.emb(x)+self.byte_drive(x) for x,_ in batches[:max_batches]]
            for i,layer in enumerate(self.layers):
                residuals=[x+layer.mixer(layer.attn_norm(x)) for x in inputs]
                scores=torch.cat([layer.ffn.gate_scores(row).flatten()
                                  for x in residuals for row in layer.ffn_norm(x).unbind(1)])
                layer.ffn.tau_eval.copy_(torch.quantile(scores,layer.ffn.target_skip_rate))
                if i+1<len(self.layers):
                    inputs=[x+torch.stack([layer.ffn(row)[0] for row in layer.ffn_norm(x).unbind(1)],1)
                            for x in residuals]
        finally:
            self.train(was_training)

    def new_stream(self):
        if self.training:
            raise ValueError('streaming requires evaluation mode')
        return PhaseStream(self)


class PhaseStream:
    def __init__(self,model):
        self.model=model
        self.reset()

    def reset(self):
        self.phases=[layer.mixer.initial(self.model.emb.weight,1) for layer in self.model.layers]

    @property
    def tensor_bytes(self):
        return sum(z.numel()*z.element_size() for z in self.phases)

    @torch.inference_mode()
    def append(self,byte):
        if type(byte) is not int or not 0<=byte<=255:
            raise ValueError('expected a raw byte')
        m=self.model
        x=(m.emb.weight[byte]+m.byte_drive.weight[byte])[None]
        for i,layer in enumerate(m.layers):
            x,self.phases[i]=layer.step(x,self.phases[i])
        return F.linear(m.out_norm(x),m.emb.weight)[0]
