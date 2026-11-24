"""Opt-in causal FFN routing; identical per-byte kernels in batch and stream."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus_expanse.kuramoto import KuramotoMonarchByteCore, KuramotoOscillatorLayer
from nexus_expanse.substrate import rank_safe_blocks


def random_deep(seed, position, layer):
    """Stateless SplitMix64 control, independent of bytes and predictions."""
    mask=(1<<64)-1
    z=(seed + 0x9E3779B97F4A7C15*(1+position) + 0xD1B54A32D192ED03*(1+layer)) & mask
    z=((z^(z>>30))*0xBF58476D1CE4E5B9)&mask
    z=((z^(z>>27))*0x94D049BB133111EB)&mask
    return bool((z^(z>>31))&1)


class CausalKuramotoLayer(KuramotoOscillatorLayer):
    def routed_step(self, x, z, policy, position, layer_index, random_seed, collect_scores=False):
        h=self.osc_norm(x)[:,0,:]
        z=self._osc_step(h,z)
        b=self.out_proj.bias
        out=F.linear(z.reshape(x.shape[0],self.dim),self.out_proj.weight.double(),b.double() if b is not None else None).to(x.dtype)
        x=x+self.drop(out.unsqueeze(1))
        h=self.ffn_norm(x)
        scores=self.ffn.gate_scores(h) if collect_scores or policy=='causal' else None
        if policy=='always':
            deep=torch.ones(h.shape[:-1],device=h.device,dtype=torch.bool)
        elif policy=='causal':
            deep=scores>=self.ffn.tau_eval
        elif policy=='random':
            deep=torch.full(h.shape[:-1],random_deep(random_seed,position,layer_index),device=h.device,dtype=torch.bool)
        else:
            raise ValueError('unknown FFN policy')
        delta=torch.zeros_like(h)
        if bool(deep.any()):
            mid=F.silu(self._monarch_fp64(self.ffn.fc1,h[deep]))
            delta[deep]=self._monarch_fp64(self.ffn.fc2,mid)
        return x+self.ffn_drop(delta),z,scores,deep,x


class CausalKuramotoMonarchByteCore(KuramotoMonarchByteCore):
    def __init__(self,n_layers=2,dim=64,ffn_mult=2,target_skip_rate=.50,enable_skip=True,dropout=0.,max_seq_len=256):
        nn.Module.__init__(self)
        self.dim=dim; self.max_seq_len=max_seq_len
        self.n_blocks=rank_safe_blocks(dim,dim*ffn_mult)
        self.emb=nn.Embedding(256,dim); self.drop=nn.Dropout(dropout)
        self.layers=nn.ModuleList([CausalKuramotoLayer(dim,ffn_mult,self.n_blocks,target_skip_rate,enable_skip,dropout) for _ in range(n_layers)])
        self.out_norm=nn.LayerNorm(dim)
        self.policy='causal' if enable_skip else 'always'
        self.random_seed=0

    def _step(self, tokens, states, position, policy=None, calibrate_layer=None, readout=True, collect_scores=False):
        x=self.drop(self.emb(tokens))
        next_states=[]; scores=[]; decisions=[]; pre=[]
        for i,(layer,z) in enumerate(zip(self.layers,states)):
            p=policy or self.policy
            if calibrate_layer is not None:
                p='causal' if i<calibrate_layer else 'always'
            x,z,s,d,u=layer.routed_step(x,z,p,position,i,self.random_seed,collect_scores or calibrate_layer is not None)
            next_states.append(z); scores.append(s); decisions.append(d); pre.append(u)
        hidden=self.out_norm(x) if readout else None
        logits=F.linear(hidden.double(),self.emb.weight.double()).to(hidden.dtype) if readout else None
        return logits,hidden,next_states,scores,decisions,pre

    def _sequence(self,tokens,targets=None,policy=None):
        if tokens.ndim!=2 or tokens.shape[1]==0: raise ValueError('nonempty [B,T] required')
        states=self.init_stream_state(tokens.shape[0],tokens.device)
        logits=[]; hidden=[]; skips=[]; aux=[]
        for i in range(tokens.shape[1]):
            l,h,states,s,d,pre=self._step(tokens[:,i:i+1],states,i,policy,collect_scores=targets is not None)
            logits.append(l); hidden.append(h); skips.append(torch.stack([(~v).float().mean() for v in d]))
            if targets is not None:
                for sc,u in zip(s,pre):
                    with torch.no_grad():
                        lp=F.linear(self.out_norm(u).double(),self.emb.weight.double()).float()
                        nll=F.cross_entropy(lp[:,0],targets[:,i],reduction='none')
                    aux.append(F.mse_loss(sc[:,0],nll))
        return torch.cat(logits,1),torch.cat(hidden,1),torch.stack(skips).mean(0).tolist(),torch.stack(aux).mean() if aux else None

    def forward_neural(self,tokens):
        return self._sequence(tokens)[:3]

    def forward_train(self,tokens,targets):
        # Shared always-on weights for all policies; only the score head receives auxiliary loss.
        l,_,_,aux=self._sequence(tokens,targets,policy='always')
        return l,aux

    @torch.inference_mode()
    def calibrate_gates(self,train_batches):
        was=self.training; self.eval()
        try:
            for layer_idx,layer in enumerate(self.layers):
                scores=[]
                for tokens,_ in train_batches:
                    z=self.init_stream_state(tokens.shape[0],tokens.device)
                    for pos in range(tokens.shape[1]):
                        _,_,z,s,_,_=self._step(tokens[:,pos:pos+1],z,pos,calibrate_layer=layer_idx,readout=False)
                        scores.append(s[layer_idx].flatten())
                if not scores: raise ValueError('nonempty training calibration required')
                layer.ffn.tau_eval.copy_(torch.quantile(torch.cat(scores),layer.ffn.target_skip_rate))
        finally: self.train(was)

    @torch.inference_mode()
    def stream_step(self,byte_val,states,position,readout=True):
        if self.training: raise ValueError('streaming requires eval mode')
        tok=torch.tensor([[byte_val]],device=self.emb.weight.device,dtype=torch.long)
        logits,_,states,_,decisions,_=self._step(tok,states,position,readout=readout)
        return logits[0,0].double().softmax(0) if readout else None,states,[bool(d.item()) for d in decisions]

    def step_streaming(self,byte_val,states,position: Optional[int]=None):
        if position is None and self.policy=='random': raise ValueError('random control requires explicit position')
        probs,states,_=self.stream_step(byte_val,states,0 if position is None else position)
        return probs,states
