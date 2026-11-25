"""Checkpoint-only Dense-to-DirectMonarch conversion for compatible byte interfaces."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import forbid_donor_forward, graft_dense_to_monarch
from nexus_expanse.substrate import DirectMonarchLinear

CONFIG = dict(dim=32, n_layers=2, n_heads=4, ffn_mult=2, max_seq_len=64,
              dropout=0., enable_skip=False)
METHODS = ('monarch', 'block2', 'spectral4', 'walsh8', 'residual8')


class CorrectedLinear(nn.Module):
    def __init__(self, base, weight, method):
        super().__init__()
        self.base = base
        self.method = method
        fit = base(torch.eye(base.in_dim)).T.detach()
        residual = weight - fit
        if method == 'block2':
            self.correction = DirectMonarchLinear(base.in_dim, base.out_dim, base.n_blocks, bias=False)
            graft_dense_to_monarch(residual, self.correction)
        elif method == 'spectral4':
            u, s, vh = torch.linalg.svd(residual.double(), full_matrices=False)
            self.left = nn.Parameter((u[:, :4] * s[:4]).float())
            self.right = nn.Parameter(vh[:4].float())
        elif method == 'walsh8':
            size = base.in_dim
            if size & (size-1):
                raise ValueError('Walsh correction requires a power-of-two input width')
            h = torch.ones(1, 1)
            while h.size(0) < size:
                h = torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
            h /= math.sqrt(size)
            coeff = residual @ h.T
            idx = coeff.square().sum(0).topk(8).indices
            self.left = nn.Parameter(coeff[:,idx])
            self.register_buffer('right', h[idx])
        elif method == 'residual16':
            self.register_buffer('half_error',residual.half())
        elif method == 'residual8':
            scale = residual.abs().amax(1,keepdim=True).clamp_min(1e-12)/127.
            self.register_buffer('codes',torch.round(residual/scale).clamp(-127,127).to(torch.int8))
            self.scale = nn.Parameter(scale)
        else:
            raise ValueError(method)

    def forward(self,x):
        if self.method == 'block2':
            delta = self.correction(x)
        elif self.method == 'residual16':
            delta=F.linear(x,self.half_error.to(x.dtype))
        elif self.method == 'residual8':
            delta = F.linear(x,self.codes.to(self.scale.dtype)*self.scale)
        else:
            delta = F.linear(F.linear(x,self.right),self.left)
        return self.base(x)+delta


class ImportedSkill(nn.Module):
    def __init__(self):
        super().__init__()
        self.core=NexusExpanseByteCore(**CONFIG)
        self.readout=nn.Linear(CONFIG['dim'],256,bias=False)

    def forward(self, raw):
        hidden=self.core.forward_neural(raw)[1]
        return self.readout(hidden)


class SkillBank(nn.Module):
    """Explicit skill selection preserves the original model's normal entry point."""
    def __init__(self, base, skill):
        super().__init__()
        self.base=base.eval().requires_grad_(False)
        self.skill=skill

    def forward(self, raw, *, use_skill=False):
        return self.skill(raw) if use_skill else self.base.forward_neural(raw)[0]


def load_skill(path, method):
    """Input is tensors and a byte-to-row vocabulary. No input examples or teacher."""
    with forbid_donor_forward() as guard, torch.no_grad():
        package=torch.load(path,weights_only=True,map_location='cpu')
        state, ids=package['weights'],package['byte_to_row']
        if not torch.equal(ids.sort().values,torch.arange(256)):
            raise ValueError('expected a bijection of all singleton bytes')
        skill=ImportedSkill()
        core=skill.core
        core.emb.weight.copy_(state['emb.weight'][ids])
        core.pos_emb.weight.copy_(state['pos_emb.weight'])
        core.out_norm.load_state_dict({k:state['out_norm.'+k] for k in ('weight','bias')})
        skill.readout.weight.copy_(state['readout.weight'][ids])
        for i,layer in enumerate(core.layers):
            prefix=f'layers.{i}.'
            for part in ('attn','attn_norm','ffn_norm'):
                target=getattr(layer,part)
                target.load_state_dict({k:state[prefix+part+'.'+k] for k in target.state_dict()})
            for name in ('fc1','fc2'):
                target=getattr(layer.ffn,name)
                target.bias.zero_()
                weight=state[prefix+name+'.weight']
                graft_dense_to_monarch(weight,target)
                if method!='monarch':
                    setattr(layer.ffn,name,CorrectedLinear(target,weight,method))
        if guard.donor_forward_calls:
            raise RuntimeError('donor execution during conversion')
        return skill
