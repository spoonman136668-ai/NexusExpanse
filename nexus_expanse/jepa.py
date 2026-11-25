"""Tokenless Kuramoto-JEPA, opt-in; never enables FFN skipping."""
import copy
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from nexus_expanse.kuramoto import KuramotoMonarchByteCore
from nexus_expanse.substrate import DirectMonarchLinear


def encoder():
    return KuramotoMonarchByteCore(n_layers=2, dim=64, ffn_mult=2, enable_skip=False)


def latent(model, x, lengths):
    """Last valid causal hidden state; padding is outside the readout."""
    _, h, _ = model.forward_neural(x)
    return h[torch.arange(len(x), device=x.device), lengths - 1]


def stream_latent(model, raw):
    states = model.init_stream_state()
    with torch.inference_mode():
        for byte in raw:
            x = model.emb(torch.tensor([[byte]], device=model.emb.weight.device))
            new = []
            for layer, old in zip(model.layers, states):
                x, z, _ = layer.forward_step(x, old)
                new.append(z)
            states = new
        return model.out_norm(x)[0, 0]


def vicreg(z):
    if z.ndim != 2 or len(z) < 2:
        raise ValueError('VICReg requires at least two examples')
    centered = z - z.mean(0)
    variance = F.relu(1.0 - torch.sqrt(centered.square().sum(0)/(len(z)-1)+1e-4)).mean()
    cov = centered.T @ centered / (len(z)-1)
    covariance = (cov.square().sum() - cov.diagonal().square().sum()) / z.shape[1]
    return variance, covariance


class KuramotoJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = encoder()
        self.target = copy.deepcopy(self.context).requires_grad_(False)
        self.target.eval()
        self.predictor = nn.Sequential(DirectMonarchLinear(64,128,8), nn.SiLU(), DirectMonarchLinear(128,64,8))

    def loss(self, ctx, clen, fut, flen):
        z = latent(self.context, ctx, clen)
        with torch.no_grad():
            tgt = latent(self.target, fut, flen)
        pred = self.predictor(z)
        v1,c1=vicreg(z); v2,c2=vicreg(pred)
        return 25*F.mse_loss(pred,tgt)+25*(v1+v2)+(c1+c2)

    @torch.no_grad()
    def update_target(self):
        for p,q in zip(self.context.parameters(),self.target.parameters()):
            q.mul_(.99).add_(p,alpha=.01)


def state_hash(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):
        h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
