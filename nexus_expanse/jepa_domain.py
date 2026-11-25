"""Gate 10 opt-in per-domain VICReg and cosine EMA, unchanged raw-byte core."""
import math
import torch
from torch.nn import functional as F
from nexus_expanse.jepa import KuramotoJEPA, latent, vicreg


def momentum(step, total=300):
    if not 0 <= step < total or total < 2:
        raise ValueError('invalid EMA step')
    return .90 + .09 * (1 - math.cos(math.pi * step / (total-1))) / 2


def domain_vicreg(z, domains):
    if domains.shape != (len(z),) or not bool(((domains>=0)&(domains<3)).all()):
        raise ValueError('invalid domains')
    terms=[vicreg(z[domains==d]) for d in range(3)]
    return tuple(torch.stack([t[i] for t in terms]).mean() for i in range(2))


class DomainKuramotoJEPA(KuramotoJEPA):
    def loss(self, ctx, clen, fut, flen, domains):
        z=latent(self.context,ctx,clen)
        with torch.no_grad(): tgt=latent(self.target,fut,flen)
        pred=self.predictor(z)
        v1,c1=domain_vicreg(z,domains); v2,c2=domain_vicreg(pred,domains)
        return 25*F.mse_loss(pred,tgt)+25*(v1+v2)+c1+c2

    @torch.no_grad()
    def update_target(self, step):
        m=momentum(step)
        for p,q in zip(self.context.parameters(),self.target.parameters()):
            q.mul_(m).add_(p,alpha=1-m)
