"""Exact raw-logit checks of prefix invariance and the constant-state stream."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nexus_expanse.kuramoto import KuramotoMonarchByteCore


@torch.inference_mode()
def sweep(seeds=100):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prefix_error=stream_error=0.
    failures=[]
    prefixes=steps=0
    for seed in range(seeds):
        torch.manual_seed(seed)
        m=KuramotoMonarchByteCore().eval()
        for layer in m.layers:
            layer.ffn.tau_eval.fill_(0.)
        length=269 if seed==0 else 24
        x=torch.randint(0,256,(1,length))
        full=m.forward_neural(x)[0]
        stream=m.new_stream()
        se=pe=0.
        for i,b in enumerate(x[0].tolist()):
            y=stream.append(b)
            se=max(se,(full[0,i]-y).abs().max().item())
            assert torch.isfinite(y).all() and stream.tensor_bytes==512
            assert all((z.norm(dim=-1)-1).abs().max()<1e-6 for z in stream.phases)
        for end in (1,7,13,length-1):
            truncated=m.forward_neural(x[:,:end])[0]
            changed=m.forward_neural(torch.cat((x[:,:end],255-x[:,end:]),1))[0]
            pe=max(pe,(full[:,:end]-truncated).abs().max().item(),
                   (full[:,:end]-changed[:,:end]).abs().max().item())
        if se or pe:
            failures.append(seed)
        steps+=length
        prefixes+=8
        prefix_error=max(prefix_error,pe)
        stream_error=max(stream_error,se)
    return dict(seeds=seeds,prefix_checks=prefixes,stream_steps=steps,prefix_max_diff=prefix_error,
                stream_max_diff=stream_error,state_bytes=512,fails=failures)


if __name__=='__main__':
    result=sweep()
    print(json.dumps(result))
    sys.exit(bool(result['fails']))
