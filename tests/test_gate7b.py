import torch
from nexus_expanse.kuramoto_skip import CausalKuramotoMonarchByteCore,random_deep


def test_causal_skip_and_streaming_smoke():
    torch.manual_seed(42); torch.set_num_threads(1)
    m=CausalKuramotoMonarchByteCore(dim=16,n_layers=2).eval()
    x=torch.tensor([[1,2,3,4,5,6]])
    m.calibrate_gates([(x[:,:-1],x[:,1:])])
    for policy in ('always','causal','random'):
        m.policy=policy
        with torch.inference_mode():
            l=m.forward_neural(x)[0]
            assert torch.equal(l[:,:3],m.forward_neural(x[:,:3])[0])
            z=m.init_stream_state()
            for i,b in enumerate(x[0].tolist()):
                p,z,_=m.stream_step(b,z,i)
                assert torch.equal(p,l[0,i].double().softmax(0))


def test_skip_extremes_and_fold_state():
    m=CausalKuramotoMonarchByteCore(dim=16).eval()
    for layer in m.layers: layer.ffn.tau_eval.fill_(float('inf'))
    p,z,d=m.stream_step(7,m.init_stream_state(),0)
    q,w,e=m.stream_step(7,m.init_stream_state(),0,readout=False)
    assert q is None and not any(d) and d==e
    assert all(torch.equal(a,b) for a,b in zip(z,w))
    assert 400<sum(random_deep(42,i,0) for i in range(1000))<600


def test_train_aux_gradient_and_calibration():
    torch.manual_seed(7)
    m=CausalKuramotoMonarchByteCore(dim=16)
    x=torch.tensor([[1,2,3,4]]); y=torch.tensor([[2,3,4,5]])
    l,a=m.forward_train(x,y)
    (torch.nn.functional.cross_entropy(l.reshape(-1,256),y.flatten())+.1*a).backward()
    assert all(layer.ffn.gate_proj.weight.grad is not None for layer in m.layers)
    m.calibrate_gates([(x,y)])
    assert all(torch.isfinite(layer.ffn.tau_eval) for layer in m.layers)
    assert m.training
