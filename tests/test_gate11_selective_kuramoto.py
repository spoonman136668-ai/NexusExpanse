"""Gate 11 engineering and protocol negative controls."""
import torch
from benchmarks.gate10_jepa_corpus import corpus
from benchmarks.gate11_selective_kuramoto_dev import (
    capacity_batches, expanded_train, model_for, parity, classify, needs_expansion, setup)
from nexus_expanse.kuramoto import KuramotoMonarchByteCore
from nexus_expanse.selective_kuramoto import (SelectiveKuramotoByteCore, SelectiveKuramotoLayer,
                                              InfoNCEKuramotoJEPA, contrastive_loss)


def test_selective_shared_initialization_and_state():
    setup()
    torch.manual_seed(42)
    old=KuramotoMonarchByteCore()
    torch.manual_seed(42)
    new=SelectiveKuramotoByteCore()
    for name,tensor in old.state_dict().items():
        assert torch.equal(tensor,new.state_dict()[name]),name
    assert sum(z.numel()*z.element_size() for z in new.init_stream_state()) == 1024
    assert new.layers[0].gate_proj.out_features == 32


def test_gate_can_lock_or_write_and_receive_gradients():
    setup()
    torch.manual_seed(7)
    m=SelectiveKuramotoLayer(16)
    z=m.init_state(2,torch.device("cpu"))
    h=torch.randn(2,16)
    with torch.no_grad():
        m.gate_proj.bias.fill_(-80)
        locked=m._osc_step(h,z)
        m.gate_proj.bias.fill_(80)
        written=m._osc_step(h,z)
        expected=super(SelectiveKuramotoLayer,m)._osc_step(h,z)
        assert torch.allclose(locked,z,atol=2e-12,rtol=0)
        assert torch.equal(written,expected)
        assert not torch.allclose(locked,written)
        m.gate_proj.bias.zero_()
    m._osc_step(h,z)[...,1].sum().backward()
    assert m.gate_proj.weight.grad.abs().sum()>0


def test_exact_parity_50_seeds_both_dimensions():
    setup()
    p=parity(50)
    assert p["passed"],p
    assert p["prefix_logits"]==p["prefix_hidden"]==p["streaming_probabilities"]==p["streaming_hidden"]==0.
    assert p["recurrent_state_bytes"]=={"64":1024,"128":2048}


def test_infonce_positive_vs_permuted_and_target_isolation():
    setup()
    target=torch.eye(8,requires_grad=True)
    pred=torch.eye(8,requires_grad=True)
    good=contrastive_loss(pred,target)
    assert good < contrastive_loss(pred,target.roll(1,0))
    good.backward()
    assert target.grad is None
    torch.manual_seed(42)
    model=InfoNCEKuramotoJEPA()
    model.train()
    assert not model.target.training
    x=torch.tensor([[1,2,3],[2,3,4],[5,6,7]])
    lengths=torch.tensor([3,3,3])
    model.loss(x,lengths,x,lengths).backward()
    assert all(p.grad is None for p in model.target.parameters())
    assert any(p.grad is not None for p in model.context.parameters())
    p,q=next(model.context.parameters()),next(model.target.parameters())
    old=q.detach().clone()
    with torch.no_grad():
        p.add_(1)
    model.update_target(0)
    assert torch.allclose(q,old*.9+p*.1)


def test_capacity_exact_splits_and_expansion():
    setup()
    s=corpus()
    train,ev=s["train"],s["eval"]
    assert len(list(capacity_batches(train,42,1)))==20
    large=expanded_train(train,ev)
    assert len(list(capacity_batches(large,42,1)))==200
    for d in ("math","science"):
        assert sum(r["domain"]==d for r in large)==1200
    assert not {r["ctx"] for r in large}&{r["ctx"] for r in ev}
    assert model_for(42,"capacity","causal_attention_monarch")(*(
        torch.tensor([[1,2,3]]),torch.tensor([3]))).shape==(1,64)


def test_disposition_negative_controls():
    def c(m,s):
        return {"heldout":{"math":{"accuracy":m},"science":{"accuracy":s}},
                "train":{"math":{"accuracy":1.},"science":{"accuracy":1.}}}
    cap={"42":{"selective_kuramoto":c(.45,.70),"ungated_kuramoto":c(.25,.50)}}
    obj={"42":{"infonce_jepa":{"max_stream_vs_batch_diff":0.,"retrieval":{"top1":.25},
                              "speed":{"64":{"speedup":2.},"128":{"speedup":2.}}},
               "next_byte_ce":{"max_stream_vs_batch_diff":0.,"retrieval":{"top1":.10}}}}
    p={"passed":True}; hist={"retrieval":{"top1":.20}}
    assert not needs_expansion(cap)
    assert classify(cap,obj,p,hist)[0]=="PASS"
    obj["42"]["infonce_jepa"]["speed"]["64"]["speedup"]=1.99
    assert classify(cap,obj,p,hist)[0]=="MIXED"
    obj["42"]["infonce_jepa"]["retrieval"]["top1"]=.20
    assert classify(cap,obj,p,hist)[0]=="FAIL"
    cap["42"]["selective_kuramoto"]["train"]["math"]["accuracy"]=.94
    assert needs_expansion(cap)
