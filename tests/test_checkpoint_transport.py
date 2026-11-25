import torch
from nexus_expanse.checkpoint_transport import CONFIG, METHODS, load_skill
from nexus_expanse.donor_assimilation import DenseTokenDonorLM, forbid_donor_forward


def test_static_transport_and_semantic_reindex(tmp_path):
    torch.manual_seed(13)
    donor=DenseTokenDonorLM(256,32,2,4,2,64).eval()
    ids=torch.randperm(256)
    path=tmp_path/'donor.pt'
    torch.save({'weights':donor.state_dict(),'byte_to_row':ids},path)
    raw=torch.tensor([[1,37,82,125]])
    with torch.inference_mode():
        target=donor(ids[raw])[...,ids]
    with forbid_donor_forward() as guard, torch.no_grad():
        for method in (*METHODS,'residual16'):
            student=load_skill(path,method).eval()
            actual=student(raw)
            assert torch.equal(student.core.emb.weight,donor.emb.weight[ids])
            assert actual.shape==target.shape and torch.isfinite(actual).all()
            if method=='residual16':
                assert (actual-target).abs().max()<.005
    assert guard.donor_forward_calls==0
