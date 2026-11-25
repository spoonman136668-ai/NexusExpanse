import torch

from benchmarks.gate15_hybrid_dev import decision, parity
from nexus_expanse.hybrid_backbone import HybridDeltaAttentionCore


def test_exact_default_hybrid_and_state_footprint():
    torch.set_num_threads(1)
    result = parity(2)
    assert result["passed"], result
    assert [r["state_tensor_bytes"] for r in result["runs"]] == [139264, 139264, 294912, 294912]


def test_optional_schedule_rollover_and_independent_rows():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    net = HybridDeltaAttentionCore(dim=32, max_seq_len=8, schedule=("delta", "delta", "delta", "attention")).eval()
    a, b = net.new_stream(), net.new_stream()
    tokens = torch.arange(24).reshape(2, 12)
    for row, stream in zip(tokens, (a, b), strict=True):
        for i, byte in enumerate(row.tolist(), 1):
            out, hidden = stream.step(byte)
            expected, latent, _ = net.forward_neural(row[max(0, i - 8):i][None, :])
            assert torch.equal(out, expected[0, -1])
            assert torch.equal(hidden, latent[0, -1])
    batch = net.forward_neural(tokens[:, -8:])[0]
    assert torch.equal(batch[0], net.forward_neural(tokens[0:1, -8:])[0][0])


def test_train_and_stream_paths_agree_and_backpropagate():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    model = HybridDeltaAttentionCore(dim=32, max_seq_len=16)
    tokens = torch.randint(256, (2, 11))
    train = model.forward_neural(tokens)[0]
    train.square().mean().backward()
    assert model.layers[0].input_proj.weight.grad.abs().sum() > 0
    assert model.layers[1].attn.in_proj_weight.grad.abs().sum() > 0
    torch.testing.assert_close(train.detach(), model.eval().forward_neural(tokens)[0], atol=2e-5, rtol=1e-5)


def test_gate15_decision_preserves_fail_floor():
    def run(math, causal, speed, reduction):
        return dict(capacity=dict(heldout=dict(math=dict(accuracy=math), science=dict(accuracy=causal))),
                    timing={"64": dict(speedup=speed, cache_reduction=reduction)})
    assert decision([run(.95, .95, 1.5, .40)], True)["status"] == "PASS"
    assert decision([run(.90, .95, 1.5, .40)], True)["status"] == "MIXED"
    assert decision([run(.85, .95, 2., .50)], True)["status"] == "FAIL"
    assert decision([run(.95, .95, 1., .50)], True)["status"] == "FAIL"
