import pytest
import torch

from nexus_expanse.checkpoint_transport import lift_dense_state
from nexus_expanse.donor_assimilation import DenseTokenDonorLM


def test_lift_preserves_dense_function_before_structured_approximation():
    torch.manual_seed(27)
    source = DenseTokenDonorLM(256, 32, 2, 4, 2, 64).eval()
    wide = DenseTokenDonorLM(256, 64, 2, 4, 2, 64).eval()
    wide.load_state_dict(lift_dense_state(source.state_dict(), 2))
    with torch.inference_mode():
        for length in (1, 7, 31):
            inputs = torch.randint(256, (3, length))
            torch.testing.assert_close(wide(inputs), source(inputs), rtol=1e-5, atol=1e-5)


def test_width_factor_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        lift_dense_state({}, 0)
