"""Gate 25 DEV donor training and static assimilation into the native byte core."""

from collections import Counter, OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.substrate import DirectMonarchLinear


MODES = ("graft_and_bridge", "graft_only", "shuffled_donor_control", "scratch_nexus")


def fit_byte_bpe(spans: list[bytes], vocab_size: int = 512) -> tuple[list[bytes], list[list[int]]]:
    """Learn raw byte merges from independent spans, then return vocab and training IDs."""
    if vocab_size < 256:
        raise ValueError("BPE must retain all 256 singleton bytes")
    vocab = [bytes([value]) for value in range(256)]
    spellings = set(vocab)
    encoded = [list(span) for span in spans]
    while len(vocab) < vocab_size:
        counts = Counter(pair for span in encoded for pair in zip(span, span[1:]))
        pairs = (pair for pair in counts if vocab[pair[0]] + vocab[pair[1]] not in spellings)
        pair = min(pairs, key=lambda item: (-counts[item], item), default=None)
        if pair is None:
            raise ValueError("training bytes cannot support the requested BPE vocabulary")
        left, right = pair
        new_id = len(vocab)
        vocab.append(vocab[left] + vocab[right])
        spellings.add(vocab[-1])
        for index, span in enumerate(encoded):
            merged = []
            offset = 0
            while offset < len(span):
                if offset + 1 < len(span) and (span[offset], span[offset + 1]) == pair:
                    merged.append(new_id)
                    offset += 2
                else:
                    merged.append(span[offset])
                    offset += 1
            encoded[index] = merged
    return vocab, encoded


@dataclass
class DonorExecution:
    donor_forward_calls: int = 0


_active_guard: ContextVar[DonorExecution | None] = ContextVar("gate25_donor_guard", default=None)


@contextmanager
def forbid_donor_forward():
    """Count and reject donor execution attempts in the guarded assimilation phase."""
    current = _active_guard.get()
    guard = current if current is not None else DonorExecution()
    token = _active_guard.set(guard)
    try:
        yield guard
    finally:
        _active_guard.reset(token)
        if guard.donor_forward_calls:
            raise RuntimeError(f"donor forward attempted during assimilation ({guard.donor_forward_calls})")


class _DenseDonorLayer(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.attn_norm(x)
        attended, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attended
        return x + self.fc2(F.silu(self.fc1(self.ffn_norm(x))))


class DenseTokenDonorLM(nn.Module):
    """Dense causal Transformer trained on byte-BPE IDs for this DEV experiment."""

    def __init__(self, vocab_size=512, dim=128, n_layers=2, n_heads=4, ffn_mult=4, max_seq_len=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.layers = nn.ModuleList(_DenseDonorLayer(dim, n_heads, dim * ffn_mult) for _ in range(n_layers))
        self.out_norm = nn.LayerNorm(dim)
        self.readout = nn.Linear(dim, vocab_size, bias=False)
        self.forward_calls = 0

    def forward(self, token_ids: Tensor) -> Tensor:
        guard = _active_guard.get()
        if guard is not None:
            guard.donor_forward_calls += 1
            raise RuntimeError("donor forward is forbidden during assimilation")
        self.forward_calls += 1
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        mask = positions[:, None] < positions[None, :]
        x = self.emb(token_ids) + self.pos_emb(positions)
        for layer in self.layers:
            x = layer(x, mask)
        return self.readout(self.out_norm(x))

    def export_static(self, vocab_bytes: list[bytes]) -> tuple[dict[str, Tensor], list[bytes]]:
        _validate_vocab(vocab_bytes, self.emb.num_embeddings)
        return {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}, list(vocab_bytes)


def _validate_vocab(vocab_bytes: list[bytes], size: int) -> None:
    if (type(vocab_bytes) is not list or len(vocab_bytes) != size
            or any(type(item) is not bytes or not item for item in vocab_bytes)):
        raise TypeError("vocab_bytes must be a list of nonempty bytes with one entry per row")
    if vocab_bytes[:256] != [bytes([value]) for value in range(256)] or len(set(vocab_bytes)) != size:
        raise ValueError("vocab_bytes must begin with the 256 singleton bytes and contain unique entries")


@dataclass(frozen=True)
class GraftError:
    initial_relative_error: float
    relative_error: float


@torch.no_grad()
def graft_dense_to_monarch(weight: Tensor, target: DirectMonarchLinear, n_iters: int = 8) -> GraftError:
    """Fit one rank-1 block per group pair with a closed-form SVD.

    n_iters is accepted for compatibility with iterative solvers. This exact
    solver needs no iterations.
    """
    if n_iters < 1:
        raise ValueError("n_iters must be positive")
    if tuple(weight.shape) != (target.out_dim, target.in_dim) or not torch.isfinite(weight).all():
        raise ValueError("donor weight must be finite and have the target's dense shape")

    groups = target.n_blocks
    dense = weight.detach().to(device=target.b1.device, dtype=torch.float64)
    blocks = dense.reshape(groups, target.b_out, groups, target.b_in).permute(0, 2, 1, 3)
    norm = float(torch.linalg.vector_norm(dense))
    scale = norm if norm > 0 else 1.0

    def relative_error() -> float:
        actual = torch.einsum("gij,jgo->jgoi", target.b1.double(), target.b2.double())
        return float(torch.linalg.vector_norm(blocks - actual)) / scale

    initial = relative_error()
    left, singular, right = torch.linalg.svd(blocks, full_matrices=False)
    root = singular[..., :1].sqrt()
    target.b1.copy_((right[..., 0, :] * root).permute(1, 2, 0))
    target.b2.copy_(left[..., :, 0] * root)
    return GraftError(initial, relative_error())


def fit_static_embedding_bridge(
    student: NexusExpanseByteCore,
    donor_embed: Tensor,
    vocab_bytes: list[bytes],
    steps: int,
) -> tuple[float, float]:
    """Fit the native causal front-end at each vocabulary spelling's last byte."""
    if steps < 1:
        raise ValueError("bridge steps must be positive")
    if donor_embed.ndim != 2 or donor_embed.shape[1] != student.dim or not torch.isfinite(donor_embed).all():
        raise ValueError("invalid donor embedding geometry or values")
    _validate_vocab(vocab_bytes, len(donor_embed))

    width = max(map(len, vocab_bytes))
    if width > student.max_seq_len:
        raise ValueError("a vocabulary spelling exceeds the student's causal context")
    device = student.emb.weight.device
    byte_rows = torch.tensor([list(spelling.ljust(width, b"\x00")) for spelling in vocab_bytes], device=device)
    ends = torch.tensor([len(spelling) - 1 for spelling in vocab_bytes], device=device)
    positions = torch.arange(width, device=device)
    causal_mask = positions[:, None] < positions[None, :]
    targets = donor_embed.detach().to(student.emb.weight)

    layer = student.layers[0]
    params = [student.emb.weight, student.pos_emb.weight, *layer.attn_norm.parameters(), *layer.attn.parameters()]
    optimizer = torch.optim.Adam(params, lr=0.003)
    order = torch.randperm(len(vocab_bytes), generator=torch.Generator().manual_seed(25)).to(device)
    batches = order.split(32)

    def loss(rows):
        frontend = student.emb(byte_rows[rows]) + student.pos_emb(positions)
        hidden = layer.attention_residual(frontend, causal_mask)
        final = hidden[torch.arange(len(rows), device=device), ends[rows]]
        return F.mse_loss(final, targets[rows])

    def measure():
        with torch.no_grad():
            return sum(float(loss(rows)) * len(rows) for rows in batches) / len(order)

    previous = layer.training
    layer.eval()
    try:
        initial = measure()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss(batches[step % len(batches)]).backward()
            optimizer.step()
        return initial, measure()
    finally:
        optimizer.zero_grad(set_to_none=True)
        layer.train(previous)


@dataclass
class Assimilation:
    donor_forward_calls: int
    reconstruction: dict[str, GraftError] = field(default_factory=dict)
    bridge_mse_before: float | None = None
    bridge_mse_after: float | None = None


def assimilate_zero_run(
    student: NexusExpanseByteCore,
    state_dict: dict[str, Tensor],
    vocab_bytes: list[bytes],
    mode: str = "graft_and_bridge",
    *,
    bridge_steps: int = 96,
    shuffle_seed: int = 0,
) -> Assimilation:
    """Prepare one initialization. Callers perform matched byte-only adaptation afterwards."""
    if mode not in MODES:
        raise ValueError(f"unknown assimilation mode: {mode}")
    if type(state_dict) not in (dict, OrderedDict) or not all(
        type(key) is str and type(value) is Tensor and value.layout == torch.strided
        and value.is_floating_point() and value.device.type == "cpu" and not value.requires_grad
        for key, value in state_dict.items()
    ):
        raise TypeError("assimilation requires an ordinary static CPU tensor state_dict")

    with forbid_donor_forward() as guard:
        donor_embed = state_dict["emb.weight"]
        if donor_embed.ndim != 2 or donor_embed.shape[1] != student.dim:
            raise ValueError("donor embedding and byte core dimensions differ")
        _validate_vocab(vocab_bytes, len(donor_embed))
        result = Assimilation(donor_forward_calls=guard.donor_forward_calls)
        if mode == "scratch_nexus":
            return result

        targets = {}
        for index, layer in enumerate(student.layers):
            for name in ("fc1", "fc2"):
                key = f"layers.{index}.{name}.weight"
                target = getattr(layer.ffn, name)
                if not isinstance(target, DirectMonarchLinear):
                    raise TypeError(f"student {key} must be DirectMonarchLinear")
                weight = state_dict[key]
                if tuple(weight.shape) != (target.out_dim, target.in_dim) or not torch.isfinite(weight).all():
                    raise ValueError(f"invalid donor weight: {key}")
                targets[key] = target

        generator = torch.Generator().manual_seed(shuffle_seed)
        if mode in ("graft_and_bridge", "shuffled_donor_control"):
            if mode == "shuffled_donor_control":
                donor_embed = donor_embed[torch.randperm(len(donor_embed), generator=generator)]
            result.bridge_mse_before, result.bridge_mse_after = fit_static_embedding_bridge(
                student, donor_embed, vocab_bytes, bridge_steps
            )

        for key, target in targets.items():
            weight = state_dict[key]
            if mode == "shuffled_donor_control":
                rows = torch.randperm(weight.shape[0], generator=generator)
                columns = torch.randperm(weight.shape[1], generator=generator)
                weight = weight[rows][:, columns]
            result.reconstruction[key] = graft_dense_to_monarch(weight, target)
        return result
