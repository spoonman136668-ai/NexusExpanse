"""Gate 25 DEV donor training and static assimilation into the native byte core."""

from collections import Counter, OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.substrate import DirectMonarchLinear


MODES = ("graft_and_bridge", "graft_only", "shuffled_donor_control", "scratch_nexus")


def fit_byte_bpe(spans: list[bytes], vocab_size: int = 512) -> tuple[list[bytes], list[list[int]]]:
    """Learn raw byte merges from independent spans, then return vocab and training IDs."""
    vocab, encoded, _ = learn_byte_bpe(spans, vocab_size)
    return vocab, encoded


def learn_byte_bpe(spans, vocab_size=512):
    """Return the Gate 25 vocabulary, training IDs, and ordered merge pairs."""
    if vocab_size < 256:
        raise ValueError("BPE must retain all 256 singleton bytes")
    vocab = [bytes([value]) for value in range(256)]
    spellings = set(vocab)
    merges = []
    encoded = [list(span) for span in spans]
    while len(vocab) < vocab_size:
        counts = Counter(pair for span in encoded for pair in zip(span, span[1:]))
        pairs = (pair for pair in counts if vocab[pair[0]] + vocab[pair[1]] not in spellings)
        pair = min(pairs, key=lambda item: (-counts[item], item), default=None)
        if pair is None:
            raise ValueError("training bytes cannot support the requested BPE vocabulary")
        left, right = pair
        merges.append(pair)
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
    return vocab, encoded, merges


def canonical_byte_bpe(raw: bytes, ranks: dict[tuple[int, int], int]) -> tuple[int, ...]:
    """Apply learned BPE merges to observed bytes, with leftmost ties."""
    tokens = list(raw)
    if len(tokens) < 2:
        return tuple(tokens)
    length = len(tokens)
    previous = list(range(-1, length - 1))
    following = list(range(1, length + 1))
    queue = []
    for left in range(length - 1):
        rank = ranks.get((tokens[left], tokens[left + 1]))
        if rank is not None:
            queue.append((rank, left, left + 1))
    heapify(queue)
    while queue:
        rank, left, right = heappop(queue)
        if (following[left] != right or following[right] == -1
                or ranks.get((tokens[left], tokens[right])) != rank):
            continue
        tokens[left] = rank
        successor = following[right]
        following[left] = successor
        following[right] = -1
        if successor != length:
            previous[successor] = left
            new_rank = ranks.get((rank, tokens[successor]))
            if new_rank is not None:
                heappush(queue, (new_rank, left, successor))
        predecessor = previous[left]
        if predecessor != -1:
            new_rank = ranks.get((tokens[predecessor], rank))
            if new_rank is not None:
                heappush(queue, (new_rank, predecessor, left))
    encoded = []
    index = 0
    while index != length:
        encoded.append(tokens[index])
        index = following[index]
    return tuple(encoded)


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

    def forward(self, token_ids: Tensor, *, readout_positions=None) -> Tensor:
        guard = _active_guard.get()
        if guard is not None:
            guard.donor_forward_calls += 1
            raise RuntimeError("donor forward is forbidden during assimilation")
        self.forward_calls += 1
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        mask = positions[:, None] < positions[None, :]
        x = self.emb(token_ids) + self.pos_emb(positions)
        if readout_positions is not None:
            # The additive mask routes large offline batches through fused SDPA.
            mask = x.new_zeros(mask.shape).masked_fill_(mask, -torch.inf)
        for layer in self.layers:
            x = layer(x, mask)
        if readout_positions is not None:
            x = x[readout_positions]
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
    *,
    freeze_attention: bool = False,
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
    params = [student.emb.weight, student.pos_emb.weight]
    if not freeze_attention:
        params.extend(layer.attn_norm.parameters())
        params.extend(layer.attn.parameters())
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
            loss(batches[step % len(batches)]).backward(inputs=params)
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
    copied_max_abs_error: dict[str, float] = field(default_factory=dict)


_FULL_MODES = ("zero_run_full_graft", "shuffled_full_graft")


def _shuffle_matrix(weight, generator):
    rows = torch.randperm(weight.shape[0], generator=generator)
    columns = torch.randperm(weight.shape[1], generator=generator)
    return weight[rows][:, columns]


def _dense_copies(student, state_dict, shuffle):
    names = ("attn.in_proj_weight", "attn.in_proj_bias", "attn.out_proj.weight", "attn.out_proj.bias",
             "attn_norm.weight", "attn_norm.bias", "ffn_norm.weight", "ffn_norm.bias")
    targets = {}
    for index, layer in enumerate(student.layers):
        tensors = {}
        for name in names:
            key = f"layers.{index}.{name}"
            parameter = layer.get_parameter(name)
            tensor = state_dict[key]
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"invalid donor dense tensor: {key}")
            tensors[name] = tensor
        if shuffle is not None:
            dim = student.dim
            for projection, chunks in (("attn.in_proj", 3), ("attn.out_proj", 1)):
                separator = "_" if chunks == 3 else "."
                weight_key, bias_key = (projection + separator + name for name in ("weight", "bias"))
                columns = torch.randperm(dim, generator=shuffle)
                rows = torch.cat([part * dim + torch.randperm(dim, generator=shuffle)
                                  for part in range(chunks)])
                tensors[weight_key] = tensors[weight_key][rows][:, columns]
                tensors[bias_key] = tensors[bias_key][rows]
            for name in ("attn_norm", "ffn_norm"):
                rows = torch.randperm(dim, generator=shuffle)
                for kind in ("weight", "bias"):
                    key = f"{name}.{kind}"
                    tensors[key] = tensors[key][rows]
        for name, tensor in tensors.items():
            targets[f"layers.{index}.{name}"] = layer.get_parameter(name), tensor
    return targets


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
    if mode not in (*MODES, *_FULL_MODES):
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

        full = mode in _FULL_MODES
        shuffled = mode in ("shuffled_donor_control", "shuffled_full_graft")
        generator = torch.Generator().manual_seed(shuffle_seed)
        copies = {}
        if full:
            if any(layer.ffn.enable_skip for layer in student.layers):
                raise ValueError("full graft requires enable_skip=False")
            copies = _dense_copies(student, state_dict, generator if shuffled else None)
            with torch.no_grad():
                for parameter, expected in copies.values():
                    parameter.copy_(expected)

        if full or mode in ("graft_and_bridge", "shuffled_donor_control"):
            if shuffled:
                donor_embed = donor_embed[torch.randperm(len(donor_embed), generator=generator)]
            result.bridge_mse_before, result.bridge_mse_after = fit_static_embedding_bridge(
                student, donor_embed, vocab_bytes, bridge_steps, freeze_attention=full
            )

        for key, target in targets.items():
            weight = state_dict[key]
            if shuffled:
                weight = _shuffle_matrix(weight, generator)
            result.reconstruction[key] = graft_dense_to_monarch(weight, target)
        result.copied_max_abs_error = {
            key: float((actual.detach().cpu() - expected).abs().max()) for key, (actual, expected) in copies.items()
        }
        if any(result.copied_max_abs_error.values()):
            raise RuntimeError("full-graft dense tensor changed during static initialization")
        return result


_BYTE_CACHE_HEADER = struct.Struct("<8sIB32s")
_BYTE_CACHE_MAGIC = b"NXBKL01\x00"


def _span_digest(spans):
    hasher = hashlib.sha256()
    for raw in spans:
        if type(raw) is not bytes or not raw:
            raise ValueError("byte-cache spans must be nonempty bytes")
        hasher.update(struct.pack("<I", len(raw)))
        hasher.update(raw)
    return hasher.digest()


def _prefix_token_table(vocab):
    """Index the strict trie prefixes by matching token ID and next byte."""
    trie = {}
    for token, spelling in enumerate(vocab):
        for index, byte in enumerate(spelling):
            matches = trie.setdefault(spelling[:index], ([], []))
            matches[0].append(token)
            matches[1].append(byte)
    return {prefix: (torch.tensor(tokens), torch.tensor(values))
            for prefix, (tokens, values) in trie.items()}


@torch.inference_mode()
def byte_marginalized_probabilities(donor, spans, vocab_bytes, merges, first_token_prior):
    """One donor call, summing trie continuations over one canonical history per boundary."""
    _validate_vocab(vocab_bytes, donor.emb.num_embeddings)
    if donor.training or any(parameter.requires_grad for parameter in donor.parameters()):
        raise ValueError("offline cache requires a frozen evaluation donor")
    if not spans or not all(type(raw) is bytes and raw for raw in spans):
        raise ValueError("cannot score empty byte spans")
    prior = torch.as_tensor(first_token_prior, dtype=torch.float64, device="cpu")
    if (prior.shape != (len(vocab_bytes),) or not torch.isfinite(prior).all()
            or not (prior > 0).all() or abs(float(prior.sum()) - 1.0) > 1e-10):
        raise ValueError("first-token prior must be a strictly positive distribution")
    ranks = {pair: i for i, pair in enumerate(merges, 256)}
    if (len(ranks) + 256 != len(vocab_bytes)
            or any(not (0 <= left < i and 0 <= right < i)
                   or vocab_bytes[left] + vocab_bytes[right] != vocab_bytes[i]
                   for (left, right), i in ranks.items())):
        raise ValueError("BPE merge ranks must reproduce the donor vocabulary")

    histories = [[canonical_byte_bpe(raw[:t], ranks) for t in range(len(raw))] for raw in spans]
    all_paths = sorted({path for span in histories for path in span if path})
    # Even one-byte fixtures need the one offline call. Its output is unused at BOS.
    if not all_paths:
        all_paths = [(0,)]
    leaves = [path for i, path in enumerate(all_paths)
              if i + 1 == len(all_paths) or all_paths[i + 1][:len(path)] != path]
    width = max(map(len, leaves))
    if width > donor.pos_emb.num_embeddings:
        raise ValueError("canonical byte prefix exceeds the donor context")
    coordinates = {}
    for row, leaf in enumerate(leaves):
        for col in range(len(leaf)):
            coordinates.setdefault(leaf[:col + 1], (row, col))
    ordered = sorted(coordinates, key=len)
    rows, cols = zip(*(coordinates[path] for path in ordered))
    padded = torch.tensor([path + (0,) * (width - len(path)) for path in leaves])
    selected = (torch.tensor(rows), torch.tensor(cols))

    before = donor.forward_calls
    log_probs = donor(padded, readout_positions=selected).double().log_softmax(-1)
    if donor.forward_calls - before != 1 or not torch.isfinite(log_probs).all():
        raise RuntimeError("offline donor evaluation must produce finite logits in one call")
    log_probs = torch.cat((prior.log().unsqueeze(0), log_probs))
    probabilities = log_probs.exp()
    ids = {path: i for i, path in enumerate(ordered, 1)}
    ids[()] = 0
    joints = [0.0]
    for path in ordered:
        parent = ids[path[:-1]]
        joints.append(joints[parent] + float(log_probs[parent, path[-1]]))

    trie = _prefix_token_table(vocab_bytes)
    max_spelling = max(map(len, vocab_bytes))
    result = []
    for raw, paths in zip(spans, histories, strict=True):
        span_probs = torch.zeros(len(raw), 256, dtype=torch.float64)
        for t, byte_mass in enumerate(span_probs):
            events = []
            for start in range(max(0, t - max_spelling + 1), t + 1):
                match = trie.get(raw[start:t])
                if match is not None:
                    events.append((ids[paths[start]], match))
            largest = max(joints[context] for context, _ in events)
            for context, (tokens, next_bytes) in events:
                scale = math.exp(joints[context] - largest)
                byte_mass.index_add_(0, next_bytes, probabilities[context, tokens] * scale)
            byte_mass /= byte_mass.sum()
        result.append(span_probs)
    return result, len(leaves)


def export_offline_byte_kl_cache(donor, path, spans, vocab_bytes, merges, first_token_prior, *, top_k=16):
    """Publish only ranked byte IDs and float16 masses, bound to ordered FIT bytes."""
    if not 1 <= top_k < 256:
        raise ValueError("top_k must be between 1 and 255")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    source = _span_digest(spans)
    before = donor.forward_calls
    distributions, donor_rows = byte_marginalized_probabilities(donor, spans, vocab_bytes, merges, first_token_prior)
    dense = torch.cat(distributions)
    error = float((dense.sum(-1) - 1.0).abs().max())
    if not torch.isfinite(dense).all() or error > 1e-5:
        raise ValueError("offline byte distributions failed normalization")
    top_ids = dense.argsort(dim=-1, descending=True, stable=True)[:, :top_k]
    top_probs = dense.gather(-1, top_ids)
    indices = top_ids.to(torch.uint8).numpy().tobytes()
    masses = top_probs.numpy().astype("<f2").tobytes()
    payload = _BYTE_CACHE_HEADER.pack(_BYTE_CACHE_MAGIC, len(dense), top_k, source) + indices + masses
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.") as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        os.link(temporary.name, path)
    return dict(sha256=hashlib.sha256(payload).hexdigest(), cache_bytes=len(payload), top_k=top_k,
                positions=len(dense), donor_forward_calls_offline=donor.forward_calls - before,
                donor_batch_rows=donor_rows, max_normalization_error=error,
                mean_top_k_mass=float(top_probs.sum(-1).mean()))


def load_offline_byte_kl_cache(path, spans, sha256, *, top_k=16):
    """Verify cache custody and return sparse distributions in FIT span order."""
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != sha256 or len(content) < _BYTE_CACHE_HEADER.size:
        raise ValueError("offline byte cache digest mismatch")
    magic, rows, k, source = _BYTE_CACHE_HEADER.unpack_from(content)
    header = _BYTE_CACHE_HEADER.size
    if (magic != _BYTE_CACHE_MAGIC or source != _span_digest(spans) or k != top_k
            or not 1 <= k < 256 or rows != sum(map(len, spans)) or len(content) != header + 3 * rows * k):
        raise ValueError("offline byte cache geometry or FIT identity mismatch")
    shape = (rows, k)
    byte_ids = torch.from_numpy(np.frombuffer(content, dtype=np.uint8, count=rows * k, offset=header)
                                .astype(np.int64).reshape(shape))
    probs = torch.from_numpy(np.frombuffer(content, dtype="<f2", offset=header + rows * k)
                             .astype(np.float32).reshape(shape))
    sorted_ids = byte_ids.sort(-1).values
    totals = probs.sum(-1, keepdim=True)
    if (not torch.isfinite(probs).all() or (probs < 0).any() or (totals <= 0).any()
            or (sorted_ids[:, :-1] == sorted_ids[:, 1:]).any()):
        raise ValueError("invalid offline byte cache probabilities or indices")
    lengths = tuple(map(len, spans))
    return list(zip(byte_ids.split(lengths), (probs / totals).split(lengths), strict=True))


def sparse_byte_kl(logits, byte_ids, probs):
    """KL from the sparse normalized byte teacher to the full student softmax."""
    student = logits.log_softmax(-1).gather(-1, byte_ids)
    return (torch.special.xlogy(probs, probs) - probs * student).sum(-1).mean()
