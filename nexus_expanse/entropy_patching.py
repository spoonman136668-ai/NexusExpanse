"""Raw-byte prediction from a causal local window and completed patch latents."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nexus_expanse.core import NexusLayer


class CausalBytePatcher(nn.Module):
    """Frozen add-one byte bigrams fitted on explicitly supplied training spans."""

    def __init__(self, training_bytes=(), *, entropy_threshold_bits=3.5, max_patch_bytes=8):
        super().__init__()
        if not math.isfinite(entropy_threshold_bits) or entropy_threshold_bits < 0:
            raise ValueError("entropy threshold must be finite and nonnegative")
        if type(max_patch_bytes) is not int or max_patch_bytes < 1:
            raise ValueError("max_patch_bytes must be a positive integer")
        self.entropy_threshold_bits = entropy_threshold_bits
        self.max_patch_bytes = max_patch_bytes
        counts = np.ones(256 * 256, dtype=np.int64)
        for raw in (training_bytes,) if isinstance(training_bytes, bytes) else training_bytes:
            values = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
            counts += np.bincount(values[:-1] * 256 + values[1:], minlength=counts.size)
        counts = counts.reshape(256, 256)
        self.register_buffer("surprise_bits", torch.from_numpy(-np.log2(counts / counts.sum(1, keepdims=True))))

    def starts(self, tokens):
        """Return opening positions, including a forced start at position zero."""
        surprises = self.surprise_bits[tokens[:, :-1], tokens[:, 1:]]
        forced = F.pad(surprises >= self.entropy_threshold_bits, (1, 0), value=True)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        last_forced = torch.where(forced, positions, 0).cummax(dim=1).values
        return (positions - last_forced).remainder(self.max_patch_bytes) == 0

    def surprising(self, previous, current):
        return bool(self.surprise_bits[previous, current] >= self.entropy_threshold_bits)


@dataclass(frozen=True)
class PatchTrace:
    starts: torch.Tensor
    completed_ends: torch.Tensor  # Inclusive last byte. Padding is -1.
    latents: torch.Tensor
    visible_latents: torch.Tensor  # Completed-patch context used by each byte.


class EntropyPatchedByteCore(nn.Module):
    """Encode bytes locally and attend only over the completed patch sequence."""

    PATCH_BLOCK = 16
    BYTE_BLOCK = 32

    def __init__(self, *, dim=32, n_layers=2, n_heads=4, ffn_mult=2, max_seq_len=1024,
                 local_kernel_size=3, entropy_threshold_bits=3.5, max_patch_bytes=8, training_bytes=()):
        super().__init__()
        geometry = (dim, n_layers, n_heads, ffn_mult, max_seq_len, local_kernel_size)
        if any(type(size) is not int or size < 1 for size in geometry) or dim % n_heads:
            raise ValueError("invalid byte-core geometry")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.patcher = CausalBytePatcher(training_bytes, entropy_threshold_bits=entropy_threshold_bits,
                                        max_patch_bytes=max_patch_bytes)
        self.pool_width = min(max_patch_bytes, max_seq_len)
        self.emb = nn.Embedding(256, dim)
        self.patch_position = nn.Embedding(max_seq_len, dim)
        self.local_kernel = nn.Parameter(torch.full((local_kernel_size, dim), 1.0 / local_kernel_size))
        self.layers = nn.ModuleList(NexusLayer(dim, n_heads, ffn_mult, n_blocks=1, target_skip_rate=0.0,
                                              enable_skip=False, dropout=0.0, ffn_kind="dense", use_sdpa=True)
                                    for _ in range(n_layers))
        self.out_norm = nn.LayerNorm(dim)
        self.readout = nn.Linear(dim, 256, bias=False)

    def _tokens(self, tokens):
        if (tokens.ndim != 2 or not tokens.shape[0] or not 1 <= tokens.shape[1] <= self.max_seq_len
                or tokens.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
            raise ValueError("expected a nonempty (batch, time) tensor of raw bytes within the context limit")
        low, high = torch.aminmax(tokens)
        if int(low) < 0 or int(high) > 255:
            raise ValueError("raw byte values must be between 0 and 255")
        return tokens.long()

    def _local(self, lags):
        state = lags[0] * self.local_kernel[0]
        for index, lag in enumerate(lags[1:], 1):
            state = state + lag * self.local_kernel[index]
        return state

    def _pool(self, padded, lengths):
        return padded.sum(dim=-2) / lengths.clamp_min(1).unsqueeze(-1)

    def _patch_block(self, pooled, start, cache):
        # Absolute patch blocks use the same kernel shape for any suffix. The
        # causal mask hides later slots, including the padding of the last block.
        positions = torch.arange(start, start + self.PATCH_BLOCK, device=pooled.device)
        mask = torch.arange(start + self.PATCH_BLOCK, device=pooled.device) <= positions[:, None]
        x = pooled + self.patch_position(positions.clamp_max(self.max_seq_len - 1))
        history = []
        for layer, previous in zip(self.layers, cache, strict=True):
            normalized = layer.attn_norm(x)
            packed = F.linear(normalized, layer.attn.in_proj_weight, layer.attn.in_proj_bias)
            q, k, v = (part.unflatten(-1, (self.n_heads, self.dim // self.n_heads)).transpose(1, 2)
                       for part in packed.chunk(3, dim=-1))
            if previous is not None:
                k = torch.cat((previous[0], k), dim=-2)
                v = torch.cat((previous[1], v), dim=-2)
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = x + layer.attn.out_proj(context.transpose(1, 2).reshape_as(x))
            x = x + layer.ffn.fc2(F.silu(layer.ffn.fc1(layer.ffn_norm(x))))
            history.append((k, v))
        return x, history

    def _completed(self, embedded, starts):
        batch, length, _ = embedded.shape
        groups = starts.long().cumsum(dim=1) - 1
        rows, positions = starts.nonzero(as_tuple=True)
        table = groups.new_full((batch, int(groups[:, -1].max()) + 2), length)
        table[rows, groups[rows, positions]] = positions
        lengths = table[:, 1:] - table[:, :-1]
        closed = (table[:, 1:] < length) | (lengths == self.patcher.max_patch_bytes)
        count = int(closed.sum(dim=1).max())
        if not count:
            return groups, table[:, :0], embedded[:, :0]
        offsets = torch.arange(self.pool_width, device=embedded.device)
        indices = (table[:, :count, None] + offsets).clamp_max(length - 1)
        features = embedded[torch.arange(batch, device=embedded.device)[:, None, None], indices]
        lengths = lengths[:, :count].masked_fill(~closed[:, :count], 0)
        padded = features.masked_fill((offsets >= lengths.unsqueeze(-1)).unsqueeze(-1), 0)
        pooled = self._pool(padded, lengths)

        cache = [None] * len(self.layers)
        encoded = []
        for start in range(0, count, self.PATCH_BLOCK):
            block = pooled[:, start:start + self.PATCH_BLOCK]
            state, cache = self._patch_block(F.pad(block, (0, 0, 0, self.PATCH_BLOCK - block.shape[1])), start, cache)
            encoded.append(state[:, :block.shape[1]])
        present = closed[:, :count]
        ends = (table[:, 1:count + 1] - 1).masked_fill(~present, -1)
        return groups, ends, torch.cat(encoded, dim=1).masked_fill(~present.unsqueeze(-1), 0)

    def _decode(self, local, context):
        hidden = self.out_norm(local + context)
        return self.readout(hidden), hidden

    def _forward(self, tokens):
        tokens = self._tokens(tokens)
        length = tokens.shape[1]
        embedded = self.emb(tokens)
        lags = [embedded, *(F.pad(embedded, (0, 0, lag, 0))[:, :length]
                            for lag in range(1, len(self.local_kernel)))]
        local = self._local(lags)
        starts = self.patcher.starts(tokens)
        groups, ends, latents = self._completed(embedded, starts)
        completed = F.pad(latents, (0, 0, 1, 0))
        # Group zero sees no patch. Group j sees the latent of group j - 1.
        context = completed.gather(1, groups.unsqueeze(-1).expand(-1, -1, self.dim))
        logits, hidden = [], []
        for loc, ctx in zip(local.split(self.BYTE_BLOCK, dim=1), context.split(self.BYTE_BLOCK, dim=1), strict=True):
            pad = (0, 0, 0, self.BYTE_BLOCK - loc.shape[1])
            decoded, state = self._decode(F.pad(loc, pad), F.pad(ctx, pad))
            logits.append(decoded[:, :loc.shape[1]])
            hidden.append(state[:, :loc.shape[1]])
        return torch.cat(logits, dim=1), torch.cat(hidden, dim=1), PatchTrace(starts, ends, latents, context)

    def forward_neural(self, tokens):
        logits, hidden, _ = self._forward(tokens)
        return logits, hidden

    forward = forward_neural

    def patch_trace(self, tokens):
        return self._forward(tokens)[2]

    def patch_stats(self, tokens):
        tokens = self._tokens(tokens)
        patches = int(self.patcher.starts(tokens).sum())
        return dict(byte_count=tokens.numel(), patch_count=patches, compression_ratio=tokens.numel() / patches)

    def new_stream(self):
        if self.training:
            raise ValueError("streaming requires an eval model")
        return EntropyPatchStream(self)


class EntropyPatchStream:
    """Incremental byte window, one open patch, and immutable KV blocks."""

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.window = deque(maxlen=len(self.model.local_kernel))
        self.open_patch = []
        self.block = []
        self.cache = [None] * len(self.model.layers)
        self.byte_count = self.completed_count = 0
        self.previous = None
        self.latest = self.model.emb.weight.new_zeros(self.model.dim)

    def _close_patch(self):
        model = self.model
        size = len(self.open_patch)
        padded = F.pad(torch.stack(self.open_patch), (0, 0, 0, model.pool_width - size))
        self.block.append(model._pool(padded, padded.new_tensor(size)))
        self.open_patch.clear()
        pooled = F.pad(torch.stack(self.block)[None], (0, 0, 0, model.PATCH_BLOCK - len(self.block)))
        state, cache = model._patch_block(pooled, self.completed_count - len(self.block) + 1, self.cache)
        self.latest = state[0, len(self.block) - 1].clone()
        self.completed_count += 1
        if len(self.block) == model.PATCH_BLOCK:
            self.cache = cache
            self.block.clear()

    @torch.inference_mode()
    def append(self, byte):
        model = self.model
        if type(byte) is not int or not 0 <= byte <= 255:
            raise ValueError("expected a raw byte")
        if self.byte_count == model.max_seq_len:
            raise ValueError("stream exceeds context limit")
        if self.open_patch and model.patcher.surprising(self.previous, byte):
            self._close_patch()
        embedded = model.emb.weight[byte]
        self.window.appendleft(embedded)
        zeros = torch.zeros_like(embedded)
        local = model._local([*self.window, *([zeros] * (len(model.local_kernel) - len(self.window)))])
        offset = self.byte_count % model.BYTE_BLOCK
        pad = (0, 0, offset, model.BYTE_BLOCK - offset - 1)
        logits, hidden = model._decode(F.pad(local[None, None], pad), F.pad(self.latest[None, None], pad))
        self.open_patch.append(embedded)
        if len(self.open_patch) == model.patcher.max_patch_bytes:
            self._close_patch()
        self.previous = byte
        self.byte_count += 1
        return logits[0, offset], hidden[0, offset]
