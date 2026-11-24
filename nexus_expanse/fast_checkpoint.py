"""Opt-in bounded fast-weight checkpoints, bound to an exact neural backbone.

Checksums detect accidental corruption, not malicious replacement or provenance.
Only CPU float32 fast-state persistence is qualified by the development tests.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import torch
from nexus_expanse.memory import HebbianFastWeightAdapter, ContextBankFastWeightAdapter, ProjectedFastWeightAdapter

_MAGIC = b'NEXUS-FAST-1\n'


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def _tensor_bytes(t):
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def backbone_identity(model):
    """Hash tensor content, neural topology and non-tensor inference settings."""
    settings = {'dim': model.dim, 'max_seq_len': model.max_seq_len,
                'layers': [{'heads': l.attn.num_heads, 'skip': l.ffn.enable_skip,
                            'ffn': type(l.ffn.fc1).__name__, 'repr': repr(l)} for l in model.layers],
                'schema': 'nexus-neural-identity-v1', 'torch': str(torch.__version__),
                'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ('core.py', 'substrate.py', 'memory.py', 'fast_checkpoint.py')}}
    h = hashlib.sha256(_canonical(settings))
    for name, t in sorted(model.state_dict().items()):
        if name.startswith('fast_weights.'):
            continue
        h.update(_canonical([name, str(t.dtype), list(t.shape)]))
        h.update(_tensor_bytes(t))
    return h.hexdigest()


def _config(model):
    a = model.fast_weights
    if (any(t.device.type != 'cpu' for t in model.state_dict().values()) or
            a.w_fast.dtype != torch.float32 or a.w_fast.device.type != 'cpu' or
            a.vocab_size != 256 or tuple(a.w_fast.shape) != (model.dim, 256) or
            a.dim != model.dim or not math.isfinite(a.lr) or a.lr <= 0 or
            not math.isfinite(a.decay) or not 0 <= a.decay <= 1):
        raise ValueError('unsupported fast-weight configuration (CPU float32 required)')
    cfg = {'dim': a.dim, 'vocab': a.vocab_size, 'lr': a.lr, 'decay': a.decay,
           'backbone': backbone_identity(model)}
    if type(a) is ContextBankFastWeightAdapter:
        cfg['routing'] = {'kind': 'blake2s-suffix-v1', 'banks': a.banks, 'order': a.context_order}
    elif type(a) is ProjectedFastWeightAdapter:
        if a.decay != 1. or a.basis.dtype != torch.float32 or tuple(a.basis.shape) != (a.rank, a.dim):
            raise ValueError('invalid protection configuration')
        cfg['protection'] = {'kind': 'causal-orthogonal-v1', 'rank': a.rank}
    elif type(a) is not HebbianFastWeightAdapter:
        raise ValueError('unsupported fast adapter')
    return cfg


def _budget(max_bytes):
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('positive byte budget required')


def _validate_basis(b):
    if not bool(torch.isfinite(b).all()):
        raise ValueError('nonfinite protection basis')
    active = b.square().sum(dim=1) != 0
    if bool((active[1:] & ~active[:-1]).any()):
        raise ValueError('noncanonical protection basis')
    q = b[active]
    if not torch.allclose(q @ q.T, torch.eye(len(q), dtype=b.dtype), atol=1e-5, rtol=0):
        raise ValueError('nonorthonormal protection basis')


def snapshot_fast_weights(model, *, max_bytes: int) -> bytes:
    _budget(max_bytes)
    a = model.fast_weights
    cfg = _config(model)
    if type(a.updates_applied) is not int or not 0 <= a.updates_applied < 2**63:
        raise ValueError('invalid update count')
    if not bool(torch.isfinite(a.w_fast).all()) or bool((a.w_fast.abs() > 2).any()):
        raise ValueError('invalid fast-weight values')
    header = _canonical({'config': cfg, 'updates': a.updates_applied})
    raw = a.w_fast.detach().numpy().astype('<f4', copy=False).tobytes()
    if type(a) is ProjectedFastWeightAdapter:
        _validate_basis(a.basis)
        raw += a.basis.detach().numpy().astype('<f4', copy=False).tobytes()
    body = _MAGIC + struct.pack('<I', len(header)) + header + raw
    if len(body) + 32 > max_bytes:
        raise ValueError('fast-weight checkpoint exceeds byte budget')
    return body + hashlib.sha256(body).digest()


@torch.no_grad()
def restore_fast_weights(model, payload: bytes, *, max_bytes: int):
    """Validate completely before mutating the destination model."""
    _budget(max_bytes)
    if not isinstance(payload, bytes) or len(payload) > max_bytes or len(payload) < len(_MAGIC) + 36:
        raise ValueError('invalid checkpoint size')
    body, digest = payload[:-32], payload[-32:]
    if not body.startswith(_MAGIC) or hashlib.sha256(body).digest() != digest:
        raise ValueError('invalid checkpoint magic or checksum')
    offset = len(_MAGIC) + 4
    n = struct.unpack_from('<I', body, len(_MAGIC))[0]
    if n > 4096 or offset + n > len(body):
        raise ValueError('invalid checkpoint header length')
    try:
        header = json.loads(body[offset:offset+n])
        if not isinstance(header, dict) or set(header) != {'config', 'updates'} or _canonical(header) != body[offset:offset+n]:
            raise ValueError('invalid or noncanonical checkpoint header')
    except (TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        raise ValueError('invalid checkpoint header') from exc
    if header['config'] != _config(model):
        raise ValueError('checkpoint backbone or adapter configuration mismatch')
    updates = header['updates']
    if type(updates) is not int or not 0 <= updates < 2**63:
        raise ValueError('invalid update count')
    raw = body[offset+n:]
    a = model.fast_weights
    wbytes = a.w_fast.numel() * 4
    bbytes = a.basis.numel() * 4 if type(a) is ProjectedFastWeightAdapter else 0
    if len(raw) != wbytes + bbytes:
        raise ValueError('invalid tensor length')
    import numpy as np
    t = torch.from_numpy(np.frombuffer(raw[:wbytes], dtype='<f4').astype('float32', copy=True)).reshape_as(a.w_fast)
    if not bool(torch.isfinite(t).all()) or bool((t.abs() > 2).any()):
        raise ValueError('invalid tensor values')
    if bbytes:
        b = torch.from_numpy(np.frombuffer(raw[wbytes:], dtype='<f4').astype('float32', copy=True)).reshape_as(a.basis)
        _validate_basis(b)
        a.basis.copy_(b)
    a.w_fast.copy_(t)
    a.updates_applied = updates


def save_fast_weights(model, path: Path, *, max_bytes: int):
    payload = snapshot_fast_weights(model, max_bytes=max_bytes)
    with Path(path).open('xb') as f:
        f.write(payload)


def load_fast_weights(model, path: Path, *, max_bytes: int):
    _budget(max_bytes)
    with Path(path).open('rb') as f:
        payload = f.read(max_bytes + 1)
    restore_fast_weights(model, payload, max_bytes=max_bytes)
