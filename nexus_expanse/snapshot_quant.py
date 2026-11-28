"""Weight-only codecs for unified byte-runtime snapshots."""

import torch


SNAPSHOT_QUANTS = ("fp32", "fp16", "int8")


def _channel_reduction(name, weight):
    # A Monarch factor is [group, inputs, outputs]. Dense and embedding
    # matrices use axis zero for output channels or embedding rows.
    if weight.ndim == 3 and name.rsplit(".", 1)[-1] in ("b1", "b2"):
        return (1,)
    return tuple(range(1, weight.ndim))


def pack_weights(backbone, state, mode):
    scales = {}
    for name, parameter in backbone.named_parameters():
        if parameter.ndim < 2:
            continue
        weight = state[name]
        if mode == "fp16":
            encoded = weight.to(torch.float16)
            if not torch.isfinite(encoded).all():
                raise ValueError(f"weight exceeds the finite fp16 range: {name}")
        else:
            peak = weight.abs().amax(dim=_channel_reduction(name, weight), keepdim=True)
            scale = torch.where(peak == 0, 1.0, (peak / 127.0).clamp_min(torch.finfo(torch.float32).tiny))
            # Canonicalize the scale under the FP32 multiply/divide used when
            # the peak int8 code is dequantized and the weights are packed again.
            scale = (scale * 127.0) / 127.0
            if not torch.isfinite(scale * 127.0).all():
                raise ValueError(f"weight exceeds the finite int8 dequantization range: {name}")
            encoded = (weight / scale).round().clamp(-127, 127).to(torch.int8)
            scales[name] = scale
        state[name] = encoded
    return scales


def unpack_weights(backbone, state, mode, scales):
    expected = backbone.state_dict()
    weights = {name for name, parameter in backbone.named_parameters() if parameter.ndim >= 2}
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("quantized snapshot has invalid backbone keys")
    if mode == "int8" and (not isinstance(scales, dict) or scales.keys() != weights):
        raise ValueError("int8 snapshot has invalid scale keys")
    if mode == "fp16" and scales is not None:
        raise ValueError("fp16 snapshot cannot contain int8 scales")
    for name, reference in expected.items():
        value = state[name]
        dtype = reference.dtype if name not in weights else (torch.float16 if mode == "fp16" else torch.int8)
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != dtype:
            raise ValueError(f"quantized snapshot has invalid tensor: {name}")
        if name not in weights:
            continue
        decoded = value.float()
        if mode == "int8":
            scale = scales[name]
            axes = _channel_reduction(name, value)
            shape = tuple(1 if axis in axes else size for axis, size in enumerate(value.shape))
            if (not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32 or scale.shape != shape
                    or not torch.isfinite(scale).all() or (scale <= 0).any() or (value == -128).any()):
                raise ValueError(f"int8 snapshot has invalid channel data: {name}")
            decoded.mul_(scale)
        if not torch.isfinite(decoded).all():
            raise ValueError(f"quantized snapshot has nonfinite weight: {name}")
        state[name] = decoded
