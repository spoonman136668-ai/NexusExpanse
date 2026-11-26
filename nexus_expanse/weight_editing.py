"""Inference-only inspection, closed-form edits, and standalone skill verification."""

from copy import deepcopy
from io import BytesIO
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from nexus_expanse.checkpoint_transport import CorrectedLinear, ImportedSkill
from nexus_expanse.substrate import DirectMonarchLinear

SCHEMA = "nexus.standalone-skill.v1"
FFN_NAMES = ("fc1", "fc2")
TARGETS = ("attn.out_proj", "ffn.fc1", "ffn.fc2")


def _ffn_layout(module):
    if isinstance(module, CorrectedLinear):
        return module.method
    if isinstance(module, nn.Linear):
        return "dense" if module.bias is not None else "dense_without_bias"
    if isinstance(module, DirectMonarchLinear):
        return "monarch"
    raise ValueError("unsupported standalone skill affine layer")


def skill_payload(skill):
    return dict(
        schema=SCHEMA,
        width=skill.readout.in_features,
        fixed_eval_geometry=skill.core.fixed_eval_geometry,
        prior_alpha=skill.prior_alpha,
        ffn_layouts=[[_ffn_layout(getattr(layer.ffn, name)) for name in FFN_NAMES]
                     for layer in skill.core.layers],
        state={key: value.detach().cpu() for key, value in skill.state_dict().items()},
    )


@torch.inference_mode()
def skill_snapshot(skill):
    packed = BytesIO()
    torch.save(skill_payload(skill), packed)
    return packed.getvalue()


@torch.inference_mode()
def restore_skill(source):
    """Read Phase 9 and legacy import-skill snapshots without a donor."""
    payload = torch.load(BytesIO(source) if isinstance(source, bytes) else Path(source),
                         weights_only=True, map_location="cpu")
    return restore_skill_payload(payload)


def restore_skill_payload(payload, *, assign=False):
    """Rebuild the native skill used by both standalone and bank archives."""
    # Restored parameters remain eligible for a later native adaptation step.
    with torch.inference_mode(False), torch.no_grad(), torch.random.fork_rng(devices=[]):
        skill = ImportedSkill(payload["width"])
        schema = payload.get("schema")
        if schema == SCHEMA:
            fixed_eval_geometry = payload["fixed_eval_geometry"]
            if type(fixed_eval_geometry) is not bool:
                raise ValueError("invalid fixed evaluation geometry")
            skill.core.fixed_eval_geometry = fixed_eval_geometry
            layouts = payload["ffn_layouts"]
        elif schema is None and "method" in payload:
            layouts = [[payload["method"]] * len(FFN_NAMES) for _ in skill.core.layers]
        else:
            raise ValueError("unknown standalone skill snapshot format")
        for layer, layout in zip(skill.core.layers, layouts, strict=True):
            for name, kind in zip(FFN_NAMES, layout, strict=True):
                base = getattr(layer.ffn, name)
                if kind in ("dense", "dense_without_bias"):
                    replacement = nn.Linear(base.in_dim, base.out_dim, bias=kind == "dense")
                elif kind != "monarch":
                    replacement = CorrectedLinear(base, torch.zeros(base.out_dim, base.in_dim), kind)
                else:
                    continue
                setattr(layer.ffn, name, replacement)
        state = payload["state"]
        alpha = payload.get("prior_alpha", 0.0)
        prior = state.get("static_byte_prior")
        if (type(alpha) not in (int, float) or not 0 <= alpha < 1
                or (alpha and prior is None)):
            raise ValueError("invalid standalone skill prior alpha")
        if prior is not None:
            if (not isinstance(prior, torch.Tensor) or prior.shape != (256, 256) or not prior.is_floating_point()
                    or not torch.isfinite(prior).all()):
                raise ValueError("invalid standalone static byte prior")
            skill.static_byte_prior = torch.empty_like(prior)
        skill.prior_alpha = alpha
        skill.load_state_dict(state, strict=True, assign=assign)
    return skill.eval().requires_grad_(False)


def _layer(skill, layer_idx):
    if not isinstance(skill, ImportedSkill) or not 0 <= layer_idx < len(skill.core.layers):
        raise ValueError("expected an ImportedSkill and a valid layer index")
    return skill.core.layers[layer_idx]


def _attention_contexts(skill, raw):
    # MultiheadAttention applies out_proj functionally. An identity projection
    # exposes its input without changing the skill.
    result = []
    work = deepcopy(skill).eval()
    hooks = []

    def capture(module, args, kwargs):
        hidden = args[0].transpose(0, 1)
        identity = torch.eye(module.embed_dim, device=raw.device, dtype=hidden.dtype)
        context, _ = F.multi_head_attention_forward(
            hidden, hidden, hidden, module.embed_dim, module.num_heads,
            module.in_proj_weight, module.in_proj_bias, None, None, False,
            0.0, identity, None, training=False, attn_mask=kwargs["attn_mask"], need_weights=False,
        )
        result.append(context[:raw.size(1)].double().mean(dim=0))

    for layer in work.core.layers:
        hooks.append(layer.attn.register_forward_pre_hook(capture, with_kwargs=True))
    try:
        work(raw)
    finally:
        for hook in hooks:
            hook.remove()
    return result


@torch.inference_mode()
def inspect_fault_direction(skill, fault_x, retain_x, layer_idx):
    """Find the top right singular vector of paired sequence-mean contrasts."""
    _layer(skill, layer_idx)
    if (fault_x.ndim != 2 or retain_x.shape != fault_x.shape or not fault_x.numel()
            or fault_x.shape[1] > skill.core.max_seq_len):
        raise ValueError("fault and retain inputs must be matched nonempty raw-byte batches")
    for raw in (fault_x, retain_x):
        if raw.is_floating_point() or raw.is_complex() or (raw < 0).any() or (raw > 255).any():
            raise ValueError("fault and retain inputs must contain only integer bytes")
    device = skill.readout.weight.device
    faults = _attention_contexts(skill, fault_x.to(device=device, dtype=torch.long))
    retains = _attention_contexts(skill, retain_x.to(device=device, dtype=torch.long))
    layers = []
    direction = None
    for index, (fault, retain) in enumerate(zip(faults, retains, strict=True)):
        _, singular, axes = torch.linalg.svd(fault - retain, full_matrices=False)
        if index == layer_idx:
            direction = axes[0]
        layers.append(dict(
            layer=index, fault_mean=fault.mean(0).tolist(), retain_mean=retain.mean(0).tolist(),
            mean_difference=(fault - retain).mean(0).tolist(), singular_values=singular.tolist(),
        ))
    return dict(direction_v=direction, layers=layers)


def _affine_parameters(module):
    if isinstance(module, nn.Linear):
        return module.weight, module.bias
    if isinstance(module, (DirectMonarchLinear, CorrectedLinear)):
        base = module.base if isinstance(module, CorrectedLinear) else module
        eye = torch.eye(base.in_dim, dtype=base.b1.dtype, device=base.b1.device)
        bias = base.bias
        responses = module(eye)
        if bias is not None:
            responses = responses - bias
        return responses.T, bias
    raise ValueError("unsupported surgical target")


def _edit_vector(value, width, reference, *, unit=False):
    if value is None:
        raise ValueError("edit requires a direction vector")
    vector = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if vector.shape != (width,) or not torch.isfinite(vector).all():
        raise ValueError("edit vector has an invalid width or nonfinite values")
    if unit and not math.isclose(vector.norm().item(), 1.0, rel_tol=1e-5):
        raise ValueError("subspace ablation requires a unit direction")
    return vector


@torch.inference_mode()
def apply_surgical_edit(skill, layer_idx, target_module, op, direction_v, alpha, direction_u=None):
    """Apply an affine edit to a detached copy, materializing only the targeted affine if needed."""
    if target_module not in TARGETS or not math.isfinite(alpha):
        raise ValueError("invalid surgical target or alpha")
    if op not in {"subspace_ablate", "rank1_steer", "bias_shift"}:
        raise ValueError("unknown surgical edit operation")
    edited = deepcopy(skill).eval().requires_grad_(False)
    layer = _layer(edited, layer_idx)
    parent_name, name = target_module.split(".")
    parent = getattr(layer, parent_name)
    module = getattr(parent, name)
    weight, bias = _affine_parameters(module)
    out_width, in_width = weight.shape
    if op == "subspace_ablate":
        v = _edit_vector(direction_v, in_width, weight, unit=True)
        weight = weight - alpha * torch.outer(weight @ v, v)
    elif op == "rank1_steer":
        v = _edit_vector(direction_v, in_width, weight)
        u = _edit_vector(direction_u, out_width, weight)
        weight = weight + alpha * torch.outer(u, v)
    else:
        u = _edit_vector(direction_u if direction_u is not None else direction_v, out_width, weight)
        bias = alpha * u if bias is None else bias + alpha * u
    if alpha == 0:
        return edited
    if isinstance(module, nn.Linear) and (module.bias is None) == (bias is None):
        replacement = module
    else:
        with torch.random.fork_rng(devices=[]):
            replacement = nn.Linear(in_width, out_width, bias=bias is not None,
                                    dtype=weight.dtype, device="cpu").to(weight.device)
        setattr(parent, name, replacement)
    replacement.weight.copy_(weight)
    if bias is not None:
        replacement.bias.copy_(bias)
    return edited.requires_grad_(False)


@torch.inference_mode()
def prefix_diff(skill, *, seed=0, probes=16):
    rng = torch.Generator().manual_seed(seed)
    width = min(32, skill.core.max_seq_len)
    raw = torch.randint(256, (probes, width), generator=rng)
    cut = width // 2
    alternate = raw.clone()
    alternate[:, cut:] = (alternate[:, cut:] + 17) % 256
    device = skill.readout.weight.device
    original = skill(raw.to(device))[:, :cut]
    changed = skill(alternate.to(device))[:, :cut]
    difference = _max_diff(original, changed)
    if skill.core.fixed_eval_geometry:
        difference = max(difference, _max_diff(original, skill(raw[:, :cut].to(device))))
    return difference


def _max_diff(before, after):
    difference = (before - after).abs().max().item()
    if not math.isfinite(difference):
        raise ValueError("skill produced nonfinite verification outputs")
    return difference


@torch.inference_mode()
def verify_skill(skill):
    """Use the real standalone archive and compare every state tensor plus probe forecasts."""
    if skill.training or skill.readout.weight.device.type != "cpu":
        raise ValueError("verify an evaluation skill on CPU")
    archive = skill_snapshot(skill)
    rebuilt = restore_skill(archive)
    probe = torch.arange(24, dtype=torch.long).unsqueeze(0)
    before, after = skill.state_dict(), rebuilt.state_dict()
    exact_state = (
        skill.prior_alpha == rebuilt.prior_alpha and before.keys() == after.keys()
        and all(torch.equal(before[key], after[key]) for key in before)
    )
    receipt = dict(
        prefix_probes=16, prefix_max_abs_diff=prefix_diff(skill),
        round_trip_max_abs_diff=_max_diff(skill(probe), rebuilt(probe)),
        exact_round_trip_state=exact_state,
    )
    return archive, receipt


def exact_skill_receipt(receipt):
    return (receipt["prefix_max_abs_diff"] == 0.0 and receipt["round_trip_max_abs_diff"] == 0.0
            and receipt["exact_round_trip_state"])


@torch.inference_mode()
def verified_surgical_edit(skill, edit_fn, fault_eval_fn, retain_eval_fn, max_retain_drift_bpb=0.02):
    """Gate a private edit on exactness, fault improvement, and retain non-degradation."""
    if not math.isfinite(max_retain_drift_bpb) or max_retain_drift_bpb < 0:
        raise ValueError("retain drift bound must be finite and nonnegative")
    original = deepcopy(skill).cpu().eval().requires_grad_(False)
    fault_before = float(fault_eval_fn(original))
    retain_before = float(retain_eval_fn(original))
    edited = edit_fn(deepcopy(original)).cpu().eval().requires_grad_(False)
    try:
        _, receipt = verify_skill(edited)
    except ValueError as exc:
        return False, skill, dict(accepted=False, verification_error=str(exc))
    fault_after = float(fault_eval_fn(edited))
    retain_after = float(retain_eval_fn(edited))
    finite = all(math.isfinite(value) for value in (fault_before, fault_after, retain_before, retain_after))
    drift = retain_after - retain_before if finite else None
    accepted = (finite and exact_skill_receipt(receipt) and fault_after < fault_before
                and drift <= max_retain_drift_bpb)
    receipt.update(
        accepted=accepted, fault_before_bpb=fault_before, fault_after_bpb=fault_after,
        retain_before_bpb=retain_before, retain_after_bpb=retain_after, retain_drift_bpb=drift,
    )
    return accepted, edited if accepted else skill, receipt
