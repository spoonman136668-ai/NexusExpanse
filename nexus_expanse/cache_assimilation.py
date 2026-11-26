"""Train a native byte skill from validated static cache tensors."""

import math

import torch

from nexus_expanse.checkpoint_transport import ImportedSkill
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.donor_cache_io import read_cache
from nexus_expanse.offline_byte_kl import offline_kl_loss
from nexus_expanse.weight_editing import exact_skill_receipt, verify_skill


def assimilate(cache_dir, *, device, dtype, steps, kl_weight, seed):
    if steps < 1 or not math.isfinite(kl_weight) or not 0 <= kl_weight <= 1:
        raise ValueError("steps must be positive and KL weight must be in [0, 1]")
    if dtype not in ("fp32", "bf16") or device not in ("auto", "cpu", "cuda"):
        raise ValueError("unknown assimilation device or dtype")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device == "cuda" and dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA bfloat16 is unavailable")

    with forbid_donor_forward() as guard:
        tensors, manifest_hash = read_cache(cache_dir)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            skill = ImportedSkill().to(device)
        batch_rng = torch.Generator().manual_seed(seed + 1)
        optimizer = torch.optim.AdamW(skill.parameters(), lr=0.004, weight_decay=0.01)
        count = len(tensors["raw"])
        for _ in range(steps):
            rows = torch.randint(count, (min(8, count),), generator=batch_rng)
            raw = tensors["raw"][rows].to(device=device, dtype=torch.long)
            indices = tensors["indices"][rows].to(device)
            probs = tensors["probs"][rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=dtype == "bf16"):
                out = skill(raw[:, :-1])
            loss = offline_kl_loss(out, raw[:, 1:], indices, probs, kl_weight)
            if not torch.isfinite(loss):
                raise ValueError("static assimilation produced nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(skill.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

        skill.zero_grad(set_to_none=True)
        skill.cpu().eval().requires_grad_(False)
        skill.core.fixed_eval_geometry = True
        archive, receipt = verify_skill(skill)
        if guard.donor_forward_calls or not exact_skill_receipt(receipt):
            raise ValueError("static assimilation failed guarded skill verification")
        receipt.update(
            donor_forward_calls=guard.donor_forward_calls, cache_manifest_sha256=manifest_hash,
            steps=steps, windows=count, seed=seed, device=device, dtype=dtype,
            kl_weight=kl_weight, final_loss=float(loss.detach()),
        )
    return archive, receipt
