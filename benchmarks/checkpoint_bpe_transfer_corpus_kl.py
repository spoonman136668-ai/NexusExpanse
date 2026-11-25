"""Benchmark: Fresh-Corpus, Validation-Stopped Offline Byte-KL vs Position-Lift."""

from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import checkpoint_bpe_challenge as bpe_chal
from benchmarks import checkpoint_transfer_study as study
from nexus_expanse.checkpoint_transport import CONFIG, SkillBank
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import DenseTokenDonorLM, forbid_donor_forward, learn_byte_bpe
from nexus_expanse.offline_byte_kl import (
    build_topk_byte_cache,
    load_offline_cache,
    offline_kl_loss,
    save_offline_cache,
)

CLOCK = time.perf_counter
ARMS = ("transfer_kl_from_position_lift", "transfer_kl_from_scratch", "position_lift", "shuffled_transfer_kl")
TRANSFER_FILES = [
    "docs/GATE15_PREREG.md",
    "docs/GATE16_PREREG.md",
    "docs/GATE18_PREREG.md",
    "docs/GATE19_PREREG.md",
]


def data_with_transfer():
    paths = dict(
        study.TEXT_FILES,
        fit=["docs/GATE17_PREREG.md", "docs/GATE20_PREREG.md"],
        val=["docs/GATE22_PREREG.md"],
        test=["docs/GATE23_PREREG.md"],
    )
    original = study.TEXT_FILES
    try:
        study.TEXT_FILES = paths
        splits, manifest = study.corpus()
        # Build disjoint transfer windows from TRANSFER_FILES excluding any 32-byte overlap with val/test.
        held = set()
        for name in ("val", "test"):
            for row in torch.cat((splits[name][0], splits[name][1][:, -1:]), 1).tolist():
                raw = bytes(row)
                for i in range(0, len(raw) - 31):
                    held.add(raw[i : i + 32])
        windows = []
        for path in TRANSFER_FILES:
            raw = study.git("show", f"{study.BASE}:{path}")
            for i in range(0, len(raw) - 64, 32):
                win = raw[i : i + 65]
                if not any(win[j : j + 32] in held for j in range(0, len(win) - 31)):
                    windows.append(list(win))
                    if len(windows) >= 64:
                        break
            if len(windows) >= 64:
                break
        t_tensor = torch.tensor(windows, dtype=torch.long)
        splits["transfer"] = (t_tensor[:, :-1], t_tensor[:, 1:])
        manifest["transfer_windows"] = len(windows)
        return splits, manifest
    finally:
        study.TEXT_FILES = original


def producer_with_transfer_cache(tmp: Path, seed: int, splits: dict) -> None:
    t = CLOCK()
    vocab, encoded, merges = learn_byte_bpe(bpe_chal.blocks(splits["fit"]), 320)
    assert len(vocab) == 320
    torch.manual_seed(seed)
    model = DenseTokenDonorLM(320, 32, 2, 4, 2, 64)
    opt = study.optim(model)
    for step in range(96):
        rng = random.Random(seed * 1000 + step)
        group = rng.sample(encoded, min(8, len(encoded)))
        max_len = max(map(len, group)) - 1
        x = torch.zeros(len(group), max_len, dtype=torch.long)
        y = torch.full_like(x, -100)
        for i, row in enumerate(group):
            x[i, : len(row) - 1] = torch.tensor(row[:-1])
            y[i, : len(row) - 1] = torch.tensor(row[1:])
        logits = model(x)
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    torch.save(
        dict(weights=model.state_dict(), byte_to_row=torch.arange(256), spellings=[v.hex() for v in vocab]),
        tmp / "donor.pt",
    )
    fit_seconds = CLOCK() - t
    model.eval()
    tx, _ = splits["transfer"]
    t_cache = build_topk_byte_cache(model, vocab, merges, tx, top_k=16)
    save_offline_cache(t_cache, tmp / "transfer_cache.pt")
    refs = {}
    ratio = {}
    for split in ("val", "test"):
        refs[split], ratio[split] = bpe_chal.producer_reference(model, vocab, merges, splits[split])
    torch.save(refs, tmp / "reference.pt")
    study.atomic(
        tmp / "producer.json",
        dict(
            fit_export_seconds=fit_seconds,
            total_seconds=CLOCK() - t,
            fit_steps=96,
            producer_forward_calls=model.forward_calls,
            vocab_size=len(vocab),
            cache_payload_bytes=t_cache["payload_bytes"],
            cache_sha256=t_cache["sha256"],
            donor_sha=study.digest((tmp / "donor.pt").read_bytes()),
            test=study.score_logits(refs["test"], splits["test"][1]),
        ),
    )


def run_seed(tmp: Path, seed: int, method: str, splits: dict) -> dict:
    subprocess.run([sys.executable, __file__, "producer", str(tmp), str(seed)], check=True)
    receipt = json.loads((tmp / "producer.json").read_text())
    refs = torch.load(tmp / "reference.pt", weights_only=True)
    t_cache = load_offline_cache(tmp / "transfer_cache.pt")

    torch.manual_seed(seed + 301)
    old = NexusExpanseByteCore(**CONFIG)
    opt = study.optim(old)
    for step in range(32):
        x, y = study.batch("text", splits["python"], seed, step, None)
        study.update(old, opt, old.forward_neural(x)[0], y)
    torch.save(old.eval().state_dict(), tmp / "old.pt")

    tx, ty = splits["transfer"]
    limit = study.score_logits(refs["val"], splits["val"][1])["bpb"] + 0.02
    evidence = {}

    with forbid_donor_forward() as guard:
        order = list(ARMS)
        random.Random(seed).shuffle(order)
        for arm in order:
            torch.manual_seed(seed)
            t = CLOCK()
            if arm == "transfer_kl_from_scratch":
                model = study.scratch(tmp)
                bridge_kind = "scratch"
            else:
                model = bpe_chal.BoundarySkill(tmp, method, "position_lift")
                bridge_kind = "position_lift"
            prep = CLOCK() - t
            pre, _ = study.evaluate(model, splits["test"], refs["test"])
            start = CLOCK()

            idx_tensor = t_cache["indices"]
            prob_tensor = t_cache["probs"]
            if arm == "shuffled_transfer_kl":
                perm = torch.randperm(tx.size(0), generator=torch.Generator().manual_seed(seed + 9001))
                idx_tensor = idx_tensor[perm]
                prob_tensor = prob_tensor[perm]

            steps = 0
            opt = study.optim(model)
            if arm == "position_lift":
                val, _ = study.evaluate(model, splits["val"])
                if val["bpb"] > limit:
                    for steps in range(1, 161):
                        model.train()
                        x, y = study.batch("text", splits["fit"], seed, steps, None)
                        study.update(model, opt, model(x), y)
                        if steps % 8 == 0:
                            val, _ = study.evaluate(model, splits["val"])
                            if val["bpb"] <= limit:
                                break
            else:
                val0, _ = study.evaluate(model, splits["val"], refs["val"])
                best_kl = val0["kl_bits"]
                best_state = copy.deepcopy(model.state_dict())
                best_step = 0
                for step in range(1, 97):
                    model.train()
                    rng = random.Random(seed * 1000 + step)
                    picks = rng.sample(range(tx.size(0)), min(8, tx.size(0)))
                    opt.zero_grad(set_to_none=True)
                    loss = offline_kl_loss(model(tx[picks]), ty[picks], idx_tensor[picks], prob_tensor[picks], kl_weight=0.85)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    if step % 8 == 0:
                        val_m, _ = study.evaluate(model, splits["val"], refs["val"])
                        if val_m["kl_bits"] < best_kl:
                            best_kl = val_m["kl_bits"]
                            best_state = copy.deepcopy(model.state_dict())
                            best_step = step
                model.load_state_dict(best_state)
                steps = best_step

            total = prep + CLOCK() - start
            selected, logits = study.evaluate(model, splits["test"], refs["test"])
            with torch.inference_mode():
                probes = splits["retention"][0]
                assert torch.equal(SkillBank(old, model).eval()(probes), old.forward_neural(probes)[0])

            evidence[arm] = dict(
                zero=pre,
                selected=selected,
                steps=steps,
                prepare_seconds=prep,
                total_seconds=total,
                old_exact=True,
                skill_bytes=study.nbytes(model),
                prediction_sha=study.digest(logits.numpy().tobytes()),
                near=study.near(selected, receipt["test"], "text"),
            )
            torch.save(
                dict(state=model.state_dict(), bridge=bridge_kind, sha=evidence[arm]["prediction_sha"]),
                tmp / (arm + ".cold"),
            )

    restored = int(subprocess.check_output([sys.executable, str(ROOT / "benchmarks/checkpoint_bpe_challenge.py"), "cold", str(tmp), method], text=True))
    assert restored == len(evidence)
    return dict(seed=seed, producer=receipt, arms=evidence, cold_exact=restored, forbidden_calls=guard.donor_forward_calls)


def run(dest: Path) -> None:
    splits, manifest = data_with_transfer()
    method = json.loads((study.OUT / "checkpoint-transfer-screen.json").read_text())["ranking"][0][-1]
    seeds = tuple(range(1201, 1206))
    results = []
    for seed in seeds:
        with tempfile.TemporaryDirectory() as d:
            results.append(run_seed(Path(d), seed, method, splits))
    obj = dict(
        source=study.git("rev-parse", "HEAD").decode().strip(),
        stage="bpe-transfer-corpus-kl",
        method=method,
        manifest=manifest,
        runs=results,
    )
    study.atomic(dest, obj)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1] == "producer":
        producer_with_transfer_cache(Path(sys.argv[2]), int(sys.argv[3]), data_with_transfer()[0])
    else:
        run(Path(sys.argv[1]))
