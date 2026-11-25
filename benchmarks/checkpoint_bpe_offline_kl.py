"""Benchmark: One-Time Offline Byte-Marginalized KL vs Static Position-Lift on BPE->Byte."""

from __future__ import annotations

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
from nexus_expanse.checkpoint_transport import CONFIG, CorrectedLinear, ImportedSkill, SkillBank
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import DenseTokenDonorLM, forbid_donor_forward, learn_byte_bpe
from nexus_expanse.offline_byte_kl import (
    build_topk_byte_cache,
    load_offline_cache,
    offline_kl_loss,
    save_offline_cache,
)

CLOCK = time.perf_counter
ARMS = ("offline_kl_from_position_lift", "offline_kl_from_scratch", "position_lift", "shuffled_offline_kl")


def producer_with_cache(tmp: Path, seed: int, splits: dict) -> None:
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
    fit_x, _ = splits["fit"]
    fit_cache = build_topk_byte_cache(model, vocab, merges, fit_x, top_k=16)
    save_offline_cache(fit_cache, tmp / "fit_cache.pt")
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
            cache_payload_bytes=fit_cache["payload_bytes"],
            cache_sha256=fit_cache["sha256"],
            donor_sha=study.digest((tmp / "donor.pt").read_bytes()),
            test=study.score_logits(refs["test"], splits["test"][1]),
        ),
    )


def run_seed(tmp: Path, seed: int, method: str, splits: dict) -> dict:
    subprocess.run([sys.executable, __file__, "producer", str(tmp), str(seed)], check=True)
    receipt = json.loads((tmp / "producer.json").read_text())
    refs = torch.load(tmp / "reference.pt", weights_only=True)
    fit_cache = load_offline_cache(tmp / "fit_cache.pt")

    torch.manual_seed(seed + 301)
    old = NexusExpanseByteCore(**CONFIG)
    opt = study.optim(old)
    for step in range(32):
        x, y = study.batch("text", splits["python"], seed, step, None)
        study.update(old, opt, old.forward_neural(x)[0], y)
    torch.save(old.eval().state_dict(), tmp / "old.pt")

    fit_x, fit_y = splits["fit"]
    limit = study.score_logits(refs["val"], splits["val"][1])["bpb"] + 0.02
    evidence = {}

    with forbid_donor_forward() as guard:
        order = list(ARMS)
        random.Random(seed).shuffle(order)
        for arm in order:
            torch.manual_seed(seed)
            t = CLOCK()
            if arm == "offline_kl_from_scratch":
                model = study.scratch(tmp)
                bridge_kind = "scratch"
            else:
                model = bpe_chal.BoundarySkill(tmp, method, "position_lift")
                bridge_kind = "position_lift"
            prep = CLOCK() - t
            pre, _ = study.evaluate(model, splits["test"], refs["test"])
            start = CLOCK()
            val, _ = study.evaluate(model, splits["val"])

            idx_tensor = fit_cache["indices"]
            prob_tensor = fit_cache["probs"]
            if arm == "shuffled_offline_kl":
                perm = torch.randperm(fit_x.size(0), generator=torch.Generator().manual_seed(seed + 9001))
                idx_tensor = idx_tensor[perm]
                prob_tensor = prob_tensor[perm]

            steps = 0
            opt = study.optim(model)
            # For offline KL arms, run full 160 steps to observe distribution transfer; also track val-stopped snapshot.
            if arm == "position_lift":
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
                for steps in range(1, 161):
                    model.train()
                    rng = random.Random(seed * 1000 + steps)
                    picks = rng.sample(range(fit_x.size(0)), min(8, fit_x.size(0)))
                    xb = fit_x[picks]
                    yb = fit_y[picks]
                    ib = idx_tensor[picks]
                    pb = prob_tensor[picks]
                    opt.zero_grad(set_to_none=True)
                    loss = offline_kl_loss(model(xb), yb, ib, pb, kl_weight=0.5)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

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
    splits, manifest = bpe_chal.data()
    method = json.loads((study.OUT / "checkpoint-transfer-screen.json").read_text())["ranking"][0][-1]
    seeds = tuple(range(1101, 1106))
    results = []
    for seed in seeds:
        with tempfile.TemporaryDirectory() as d:
            results.append(run_seed(Path(d), seed, method, splits))
    obj = dict(
        source=study.git("rev-parse", "HEAD").decode().strip(),
        stage="bpe-offline-kl",
        method=method,
        manifest=manifest,
        runs=results,
    )
    study.atomic(dest, obj)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1] == "producer":
        producer_with_cache(Path(sys.argv[2]), int(sys.argv[3]), bpe_chal.data()[0])
    else:
        run(Path(sys.argv[1]))
