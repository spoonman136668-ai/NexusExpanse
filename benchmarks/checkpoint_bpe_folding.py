"""Benchmark for BPE Causal-History Folding (Screen + Confirmation)."""

from __future__ import annotations

import json
import random
import statistics
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
from nexus_expanse.bpe_folding import FoldedBoundarySkill, rebuild_folded_skill_from_state
from nexus_expanse.checkpoint_transport import CONFIG, SkillBank
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import forbid_donor_forward

CLOCK = time.perf_counter
FOLD_METHODS = ("position_lift", "entropy_fold", "static_trie_fold")


def _build_arm(tmp: Path, method: str, arm: str):
    if arm == "scratch":
        return study.scratch(tmp)
    if arm == "position_lift":
        return bpe_chal.BoundarySkill(tmp, method, "position_lift")
    return FoldedBoundarySkill(tmp / "donor.pt", method, arm)


@torch.inference_mode()
def replay_cold(tmp: Path, method: str, splits: dict) -> int:
    count = 0
    for file in sorted(tmp.glob("*.cold")):
        saved = torch.load(file, weights_only=True, map_location="cpu")
        arm = saved["arm"]
        if arm in ("position_lift", "scratch"):
            # Reuse checkpoint_bpe_challenge replay logic inline
            from nexus_expanse.checkpoint_transport import CorrectedLinear, ImportedSkill
            from nexus_expanse.bpe_folding import SuffixEmbedding as FoldSuffix

            state = saved["state"]
            skill = ImportedSkill()
            if arm != "scratch":
                for layer in skill.core.layers:
                    for name in ("fc1", "fc2"):
                        base = getattr(layer.ffn, name)
                        if method != "monarch":
                            setattr(layer.ffn, name, CorrectedLinear(base, torch.zeros(base.out_dim, base.in_dim), method))
                if "skill.core.emb.keys" in state:
                    skill.core.emb = FoldSuffix(skill.core.emb, state["skill.core.emb.keys"], state["skill.core.emb.delta"])
                m = bpe_chal.BoundarySkill.__new__(bpe_chal.BoundarySkill)
                torch.nn.Module.__init__(m)
                m.skill = skill
                m.skill.readout = torch.nn.Linear(CONFIG["dim"], len(state["first"]), bias=False)
                m.register_buffer("first", state["first"])
            else:
                m = skill
            m.load_state_dict(state)
        else:
            m = rebuild_folded_skill_from_state(saved["state"], method, arm)
        logits = m.eval()(splits["test"][0])
        assert study.digest(logits.numpy().tobytes()) == saved["sha"]
        count += 1
    return count


def run_seed(tmp: Path, seed: int, method: str, arms: list[str], screen: bool, splits: dict) -> dict:
    subprocess.run([sys.executable, str(ROOT / "benchmarks/checkpoint_bpe_challenge.py"), "producer", str(tmp), str(seed)], check=True)
    receipt = json.loads((tmp / "producer.json").read_text())
    refs = torch.load(tmp / "reference.pt", weights_only=True)

    torch.manual_seed(seed + 301)
    old = NexusExpanseByteCore(**CONFIG)
    opt = study.optim(old)
    for step in range(32):
        x, y = study.batch("text", splits["python"], seed, step, None)
        study.update(old, opt, old.forward_neural(x)[0], y)

    limit = study.score_logits(refs["val"], splits["val"][1])["bpb"] + 0.02
    evidence = {}
    with forbid_donor_forward() as guard:
        order = list(arms)
        random.Random(seed).shuffle(order)
        for arm in order:
            torch.manual_seed(seed)
            t = CLOCK()
            model = _build_arm(tmp, method, arm)
            prep = CLOCK() - t
            pre, _ = study.evaluate(model, splits["test"], refs["test"])
            start = CLOCK()
            val, _ = study.evaluate(model, splits["val"])
            steps = 0
            if not screen and val["bpb"] > limit:
                opt = study.optim(model)
                for steps in range(1, 161):
                    model.train()
                    x, y = study.batch("text", splits["fit"], seed, steps, None)
                    study.update(model, opt, model(x), y)
                    if steps % 8 == 0:
                        val, _ = study.evaluate(model, splits["val"])
                        if val["bpb"] <= limit:
                            break
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
                matched=val["bpb"] <= limit and selected["bpb"] <= receipt["test"]["bpb"] + 0.02,
                near=study.near(selected, receipt["test"], "text"),
            )
            if not screen:
                torch.save(dict(state=model.state_dict(), arm=arm, sha=evidence[arm]["prediction_sha"]), tmp / (arm + ".cold"))

    restored = 0
    if not screen:
        restored = int(subprocess.check_output([sys.executable, __file__, "cold", str(tmp), method], text=True))
        assert restored == len(evidence)
    return dict(seed=seed, producer=receipt, arms=evidence, cold_exact=restored, forbidden_calls=guard.donor_forward_calls)


def run(stage: str, dest: Path) -> None:
    splits, manifest = bpe_chal.data()
    screen_file = json.loads((study.OUT / "checkpoint-transfer-screen.json").read_text())
    method = screen_file["ranking"][0][-1]
    if stage == "fold-screen":
        arms = list(FOLD_METHODS)
        seeds = (901, 902, 903)
    else:
        ranking = json.loads((study.OUT / "checkpoint-transfer-bpe-fold-screen.json").read_text())["ranking"]
        winner = ranking[0][-1]
        if winner == "position_lift":
            winner = ranking[1][-1]
        arms = [winner, "position_lift", "shuffled_" + winner, "scratch"]
        seeds = tuple(range(1001, 1006))

    results = []
    for seed in seeds:
        with tempfile.TemporaryDirectory() as d:
            results.append(run_seed(Path(d), seed, method, arms, stage == "fold-screen", splits))

    obj = dict(
        source=study.git("rev-parse", "HEAD").decode().strip(),
        stage=stage,
        method=method,
        manifest=manifest,
        runs=results,
    )
    if stage == "fold-screen":
        obj["ranking"] = sorted(
            (
                max(r["arms"][a]["zero"]["kl_bits"] for r in results),
                statistics.mean(r["arms"][a]["zero"]["kl_bits"] for r in results),
                a,
            )
            for a in FOLD_METHODS
        )
    else:
        obj["winner"] = arms[0]
        obj["near_cases"] = sum(r["arms"][arms[0]]["near"] for r in results)
    study.atomic(dest, obj)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1] == "cold":
        print(replay_cold(Path(sys.argv[2]), sys.argv[3], bpe_chal.data()[0]))
    else:
        run(sys.argv[1], Path(sys.argv[2]))
