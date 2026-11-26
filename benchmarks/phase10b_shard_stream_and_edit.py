"""Frozen Phase 10B Linux CPU DEV confirmation with local three-shard BPE donors."""

import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.checkpoint_transfer_study import optim
from benchmarks.dev_artifacts import atomic, digest
from benchmarks.phase3_multi_skill_routing import SOURCE, git
from benchmarks.phase9_weight_edit_and_assimilate import (
    DOMAINS, bpb, combine, edit_experiment, frozen_data, produce_donor,
)
from nexus_expanse.checkpoint_transport import CorrectedLinear
from nexus_expanse.donor_assimilation import DenseTokenDonorLM, forbid_donor_forward, learn_byte_bpe
from nexus_expanse.donor_cache_io import GPT2_BYTES
from nexus_expanse.shard_stream_assimilation import adapt_skill, assimilate_zero_exec, make_student
from nexus_expanse.weight_editing import exact_skill_receipt, restore_skill, verify_skill

BASE = "7b1f81a07ddbd0f37682a8132f483d29298a0408"
PREREG = "docs/PHASE10B_SHARD_STREAM_AND_EDIT_PREREG.md"
SEEDS = tuple(range(2401, 2406))
TEACHER_STEPS = 320
ADAPT_STEPS = 64
MOCK_FILES = tuple(f"model-{i:05d}-of-00003.pt" for i in range(1, 4))
SUITE_B_ARMS = ("shard_stream_decay_stack", "shuffled_donor_control", "scratch_nexus")


def write_mock_shards(directory, tensors, vocab, merges=()):
    """Write a local fixture in the same name-index shape as HF safetensors."""
    directory.mkdir()
    names = {}
    for name, shard in zip(MOCK_FILES, tensors, strict=True):
        torch.save(shard, directory / name)
        for key in shard:
            if key in names:
                raise ValueError("duplicate fixture tensor")
            names[key] = name
    atomic(directory / "model.pt.index.json", {"weight_map": names})
    encoder = {byte: char for char, byte in GPT2_BYTES.items()}
    words = ["".join(encoder[byte] for byte in token) for token in vocab]
    atomic(directory / "tokenizer.json", {
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": dict(zip(words, range(len(words)), strict=True)),
                  "merges": [[words[a], words[b]] for a, b in merges]},
    })


def produce_cross_width_donor(directory, fit, seed, *, dim=64, layers=2, engram=False):
    """Producer process only. The consumer sees three tensor files and tokenizer JSON."""
    started = time.perf_counter()
    spans = [bytes(row) for row in fit.to(torch.uint8).tolist()]
    vocab, encoded, merges = learn_byte_bpe(spans, vocab_size=320)
    torch.manual_seed(seed + 11000)
    model = DenseTokenDonorLM(vocab_size=320, dim=dim, n_layers=layers, n_heads=4, ffn_mult=2, max_seq_len=64)
    optimizer = optim(model)
    rng = torch.Generator().manual_seed(seed + 12000)
    tokens = [torch.tensor(row) for row in encoded]
    for _ in range(TEACHER_STEPS):
        examples = [tokens[i] for i in torch.randint(len(tokens), (8,), generator=rng).tolist()]
        inputs = torch.nn.utils.rnn.pad_sequence([row[:-1] for row in examples], batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence(
            [row[1:] for row in examples], batch_first=True, padding_value=-100,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()

    state = model.state_dict()
    shards = [
        {"model.embed_tokens.weight": state["emb.weight"]},
        {"model.embed_positions.weight": state["pos_emb.weight"],
         "model.norm.weight": state["out_norm.weight"], "lm_head.weight": state["readout.weight"]},
        {},
    ]
    if engram:
        pairs = fit[:, :-1].flatten() * 256 + fit[:, 1:].flatten()
        counts = (torch.bincount(pairs, minlength=256 * 256).reshape(256, 256) + 1).float()
        shards[1]["engram.weight"] = counts.log() - counts.sum(1, keepdim=True).log()
    for i in range(layers):
        prefix = f"layers.{i}."
        out = f"model.layers.{i}."
        q, k, v = state[prefix + "attn.in_proj_weight"].chunk(3, dim=0)
        shards[1 + i % 2].update({
            out + "self_attn.q_proj.weight": q, out + "self_attn.k_proj.weight": k,
            out + "self_attn.v_proj.weight": v,
            out + "self_attn.o_proj.weight": state[prefix + "attn.out_proj.weight"],
            out + "mlp.up_proj.weight": state[prefix + "fc1.weight"],
            out + "mlp.down_proj.weight": state[prefix + "fc2.weight"],
            out + "input_layernorm.weight": state[prefix + "attn_norm.weight"],
            out + "post_attention_layernorm.weight": state[prefix + "ffn_norm.weight"],
        })
    write_mock_shards(directory, shards, vocab, merges)
    return dict(
        seconds=time.perf_counter() - started, forward_calls=model.forward_calls, steps=TEACHER_STEPS,
        files={path.name: dict(sha256=digest(path), bytes=path.stat().st_size)
               for path in sorted(directory.iterdir())},
    )


@torch.no_grad()
def make_matched_scratch(seed, **geometry):
    """Match trainable spectral residual capacity while keeping the initial random native function."""
    skill = make_student(seed, **geometry)
    for layer in skill.core.layers:
        for name in ("fc1", "fc2"):
            base = getattr(layer.ffn, name)
            uncorrected = base(torch.eye(base.in_dim)).T
            setattr(layer.ffn, name, CorrectedLinear(base, uncorrected, "spectral4"))
    return skill.eval()


def checked_score(archive, check):
    with forbid_donor_forward() as guard:
        skill = restore_skill(archive)
        result = dict(
            combined=bpb(skill, combine(check)),
            by_domain={name: bpb(skill, check[name]) for name in DOMAINS},
        )
        return result, guard.donor_forward_calls


def stream_experiment(donor_dir, seed, data):
    output = {}
    for arm in SUITE_B_ARMS:
        if arm == "scratch_nexus":
            initial, initial_checks = verify_skill(make_matched_scratch(seed))
        else:
            initial, initial_checks = assimilate_zero_exec(
                donor_dir, mock_pt=True, seed=seed,
                shuffle_seed=seed + 50000 if arm == "shuffled_donor_control" else None,
            )
        skill = restore_skill(initial)
        adapted, final_checks = adapt_skill(skill, combine(data["fit"]), seed=seed + 60000, steps=ADAPT_STEPS)
        initial_bpb, initial_calls = checked_score(initial, data["check"])
        final_bpb, final_calls = checked_score(adapted, data["check"])
        output[arm] = dict(
            zero_update_bpb=initial_bpb, post_adapt_bpb=final_bpb,
            initial=initial_checks, final=final_checks,
            cold_scoring_donor_forward_calls=initial_calls + final_calls,
        )
    return output


def score(rows):
    accepted = superiority = recovered = retain = zero_update = post_update = cold_a = cold_b = 0
    for row in rows:
        arms, stream = row["edit"]["arms"], row["stream"]
        svd = arms["svd_subspace_ablate"]
        random = arms["random_subspace_ablate"]
        fault = arms["unrepaired_faulted"]
        accepted += svd["accepted"]
        superiority += svd["fault_bpb"] < min(random["fault_bpb"], fault["fault_bpb"])
        recovered += svd["fraction_damage_recovered"] is not None and svd["fraction_damage_recovered"] >= 0.5
        retain += svd["retain_delta_bpb"] <= 0.02
        cold_a += all(exact_skill_receipt(arms[name]["verification"])
                      for name in ("svd_subspace_ablate", "random_subspace_ablate"))
        stack, shuffled, scratch = [stream[name] for name in SUITE_B_ARMS]
        zero_update += shuffled["zero_update_bpb"]["combined"] - stack["zero_update_bpb"]["combined"] >= 0.5
        post_update += stack["post_adapt_bpb"]["combined"] < min(
            shuffled["post_adapt_bpb"]["combined"], scratch["post_adapt_bpb"]["combined"],
        )
        cold_b += all(exact_skill_receipt(arm[stage]) for arm in stream.values() for stage in ("initial", "final"))
    suite_a = "FAIL"
    if accepted == superiority == recovered == retain == 5:
        suite_a = "PASS"
    elif accepted >= 4 and superiority >= 4 and retain == 5:
        suite_a = "MIXED"
    suite_b = "FAIL"
    if zero_update == post_update == 5:
        suite_b = "PASS"
    elif 5 in (zero_update, post_update):
        suite_b = "MIXED"
    return dict(
        suite_a=dict(verdict=suite_a, accepted=accepted, beats_both=superiority,
                     half_damage_recovered=recovered, retain_within_bound=retain, exact_cold_restores=cold_a),
        suite_b=dict(verdict=suite_b, zero_update_margin=zero_update,
                     post_adaptation_beats_both=post_update, exact_cold_restores=cold_b),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--produce-fit", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.out.exists():
        parser.error("the requested receipt or donor directory already exists")
    if args.produce_fit is not None:
        fit = torch.load(args.produce_fit, map_location="cpu", weights_only=True)
        print(json.dumps(produce_cross_width_donor(args.out, fit, args.seed), allow_nan=False))
        return

    source = git("rev-parse", "HEAD").decode().strip()
    if (git("rev-parse", "HEAD^").decode().strip() != BASE
            or git("status", "--porcelain") or not git("ls-tree", source, PREREG)):
        parser.error("run scored confirmation on the clean first Phase 10B commit")
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix="nexus-phase10b-") as scratch:
        for seed in SEEDS:
            seed_started = time.perf_counter()
            directory = Path(scratch) / str(seed)
            directory.mkdir()
            data, hashes = frozen_data(seed)
            fit = combine(data["fit"])

            edit_donor = directory / "edit.pt"
            edit_producer = produce_donor(edit_donor, seed, fit)
            edit = edit_experiment(edit_donor, seed, data)
            arms = edit["arms"]
            retain = arms["unrepaired_faulted"]["retain_bpb"]
            for name in ("svd_subspace_ablate", "random_subspace_ablate"):
                arms[name]["retain_delta_bpb"] = arms[name]["retain_bpb"] - retain
            fit_path = directory / "fit.pt"
            torch.save(fit, fit_path)
            donor_dir = directory / "bpe-shards"
            producer = json.loads(subprocess.check_output(
                [sys.executable, __file__, "--produce-fit", str(fit_path), "--seed", str(seed), "--out", str(donor_dir)],
                cwd=ROOT, text=True,
            ))
            stream = stream_experiment(donor_dir, seed, data)
            rows.append(dict(seed=seed, split_provenance=hashes, edit_producer=edit_producer, edit=edit,
                             stream_producer=producer, stream=stream, seconds=time.perf_counter() - seed_started))
            print(f"Phase 10B scored seed {seed} recorded", flush=True)
    if git("rev-parse", "HEAD").decode().strip() != source or git("status", "--porcelain"):
        parser.error("checkout changed during scored confirmation")

    summary = score(rows)
    streams = [r["stream"][arm]["initial"] for r in rows for arm in SUITE_B_ARMS[:2]]
    calls = sum(
        arm["initial"].get("donor_forward_calls", 0) + arm["final"]["donor_forward_calls"]
        + arm["cold_scoring_donor_forward_calls"]
        for row in rows for arm in row["stream"].values()
    )
    result = dict(
        schema="nexus.phase10b-shard-stream-and-edit.v1", source_commit=source, source_tree=tree,
        baseline=BASE, fixture_source_sha=SOURCE, prereg_sha256=digest(ROOT / PREREG),
        stack_provenance="Phase 10B implementation from operator-named techniques",
        platform=platform.platform(), python=platform.python_version(), torch=str(torch.__version__),
        device="cpu", threads=torch.get_num_threads(), source_format="mock_pt",
        edit_autograd_calls=sum(row["edit"]["autograd_calls"] for row in rows),
        guarded_donor_forward_calls=calls + sum(row["edit"]["donor_forward_calls"] for row in rows),
        peak_loaded_shards=max(r["peak_loaded_shards"] for r in streams),
        all_streams_three_shards=all(r["loaded_shard_count"] == 3 for r in streams),
        frozen_edit_parameters_all_seeds=all(row["edit"]["all_parameters_frozen"] for row in rows),
        fit_val_check_32byte_overlap=0, external_downloads=0, paid_api_calls=0,
        wall_seconds=time.perf_counter() - started,
        max_rss_kib=max(resource.getrusage(mode).ru_maxrss for mode in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)),
        seeds=rows, summary=summary,
    )
    atomic(args.out, result)
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
