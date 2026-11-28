"""Offline regression coverage for verifier-gated byte harvesting."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.phase15_cosine_lr_and_sampling import call_cli, space_snapshot
from nexus_expanse.livebench_eval import CATEGORIES, suite_from_rows
from nexus_expanse.patching import CompactBigram
from nexus_expanse.self_distill import DELIMITER, builtin_train_suite, distillation_partitions, harvest_verified_traces, write_harvest
from nexus_expanse.ssd_engram import SSDEngramStore
from nexus_expanse.unified import NexusExpanseUnified
from nexus_expanse.verified_search import ByteCandidate, SearchResult, Verification, VerifiedByteSearch
from scripts.prepare_gpu_corpus import CorpusOverlapError


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def suite(*prompts):
    return suite_from_rows([dict(question_id=f"task-{i}", category="math", verifier="arithmetic", turns=[p])
                            for i, p in enumerate(prompts)])


class Stub:
    snapshot_sha256 = "0" * 64

    def __init__(self, raw, *, forge=False, calls=0):
        self.raw = raw
        self.forge = forge
        self.calls = calls
        self.seeds = []

    def run(self, prompt, verify, *, seed, **options):
        self.seeds.append(seed)
        data = self.raw[prompt]
        checks = [Verification(True, 1, 1, None) if self.forge else verify(item) for item in data]
        return SearchResult(tuple(ByteCandidate(item, -float(i+1), result, 0.0)
                                  for i, (item, result) in enumerate(zip(data, checks, strict=True))), self.calls)


def test_only_verified_exact_bytes_enter_training_stream_and_audit(tmp_path):
    tasks = suite("1+2=", "1+4=", "6*7=")
    engine = Stub({b"1+2=":(b"3\nunchecked tail", b"3\n"), b"1+4=":(b"not verified\x00\xff",), b"6*7=":(b"42\n",)})
    got = harvest_verified_traces(engine, tasks, seed=7)
    assert engine.seeds == [7, 71, 135] and got == harvest_verified_traces(engine, tasks, seed=7)
    out, audit, eval_file = (tmp_path / name for name in ("replay.bin", "audit.jsonl", "eval.bin"))
    eval_file.write_bytes(b"123+45=168\n")
    write_harvest(got, out, out_jsonl=audit, evaluation=eval_file.read_bytes())
    assert out.read_bytes() == b"1+2=3\n" + DELIMITER + b"6*7=42\n" + DELIMITER
    receipt = got.report()
    assert receipt["accepted"] == 2 and receipt["rejected"] == 1 and receipt["donor_forward_calls"] == 0
    assert receipt["counts"]["math"] == dict(accepted=2, rejected=1)
    data = out.read_bytes()
    tasks_by_id = {task.id:task for task in tasks.tasks}
    rows = [json.loads(line) for line in audit.read_bytes().splitlines()]
    assert [r["question_id"] for r in rows] == ["task-0", "task-2"]
    for record in rows:
        raw = data[record["offset"]:record["offset"]+record["length"]]
        task = tasks_by_id[record["question_id"]]
        value = raw[len(task.prompt):len(task.prompt)+record["solution_bytes"]]
        assert raw == task.prompt+value+DELIMITER and task.verify(value).passed
        assert record["output_sha256"] == receipt["output_sha256"] == hashlib.sha256(data).hexdigest()
    train, evaluation, manifest = scaling.load_corpora(SimpleNamespace(smoke=False, train_file=out, eval_file=eval_file))
    assert train["stream"] == data and evaluation["stream"] == eval_file.read_bytes()
    assert manifest["train"]["stream"]["sha256"] == receipt["output_sha256"]


def test_independent_verification_blocks_forged_pass_and_donor_result(tmp_path):
    tasks = suite("1+2=")
    result = harvest_verified_traces(Stub({b"1+2=":(b"900\n",)}, forge=True), tasks)
    assert result.payload == b"" and not result.records and result.report()["rejected"] == 1
    write_harvest(result, tmp_path / "empty.bin", out_jsonl=tmp_path / "empty.jsonl")
    assert (tmp_path / "empty.bin").read_bytes() == (tmp_path / "empty.jsonl").read_bytes() == b""
    with pytest.raises(RuntimeError, match="donor forward"):
        harvest_verified_traces(Stub({b"1+2=":(b"3\n",)}, calls=1), tasks)


def test_overlap_rejects_even_a_window_crossing_trace_join(tmp_path):
    prompts = ("This is the first raw prompt window\n1+2=", "This is the next raw prompt reference\n3+5=")
    tasks = suite(*prompts)
    result = harvest_verified_traces(Stub({prompts[0].encode():(b"3\n",), prompts[1].encode():(b"8\n",)}), tasks)
    out, audit = tmp_path / "out.bin", tmp_path / "audit.jsonl"
    middle = result.records[1]["offset"]
    with pytest.raises(CorpusOverlapError, match="32-byte"):
        write_harvest(result, out, out_jsonl=audit, evaluation=result.payload[middle-16:middle+16])
    assert not out.exists() and not audit.exists()
    out.write_bytes(b"must remain intact")
    with pytest.raises(FileExistsError):
        write_harvest(result, out, out_jsonl=audit)
    assert out.read_bytes() == b"must remain intact" and not audit.exists()


def test_cli_harvest_int8_engram_replays_files_exactly(tmp_path):
    snapshot, source, index, tasks, eval_file = (tmp_path / k for k in ("model.pt", "fit.bin", "fit.db", "tasks.jsonl", "eval.bin"))
    snapshot.write_bytes(NexusExpanseUnified.restore(space_snapshot(22)).snapshot(snapshot_quant="int8"))
    source.write_bytes(b"1+2=3\n" * 60 + b"1+4=7\n" * 60)
    store = SSDEngramStore.build(source, index, max_order=4)
    tasks.write_text("\n".join(json.dumps(dict(question_id=i, category="math", turns=[p]))
                                for i,p in enumerate(("1+2=", "1+4="))) + "\n")
    eval_file.write_bytes(b"6*7=42\n")
    args = ("harvest-verified", "--snapshot", snapshot, "--tasks-jsonl", tasks, "--candidates", "1", "--max-bytes", "4",
            "--always-wake", "--engram-index", index, "--engram-index-sha256", store["index_sha256"],
            "--engram-alpha", "0.9", "--eval-file", eval_file)
    receipts = []
    for i in range(2):
        output, audit = tmp_path / f"out{i}.bin", tmp_path / f"out{i}.jsonl"
        report = call_cli(*args, "--out-bin", output, "--out-jsonl", audit)
        assert report.pop("wall_seconds") >= 0
        assert report["accepted"] == 1 and report["rejected"] == 1 and report["shared_32_byte_windows"] == 0
        assert report["donor_forward_calls"] == 0
        assert output.read_bytes() == b"1+2=3\n\n\n"
        receipts.append((report, output.read_bytes(), audit.read_bytes()))
    assert receipts[0] == receipts[1]


def test_builtin_train_pool_covers_all_categories_without_held_prompts():
    result = builtin_train_suite()
    _, _, held = distillation_partitions()
    assert len(result.tasks) == 40 and not result.unsupported
    assert {t.category for t in result.tasks} == set(CATEGORIES)
    assert {t.prompt for t in result.tasks}.isdisjoint(row["turns"][0].encode() for row,_ in held)


@pytest.fixture
def conflicting_memory(tmp_path):
    base = NexusExpanseUnified.restore(space_snapshot(22))
    counts = np.zeros((256, 256), dtype="<u4")
    counts[ord("2"), ord("=")] = 64
    counts[ord("="), ord("3")] = 64
    model = NexusExpanseUnified(base.backbone, CompactBigram.from_dense(counts))
    clear = model.snapshot(snapshot_quant="int8")
    text = b"1+2=9\nRepeated conflicting phrase.\n" * 40
    for i in range(1, len(text)):
        model.episodic.observe_transition(text[max(0, i-8):i], text[i])
    path = tmp_path / "conflicting.pt"
    path.write_bytes(model.snapshot(snapshot_quant="int8"))
    return path, clear


@pytest.mark.parametrize("command", ("generate", "search-verified", "eval-livebench", "harvest-verified"))
def test_no_episodic_cli_ignores_conflicting_memory_and_replays(command, tmp_path, conflicting_memory):
    path, _ = conflicting_memory
    args = (command, "--snapshot", path, "--max-bytes", "1")
    if command == "generate":
        args += ("--prompt-text", "1+2=")
    elif command == "search-verified":
        args += ("--prompt-text", "1+2=", "--verifier", "arithmetic", "--candidates", "1")
    else:
        tasks = tmp_path / "tasks.jsonl"
        tasks.write_text(json.dumps(dict(question_id="math", category="math", turns=["1+2="])) + "\n")
        args += ("--tasks-jsonl", tasks, "--candidates", "1")

    def run(index, *flags):
        output = tmp_path / f"harvest{index}.bin"
        extra = ("--out-bin", output) if command == "harvest-verified" else ()
        result = call_cli(*args, *extra, *flags)
        assert result["donor_forward_calls"] == 0
        timed = {"prefill_ms", "ms_per_byte", "wall_ms", "wall_seconds", "candidate_seconds"}
        return {key: value for key, value in result.items() if key not in timed}, (
            output.read_bytes() if extra else b"")

    legacy, legacy_bin = run(0)
    clean, clean_bin = run(1, "--no-episodic")
    assert clean["no_episodic"] is True and legacy["no_episodic"] is False
    assert (clean, clean_bin) == run(2, "--no-episodic")
    if command == "generate":
        stream = call_cli("stream", "--snapshot", path, "--input-hex", b"1+2=".hex(), "--no-adapt", "--continuation", "1")
        assert legacy["generated_hex"] == stream["continuation_hex"] == b"9".hex()
        assert clean["generated_hex"] == b"3".hex()
        assert (clean, clean_bin) == run(3, "--no-episodic", "--adapt")
    elif command == "search-verified":
        assert not legacy["verified"] and legacy["selected_text"] == "9"
        assert clean["verified"] and clean["selected_text"] == "3"
    elif command == "eval-livebench":
        assert legacy["overall_score"] == 0.0 and clean["overall_score"] == 1.0
        assert legacy["tasks"][0]["selected_hex"] == b"9".hex()
        assert clean["tasks"][0]["selected_hex"] == b"3".hex()
    else:
        assert legacy["accepted"] == 0 and legacy_bin == b""
        assert clean["accepted"] == 1 and clean_bin == b"1+2=3\n\n"


def test_clean_decode_prefix_and_active_engram_are_independent_of_episodic_state(tmp_path, conflicting_memory, monkeypatch):
    path, clear = conflicting_memory
    raw = path.read_bytes()
    runtime = NexusExpanseUnified.restore(raw, fast_decode=True)
    assert runtime.no_episodic is False
    runtime.no_episodic = True

    def forbidden(*_args):
        pytest.fail("clean decode consulted or updated episodic memory")

    monkeypatch.setattr(type(runtime.episodic), "predict_probs", forbidden)
    monkeypatch.setattr(type(runtime.episodic), "observe_transition", forbidden)
    forecast = runtime.prefill(b"1+2=", adapt=True)
    assert torch.equal(forecast.mixed, forecast.patch)
    after = runtime.step(ord("3"), adapt=True)
    assert torch.equal(after.mixed, after.patch)
    assert runtime.arbitrator.updates == 0

    source, index = tmp_path / "training.bin", tmp_path / "training.sqlite"
    source.write_bytes(b"1+2=3\n" * 50)
    SSDEngramStore.build(source, index)
    prompt = b"1+2="
    with SSDEngramStore(index) as store:
        assert store.lookup(prompt)[0] > 0
        opts = dict(no_episodic=True, always_wake=True, engram=store)
        engine = VerifiedByteSearch(raw, **opts)
        clean = VerifiedByteSearch(clear, **opts)
        prefix = b"3\x00\xff"
        a = engine.forecast_trace(prompt, prefix + b"first")
        b = engine.forecast_trace(prompt, prefix + b"other")
        n = len(prefix) + 1
        assert torch.equal(a, clean.forecast_trace(prompt, prefix + b"first"))
        assert torch.equal(a[:n], b[:n])
        assert torch.equal(a[:n], engine.forecast_trace(prompt, prefix))
