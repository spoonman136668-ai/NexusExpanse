"""Offline structured verifier and local LiveBench integration coverage."""

import json

import pytest
import torch

from benchmarks.phase15_cosine_lr_and_sampling import call_cli, space_snapshot
from nexus_expanse.livebench_eval import CATEGORIES, dev_records, load_tasks_jsonl, suite_from_rows
from nexus_expanse.ssd_engram import SSDEngramStore
from nexus_expanse.unified import NexusExpanseUnified
from nexus_expanse.verified_search import VerifiedByteSearch, make_verifier


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


def test_csp_solves_both_assignment_formats_and_requires_every_constraint():
    spec = {"variables": {k: [1, 2, 3] for k in ("A", "B", "C")}, "constraints": [
        {"type": "all_different", "variables": ["A", "B", "C"]},
        {"type": "equals", "left": "A", "value": 1}, {"type": "not_equals", "left": "A", "right": "C"},
        {"type": "adjacent", "left": "B", "right": "C"}, {"type": "before", "left": "B", "right": "C"},
        {"type": "implies", "if": {"type": "equals", "left": "A", "value": 1},
         "then": {"type": "equals", "left": "B", "value": 2}},
    ]}
    check = make_verifier("logic_csp", b"Solve", spec=spec)
    for raw in (b"A=1,B=2,C=3\n", b'{"C":3,"A":1,"B":2}', b"A=1\nB=2\nC=3"):
        assert check(raw) == (True, 6, 6, None)
    assert check(b"A=1,B=3,C=2") == (False, 4, 6, None)
    for raw in (b"A=1,B=2", b"A=1,A=1,B=2,C=3", b"A=1,B=2,C=4", b"A=1,B=2,C=3,D=2",
                b'{"A":true,"B":2,"C":3}', b'{"A":1,"A":2,"B":2,"C":3}', b"A=1,B=2,C=3,__class__=1"):
        assert check(raw)[:3] == (False, 0, 6)
    malformed = {**spec, "constraints": [{"type": "before", "left": "A", "right": "D"}]}
    with pytest.raises(ValueError, match="unknown CSP"):
        make_verifier("logic_csp", b"Solve", spec=malformed)


@pytest.mark.parametrize("target,valid,invalid", [
    ("json", b'[{"name":"red","value":2},{"name":"blue","value":4}]',
     b'[{"name":"red","value":2},{"name":"blue","value":"4"}]'),
    ("csv", b'name,value\r\n"red",2\r\nblue,4\r\n', b'name,value\n"red,2\nblue,4'),
    ("tsv", b'name\tvalue\nred\t2\nblue\t4', b'name\tvalue\nred\t2\nblue\t5'),
    ("markdown", b'| name | value |\n| :--- | ---: |\n| red | 2 |\n| blue | 4 |\n',
     b'| name | value |\n| bad | bad |\n| red | 2 |\n| blue | 4 |'),
])
def test_tables_require_exact_headers_counts_rows_and_target_grammar(target, valid, invalid):
    spec = {"format": target, "columns": ["name", "value"], "rows": [["red", 2], ["blue", 4]]}
    check = make_verifier("table_format", b"Format the table", spec=spec)
    assert check(valid) == (True, 4, 4, None)
    assert check(valid) == check(valid)
    assert not check(invalid).passed and not check(valid.replace(b"name", b"names")).passed
    assert not check(b"```\n" + valid + b"\n```").passed
    assert not check(valid + b"\x00").passed


def test_text_constraints_count_individual_checks():
    spec = dict(min_words=4, max_words=4, required_substrings=["red", "blue"], forbidden_substrings=["green"],
                must_start_with="- red", must_end_with="blue", exact_lines=2, bullet_count=2)
    check = make_verifier("instruction_constraints", b"Follow the constraints", spec=spec)
    assert check(b"- red\r\n- blue\r\n\r\n") == (True, 9, 9, None)
    assert check(b"- red\n- green") == (False, 6, 9, None)
    assert check(b"- red\n- blue\x1b")[:3] == (False, 0, 9)
    with pytest.raises(ValueError):
        make_verifier("instruction_constraints", b"Prompt", spec={"open": "anything"})
    with pytest.raises(ValueError):
        make_verifier("instruction_constraints", b"Prompt", spec={"min_words": 5, "max_words": 1})


def test_official_row_adapters_and_explicit_unsupported(tmp_path):
    source = tmp_path / "rows.jsonl"
    rows = [
        dict(question_id=1, category="math", task="math_comp", turns=["What is six times eight?"], ground_truth="48"),
        dict(question_id=2, category="coding", turns=["def f(x):\n    "], tests=[{"args":[2],"expected":3}]),
        dict(question_id=3, category="reasoning", turns=["Locate A and B"], ground_truth={"A": 1, "B": 2}),
        dict(question_id=4, category="data_analysis", task="tablereformat", turns=["Target Format: csv"], ground_truth="name,value\na,2\n"),
        dict(question_id=5, category="instruction_following", turns=["Write a bullet list"],
             instruction_id_list=["detectable_format:number_bullet_lists"], kwargs=[{"num_bullets":1}]),
        dict(question_id=6, category="language", turns=["Read this"], ground_truth="yes"),
        dict(question_id=7, category="coding", turns=["Code this"], tests="assert generated(1)==1"),
        dict(question_id=8, category="math", turns=["first", "second"], ground_truth=1),
    ]
    source.write_text("\n".join(map(json.dumps, rows)) + "\n")
    suite = load_tasks_jsonl(source)
    assert tuple(t.category for t in suite.tasks) == CATEGORIES
    for t, answer in zip(suite.tasks, (b"48\n", b"return x+1", b"A=1,B=2", b"name,value\na,2", b"- ready"), strict=True):
        assert t.verify(answer).passed
        assert not t.verify(b"not an answer").passed
    assert [r["question_id"] for r in suite.unsupported] == ["6", "7", "8"]
    source.write_text('{"question_id":1,"question_id":2}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_tasks_jsonl(source)
    with pytest.raises(ValueError, match="duplicate question ID"):
        suite_from_rows([rows[0], rows[0]])


def canonical(report):
    return {k:v for k,v in report.items() if k not in ("wall_seconds", "candidate_seconds")}


def test_eval_cli_builtin_and_custom_determinism_and_coverage(tmp_path):
    raw = NexusExpanseUnified.restore(space_snapshot(21)).snapshot(snapshot_quant="int8")
    archive = tmp_path / "weights.pt"
    archive.write_bytes(raw)
    common = ("eval-livebench", "--snapshot", archive, "--device", "cpu", "--max-bytes", "2", "--candidates", "1", "--always-wake")
    built = [call_cli(*common, "--builtin-dev-suite") for _ in range(2)]
    assert canonical(built[0]) == canonical(built[1])
    assert built[0]["graded_categories"] == 5
    assert [built[0]["counts"][key]["total"] for key in CATEGORIES] == [6] * 5
    _, _, held = dev_records()
    path = tmp_path / "suite.jsonl"
    extra = {"question_id":"uncovered", "category":"coding", "turns":["Solve"], "tests":"not pure cases"}
    path.write_text("\n".join(json.dumps(row) for row in [held[0][0], extra]) + "\n")
    local = [call_cli(*common, "--tasks-jsonl", path) for _ in range(2)]
    assert canonical(local[0]) == canonical(local[1])
    item = local[0]
    assert item["math"] == item["overall_score"] and item["reasoning"] is None
    assert item["graded_categories"] == 1 and item["counts"]["coding"]["unsupported"] == 1
    assert item["donor_forward_calls"] == built[0]["donor_forward_calls"] == 0


def test_search_cli_spec_and_active_engram_prefix_are_exact(tmp_path):
    prompt = "Use equal slots A=1.\n"
    archive, source, index = (tmp_path / name for name in ("s.pt", "fit.bin", "fit.db"))
    raw = NexusExpanseUnified.restore(space_snapshot(5)).snapshot(snapshot_quant="int8")
    archive.write_bytes(raw)
    source.write_bytes((prompt.encode()+b"A=1\n\n")*60)
    SSDEngramStore.build(source, index)
    spec = {"variables":{"A":[1]}, "constraints":[{"type":"equals", "left":"A", "value":1}]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    args = ("search-verified", "--snapshot", archive, "--verifier", "logic_csp", "--prompt-text", prompt, "--spec-json")
    rest = ("--candidates", "1", "--max-bytes", "8", "--always-wake", "--engram-index", index, "--engram-alpha", "0.9")
    a, b = [call_cli(*args, value, *rest) for value in (json.dumps(spec), path)]
    assert a.pop("wall_ms") >= 0 and b.pop("wall_ms") >= 0
    assert a == b and a["verified"] and a["passed_tests"] == a["total_tests"] == 1
    with SSDEngramStore(index) as store:
        engine = VerifiedByteSearch(raw, engram=store, always_wake=True)
        assert store.lookup(prompt.encode())[0] > 0
        first = engine.forecast_trace(prompt.encode(), b"A=1\x00\xff")
        second = engine.forecast_trace(prompt.encode(), b"A=1\xff\x00")
        assert torch.equal(first[:4], second[:4])
        assert torch.equal(first[:4], engine.forecast_trace(prompt.encode(), b"A=1"))


def test_builtin_fixtures_have_all_five_objective_reference_answers():
    fit, val, held = dev_records()
    groups = [{r["turns"][0] for r, _ in split} for split in (fit, val, held)]
    assert sum(map(len, groups)) == len(set.union(*groups))
    suite = suite_from_rows(row for row, _ in held)
    assert len(suite.tasks) == 30 and not suite.unsupported
    assert all(task.verify(answer).passed for task, (_, answer) in zip(suite.tasks, held, strict=True))
