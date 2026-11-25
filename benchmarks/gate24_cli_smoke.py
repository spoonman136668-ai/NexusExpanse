"""Fresh-command Gate 24 CLI checks on a pinned Linux CPU profile."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.dev_artifacts import atomic, digest
from benchmarks.gate_custody import prepare_directory, writer
from nexus_expanse.cli import corpus_paths


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run_smoke(directory):
    data = directory / "corpus"
    data.mkdir()
    corpus = {
        "train_math": b"2+4=6\n6*3=18\n4+5=9\n", "eval_math": b"8+7=15\n7*2=14\n",
        "train_code": b"def twice(x):\n    return x * 2\n", "eval_code": b"def next(x):\n    return x + 1\n",
        "train_prose": b"Raw bytes form bounded records.\n", "eval_prose": b"Each frame has an observed end.\n",
    }
    for key, path in corpus_paths(data).items():
        path.write_bytes(corpus[key])
    training = directory / "training"
    unified = directory / "verified-unified.pt"
    adapted = directory / "adapted-unified.pt"
    skills = directory / "verified-expression.bin"
    steps = []

    def invoke(*args, stdin=None):
        started = time.monotonic()
        args = list(map(str, args))
        r = subprocess.run([sys.executable, "-m", "nexus_expanse.cli", *args], cwd=ROOT,
                           input=stdin, capture_output=True, text=True, timeout=60)
        steps.append(dict(args=args, stdin=stdin, exit_code=r.returncode,
                          stdout=r.stdout, stderr=r.stderr, wall_seconds=time.monotonic() - started))
        if r.returncode:
            raise RuntimeError(f"{args[0]} smoke failed: {r.stdout}{r.stderr}")
        return [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]

    params = ("train", "--corpus-dir", data, "--directory", training, "--device", "cpu", "--dtype", "fp32",
              "--dim", 32, "--context", 16, "--epochs", 2, "--steps-per-epoch", 1,
              "--batch-per-domain", 1, "--eval-blocks", 1, "--epoch-limit", 1, "--save-snapshot", unified)
    invoke(*params)
    if not (training / "gate17-epoch-01.pt").is_file() or unified.exists():
        raise ValueError("first slice did not stop at the initial checkpoint")
    trained = invoke(*params, "--until-complete")[-1]
    receipt = json.loads((training / "gate17.json").read_text())
    first, second = [x["telemetry"] for x in receipt["epochs"]]
    if (first["worker_pid"] == second["worker_pid"]
            or first["end_model_hash"] != second["start_model_hash"]
            or second["optimizer_steps_at_start"] != 1
            or not receipt["cold_equal"] or trained["neural_state_sha256"] != receipt["final_hash"]):
        raise ValueError("exact training resume proof failed")
    checked = invoke("verify", "--snapshot", unified, "--snapshot-sha256", digest(unified))[-1]
    if not checked["exact_state"] or any(checked["parity"].values()) or checked["round_trip_probabilities"] != 0.:
        raise ValueError("verify command did not establish exactness")

    stream = invoke("stream", "--snapshot", unified, "--snapshot-sha256", digest(unified),
                    "--continuation", 3, "--save-snapshot", adapted, stdin="000102\n030405\n")
    actual = [r for r in stream if "observed_hex" in r]
    if (len(actual) != 2 or [r["arbitration_updates"] for r in actual] != [2, 5]
            or any(len(r["next"]) != 5 or len(r["continuation_hex"]) != 6 for r in actual)
            or stream[-1]["snapshot_sha256"] != digest(adapted)
            or stream[-1]["round_trip_probabilities"] != 0.):
        raise ValueError("stream prediction, observed-only update or snapshot failed")

    solved = invoke("solve", "--source-hex", "010203", "--target-hex", "040302",
                    "--depth", 2, "--save-agent-snapshot", skills)[-1]
    if not solved["solved"] or solved["residual_hex"] != "000000" or not 1 <= solved["calls"] <= 6:
        raise ValueError("solve did not produce the exact expression target")
    return dict(status="PASS", steps=steps, epoch_workers=[first["worker_pid"], second["worker_pid"]],
                training_receipt=receipt, verify=checked, stream=stream, solve=solved,
                corpus_sha256={k: digest(p) for k, p in corpus_paths(data).items()},
                snapshots={p.name: digest(p) for p in (unified, adapted, skills)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if git("status", "--porcelain"):
        raise ValueError("smoke requires clean committed source")
    directory = prepare_directory(args.directory, ROOT)
    with writer(directory, "gate24"):
        result = run_smoke(directory)
        atomic(args.output, dict(schema="nexus-gate24-v1-rc1-cli-smoke-v1", qualification="LINUX_CPU_DEV_SMOKE",
                                 identity=dict(source_sha=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"),
                                               platform=platform.platform(), torch=str(torch.__version__)),
                                 **result))
    print(json.dumps(dict(status=result["status"], commands=[r["args"][0] for r in result["steps"]],
                          verify_parity=result["verify"]["parity"],
                          snapshot_round_trip=result["verify"]["round_trip_probabilities"],
                          solve_exact=result["solve"]["solved"])))


if __name__ == "__main__":
    main()
