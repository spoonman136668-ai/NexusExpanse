"""Development-only matched ablation on embedded synthetic fixtures; never held-out science."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.memory import EpisodicByteMemory
from nexus_expanse.substrate import static_footprint_bytes

# Known, deliberately small, artificial patterns. Not a generalization test.
TRAIN = b"a=1;b=2;c=3;\n" * 24 + b"red:up;blue:down;\n" * 16
DEV = b"a=1;b=2;c=4;\n" * 8 + b"red:down;blue:up;\n" * 8
CONFIG = dict(n_layers=2, dim=32, n_heads=4, ffn_mult=2, dropout=0.0,
              max_seq_len=32, memory_max_order=8, fast_lr=0.08)
# Each neural arm gets the same pretrained state and starting episodic table.
ARMS = {
    "neural": (False, False, False),
    "static_episodic": (True, False, False),
    "online_episodic": (True, True, False),
    "hebbian": (False, False, True),
    "static_episodic_hebbian": (True, False, True),
    "online_episodic_hebbian": (True, True, True),
}


def batches(data: bytes, seq_len: int = 32, batch_size: int = 8):
    """All training transitions, including remainder (no padded-target leakage)."""
    t = torch.tensor(list(data), dtype=torch.long)
    n = (len(t) - 1) // seq_len
    x = t[:n * seq_len].reshape(n, seq_len)
    y = t[1:n * seq_len + 1].reshape(n, seq_len)
    out = [(x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, n, batch_size)]
    if n * seq_len < len(t) - 1:
        out.append((t[n * seq_len:-1][None], t[n * seq_len + 1:][None]))
    return out


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode() + b"\0")
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def shared_init(seed, kind, skip, reference):
    torch.manual_seed(seed)
    model = NexusExpanseByteCore(**CONFIG, ffn_kind=kind, enable_skip=skip)
    if reference is not None:
        state = model.state_dict()
        for name, value in reference.state_dict().items():
            if ".ffn.fc" not in name:
                state[name] = value.clone()
        model.load_state_dict(state)
    return model


def memory_only(mem, data, online):
    mem = copy.deepcopy(mem)
    loss = 0.0
    for i in range(len(data) - 1):
        ctx = data[max(0, i + 1 - mem.max_order):i + 1]
        p, _ = mem.predict_probs(ctx)
        target = data[i + 1]
        loss -= math.log2(max(float(p[target]), 1e-12))
        if online:
            mem.observe_transition(ctx, target)
    return {"bpb": loss / (len(data) - 1), "bytes": len(data) - 1,
            "episodic_entries": mem.memory_entry_count()}


def run_dev(seed=42, epochs=3):
    if epochs < 1:
        raise ValueError("positive training epochs required")
    train_batches = batches(TRAIN)
    reference = shared_init(seed, "monarch", False, None)
    memory = EpisodicByteMemory(max_order=CONFIG["memory_max_order"])
    memory.ingest_stream(TRAIN)
    result = {"seed": seed, "epochs": epochs, "train_transitions_per_epoch": len(TRAIN) - 1,
              "optimizer_steps_per_backbone": epochs * len(train_batches), "backbones": {},
              "memory_only": {"static": memory_only(memory, DEV, False),
                              "online": memory_only(memory, DEV, True)}}
    for kind in ("monarch", "dense"):
        for skip in (False, True):
            model = shared_init(seed, kind, skip, reference)
            model.episodic = copy.deepcopy(memory)
            opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
            init_hash = state_hash(model)
            start = time.perf_counter()
            for _ in range(epochs):
                model.train()
                for x, y in train_batches:
                    opt.zero_grad(set_to_none=True)
                    logits, _, aux, _ = model.forward_neural(x, targets=y)
                    loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) + 0.1 * aux
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                if skip:
                    model.calibrate_gates(train_batches)
            info = {"init_sha256": init_hash, "trained_state_sha256": state_hash(model),
                    "train_seconds": time.perf_counter() - start,
                    "ffn_bytes": model.total_ffn_footprint()[1],
                    "module_tensor_bytes": static_footprint_bytes(model)[1], "arms": {}}
            for name, (episodic, online, hebbian) in ARMS.items():
                arm = copy.deepcopy(model)
                assert state_hash(arm) == info["trained_state_sha256"]
                start = time.perf_counter()
                score = arm.evaluate_streaming(DEV, seq_len=32, use_episodic=episodic,
                    online_episodic_update=online, online_hebbian_update=hebbian)
                score["eval_seconds"] = time.perf_counter() - start
                info["arms"][name] = score
            assert state_hash(model) == info["trained_state_sha256"]
            result["backbones"][f"{kind}_{'skip' if skip else 'full'}"] = info
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    if git("status", "--porcelain"):
        raise RuntimeError("development evidence requires a clean Git tree")
    result = {"schema": "nexus-development-ablation-v1", "source_sha": git("rev-parse", "HEAD"),
              "source_tree": git("rev-parse", "HEAD^{tree}"), "scientific_trial_executed": False,
              "qualification": "DEVELOPMENT_ONLY", "device": "cpu", "torch": torch.__version__,
              "python": platform.python_version(), "platform": platform.platform(), "threads": 1,
              "config": CONFIG, "fixture": {"train_bytes": len(TRAIN), "dev_bytes": len(DEV),
              "train_sha256": hashlib.sha256(TRAIN).hexdigest(), "dev_sha256": hashlib.sha256(DEV).hexdigest()},
              "runs": [run_dev(seed) for seed in (42, 43, 44)]}
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
