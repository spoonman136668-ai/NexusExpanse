"""Immutable, resumable CPU training receipts for the Gate 12–14 DEV suite."""
import fcntl
import json
import time
from contextlib import contextmanager

import torch

from benchmarks.gate9_jepa_dev import atomic, digest
from benchmarks.gate10_jepa_dev import save_pt


def prepare_directory(path, root):
    path = path.resolve()
    if path.is_relative_to(root):
        raise ValueError("checkpoint directory must be outside the source tree")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def writer(directory, name):
    with (directory / f"{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def peak_rss():
    # Linux's per-exec high-water mark excludes an unrelated parent's heap.
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("Linux peak RSS unavailable")


class EpochRun:
    def __init__(self, directory, name, identity, epochs, steps_per_epoch):
        self.directory = directory
        self.name = name
        self.identity = identity
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

    def paths(self, epoch):
        base = self.directory / f"{self.name}-epoch-{epoch:02}"
        return (base.with_suffix(".intent.json"), base.with_suffix(".pt"),
                base.with_suffix(".json"))

    @property
    def receipt(self):
        return self.directory / f"{self.name}.json"

    @property
    def checkpoint(self):
        return self.directory / f"{self.name}-final.pt"

    def history(self):
        parent = None
        history = []
        for epoch in range(1, self.epochs + 1):
            intent, checkpoint, meta = self.paths(epoch)
            exists = [p.exists() for p in (intent, checkpoint, meta)]
            if not any(exists):
                continue
            if not all(exists) or epoch != len(history) + 1:
                raise RuntimeError(f"ambiguous epoch custody, replay refused: {checkpoint}")
            expected = dict(identity=self.identity, name=self.name, epoch=epoch, parent=parent)
            record = json.loads(meta.read_text())
            if (json.loads(intent.read_text()) != expected
                    or any(record.get(k) != v for k, v in expected.items())
                    or record["sha256"] != digest(checkpoint)
                    or len(record["losses"]) != self.steps_per_epoch):
                raise ValueError(f"epoch identity or hash mismatch: {checkpoint}")
            parent = record["sha256"]
            history.append(record)
        return history

    def read(self):
        history = self.history()
        record = json.loads(self.receipt.read_text())
        if (len(history) != self.epochs or record["identity"] != self.identity
                or record["name"] != self.name or not record["cold_equal"]
                or record["epochs"] != history or record["checkpoint"] != self.checkpoint.name
                or record["checkpoint_sha256"] != digest(self.checkpoint)
                or record["steps"] != self.epochs * self.steps_per_epoch):
            raise ValueError(f"invalid completed receipt: {self.receipt}")
        return record

    def advance(self, model, optimizer, epoch_limit, train_epoch, cold, evaluate, initial_hash):
        with writer(self.directory, self.name):
            if self.receipt.exists():
                return self.read()
            start = time.monotonic()
            history = self.history()
            completed = len(history)
            if history:
                restored = torch.load(self.paths(completed)[1], weights_only=True)
                model.load_state_dict(restored["model"])
                optimizer.load_state_dict(restored["optimizer"])
                torch.set_rng_state(restored["rng"])
            for epoch in range(completed + 1, min(completed + epoch_limit, self.epochs) + 1):
                # A slice finishes at an epoch boundary. Leave a conservative
                # multiple of the slowest measured epoch for the next one.
                if epoch > completed + 1:
                    estimate = max(item["wall_seconds"] for item in history)
                    if time.monotonic() - start + 2 * estimate > 48:
                        break
                intent, checkpoint, meta = self.paths(epoch)
                data = dict(identity=self.identity, name=self.name, epoch=epoch,
                            parent=history[-1]["sha256"] if history else None)
                atomic(intent, data)
                wall, cpu = time.monotonic(), time.process_time()
                model.train()
                losses = train_epoch(epoch)
                if len(losses) != self.steps_per_epoch:
                    raise ValueError("wrong number of optimizer steps")
                save_pt(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                         rng=torch.get_rng_state()))
                data.update(sha256=digest(checkpoint), losses=losses,
                            wall_seconds=time.monotonic() - wall,
                            cpu_seconds=time.process_time() - cpu, peak_rss_bytes=peak_rss())
                atomic(meta, data)
                history.append(data)
                print(f"{self.name} epoch {epoch}/{self.epochs} {data['wall_seconds']:.2f}s", flush=True)
            if len(history) != self.epochs:
                return None

            from nexus_expanse.jepa import state_hash
            model.eval()
            final_hash = state_hash(model)
            if not self.checkpoint.exists():
                save_pt(self.checkpoint, model.state_dict())
            cold_hash, cold_seconds, cold_rss = cold(self.checkpoint)
            if cold_hash != final_hash:
                raise ValueError("fresh-process checkpoint digest mismatch")
            metrics = evaluate()
            if state_hash(model) != final_hash:
                raise ValueError("evaluation modified frozen weights")
            record = dict(
                identity=self.identity, name=self.name, initial_hash=initial_hash,
                epochs=history, steps=self.epochs * self.steps_per_epoch,
                training_seconds=sum(item["wall_seconds"] for item in history),
                training_cpu_seconds=sum(item["cpu_seconds"] for item in history),
                checkpoint=self.checkpoint.name, checkpoint_sha256=digest(self.checkpoint),
                final_hash=final_hash, cold_equal=True, cold_restore_seconds=cold_seconds,
                cold_peak_rss_bytes=cold_rss, evaluation_peak_rss_bytes=peak_rss(), **metrics,
            )
            atomic(self.receipt, record)
            return record


def update(model, optimizer, loss):
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())
