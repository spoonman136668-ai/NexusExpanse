"""Frozen, locally exported next-byte distributions and their static shard format."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

import torch

from nexus_expanse.donor_assimilation import DenseTokenDonorLM, canonical_byte_bpe

SCHEMA = "nexus.donor-byte-cache.v1"
MANIFEST = "cache_manifest.json"
TOP_K = 32
EXPORT_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _gpt2_decoder():
    direct = [*range(33, 127), *range(161, 173), *range(174, 256)]
    extra = [b for b in range(256) if b not in direct]
    return dict(zip(map(chr, direct + list(range(256, 256 + len(extra)))), direct + extra, strict=True))


GPT2_BYTES = _gpt2_decoder()
FALLBACK_TOKEN = re.compile(r"<0x([A-Fa-f0-9]{2})>")


def token_bytes(token, family):
    """Decode one vocabulary token without Unicode replacement or per-token stripping."""
    if family == "gpt2":
        try:
            return bytes(GPT2_BYTES[c] for c in token)
        except KeyError:
            return token.encode("utf-8")
    if family == "sentencepiece":
        match = FALLBACK_TOKEN.fullmatch(token)
        if match:
            return bytes([int(match[1], 16)])
        return token.replace("▁", " ").encode("utf-8")
    raise ValueError(f"unsupported tokenizer family {family}")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_nexus_donor(path, n_heads):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["weights"]
    size, dim = state["emb.weight"].shape
    depth = sum(key.endswith(".attn.in_proj_weight") for key in state)
    donor = DenseTokenDonorLM(
        vocab_size=size, dim=dim, n_layers=depth, n_heads=n_heads,
        ffn_mult=state["layers.0.fc1.weight"].shape[0] // dim,
        max_seq_len=state["pos_emb.weight"].shape[0],
    )
    donor.load_state_dict(state)
    if "byte_to_row" in payload:
        rows = payload["byte_to_row"].tolist()
        if size != 256 or sorted(rows) != list(range(256)):
            raise ValueError("singleton donor must map every byte to a unique row")
        first_bytes = torch.tensor(rows).argsort()
        encode = lambda raw: tuple(rows[byte] for byte in raw)
    else:
        vocab = payload["vocab_bytes"]
        merges = payload["merges"]
        ranks = {tuple(pair): i for i, pair in enumerate(merges, 256)}
        if (len(vocab) != size or len(ranks) != size - 256 or len(set(vocab)) != size
                or vocab[:256] != [bytes([byte]) for byte in range(256)]
                or any(not (0 <= a < i and 0 <= b < i) or vocab[i] != vocab[a] + vocab[b]
                       for (a, b), i in ranks.items())):
            raise ValueError("invalid dense donor BPE vocabulary or merges")
        first_bytes = torch.tensor([spelling[0] for spelling in vocab])
        encode = lambda raw: canonical_byte_bpe(raw, ranks)
    return donor.eval().requires_grad_(False), encode, first_bytes, lambda ids: donor(ids)


def _decoder_types(decoder):
    return {decoder.get("type")} | {
        kind for child in decoder.get("decoders", []) for kind in _decoder_types(child)
    }


def _require_safetensors(directory):
    """Resolve only local model files; accept standard unsharded and sharded layouts."""
    for name in ("config.json", "tokenizer.json"):
        path = directory / name
        if not path.is_file() or path.resolve().parent != directory:
            raise ValueError(f"missing local Hugging Face {name}")
    if (directory / "adapter_config.json").exists():
        raise ValueError("Hugging Face backend requires a standalone model, not an adapter")
    index = directory / "model.safetensors.index.json"
    if (directory / "model.safetensors").exists():
        names = {"model.safetensors"}
    elif index.is_file() and index.resolve().parent == directory:
        names = set(json.loads(index.read_bytes())["weight_map"].values())
    else:
        raise ValueError("Hugging Face backend requires local model.safetensors weights")
    if not names or any(
        Path(name).name != name or not name.endswith(".safetensors")
        or not (directory / name).is_file() or (directory / name).resolve().parent != directory
        for name in names
    ):
        raise ValueError("Hugging Face safetensors index references missing or nonlocal weights")


def load_hf_donor(path, *, dtype=None):
    directory = path.resolve(strict=True)
    _require_safetensors(directory)
    config = json.loads((directory / "tokenizer.json").read_bytes())
    decoders = _decoder_types(config.get("decoder") or {})
    if "ByteLevel" in decoders:
        family = "gpt2"
    elif decoders & {"Metaspace", "ByteFallback", "Replace"}:
        family = "sentencepiece"
    else:
        raise ValueError("tokenizer.json must use GPT-2 ByteLevel or SentencePiece decoding")
    try:
        from transformers import AutoModelForCausalLM
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ValueError("the Hugging Face backend requires local transformers and tokenizers") from exc

    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    precision = {"torch_dtype": dtype} if dtype is not None else {}
    model = AutoModelForCausalLM.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False, use_safetensors=True, **precision,
    ).eval().requires_grad_(False)
    size = model.get_output_embeddings().weight.shape[0]
    spellings = {}
    first = torch.full((size,), -1, dtype=torch.long)
    vocab = tokenizer.get_vocab()
    special_ids = {item["id"] for item in config.get("added_tokens", []) if item.get("special")}
    for token, index in vocab.items():
        if 0 <= index < size and index not in special_ids:
            spelling = token_bytes(token, family)
            if spelling:
                spellings[index] = spelling
                first[index] = spelling[0]
    singleton = {spelling[0]: index for index, spelling in spellings.items() if len(spelling) == 1}
    if family == "sentencepiece":
        # Use explicit byte fallback for invalid UTF-8 and incomplete code points.
        singleton = {
            int(match[1], 16): index
            for token, index in vocab.items()
            if index in spellings and (match := FALLBACK_TOKEN.fullmatch(token))
        }

    def encode(raw):
        # Surrogate escape preserves invalid and incomplete UTF-8 bytes as individual bytes.
        decoded = raw.decode("utf-8", errors="surrogateescape")
        text = re.split(r"([\udc80-\udcff]+)", decoded)
        ids = []
        for segment in text:
            if not segment:
                continue
            binary = 0xDC80 <= ord(segment[0]) <= 0xDCFF
            if binary:
                try:
                    ids.extend(singleton[ord(c) - 0xDC00] for c in segment)
                except KeyError as exc:
                    raise ValueError("tokenizer lacks a required singleton byte") from exc
            else:
                part = tokenizer.encode(segment, add_special_tokens=False).ids
                actual = b"".join(spellings.get(index, b"") for index in part)
                expected = segment.encode()
                # SentencePiece can prepend one virtual space at the start of a coded segment.
                if (any(index not in spellings for index in part)
                        or (actual != expected and not (
                            family == "sentencepiece" and actual == b" " + expected
                        ))):
                    raise ValueError("tokenizer does not preserve input bytes")
                ids.extend(part)
        return tuple(ids)

    def forward(ids):
        return model(input_ids=ids, use_cache=False).logits

    return model, encode, first, forward


@torch.inference_mode()
def _compact_windows(windows, encode, first_bytes, forward, *, device="cpu"):
    # Each query encodes only its observed byte prefix. Vocabulary rows with the
    # same leading byte share probability mass. Special tokens are excluded.
    queries = [encode(window[:t]) for window in windows for t in range(1, len(window))]
    unique = list(dict.fromkeys(queries))
    if not unique or any(not ids for ids in unique):
        raise ValueError("donor tokenizer produced an empty observed context")
    valid = first_bytes >= 0
    if valid.sum().item() < TOP_K or first_bytes[valid].unique().numel() < TOP_K:
        raise ValueError("donor vocabulary must represent at least 32 distinct first bytes")
    predictions = []
    for start in range(0, len(unique), 64):
        group = unique[start:start + 64]
        width = max(map(len, group))
        padded = torch.tensor([ids + (0,) * (width - len(ids)) for ids in group], device=device)
        last = torch.tensor([len(ids) - 1 for ids in group], device=device)
        selected = forward(padded)[torch.arange(len(group), device=device), last]
        # Move only the queried rows across devices. Accumulate byte mass on CPU in FP64.
        logits = selected.cpu().double()[:, valid]
        if not torch.isfinite(logits).all():
            raise ValueError("donor emitted nonfinite logits")
        probs = logits.softmax(-1)
        mass = torch.zeros((len(group), 256), dtype=probs.dtype)
        mass.scatter_add_(1, first_bytes[valid].expand_as(probs), probs)
        predictions.append(mass)
    indices = {ids: i for i, ids in enumerate(unique)}
    mass = torch.cat(predictions)[[indices[ids] for ids in queries]]
    values, top = mass.topk(TOP_K)
    shape = (len(windows), len(windows[0]) - 1, TOP_K)
    return {
        "raw": torch.tensor([list(window) for window in windows], dtype=torch.uint8),
        "indices": top.reshape(shape).to(torch.uint8),
        "probs": (values / values.sum(dim=-1, keepdim=True)).reshape(shape).to(torch.float16),
    }


def export_donor_cache(directory, windows, *, backend, checkpoint, shard_windows=16, n_heads=4,
                       device="auto", dtype="fp32"):
    """Publish a complete cache directory only after all shards have been written."""
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(directory)
    if (not windows or not 2 <= len(windows[0]) <= 64 or shard_windows < 1
            or any(type(window) is not bytes or len(window) != len(windows[0]) for window in windows)):
        raise ValueError("expected nonempty equal-size raw windows of 2–64 bytes and positive shard size")
    if device not in ("auto", "cpu", "cuda") or dtype not in EXPORT_DTYPES:
        raise ValueError("invalid donor export device or dtype")
    if device == "auto":
        device = "cuda" if backend == "hf" and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if dtype == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("the CUDA device lacks native BF16 support")
    precision = EXPORT_DTYPES[dtype]
    if backend == "nexus_donor":
        donor, encode, first, forward = load_nexus_donor(checkpoint, n_heads)
        if len(windows[0]) - 1 > donor.pos_emb.num_embeddings:
            raise ValueError("raw window can exceed the dense donor context")
    elif backend == "hf":
        donor, encode, first, forward = load_hf_donor(checkpoint, dtype=precision)
    else:
        raise ValueError("unknown donor backend")
    donor.to(device=device, dtype=precision)

    manifest = dict(schema=SCHEMA, backend=backend, top_k=TOP_K, window_bytes=len(windows[0]), shards=[])
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".donor-cache-") as scratch:
        stage = Path(scratch) / "cache"
        stage.mkdir()
        for start in range(0, len(windows), shard_windows):
            group = windows[start:start + shard_windows]
            path = stage / f"shard-{len(manifest['shards']):06d}.pt"
            torch.save(_compact_windows(group, encode, first, forward, device=device), path)
            data = path.read_bytes()
            manifest["shards"].append(dict(file=path.name, bytes=len(data), sha256=sha256(data), windows=len(group)))
        manifest["windows"] = len(windows)
        manifest["payload_bytes"] = sum(shard["bytes"] for shard in manifest["shards"])
        (stage / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(directory)
    return manifest


export_cache = export_donor_cache


def read_cache(directory):
    """Verify the exact stored bytes before a weights-only load of each static shard."""
    manifest_path = directory / MANIFEST
    if manifest_path.is_symlink():
        raise ValueError("donor cache manifest must be a local file")
    metadata = manifest_path.read_bytes()
    manifest = json.loads(metadata)
    if not isinstance(manifest, dict):
        raise ValueError("invalid donor cache manifest")
    width = manifest.get("window_bytes")
    if (manifest.get("schema") != SCHEMA or manifest.get("top_k") != TOP_K
            or type(width) is not int or not 2 <= width <= 64
            or not isinstance(manifest.get("shards"), list) or not manifest["shards"]):
        raise ValueError("invalid donor cache manifest")
    parts = []
    totals = 0
    for i, entry in enumerate(manifest["shards"]):
        name = f"shard-{i:06d}.pt"
        path = directory / name
        if not isinstance(entry, dict) or entry.get("file") != name or path.is_symlink():
            raise ValueError("invalid donor cache shard path")
        data = path.read_bytes()
        if len(data) != entry.get("bytes") or sha256(data) != entry.get("sha256"):
            raise ValueError(f"donor cache integrity mismatch for {name}")
        shard = torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
        if (not isinstance(shard, dict) or set(shard) != {"raw", "indices", "probs"}
                or any(not isinstance(value, torch.Tensor) for value in shard.values())):
            raise ValueError(f"invalid donor cache tensors in {name}")
        raw, indices, probs = shard["raw"], shard["indices"], shard["probs"]
        count = entry.get("windows")
        shape = (count, width - 1, TOP_K)
        if (type(count) is not int or count < 1 or raw.shape != (count, width)
                or raw.dtype != torch.uint8 or indices.shape != shape or probs.shape != shape
                or indices.dtype != torch.uint8 or probs.dtype != torch.float16
                or not torch.isfinite(probs).all() or (probs < 0).any()
                or (probs.float().sum(-1) - 1).abs().max().item() > 0.001
                or (indices.sort(-1).values.diff(dim=-1) == 0).any()):
            raise ValueError(f"invalid donor cache geometry or probabilities in {name}")
        parts.append(shard)
        totals += len(data)
    joined = {key: torch.cat([shard[key] for shard in parts]) for key in ("raw", "indices", "probs")}
    if manifest.get("windows") != len(joined["raw"]) or manifest.get("payload_bytes") != totals:
        raise ValueError("donor cache manifest total mismatch")
    return joined, sha256(metadata)
