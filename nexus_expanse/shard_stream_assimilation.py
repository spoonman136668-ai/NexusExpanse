"""One-shard-at-a-time checkpoint transport into a native raw-byte skill."""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from torch.nn import functional as F

from nexus_expanse.checkpoint_transport import CorrectedLinear, ImportedSkill
from nexus_expanse.donor_assimilation import forbid_donor_forward, graft_dense_to_monarch
from nexus_expanse.donor_cache_io import _decoder_types, token_bytes
from nexus_expanse.weight_editing import exact_skill_receipt, verify_skill

INITIAL_ALPHA = 0.25
ROW_CHUNK = 256
LAYER_NUMBER = re.compile(r"(?:^|\.)(?:layers|blocks|h)\.(\d+)\.")
LAYER_WEIGHTS = ("v_proj", "o_proj", "up_proj", "down_proj")
OPTIONAL_WEIGHTS = ("q_proj", "k_proj", "input_layernorm", "post_attention_layernorm")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _local_file(root, name):
    path = root / name
    if Path(name).name != name or path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a local regular checkpoint file: {name}")
    return path


class ShardStream:
    """Map checkpoint keys and expose tensors only within the open shard's lifetime."""

    def __init__(self, directory, *, mock_pt=False):
        self.root = Path(directory).resolve(strict=True)
        self.mock_pt = mock_pt
        index = "model.pt.index.json" if mock_pt else "model.safetensors.index.json"
        data = _local_file(self.root, index).read_bytes()
        self.index_sha256 = _digest(data)
        index = json.loads(data)
        self.weight_map = index["weight_map"]
        suffix = ".pt" if mock_pt else ".safetensors"
        if not isinstance(self.weight_map, dict) or not self.weight_map:
            raise ValueError("checkpoint index must map tensor names to shards")
        if any(not isinstance(key, str) or not isinstance(name, str) or not name.endswith(suffix)
               for key, name in self.weight_map.items()):
            raise ValueError("checkpoint index contains invalid tensor or shard names")
        self.paths = {name: _local_file(self.root, name) for name in set(self.weight_map.values())}
        self.live = 0
        self.peak = 0
        self.visits = 0
        self.files = {}

    @contextmanager
    def _open(self, name):
        if self.live:
            raise RuntimeError("attempt to load a second donor shard")
        path = self.paths[name]
        self.live += 1
        self.visits += 1
        self.peak = max(self.peak, self.live)
        try:
            if name not in self.files:
                with path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                self.files[name] = dict(bytes=path.stat().st_size, sha256=digest)
            if self.mock_pt:
                state = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(state, dict):
                    raise ValueError("mock shard must contain a tensor dictionary")
                yield state.__getitem__
            else:
                try:
                    from safetensors import safe_open
                except ImportError as exc:
                    raise ValueError("the safetensors backend requires the local safetensors package") from exc
                with safe_open(str(path), framework="pt", device="cpu") as reader:
                    yield reader.get_tensor
        finally:
            self.live -= 1

    def consume(self, transforms):
        grouped = {}
        for key, transform in transforms.items():
            grouped.setdefault(self.weight_map[key], []).append((key, transform))
        for name, entries in grouped.items():
            with self._open(name) as read:
                for key, transform in entries:
                    raw = read(key)
                    if not isinstance(raw, torch.Tensor) or not raw.is_floating_point():
                        raise ValueError(f"expected a floating-point tensor at {key}")
                    transform(raw)
                    del raw
            # The mock getter owns its loaded dict. Drop it before opening another file.
            del read

    def receipt(self):
        return dict(
            format="mock_pt" if self.mock_pt else "safetensors", index_sha256=self.index_sha256,
            peak_loaded_shards=self.peak, loaded_shard_count=len(self.files),
            indexed_shard_count=len(self.paths), shard_visits=self.visits, shards=self.files,
        )


def _lexicon(data):
    config = json.loads(data)
    decoder = _decoder_types(config.get("decoder") or {})
    if "ByteLevel" in decoder:
        family = "gpt2"
    elif decoder & {"Metaspace", "ByteFallback", "Replace"}:
        family = "sentencepiece"
    else:
        raise ValueError("expected GPT-2 ByteLevel or SentencePiece tokenizer.json")
    added = config.get("added_tokens", [])
    special = {item["id"] for item in added if item.get("special")}
    vocab = config["model"]["vocab"]
    entries = list(vocab.items()) if isinstance(vocab, dict) else [
        (entry[0], index) for index, entry in enumerate(vocab)
    ]
    entries.extend((item["content"], item["id"]) for item in added if not item.get("special"))
    spellings = {}
    for token, index in entries:
        if type(index) is not int or index < 0:
            raise ValueError("invalid tokenizer vocabulary index")
        if index in special:
            continue
        spelling = token_bytes(token, family)
        if spelling:
            if index in spellings and spellings[index] != spelling:
                raise ValueError("tokenizer index maps to conflicting byte spellings")
            spellings[index] = spelling
    singles = {}
    for index, spelling in sorted(spellings.items()):
        if len(spelling) == 1:
            singles.setdefault(spelling[0], index)
    if len(singles) != 256:
        raise ValueError("tokenizer must provide one standalone token for every raw byte")
    ids = sorted(spellings)
    return (
        torch.tensor([singles[b] for b in range(256)]),
        torch.tensor(ids),
        torch.tensor([spellings[i][0] for i in ids]),
    )


def _find(keys, suffix, *, required=False):
    candidates = [key for key in keys if key == suffix or key.endswith("." + suffix)]
    if len(candidates) > 1:
        raise ValueError(f"ambiguous checkpoint mapping for {suffix}")
    if not candidates and required:
        raise ValueError(f"checkpoint is missing {suffix}")
    return next(iter(candidates), None)


def _layer_keys(keys, depth):
    groups = {}
    for key in keys:
        if match := LAYER_NUMBER.search(key):
            groups.setdefault(int(match[1]), []).append(key)
    if len(groups) < depth:
        raise ValueError("checkpoint has insufficient donor layers")
    for index in sorted(groups)[:depth]:
        yield {
            name: key for name in (*LAYER_WEIGHTS, *OPTIONAL_WEIGHTS)
            if (key := _find(groups[index], f"{name}.weight", required=name in LAYER_WEIGHTS))
        }


def _finite(x):
    x = x.float()
    if not torch.isfinite(x).all():
        raise ValueError("checkpoint contains a nonfinite mapped tensor")
    return x


def _right_project(weight, projection):
    if weight.ndim != 2 or weight.shape[1] != projection.shape[0]:
        raise ValueError("donor matrix has an invalid input width")
    return torch.cat([_finite(block) @ projection for block in weight.split(ROW_CHUNK)])


def _left_project(weight, projection):
    if weight.ndim != 2 or weight.shape[0] != projection.shape[0]:
        raise ValueError("donor matrix has an invalid output width")
    return torch.cat([projection.T @ _finite(block) for block in weight.split(ROW_CHUNK, dim=1)], dim=1)


def _aligned_basis(embeddings, reference):
    width = reference.shape[1]
    if embeddings.shape[1] < width:
        raise ValueError("donor embedding width must be at least the native student width")
    _, singular, right = torch.linalg.svd(embeddings, full_matrices=False)
    basis = right[:width].T.contiguous()
    u, _, vh = torch.linalg.svd((embeddings @ basis).T @ reference, full_matrices=False)
    return basis @ (u @ vh), singular[:width].tolist()


def _attention_project(weight, basis, config):
    query_width, _ = basis.shape
    result = _right_project(weight, basis)
    if result.shape[0] != query_width:
        heads = config.get("num_attention_heads")
        kv_heads = config.get("num_key_value_heads")
        if (type(heads) is not int or type(kv_heads) is not int or kv_heads < 1
                or heads < kv_heads or heads % kv_heads or query_width % heads
                or result.shape[0] != kv_heads * (query_width // heads)):
            raise ValueError("grouped-query donor needs compatible attention heads in config.json")
        result = result.unflatten(0, (kv_heads, query_width // heads)).repeat_interleave(heads // kv_heads, 0)
        result = result.flatten(0, 1)
    return basis.T @ result


def _shuffle(matrix, rng):
    if rng is None:
        return matrix
    rows = torch.randperm(matrix.shape[0], generator=rng)
    cols = torch.randperm(matrix.shape[1], generator=rng)
    return matrix[rows][:, cols]


def make_student(seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        student = ImportedSkill()
    student.core.fixed_eval_geometry = True
    return student


def _spectral_weighted_monarch(module, weight):
    graft_dense_to_monarch(weight, module)
    return CorrectedLinear(module, weight, "spectral4")


def _mlp_kv_mine(up, down, count):
    if up.shape[0] != down.shape[1] or not up.shape[0]:
        raise ValueError("MLP up/down projections do not share their inner dimension")
    importance = up.norm(dim=1) * down.norm(dim=0)
    ids = importance.argsort(descending=True, stable=True)[:count]
    keys = F.pad(up[ids], (0, 0, 0, count - len(ids)))
    values = F.pad(down[:, ids], (0, count - len(ids)))
    return keys, values


def _static_ov_readout(embeddings, projected_head, first_bytes, transport, engram):
    width = embeddings.shape[-1]
    context = F.layer_norm(embeddings + embeddings @ transport.T, (width,))
    # Sum probability mass over all distinct vocabulary rows that can emit the same byte.
    # Log reductions keep the conversion stable for high-magnitude donor heads.
    log_mass = torch.full((256, 256), -torch.inf, dtype=torch.float64)
    for head, leading in zip(projected_head.split(ROW_CHUNK), first_bytes.split(ROW_CHUNK), strict=True):
        scores = context.double() @ head.double().T
        for byte in leading.unique().tolist():
            log_mass[:, byte] = torch.logaddexp(log_mass[:, byte], scores[:, leading == byte].logsumexp(1))
    if engram is not None:
        log_mass += engram
    return log_mass.log_softmax(-1).float()


@torch.no_grad()
def _transport(student, stream, directory, *, shuffle_seed):
    keys = stream.weight_map
    tokenizer = _local_file(directory, "tokenizer.json").read_bytes()
    singles, ids, leading = _lexicon(tokenizer)
    embed_key = _find(keys, "embed_tokens.weight", required=True)
    head_key = _find(keys, "lm_head.weight")
    config_path = directory / "config.json"
    config = json.loads(_local_file(directory, "config.json").read_bytes()) if config_path.exists() else {}
    width = student.readout.in_features
    transforms = {}
    reduced = {}
    rng = None if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed)
    embedded = {}

    def embed(weight):
        if weight.ndim != 2 or weight.shape[0] <= ids.max().item():
            raise ValueError("embedding tensor is missing tokenizer rows")
        rows = _finite(weight[singles])
        basis, spectrum = _aligned_basis(rows, student.core.emb.weight)
        embedded.update(basis=basis, spectrum=spectrum, input=_shuffle(rows @ basis, rng))

    stream.consume({embed_key: embed})
    basis = embedded["basis"]

    def read_head(weight):
        if weight.ndim != 2 or weight.shape[0] <= ids.max().item():
            raise ValueError("output head is missing tokenizer rows")
        projected = torch.cat([_right_project(weight[group], basis) for group in ids.split(ROW_CHUNK)])
        reduced["head"] = _shuffle(projected, rng)

    transforms[head_key or embed_key] = read_head
    pos = _find(keys, "embed_positions.weight")
    final_norm = _find((key for key in keys if not LAYER_NUMBER.search(key)), "norm.weight")
    engram_key = _find(keys, "engram.weight")
    if pos:
        transforms[pos] = lambda weight: reduced.update(pos=_right_project(weight[:student.core.max_seq_len], basis))
    if final_norm:
        transforms[final_norm] = lambda weight: reduced.update(norm=basis.square().T @ _finite(weight))
    if engram_key:
        def read_engram(weight):
            if weight.shape != (256, 256):
                raise ValueError("static Engram must be a 256 by 256 byte logit table")
            reduced["engram"] = _shuffle(_finite(weight).clone(), rng)
        transforms[engram_key] = read_engram

    layers = list(_layer_keys(keys, len(student.core.layers)))

    def project(weight, name):
        if name in ("q_proj", "k_proj", "v_proj"):
            matrix = _attention_project(weight, basis, config)
        elif name == "o_proj":
            matrix = basis.T @ _right_project(weight, basis)
        elif name == "up_proj":
            matrix = _right_project(weight, basis)
        elif name == "down_proj":
            matrix = _left_project(weight, basis)
        else:
            return basis.square().T @ _finite(weight)
        return _shuffle(matrix, rng)

    for index, layer in enumerate(layers):
        for name, key in layer.items():
            def collect(weight, index=index, name=name):
                reduced[index, name] = project(weight, name)
            transforms[key] = collect
    stream.consume(transforms)

    student.core.emb.weight.copy_(embedded["input"])
    if "pos" in reduced:
        student.core.pos_emb.weight[:len(reduced["pos"])].copy_(reduced["pos"])
    else:
        student.core.pos_emb.weight.zero_()
    if "norm" in reduced:
        student.core.out_norm.weight.copy_(reduced["norm"])
    # The row positions are tokenizer IDs in ascending order, even across vocab gaps.
    student.readout.weight.copy_(reduced["head"][torch.searchsorted(ids, singles)])
    transport = torch.zeros(width, width)
    for index, layer in enumerate(student.core.layers):
        attn = layer.attn
        for part, name in enumerate(("q_proj", "k_proj", "v_proj")):
            if (index, name) in reduced:
                attn.in_proj_weight[part * width:(part + 1) * width].copy_(reduced[index, name])
        attn.out_proj.weight.copy_(reduced[index, "o_proj"])
        for target, name in ((layer.attn_norm, "input_layernorm"), (layer.ffn_norm, "post_attention_layernorm")):
            if (index, name) in reduced:
                target.weight.copy_(reduced[index, name])

        mined_up, mined_down = _mlp_kv_mine(reduced[index, "up_proj"], reduced[index, "down_proj"],
                                            layer.ffn.fc1.out_dim)
        layer.ffn.fc1 = _spectral_weighted_monarch(layer.ffn.fc1, mined_up)
        layer.ffn.fc2 = _spectral_weighted_monarch(layer.ffn.fc2, mined_down)
        transport += reduced[index, "o_proj"] @ reduced[index, "v_proj"] + mined_down @ mined_up
    student.static_byte_prior = _static_ov_readout(
        embedded["input"], reduced["head"], leading, transport / (2 * len(layers)), reduced.get("engram"),
    )
    student.prior_alpha = INITIAL_ALPHA
    return dict(
        tokenizer_sha256=_digest(tokenizer), donor_width=basis.shape[0], donor_vocab_size=len(ids),
        student_width=width, student_vocab_size=256, projected_embedding_singular_values=embedded["spectrum"],
        selected_layers=len(layers), uses_static_engram="engram" in reduced,
        local_files_only=True, trust_remote_code=False,
    )


def assimilate_zero_exec(directory, *, mock_pt=False, seed=0, shuffle_seed=None):
    """Initialize and verify a standalone skill from local tensor shards."""
    with forbid_donor_forward() as guard:
        stream = ShardStream(directory, mock_pt=mock_pt)
        skill = make_student(seed)
        details = _transport(skill, stream, stream.root, shuffle_seed=shuffle_seed)
        skill.eval()
        archive, checks = verify_skill(skill)
        receipt = dict(**details, **stream.receipt(), **checks, prior_alpha=skill.prior_alpha,
                       donor_forward_calls=guard.donor_forward_calls, seed=seed, shuffle_seed=shuffle_seed)
        if stream.live or receipt["peak_loaded_shards"] != 1 or not exact_skill_receipt(receipt):
            raise ValueError("shard-stream skill failed exact qualification")
    return archive, receipt


def adapt_skill(skill, raw, *, seed, steps=64):
    """Apply matched byte CE updates with a linearly decaying static-prior blend."""
    if steps < 2 or raw.ndim != 2 or not raw.numel() or not 2 <= raw.shape[1] <= skill.core.max_seq_len:
        raise ValueError("adaptation needs at least two updates and nonempty raw-byte windows")
    if raw.is_floating_point() or raw.is_complex() or (raw < 0).any() or (raw > 255).any():
        raise ValueError("adaptation windows must contain only integer bytes")
    initial = skill.prior_alpha
    rng = torch.Generator().manual_seed(seed)
    schedule = []
    with forbid_donor_forward() as guard:
        skill.train().requires_grad_(True)
        optimizer = torch.optim.AdamW(skill.parameters(), lr=0.004, weight_decay=0.01)
        for step in range(steps):
            skill.prior_alpha = initial * (1.0 - step / (steps - 1))
            schedule.append(skill.prior_alpha)
            rows = torch.randint(len(raw), (min(8, len(raw)),), generator=rng)
            packed = raw[rows].long()
            optimizer.zero_grad(set_to_none=True)
            logits = skill(packed[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1), packed[:, 1:].flatten())
            if not math.isfinite(loss.item()):
                raise ValueError("student adaptation produced nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(skill.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        skill.zero_grad(set_to_none=True)
        skill.eval().requires_grad_(False)
        archive, checks = verify_skill(skill)
        if not exact_skill_receipt(checks):
            raise ValueError("adapted student failed exact qualification")
        receipt = dict(**checks, steps=steps, prior_alpha_schedule=schedule,
                       donor_forward_calls=guard.donor_forward_calls, final_ce=float(loss.detach()))
    return archive, receipt
