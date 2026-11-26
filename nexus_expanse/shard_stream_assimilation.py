"""One-shard-at-a-time checkpoint transport into a native raw-byte skill."""

from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from nexus_expanse.checkpoint_transport import CorrectedLinear, ImportedSkill
from nexus_expanse.donor_assimilation import forbid_donor_forward, graft_dense_to_monarch
from nexus_expanse.donor_cache_io import _decoder_types, token_bytes
from nexus_expanse.weight_editing import exact_skill_receipt, verify_skill

INITIAL_ALPHA = 0.25
ROW_CHUNK = 256
LAYER_NUMBER = re.compile(r"(?:^|\.)(?:layers|blocks|h)\.(\d+)\.")
ATTENTION_WEIGHTS = ("v_proj", "o_proj")
MLP_WEIGHTS = ("up_proj", "down_proj")
OPTIONAL_WEIGHTS = ("q_proj", "k_proj", "input_layernorm", "post_attention_layernorm")
HF_INDEX = "model.safetensors.index.json"
HF_TOKENIZER = "tokenizer.json"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _local_file(root, name):
    path = root / name
    if Path(name).name != name or path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a local regular checkpoint file: {name}")
    return path


def _safetensor_reader(path):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ValueError("the safetensors backend requires the local safetensors package") from exc
    return safe_open(str(path), framework="pt", device="cpu")


def _hf_source(repo):
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:
        raise ValueError("--hf-repo requires the huggingface_hub package") from exc

    try:
        revision = get_hf_file_metadata(hf_hub_url(repo, HF_INDEX)).commit_hash
    except HfHubHTTPError as exc:
        raise ValueError(f"could not resolve {HF_INDEX} for {repo}") from exc
    if not revision or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError(f"Hugging Face did not report an immutable revision for {repo}")

    def download(name, directory):
        try:
            hf_hub_download(repo_id=repo, filename=name, revision=revision, local_dir=directory)
        except HfHubHTTPError as exc:
            raise ValueError(f"could not download {name} from {repo}") from exc
        return _local_file(directory, name)

    return download, revision


class ShardStream:
    """Map checkpoint keys and expose tensors only within the open shard's lifetime."""

    def __init__(self, directory, *, mock_pt=False, hf_repo=None):
        self.root = Path(directory).resolve(strict=True)
        self.mock_pt = mock_pt
        self.hf_repo = hf_repo
        if hf_repo is not None:
            self.download, self.revision = _hf_source(hf_repo)
            for name in (HF_INDEX, HF_TOKENIZER):
                self.download(name, self.root)
        index = "model.pt.index.json" if mock_pt else HF_INDEX
        index_path = self.root / index
        if not mock_pt and not index_path.exists() and not index_path.is_symlink():
            name = "model.safetensors"
            with _safetensor_reader(_local_file(self.root, name)) as reader:
                weight_map = dict.fromkeys(reader.keys(), name)
            data = json.dumps({"weight_map": weight_map}, sort_keys=True).encode()
        else:
            data = _local_file(self.root, index).read_bytes()
            weight_map = json.loads(data)["weight_map"]
        self.index_sha256 = _digest(data)
        self.weight_map = weight_map
        suffix = ".pt" if mock_pt else ".safetensors"
        if not isinstance(self.weight_map, dict) or not self.weight_map:
            raise ValueError("checkpoint index must map tensor names to shards")
        if any(not isinstance(key, str) or not isinstance(name, str) or not name.endswith(suffix)
               for key, name in self.weight_map.items()):
            raise ValueError("checkpoint index contains invalid tensor or shard names")
        self.shard_names = set(self.weight_map.values())
        if hf_repo is None:
            self.paths = {name: _local_file(self.root, name) for name in self.shard_names}
        elif any(Path(name).name != name or "\\" in name or ":" in name for name in self.shard_names):
            raise ValueError("checkpoint index references a nonlocal shard name")
        self.live = 0
        self.peak = 0
        self.peak_disk = 0
        self.visits = 0
        self.files = {}

    @contextmanager
    def _shard_path(self, name):
        if self.hf_repo is None:
            yield self.paths[name]
            return
        # local_dir bypasses the Hub blob cache. Each lease also removes its download metadata.
        with TemporaryDirectory(dir=self.root, prefix="shard-") as directory:
            path = self.download(name, Path(directory))
            self.peak_disk = max(self.peak_disk, sum(1 for _ in self.root.glob("shard-*/*.safetensors")))
            if self.peak_disk != 1:
                raise RuntimeError("expected exactly one downloaded donor shard on disk")
            yield path

    @contextmanager
    def _open(self, name, path):
        if self.live:
            raise RuntimeError("attempt to load a second donor shard")
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
                with _safetensor_reader(path) as reader:
                    yield reader.get_tensor
        finally:
            self.live -= 1

    def consume(self, transforms):
        grouped = {}
        for key, transform in transforms.items():
            grouped.setdefault(self.weight_map[key], []).append((key, transform))
        for name, entries in grouped.items():
            with self._shard_path(name) as path, self._open(name, path) as read:
                try:
                    for key, transform in entries:
                        raw = read(key)
                        try:
                            if not isinstance(raw, torch.Tensor) or not raw.is_floating_point():
                                raise ValueError(f"expected a floating-point tensor at {key}")
                            transform(raw)
                        finally:
                            del raw
                finally:
                    # Release the getter before closing the shard and deleting its backing file.
                    del read

    def receipt(self):
        receipt = dict(
            format="mock_pt" if self.mock_pt else "safetensors", index_sha256=self.index_sha256,
            peak_loaded_shards=self.peak, loaded_shard_count=len(self.files),
            indexed_shard_count=len(self.shard_names), shard_visits=self.visits, shards=self.files,
        )
        if self.hf_repo is not None:
            receipt.update(hf_repo=self.hf_repo, hf_revision=self.revision, peak_disk_shards=self.peak_disk)
        return receipt


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
        candidates = [key for key in groups[index] if ".experts." not in key and ".shared_experts." not in key]
        mlp = {name: _find(candidates, f"{name}.weight") for name in MLP_WEIGHTS}
        if not any(mlp.values()):
            mlp = {
                name: _find(groups[index], f"mlp.shared_experts.{name}.weight", required=True)
                for name in MLP_WEIGHTS
            }
        if not all(mlp.values()):
            raise ValueError(f"checkpoint layer {index} needs paired MLP up/down projections")
        attention = {name: _find(candidates, f"{name}.weight", required=True) for name in ATTENTION_WEIGHTS}
        optional = {
            name: key for name in OPTIONAL_WEIGHTS if (key := _find(candidates, f"{name}.weight"))
        }
        yield {**attention, **mlp, **optional}


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


def _attention_project(weight, basis, read_config):
    query_width, _ = basis.shape
    result = _right_project(weight, basis)
    if result.shape[0] != query_width:
        config = read_config()
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
    tokenizer = _local_file(directory, HF_TOKENIZER).read_bytes()
    singles, ids, leading = _lexicon(tokenizer)
    embed_key = _find(keys, "embed_tokens.weight", required=True)
    head_key = _find(keys, "lm_head.weight")
    config_path = directory / "config.json"
    config = None
    if stream.hf_repo is None:
        config = json.loads(_local_file(directory, "config.json").read_bytes()) if config_path.exists() else {}

    def read_config():
        nonlocal config
        if config is None:
            # Only grouped-query geometry needs this additional metadata.
            path = stream.download("config.json", directory)
            config = json.loads(path.read_bytes())
        return config

    width = student.readout.in_features
    transforms = {}
    reduced = {}
    rng = None if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed)
    embedded = {}
    basis = None

    def embed(weight):
        nonlocal basis
        if weight.ndim != 2 or weight.shape[0] <= ids.max().item():
            raise ValueError("embedding tensor is missing tokenizer rows")
        rows = _finite(weight[singles])
        basis, spectrum = _aligned_basis(rows, student.core.emb.weight)
        embedded.update(spectrum=spectrum, input=_shuffle(rows @ basis, rng))

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
            matrix = _attention_project(weight, basis, read_config)
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
    if stream.hf_repo is None:
        stream.consume({embed_key: embed})
        stream.consume(transforms)
    else:
        def embed_first(weight):
            embed(weight)
            if head_key is None:
                read_head(weight)

        if head_key is None:
            del transforms[embed_key]
        stream.consume({embed_key: embed_first, **transforms})

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
        local_files_only=stream.hf_repo is None, trust_remote_code=False,
    )


def assimilate_zero_exec(directory=None, *, hf_repo=None, mock_pt=False, seed=0, shuffle_seed=None):
    """Initialize and verify a standalone skill from local or streamed HF tensor shards."""
    if (directory is None) == (hf_repo is None) or (hf_repo is not None and mock_pt):
        raise ValueError("choose one donor directory or HF repo; mock PT requires a local directory")
    source = TemporaryDirectory(prefix="nexus-hf-") if hf_repo is not None else nullcontext(directory)
    with source as root, forbid_donor_forward() as guard:
        stream = ShardStream(root, mock_pt=mock_pt, hf_repo=hf_repo)
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
