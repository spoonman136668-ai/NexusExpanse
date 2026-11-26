"""Static singleton-byte lookups for the DeepSeek-V4.1 Engram hash layout."""

import json
import math
import re
import unicodedata

import numpy as np
import torch

from nexus_expanse.donor_cache_io import token_bytes


SPACE = "\t\n\v\f\r \x85\xa0\u1680" + "".join(chr(i) for i in range(0x2000, 0x200B))
SPACE += "\u2028\u2029\u202f\u205f\u3000"
ASCII_SPACE = re.compile(r"[ \t\r\n]+")


def _compressed_tokens(data):
    tokenizer = json.loads(data)
    if (tokenizer.get("decoder", {}).get("type") != "ByteLevel"
            or tokenizer.get("model", {}).get("type") != "BPE"):
        raise ValueError("DeepSeek Engram requires a ByteLevel BPE tokenizer")
    spellings = {index: token for token, index in tokenizer["model"]["vocab"].items()}
    spellings.update((item["id"], item["content"]) for item in tokenizer.get("added_tokens", []))
    if not spellings or set(spellings) != set(range(len(spellings))):
        raise ValueError("DeepSeek Engram requires a contiguous tokenizer vocabulary")
    keys = {}
    ids = []
    for index in range(len(spellings)):
        token = spellings[index]
        text = token_bytes(token, "gpt2").decode("utf-8", errors="replace")
        if "\ufffd" in text:
            key = token
        else:
            decomposed = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
            lower = "".join(char.lower() for char in decomposed if not unicodedata.category(char).startswith("M"))
            spaced = ASCII_SPACE.sub(" ", lower)
            normalized = spaced if spaced == " " else spaced.strip(SPACE).replace("\ue000", " ")
            key = normalized or text
        ids.append(keys.setdefault(key, len(keys)))
    return torch.tensor(ids, dtype=torch.int64), len(keys)


def _next_prime(start, used):
    candidate = start + 1
    while candidate in used or any(candidate % divisor == 0 for divisor in range(2, math.isqrt(candidate) + 1)):
        candidate += 1
    return candidate


def engram_rows(tokenizer, config, singles):
    """Hash a one-token prefix per byte with padded history into reference buckets."""
    names = ("engram_max_ngram_size", "engram_n_heads", "engram_head_dim", "engram_vocab_size",
             "engram_compressed_vocab_size")
    ngram, heads, dim, base, vocab_size = (config.get(name) for name in names)
    if (any(type(value) is not int or value < 1 for value in (ngram, heads, dim, base, vocab_size))
            or ngram < 2 or base < 2):
        raise ValueError("invalid DeepSeek Engram hash geometry")
    layers, totals = config.get("engram_layer_ids"), config.get("engram_num_embeddings")
    if (not isinstance(layers, list) or not isinstance(totals, list) or not layers
            or len(layers) != len(totals) or any(type(layer) is not int or layer < 0 for layer in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("invalid DeepSeek Engram layer layout")
    token_map, size = _compressed_tokens(tokenizer)
    pad = config.get("engram_pad_token_id")
    if size != vocab_size or type(pad) is not int or not 0 <= pad < len(token_map):
        raise ValueError("DeepSeek Engram compressed vocabulary or padding does not match config.json")
    limit = max(1, (np.iinfo(np.int64).max // size) // 2)
    result = {}
    used = set()
    for layer, total in zip(layers, totals, strict=True):
        rng = np.random.default_rng(10007 * layer)
        multipliers = rng.integers(0, limit, size=ngram, dtype=np.int64) * 2 + 1
        rolling = token_map[singles] * int(multipliers[0])
        offset = 0
        slots = []
        for multiplier in multipliers[1:]:
            rolling = torch.bitwise_xor(rolling, token_map[pad] * int(multiplier))
            next_start = base - 1
            for _ in range(heads):
                prime = _next_prime(next_start, used)
                used.add(prime)
                slots.append(rolling.remainder(prime) + offset)
                next_start = prime
                offset += prime
        if offset != total:
            raise ValueError(f"DeepSeek Engram bucket count disagrees with config for layer {layer}")
        result[layer] = (torch.stack(slots, dim=1), (total, dim))
    return result
