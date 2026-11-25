"""Offline consolidation and physically byte-matched pruning for DEV."""
import math
import random

from nexus_expanse.hopfield_vsa import HopfieldVSAMemory

BUDGET = 40_000
REPLAY_LIMIT = 16
EPISODIC_BYTES = 24_000


class SleepHopfieldMemory(HopfieldVSAMemory):
    SLOTS = 44


def adaptive_bytes(model):
    # Matches three fixed eight-byte arbitration fields: share, weight, count.
    return model.episodic.serialized_size() + len(model.reasoning.snapshot()) + 24


def _ranked(model):
    episodic = model.episodic
    rows = sorted(episodic._table, key=lambda key: (
        max(episodic._table[key].values()), episodic._totals[key], len(key), key,
    ))
    rel = model.reasoning
    slots = sorted(range(rel.used), key=lambda i: (int(rel.counts[i].max()), int(rel.counts[i].sum()), i))
    return rows, slots


def _remove(model, contexts, slots):
    memory, rel = model.episodic, model.reasoning
    for key in contexts:
        memory._row_bytes -= memory._sizes.pop(key)
        del memory._table[key]
        del memory._totals[key]
    if slots:
        removed = set(slots)
        keep = [i for i in range(rel.used) if i not in removed]
        used = len(keep)
        rel.keys[:used], rel.counts[:used] = rel.keys[keep], rel.counts[keep]
        rel.keys[used:].fill(0)
        rel.counts[used:].fill(0)
        rel.used = used
    return dict(episodic_entries=len(contexts), hopfield_entries=len(slots))


def sleep(model, boundary_replay):
    """Replay at most sixteen observed boundary transitions, then prune weak rows."""
    if model._previous is not None:
        raise ValueError("reset byte context before offline consolidation")
    before = adaptive_bytes(model)
    if before > BUDGET:
        raise ValueError("sleep input needs an already bounded adaptive store")
    rows = list(boundary_replay)
    if len(rows) > REPLAY_LIMIT or any(
        not isinstance(ctx, bytes) or not 1 <= len(ctx) <= model.episodic.max_order
        or type(byte) is not int or not 0 <= byte <= 255
        or not math.isfinite(surprise) or surprise < 0 for ctx, byte, surprise in rows
    ):
        raise ValueError("invalid bounded replay ledger")
    replayed = []
    for ctx, byte, surprise in sorted(rows, key=lambda row: (-row[2], row[0], row[1])):
        model.episodic.observe_transition(ctx, byte)
        model.reasoning.observe(ctx[-2:], byte)
        if adaptive_bytes(model) > BUDGET:
            raise ValueError("adaptive ceiling violated during offline replay")
        replayed.append(dict(context=ctx.hex(), byte=byte, surprisal=surprise))
    rows, slots = _ranked(model)
    removal = _remove(model, rows[:len(rows) // 8], slots[:len(slots) // 8])
    return dict(before=before, after=adaptive_bytes(model), replayed=replayed, **removal)


def _items(model, rng):
    rows, slots = _ranked(model)
    if rng is not None:
        rng.shuffle(rows)
        rng.shuffle(slots)
    # Retain one context, so each removable row saves its exact JSON bytes
    # plus one comma, independently of what other rows are removed.
    items = [(model.episodic._sizes[key] + 1, key) for key in rows[:-1]]
    item_bytes = model.reasoning.keys.shape[1] + model.reasoning.counts.shape[1]
    items.extend((item_bytes, i) for i in slots)
    if rng is not None:
        rng.shuffle(items)
    return items


def _reachable(items):
    levels = [1]
    for size, _ in items:
        levels.append(levels[-1] | (levels[-1] << size))
    return levels


def _apply(model, items, levels, removal):
    chosen = []
    for i in reversed(range(len(items))):
        if (levels[i] >> removal) & 1:
            continue
        size, element = items[i]
        chosen.append(element)
        removal -= size
    if removal != 0:
        raise ValueError("unreachable byte removal")
    return _remove(model, [x for x in chosen if isinstance(x, bytes)],
                   [x for x in chosen if isinstance(x, int)])


def match_random_prune(sleep_model, control, seed):
    """Pick the nearest jointly attainable byte count at or below sleep size."""
    a, b = adaptive_bytes(sleep_model), adaptive_bytes(control)
    ranked, random_rows = _items(sleep_model, None), _items(control, random.Random(seed))
    pa, pb = _reachable(ranked), _reachable(random_rows)
    target = next((size for size in range(min(a, b), 0, -1)
                   if (pa[-1] >> (a - size)) & (pb[-1] >> (b - size)) & 1), None)
    if target is None:
        raise ValueError("the current memory states have no exact shared byte count")
    sa, rb = _apply(sleep_model, ranked, pa, a - target), _apply(control, random_rows, pb, b - target)
    if adaptive_bytes(sleep_model) != target or adaptive_bytes(control) != target:
        raise ValueError("byte-matched physical pruning did not balance")
    return dict(target_bytes=target, sleep_extra_prune=sa, random_prune=rb,
                sleep_before_match=a, random_before_match=b)
