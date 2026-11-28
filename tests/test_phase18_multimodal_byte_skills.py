"""Offline codec, compressed receiver, and causal bank regressions."""

from itertools import accumulate

import pytest
import torch

from benchmarks import gate17_gpu_scaling_harness as scaling
from benchmarks.phase15_cosine_lr_and_sampling import call_cli
from nexus_expanse.cli import route_prefix_check
from nexus_expanse.donor_assimilation import forbid_donor_forward
from nexus_expanse.jepa import state_hash
from nexus_expanse.multi_skill_router import BANK_V3, MultiSkillBank, load_bank, save_bank
from nexus_expanse.multimodal_bytes import (
    FRAME_BYTES, decode_audio_frame, decode_image_tile, decode_telemetry_packet,
    encode_audio_frame, encode_image_tile, encode_telemetry_packet, generate_stream, modality_splits,
)
from nexus_expanse.shard_stream_assimilation import make_student
from nexus_expanse.skill_birth import EPISODIC_BIRTH, OpenWorldSkillBank
from nexus_expanse.unified_receiver import UnifiedSnapshotSkill
from scripts.prepare_gpu_corpus import assert_disjoint


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


CODECS = {"audio": (decode_audio_frame, lambda pitch, samples: encode_audio_frame(samples, pitch)),
          "image": (decode_image_tile, lambda kind, pixels: encode_image_tile(pixels, kind)),
          "telemetry": (decode_telemetry_packet, encode_telemetry_packet)}


@pytest.mark.parametrize("mode", CODECS)
def test_multimodal_codecs_are_stable_binary_and_disjoint(mode):
    rows = modality_splits(mode, 3201, train_frames=96, eval_frames=36)
    assert rows == modality_splits(mode, 3201, train_frames=96, eval_frames=36)
    assert_disjoint((rows.train,), {mode: rows.evaluation})
    width = FRAME_BYTES[mode]
    assert len(rows.train) == 96 * width and len(rows.evaluation) == 36 * width
    assert 0 in rows.train and max(rows.train) >= 0x80
    decode, encode = CODECS[mode]
    states = []
    for offset in range(0, len(rows.train), width):
        frame = rows.train[offset:offset + width]
        decoded = decode(frame)
        assert encode(*decoded) == frame
        states.append(decoded)
    if mode == "telemetry":
        counters = list(accumulate(state[1] for state in states))
        assert all(a < b for a, b in zip(counters, counters[1:]))


def test_boundary_codes_and_checksum_reject_bad_packets():
    wave = encode_audio_frame([-1.0, 1.0] * 32, 15)
    assert wave[:3] == bytes([255, 0, 239])
    kind, decoded = decode_audio_frame(wave)
    assert encode_audio_frame(decoded, kind) == wave
    tile = encode_image_tile(bytes(range(64)), 7)
    assert decode_image_tile(tile) == (7, bytes(range(64)))
    packet = encode_telemetry_packet(0, 257, (0, 128, 255))
    assert decode_telemetry_packet(packet) == (0, 257, (0, 128, 255))
    with pytest.raises(ValueError, match="checksum"):
        decode_telemetry_packet(packet[:-1] + bytes([packet[-1] ^ 1]))
    with pytest.raises(ValueError, match="audio frame"):
        decode_audio_frame(wave[:2] + b"\xf0" + wave[3:])


def snapshot(seed, raw, *, dim=32, context=16, layers=2):
    shape = scaling.Geometry(seed, dim, context, 1, 1, 1, 1, layers)
    model = scaling.model_for(shape, "causal_attention_skill", torch.device("cpu"))
    unified = scaling.build_unified(model, {"stream": raw})
    return unified.snapshot(snapshot_quant="int8"), len(unified.snapshot())


def assert_equal(first, second):
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_cli_and_bank_restore_heterogeneous_int8_experts_with_exact_prefix(tmp_path):
    samples = {"prose": b"one raw byte stream\n" * 16, "code": b"def add(x):\n    return x + 1\n" * 16,
               "audio": generate_stream("audio", 17, 8), "image": generate_stream("image", 17, 8)}
    paths = {}
    for index, (name, data) in enumerate(samples.items()):
        raw, fp32 = snapshot(index, data, dim=128 if name == "audio" else 32,
                             context=32 if name == "audio" else 16, layers=4 if name == "audio" else 2)
        if name == "audio":
            assert fp32 / len(raw) >= 3
        paths[name] = tmp_path / f"{name}.pt"
        paths[name].write_bytes(raw)
    with forbid_donor_forward() as guard:
        bank = MultiSkillBank.from_unified_snapshots(paths)
        assert bank.context == 16
        assert bank.skills["audio"].core.dim == 128
        assert all(not parameter.requires_grad for parameter in bank.parameters())
        packed = tmp_path / "bank.pt"
        save_bank(bank, packed)
        contents = torch.load(packed, weights_only=True)
        assert contents["schema"] == BANK_V3
        for name, entry in contents["entries"].items():
            assert entry["unified_snapshot"].dtype == torch.uint8
            assert entry["unified_snapshot"].numpy().tobytes() == paths[name].read_bytes()
        cold = load_bank(packed)
        data = b"".join(stream[:19] for stream in samples.values()) + bytes([0, 255])
        tokens = torch.tensor([list(data)])
        route = bank.forward_hmm_bma
        expected = route(tokens, include_next=True)
        assert route_prefix_check(route, tokens, expected) == 0.0
        assert_equal(expected, cold.forward_hmm_bma(tokens, include_next=True))
        assert {n: state_hash(s) for n, s in bank.skills.items()} == {n: state_hash(s) for n, s in cold.skills.items()}
    assert guard.donor_forward_calls == 0
    files = [value for name, path in paths.items() for value in ("--skill-snapshot", name, path)]
    common = ("route-stream", "--router", "hmm_bma", "--input-hex", data.hex())
    direct = call_cli(*common, *files)
    assert direct == call_cli(*common, "--bank", packed)
    assert direct["donor_forward_calls"] == 0 and direct["prefix_max_abs_diff"] == 0.0


def test_int8_birth_repackages_only_candidate_and_preserves_legacy_receiver(tmp_path):
    raw, _ = snapshot(42, b"prose resident\n" * 20)
    first = UnifiedSnapshotSkill(raw)
    second = make_student(51).eval()
    bank = MultiSkillBank(first, {"legacy": second}, base_name="prose")
    world = OpenWorldSkillBank(bank)
    encounter = torch.tensor([list(generate_stream("telemetry", 21, 16))])
    states = {name: state_hash(skill) for name, skill in bank.skills.items()}
    prefixes = bank.component_log_probs(encounter[:, :32])
    assert world.observe("telemetry", encounter[:, :16]) is None
    with forbid_donor_forward() as guard:
        candidate = world.observe("telemetry", encounter, min_bpb_threshold=0.1, mode=EPISODIC_BIRTH, steps=2)
        assert candidate is not None and candidate.neural.snapshot_quant == "int8"
        assert candidate.neural.archive() != raw and candidate.keys.numel()
        assert world.observe("telemetry", encounter, min_bpb_threshold=0.1) is None
        assert world.resident_names == ("prose", "legacy")
        assert all(states[d] == state_hash(bank.skills[d]) for d in states)
        assert torch.equal(prefixes, bank.component_log_probs(encounter[:, :32])[:, :, :2])
        file = tmp_path / "born.pt"
        world.save_bank(file)
        cold = OpenWorldSkillBank.load_bank(file)
        assert cold.bank.skill_names == ("prose", "legacy", "telemetry")
        assert_equal(world(encounter, include_next=True), cold(encounter, include_next=True))
    assert guard.donor_forward_calls == 0


def test_bank_save_rejects_a_stale_unified_archive(tmp_path):
    raw, _ = snapshot(42, b"native raw bytes" * 20)
    skill = UnifiedSnapshotSkill(raw)
    bank = MultiSkillBank(skill, {}, base_name="prose")
    with torch.no_grad():
        skill.core.emb.weight[0, 0].add_(1)
    with pytest.raises(ValueError, match="changed after snapshot restore"):
        save_bank(bank, tmp_path / "stale.pt")
