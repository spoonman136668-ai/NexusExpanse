"""Deterministic audio, image and sensor fixtures in the common byte alphabet."""

from dataclasses import dataclass
import math
import random
import struct


AUDIO_SAMPLES = 64
AUDIO_FRAME_BYTES = AUDIO_SAMPLES + 1
IMAGE_SIDE = 8
IMAGE_FRAME_BYTES = IMAGE_SIDE ** 2 + 1
TELEMETRY_PAYLOAD = struct.Struct("<BHBBB")
TELEMETRY_FRAME_BYTES = TELEMETRY_PAYLOAD.size + 1
MU = 255
WAVEFORM_MAX_CODE = 239
GRAM_BYTES = 32


def encode_audio_frame(samples, pitch):
    samples = tuple(samples)
    if (len(samples) != AUDIO_SAMPLES or any(not -1 <= x <= 1 for x in samples)
            or type(pitch) is not int or not 0 <= pitch <= 15):
        raise ValueError("audio requires 64 normalized samples and a four-bit pitch")
    codes = []
    for sample in samples:
        compressed = math.copysign(math.log1p(MU * abs(sample)) / math.log1p(MU), sample)
        codes.append(round((compressed + 1) * WAVEFORM_MAX_CODE / 2))
    return bytes([0xF0 + pitch, *codes])


def decode_audio_frame(frame):
    if (len(frame) != AUDIO_FRAME_BYTES or frame[0] < 0xF0
            or any(value > WAVEFORM_MAX_CODE for value in frame[1:])):
        raise ValueError("invalid audio frame")
    compressed = (2 * value / WAVEFORM_MAX_CODE - 1 for value in frame[1:])
    samples = tuple(math.copysign(math.expm1(abs(x) * math.log1p(MU)) / MU, x) for x in compressed)
    return frame[0] - 0xF0, samples


def encode_image_tile(pixels, kind):
    pixels = bytes(pixels)
    if len(pixels) != IMAGE_SIDE ** 2 or type(kind) is not int or not 0 <= kind <= 15:
        raise ValueError("image requires 64 grayscale bytes and a four-bit tile kind")
    return bytes([0xE0 + kind]) + pixels


def decode_image_tile(frame):
    if len(frame) != IMAGE_FRAME_BYTES or not 0xE0 <= frame[0] <= 0xEF:
        raise ValueError("invalid image tile")
    return frame[0] - 0xE0, bytes(frame[1:])


def encode_telemetry_packet(kind, counter_delta, channels):
    channels = tuple(channels)
    if (type(kind) is not int or not 0 <= kind <= 7 or type(counter_delta) is not int
            or not 1 <= counter_delta <= 65535 or len(channels) != 3
            or any(type(value) is not int or not 0 <= value <= 255 for value in channels)):
        raise ValueError("telemetry requires a packet kind, a positive uint16 delta and three byte channels")
    payload = TELEMETRY_PAYLOAD.pack(0xD0 + kind, counter_delta, *channels)
    return payload + bytes([sum(payload) & 0xFF])


def decode_telemetry_packet(frame):
    if (len(frame) != TELEMETRY_FRAME_BYTES or not 0xD0 <= frame[0] <= 0xD7
            or (sum(frame[:-1]) & 0xFF) != frame[-1]):
        raise ValueError("invalid telemetry packet or checksum")
    kind, delta, *channels = TELEMETRY_PAYLOAD.unpack(frame[:-1])
    if delta == 0:
        raise ValueError("telemetry counter must advance")
    return kind - 0xD0, delta, tuple(channels)


def _audio(rng, index):
    pitch = rng.randrange(16)
    frequency = 0.75 + 0.25 * pitch + rng.uniform(-0.12, 0.12)
    phase = rng.uniform(0, math.tau)
    harmonic = rng.uniform(0.18, 0.38)
    wave = []
    for position in range(AUDIO_SAMPLES):
        angle = math.tau * frequency * position / AUDIO_SAMPLES + phase
        wave.append(math.sin(angle) + harmonic * math.sin(2 * angle + 0.3) + 0.16 * math.sin(3 * angle - 0.5))
    peak = max(map(abs, wave))
    return encode_audio_frame((x / peak for x in wave), pitch)


def _image(rng, index):
    kind = rng.randrange(16)
    base = rng.randrange(256)
    dx = rng.choice((-7, -4, -2, 2, 4, 7))
    dy = rng.choice((-6, -3, -1, 1, 3, 6))
    pixels = (min(255, max(0, base + row * dy + col * dx + (row * col + kind) // 8 + rng.randrange(-2, 3)))
              for row in range(IMAGE_SIDE) for col in range(IMAGE_SIDE))
    return encode_image_tile(pixels, kind)


def _telemetry(rng, index):
    kind = (index // 16) % 8
    delta = rng.choice((1, 1, 1, 2, 3))
    level = 168 + (index % 24) * 3 + rng.randrange(-3, 4)
    return encode_telemetry_packet(kind, delta, (level, level + 7, level - 16 + rng.randrange(-2, 3)))


FRAME_BYTES = {"audio": AUDIO_FRAME_BYTES, "image": IMAGE_FRAME_BYTES, "telemetry": TELEMETRY_FRAME_BYTES}
_BUILDERS = {"audio": _audio, "image": _image, "telemetry": _telemetry}


def _windows(raw):
    return (raw[i:i + GRAM_BYTES] for i in range(len(raw) - GRAM_BYTES + 1))


def generate_stream(modality, seed, frames, *, exclude=()):
    """Generate complete frames and forbid exact windows shared with supplied splits."""
    if modality not in _BUILDERS or type(frames) is not int or frames < 1:
        raise ValueError("expected a supported modality and a positive frame count")
    build = _BUILDERS[modality]
    forbidden = {window for raw in exclude for window in _windows(raw)}
    rng = random.Random(f"nexus-multimodal:{modality}:{seed}")
    raw = bytearray()
    count = 0
    for index in range(frames * 64):
        frame = build(rng, index)
        joined = bytes(raw[-(GRAM_BYTES - 1):]) + frame
        if forbidden and any(window in forbidden for window in _windows(joined)):
            continue
        raw.extend(frame)
        count += 1
        if count == frames:
            return bytes(raw)
    raise ValueError(f"could not form {frames} leakage-free {modality} frames")


@dataclass(frozen=True)
class ModalitySplits:
    train: bytes
    evaluation: bytes


def modality_splits(modality, seed, *, train_frames, eval_frames):
    train = generate_stream(modality, f"{seed}:fit", train_frames)
    evaluation = generate_stream(modality, f"{seed}:check", eval_frames, exclude=(train,))
    return ModalitySplits(train, evaluation)
