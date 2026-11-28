"""Exact, read-only disk-backed byte suffix counts."""

from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
import errno
import hashlib
from itertools import groupby
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import tempfile


SCHEMA = "nexus-ssd-engram-v1"
READ_SIZE = 64 * 1024
PAGER_KIB = 2048
BATCH_SIZE = 16_384
PUBLISH_FALLBACK = {errno.EPERM, errno.EXDEV, errno.ENOTSUP}


@dataclass(frozen=True)
class EngramInfo:
    schema: str
    source_sha256: str
    min_order: int
    max_order: int
    min_support: int
    entry_count: int


@dataclass(frozen=True)
class EngramMatch:
    order: int
    support: int
    counts: tuple[tuple[int, int], ...]


def _parameters(min_order, max_order, min_support):
    if (type(min_order) is not int or type(max_order) is not int or not 1 <= min_order <= max_order <= 64
            or type(min_support) is not int or not 1 <= min_support <= 2**63 - 1):
        raise ValueError("require 1 <= min_order <= max_order <= 64 and positive SQLite-safe min_support")


def _alpha(value):
    if not math.isfinite(value) or not 0 <= value <= 0.9:
        raise ValueError("engram alpha must be finite and in [0.0, 0.9]")
    return float(value)


def _configure(database, *, builder=False):
    database.execute(f"PRAGMA cache_size=-{PAGER_KIB}")
    database.execute("PRAGMA mmap_size=0")
    if builder:
        # This database is disposable until commit, close, and fsynced publication.
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA temp_store=FILE")
    else:
        database.execute("PRAGMA query_only=ON")


def _identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _sources(path):
    if path.is_file():
        return False, [(b"", path)]
    if not path.is_dir():
        raise ValueError("engram input must be a file or directory")
    files = [(os.fsencode(item.relative_to(path).as_posix()), item)
             for item in path.rglob("*") if not item.is_symlink() and item.is_file()]
    if not files:
        raise ValueError("engram input directory has no regular files")
    return True, sorted(files)


def _count(database, files, directory, min_order, max_order):
    source_hash = hashlib.sha256(b"nexus-engram-directory-v1\0" if directory else b"")
    pending = Counter()

    def flush():
        database.executemany("INSERT INTO occurrences VALUES (?, ?, ?) "
                             "ON CONFLICT(suffix, byte) DO UPDATE SET n = n + excluded.n",
                             ((suffix, byte, count) for (suffix, byte), count in pending.items()))
        pending.clear()

    for name, path in files:
        prefix = b""
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            if directory:
                source_hash.update(struct.pack("<QQ", len(name), before.st_size))
                source_hash.update(name)
            while chunk := source.read(READ_SIZE):
                source_hash.update(chunk)
                raw = prefix + chunk
                for pos in range(max(len(prefix), min_order), len(raw)):
                    byte = raw[pos]
                    for order in range(min_order, min(pos, max_order) + 1):
                        pending[raw[pos - order:pos], byte] += 1
                    if len(pending) >= BATCH_SIZE:
                        flush()
                prefix = raw[-max_order:]
            if _identity(before) != _identity(os.fstat(source.fileno())):
                raise ValueError(f"engram input changed during build: {path}")
    flush()
    return source_hash.hexdigest()


def _pack_counts(counts):
    # A byte ID followed by an unsigned base-128 count, in byte-ID order.
    packed = bytearray()
    for byte, count in counts:
        packed.append(byte)
        while count >= 128:
            packed.append((count & 127) | 128)
            count >>= 7
        packed.append(count)
    return packed


def _unpack_counts(raw, support):
    counts = []
    items = iter(raw)
    previous = -1
    for byte in items:
        count = 0
        for shift in range(0, 63, 7):
            value = next(items, None)
            if value is None:
                raise ValueError("invalid engram counts")
            count |= (value & 127) << shift
            if value < 128:
                break
        else:
            raise ValueError("invalid engram count length")
        if byte <= previous or count == 0:
            raise ValueError("invalid engram counts")
        counts.append((byte, count))
        previous = byte
    if sum(n for _, n in counts) != support:
        raise ValueError("engram count does not match support")
    return tuple(counts)


def _build_database(scratch, files, directory, min_order, max_order, min_support):
    result = scratch / "engram.sqlite"
    with closing(sqlite3.connect(scratch / "counts.sqlite")) as staging, closing(sqlite3.connect(result)) as final:
        for database in (staging, final):
            _configure(database, builder=True)
        staging.execute("CREATE TABLE occurrences (suffix BLOB, byte INTEGER, n INTEGER, "
                        "PRIMARY KEY (suffix, byte)) WITHOUT ROWID")
        source_hash = _count(staging, files, directory, min_order, max_order)
        staging.commit()
        final.execute("CREATE TABLE contexts (suffix BLOB PRIMARY KEY, support INTEGER NOT NULL, "
                      "counts BLOB NOT NULL) WITHOUT ROWID")
        ordered = staging.execute("SELECT suffix, byte, n FROM occurrences ORDER BY suffix, byte")
        entries = 0
        for suffix, rows in groupby(ordered, key=lambda row: row[0]):
            counts = [(byte, n) for _, byte, n in rows]
            support = sum(n for _, n in counts)
            if support >= min_support:
                final.execute("INSERT INTO contexts VALUES (?, ?, ?)", (suffix, support, _pack_counts(counts)))
                entries += 1
        info = EngramInfo(SCHEMA, source_hash, min_order, max_order, min_support, entries)
        final.execute("CREATE TABLE metadata (schema TEXT, source_sha256 TEXT, min_order INTEGER, "
                      "max_order INTEGER, min_support INTEGER, entry_count INTEGER)")
        final.execute("INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?)", tuple(asdict(info).values()))
        final.commit()
    return result


def _publish(partial, out):
    try:
        os.link(partial, out)
    except OSError as exc:
        if not isinstance(exc, PermissionError) and exc.errno not in PUBLISH_FALLBACK:
            raise
        # The exclusively created partial serializes cooperating publishers on FUSE.
        if os.path.lexists(out):
            raise FileExistsError(out) from exc
        os.replace(partial, out)
        return True
    return False


class SSDEngramStore:
    """A bounded read-only pager over exact binary keys. Close after use."""

    @classmethod
    def build(cls, source, out, *, min_order=4, max_order=12, min_support=2):
        """Publish a new immutable index and return its metadata and exact digest."""
        _parameters(min_order, max_order, min_support)
        out = Path(out)
        if os.path.lexists(out):
            raise FileExistsError(out)
        directory, files = _sources(Path(source))
        partial = out.with_name(out.name + ".partial")
        destination = partial.open("xb")
        moved = False
        try:
            with destination, tempfile.TemporaryDirectory(prefix="nexus-ssd-engram-") as scratch:
                local = _build_database(Path(scratch), files, directory, min_order, max_order, min_support)
                with local.open("rb") as database:
                    shutil.copyfileobj(database, destination, READ_SIZE)
                destination.flush()
                os.fsync(destination.fileno())
            with cls(partial) as verified:
                report = dict(**asdict(verified.info), index_bytes=verified.index_bytes, index_sha256=verified.index_sha256)
            moved = _publish(partial, out)
            return report
        except sqlite3.Error as exc:
            raise ValueError(f"could not build engram index: {exc}") from exc
        finally:
            if not moved:
                partial.unlink(missing_ok=True)

    def __init__(self, path, *, expected_sha256=None):
        path = Path(path).resolve()
        if expected_sha256 is not None and not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
            raise ValueError("expected a lowercase 64-hex engram SHA-256")
        with path.open("rb") as raw:
            stat = os.fstat(raw.fileno())
            self.index_bytes = stat.st_size
            self.index_sha256 = hashlib.file_digest(raw, "sha256").hexdigest()
        if expected_sha256 is not None and self.index_sha256 != expected_sha256:
            raise ValueError("engram index SHA-256 mismatch")
        database = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, cached_statements=64)
        try:
            if _identity(stat) != _identity(path.stat()):
                raise ValueError("engram index changed while opening")
            _configure(database)
            records = database.execute("SELECT * FROM metadata LIMIT 2").fetchall()
            if len(records) != 1 or len(records[0]) != 6:
                raise ValueError("invalid engram metadata")
            info = EngramInfo(*records[0])
            if (info.schema != SCHEMA or not isinstance(info.source_sha256, str)
                    or not re.fullmatch(r"[a-f0-9]{64}", info.source_sha256)
                    or type(info.entry_count) is not int or info.entry_count < 0):
                raise ValueError("invalid engram metadata")
            _parameters(info.min_order, info.max_order, info.min_support)
            database.execute("SELECT suffix, support, counts FROM contexts LIMIT 0")
        except sqlite3.Error as exc:
            database.close()
            raise ValueError(f"invalid engram index: {exc}") from exc
        except BaseException:
            database.close()
            raise
        self._database = database
        self.info = info
        self.max_order = info.max_order

    def close(self):
        self._database.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def lookup(self, history, *, max_alpha=0.35, min_support=None):
        """Return (gate, match) from only the supplied observed bytes."""
        alpha = _alpha(max_alpha)
        floor = self.info.min_support if min_support is None else min_support
        _parameters(self.info.min_order, self.max_order, floor)
        if floor < self.info.min_support:
            raise ValueError("lookup min_support cannot be below the index build floor")
        tail = bytes(history[-self.max_order:])
        if not alpha or len(tail) < self.info.min_order:
            return 0.0, None
        keys = [tail[-order:] for order in range(self.info.min_order, len(tail) + 1)]
        params = ",".join("?" for _ in keys)
        try:
            record = self._database.execute(f"SELECT length(suffix), support, counts FROM contexts "
                                            f"WHERE suffix IN ({params}) AND support >= ? "
                                            "ORDER BY length(suffix) DESC LIMIT 1", (*keys, floor)).fetchone()
        except sqlite3.Error as exc:
            raise ValueError(f"could not read engram index: {exc}") from exc
        if record is None:
            return 0.0, None
        order, support, raw = record
        gate = alpha * (order / self.max_order)**4 * (support / (support + 4))
        return gate, EngramMatch(order, support, _unpack_counts(raw, support))
