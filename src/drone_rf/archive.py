"""TAR tools: header walking (local or over HTTP range requests), label indexing without
downloading images, inventory, MD5, and safe selective extraction.

The real archive is an uncompressed ustar TAR: root folder, ``images/`` (shuffled), then
``labels/`` at the very end. Because label files are tiny and sit at the end, the whole label
set can be fetched with range requests over the last few hundred MB.

We never call ``tarfile.extractall``. Every member name is validated before anything is written.
"""

from __future__ import annotations

import hashlib
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd

BLOCK = 512
ZERO_BLOCK = b"\0" * BLOCK
REGULAR_TYPES = ("0", "\0", "")


@dataclass(frozen=True)
class TarHeader:
    name: str
    size: int
    typeflag: str
    offset: int  # absolute offset of the 512-byte header in the archive

    @property
    def data_offset(self) -> int:
        return self.offset + BLOCK

    @property
    def next_offset(self) -> int:
        return self.offset + BLOCK + ((self.size + BLOCK - 1) // BLOCK) * BLOCK

    @property
    def is_regular(self) -> bool:
        return self.typeflag in REGULAR_TYPES


def parse_header(block: bytes, offset: int = 0) -> TarHeader | None:
    """Parse one 512-byte ustar header. Returns None unless magic and checksum are valid."""
    if len(block) != BLOCK or block[257:262] != b"ustar":
        return None
    try:
        stored = int(block[148:156].strip(b"\0 ") or b"0", 8)
        size = int(block[124:136].strip(b"\0 ") or b"0", 8)
    except ValueError:
        return None
    if sum(block[:148]) + 8 * 32 + sum(block[156:]) != stored:
        return None
    name = block[:100].split(b"\0", 1)[0].decode("utf-8", "replace")
    prefix = block[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
    if prefix:
        name = f"{prefix}/{name}"
    return TarHeader(name, size, chr(block[156]) if block[156] else "\0", offset)


# --------------------------------------------------------------------------- range readers


class FileRangeReader:
    """Random-access reader over a local file (also used as a fake remote in tests)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self.bytes_read = 0

    def read(self, start: int, length: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(start)
            data = f.read(length)
        self.bytes_read += len(data)
        return data


class HttpRangeReader:
    """Random-access reader over HTTP using ``Range`` requests, with retries."""

    def __init__(self, url: str, size: int | None = None, retries: int = 5, timeout: float = 120.0):
        self.url = url
        self.retries = retries
        self.timeout = timeout
        self.bytes_read = 0
        self.size = size if size is not None else self._probe_size()

    def _open(self, start: int, end: int):
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "drone-rf-range-reader"}
        )
        return urllib.request.urlopen(req, timeout=self.timeout)

    def _probe_size(self) -> int:
        with self._open(0, 0) as resp:
            if resp.status != 206:
                raise OSError("server does not support range requests")
            return int(resp.headers["Content-Range"].rsplit("/", 1)[1])

    def read(self, start: int, length: int) -> bytes:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                with self._open(start, start + length - 1) as resp:
                    if resp.status != 206:
                        raise OSError(f"expected HTTP 206, got {resp.status}")
                    data = resp.read()
                if len(data) != length:
                    raise OSError(f"short read: {len(data)} of {length} bytes")
                self.bytes_read += len(data)
                return data
            except (OSError, urllib.error.URLError) as exc:
                last = exc
                time.sleep(2.0 * (attempt + 1))
        raise OSError(f"range read failed after {self.retries} attempts: {last}")


# --------------------------------------------------------------------------- walking


def walk_buffer(buf: bytes | bytearray | memoryview, base_offset: int, start: int = 0) -> Iterator[TarHeader]:
    """Walk TAR headers sequentially inside an in-memory buffer.

    ``base_offset`` is the absolute archive offset of ``buf[0]``; ``start`` is where the first
    header sits inside the buffer. Stops at the end-of-archive zero block, at the end of the
    buffer, or raises ValueError on a corrupt header.
    """
    pos = start
    view = memoryview(buf)
    while pos + BLOCK <= len(view):
        block = bytes(view[pos : pos + BLOCK])
        if block == ZERO_BLOCK:
            return
        header = parse_header(block, base_offset + pos)
        if header is None:
            raise ValueError(f"corrupt TAR header at offset {base_offset + pos}")
        yield header
        pos = header.next_offset - base_offset


def find_first_header(buf: bytes | bytearray, base_offset: int = 0) -> int | None:
    """Scan block-aligned positions for the first valid header. Returns buffer position."""
    for pos in range(0, len(buf) - BLOCK + 1, BLOCK):
        if buf[pos + 257 : pos + 262] == b"ustar" and parse_header(bytes(buf[pos : pos + BLOCK])) is not None:
            return pos
    return None


@dataclass
class LabelIndex:
    labels: dict[str, tuple[str, str]]  # stem -> (member name, text)
    other_members: list[str]  # non-label members found in the region (e.g. data.txt)
    duplicate_members: list[str]
    region_start: int  # absolute offset where the label region starts
    archive_size: int
    bytes_fetched: int


def index_labels_from_tail(
    reader,
    chunk_bytes: int = 16 * 1024 * 1024,
    progress: Callable[[str], None] | None = None,
) -> LabelIndex:
    """Read every label file from the end of a TAR without reading the images.

    Fetches fixed-size chunks backwards from the end of the archive until a chunk contains an
    image member header; the label region starts right after that image's data. Then walks the
    region forward header by header.
    """
    if chunk_bytes % BLOCK:
        raise ValueError("chunk_bytes must be a multiple of 512")
    total = reader.size
    if total % BLOCK:
        raise ValueError("archive size is not a multiple of 512; not a plain TAR")
    chunks: list[bytes] = []
    pos = total
    region_start: int | None = None
    while pos > 0 and region_start is None:
        start = max(0, pos - chunk_bytes)
        data = reader.read(start, pos - start)
        chunks.insert(0, data)
        pos = start
        # scan this chunk backwards for the last image header
        for p in range(len(data) - BLOCK, -1, -BLOCK):
            if data[p + 257 : p + 262] != b"ustar":
                continue
            header = parse_header(data[p : p + BLOCK], start + p)
            if header is not None and header.is_regular and header.name.lower().endswith(".png"):
                region_start = header.next_offset
                break
        if progress:
            progress(f"fetched {(total - pos) / 1e6:.0f} MB from the tail")
    buf = b"".join(chunks)
    base = pos
    if region_start is None:
        region_start = 0  # whole archive is small / has no images
        first = find_first_header(buf, base)
        region_start = base + (first or 0)
    labels: dict[str, tuple[str, str]] = {}
    other: list[str] = []
    dups: list[str] = []
    for header in walk_buffer(buf, base, region_start - base):
        if not header.is_regular:
            continue
        path = PurePosixPath(header.name)
        if path.suffix == ".txt" and path.parent.name == "labels":
            if path.stem in labels:
                dups.append(header.name)
                continue
            data = buf[header.data_offset - base : header.data_offset - base + header.size]
            labels[path.stem] = (header.name, data.decode("utf-8", "replace"))
        else:
            other.append(header.name)
    return LabelIndex(labels, other, dups, region_start, total, getattr(reader, "bytes_read", len(buf)))


def fetch_region_members(
    reader,
    start: int,
    length: int,
    keep: Callable[[TarHeader], bool],
) -> Iterator[tuple[TarHeader, bytes]]:
    """Download one contiguous byte region and yield every *complete* member inside it.

    Useful because the archive is shuffled: a contiguous region is an arbitrary mix of samples,
    classes and SNR levels. Members cut off by the region's edges are skipped.
    """
    start -= start % BLOCK
    length -= length % BLOCK
    buf = reader.read(start, min(length, reader.size - start))
    first = find_first_header(buf, start)
    if first is None:
        return
    pos = first
    while pos + BLOCK <= len(buf):
        block = buf[pos : pos + BLOCK]
        if block == ZERO_BLOCK:
            return
        header = parse_header(block, start + pos)
        if header is None:
            raise ValueError(f"corrupt TAR header at offset {start + pos}")
        end = pos + BLOCK + header.size
        if end > len(buf):
            return
        if header.is_regular and keep(header):
            yield header, buf[pos + BLOCK : end]
        pos = header.next_offset - start


# --------------------------------------------------------------------------- local archives


def inventory_tar(tar_path: str | Path) -> pd.DataFrame:
    """List every member of a local TAR without extracting anything."""
    rows = []
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            rows.append(
                {
                    "member": m.name,
                    "size": m.size,
                    "offset_data": m.offset_data,
                    "type": "file" if m.isreg() else "dir" if m.isdir() else "other",
                }
            )
    df = pd.DataFrame(rows, columns=["member", "size", "offset_data", "type"])
    df["duplicate"] = df.duplicated("member", keep="first")
    return df


def read_tar_listing(tar_path: str | Path) -> tuple[dict[str, str], dict[str, tuple[str, str]], list[str]]:
    """Read image names and label contents from a local TAR.

    Returns ``(images, labels, duplicate_members)`` in the form ``build_manifest`` expects.
    """
    images: dict[str, str] = {}
    labels: dict[str, tuple[str, str]] = {}
    dups: list[str] = []
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if not m.isreg():
                continue
            path = PurePosixPath(m.name)
            if path.suffix == ".png" and path.parent.name == "images":
                if path.stem in images:
                    dups.append(m.name)
                else:
                    images[path.stem] = m.name
            elif path.suffix == ".txt" and path.parent.name == "labels":
                if path.stem in labels:
                    dups.append(m.name)
                else:
                    fobj = tf.extractfile(m)
                    labels[path.stem] = (m.name, fobj.read().decode("utf-8", "replace") if fobj else "")
    return images, labels, dups


def md5_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """MD5 of a file, streamed. For the full archive this reads 101 GB."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while data := f.read(chunk_bytes):
            h.update(data)
    return h.hexdigest()


# --------------------------------------------------------------------------- safe extraction


def unsafe_reason(name: str) -> str | None:
    """Return why a member name is unsafe to extract, or None if it is fine."""
    if not name or "\0" in name:
        return "empty_or_nul"
    if name.startswith(("/", "\\")) or (len(name) > 1 and name[1] == ":"):
        return "absolute_path"
    parts = name.replace("\\", "/").split("/")
    if ".." in parts:
        return "path_traversal"
    return None


def relative_output_path(name: str) -> PurePosixPath:
    """Drop the archive's root folder: ``Root/images/x.png`` -> ``images/x.png``."""
    parts = PurePosixPath(name).parts
    return PurePosixPath(*parts[1:]) if len(parts) > 1 else PurePosixPath(*parts)


@dataclass
class ExtractReport:
    written: list[str]
    skipped_existing: list[str]
    rejected: list[tuple[str, str]]  # (member, reason)
    not_found: list[str]


def extract_members(
    tar_path: str | Path,
    wanted: set[str],
    out_dir: str | Path,
    max_member_bytes: int = 50 * 1024 * 1024,
) -> ExtractReport:
    """Extract only ``wanted`` members from a local TAR into ``out_dir``, safely.

    Rejects absolute paths, ``..``, links and special files, oversized members, and the second
    occurrence of a duplicate name. Never overwrites an existing file.
    """
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    report = ExtractReport([], [], [], [])
    seen: set[str] = set()
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if m.name not in wanted:
                continue
            reason = unsafe_reason(m.name)
            if reason is None and not m.isreg():
                reason = "not_regular_file"
            if reason is None and m.size > max_member_bytes:
                reason = "too_large"
            if reason is None and m.name in seen:
                reason = "duplicate_member"
            if reason is not None:
                report.rejected.append((m.name, reason))
                continue
            seen.add(m.name)
            target = (out_root / Path(*relative_output_path(m.name).parts)).resolve()
            if out_root not in target.parents:
                report.rejected.append((m.name, "escapes_output_dir"))
                continue
            if target.exists():
                report.skipped_existing.append(m.name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            with open(target, "wb") as dst:
                dst.write(src.read() if src else b"")
            report.written.append(m.name)
    report.not_found = sorted(wanted - seen - {r[0] for r in report.rejected})
    return report
