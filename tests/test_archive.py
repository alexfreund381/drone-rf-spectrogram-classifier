import tarfile

import pytest

from drone_rf.archive import (
    BLOCK,
    FileRangeReader,
    extract_members,
    fetch_region_members,
    index_labels_from_tail,
    inventory_tar,
    md5_file,
    parse_header,
    read_tar_listing,
    relative_output_path,
    unsafe_reason,
    walk_buffer,
)
from drone_rf.manifest import build_manifest, build_manifest_from_dir
from drone_rf.sources import ARCHIVE_ROOT
from drone_rf.synthetic import make_unsafe_tar


def test_header_walk_matches_tarfile(fixture_tar):
    data = fixture_tar.read_bytes()
    ours = [(h.name.rstrip("/"), h.size, h.data_offset) for h in walk_buffer(data, 0)]
    with tarfile.open(fixture_tar) as tf:
        theirs = [(m.name, m.size, m.offset_data) for m in tf]
    assert ours == theirs and len(ours) > 100


def test_parse_header_rejects_garbage_and_bad_checksum(fixture_tar):
    data = bytearray(fixture_tar.read_bytes()[:BLOCK])
    assert parse_header(bytes(data)) is not None
    assert parse_header(b"\0" * BLOCK) is None
    assert parse_header(b"x" * 100) is None
    data[0] ^= 0xFF  # corrupt name -> checksum mismatch
    assert parse_header(bytes(data)) is None


def test_inventory_lists_without_extracting(fixture_tar, fixture_dir):
    inv = inventory_tar(fixture_tar)
    n_png = len(list((fixture_dir / "images").glob("*.png")))
    assert (inv.member.str.endswith(".png")).sum() == n_png
    assert not inv.duplicate.any()


@pytest.mark.parametrize("chunk", [512, 4096, 1024 * 1024])
def test_label_index_from_tail_equals_directory_labels(fixture_tar, fixture_dir, chunk):
    reader = FileRangeReader(fixture_tar)
    idx = index_labels_from_tail(reader, chunk_bytes=chunk)
    expected = {p.stem: p.read_bytes().decode() for p in (fixture_dir / "labels").glob("*.txt")}
    assert {k: v[1] for k, v in idx.labels.items()} == expected
    assert idx.other_members == [f"{ARCHIVE_ROOT}/data.txt"]
    assert idx.duplicate_members == []
    # the point of the tool: it must not read the image region (small overshoot allowed)
    assert idx.bytes_fetched <= (idx.archive_size - idx.region_start) + chunk + 8192


def test_label_index_manifest_matches_directory_manifest(fixture_tar, fixture_dir):
    idx = index_labels_from_tail(FileRangeReader(fixture_tar), chunk_bytes=2048)
    from_tail, _ = build_manifest(None, idx.labels)
    from_dir, _ = build_manifest_from_dir(fixture_dir)
    cols = ["filename_stem", "class_tag", "source_sample_id", "snr_db", "n_valid_boxes", "binary_label"]
    assert from_tail[cols].equals(from_dir[cols])


def test_tar_listing_matches_directory(fixture_tar, fixture_dir):
    images, labels, dups = read_tar_listing(fixture_tar)
    a, pa = build_manifest(images, labels)
    b, pb = build_manifest_from_dir(fixture_dir)
    assert dups == [] and len(pa) == len(pb) == 0
    drop = ["image_member", "label_member"]
    assert a.drop(columns=drop).equals(b.drop(columns=drop))


def test_fetch_region_returns_only_complete_members(fixture_tar):
    reader = FileRangeReader(fixture_tar)
    # start in the middle of some member: the scanner must resynchronise on a real header
    got = list(fetch_region_members(reader, 3 * BLOCK, 40 * BLOCK, lambda h: h.name.endswith(".png")))
    assert got
    with tarfile.open(fixture_tar) as tf:
        for header, data in got:
            assert tf.extractfile(header.name).read() == data
    assert reader.bytes_read == 40 * BLOCK


@pytest.mark.parametrize(
    "name,reason",
    [
        ("root/images/a.png", None),
        ("/abs/a.png", "absolute_path"),
        ("\\abs\\a.png", "absolute_path"),
        ("C:/x/a.png", "absolute_path"),
        ("root/../../a.png", "path_traversal"),
        ("root\\..\\a.png", "path_traversal"),
        ("", "empty_or_nul"),
    ],
)
def test_unsafe_reason(name, reason):
    assert unsafe_reason(name) == reason


def test_relative_output_path_drops_root():
    assert relative_output_path("Root/images/x.png").as_posix() == "images/x.png"


def test_selective_extraction_writes_only_requested(fixture_tar, tmp_path):
    with tarfile.open(fixture_tar) as tf:
        pngs = [m.name for m in tf if m.name.endswith(".png")]
    wanted = set(pngs[:5])
    rep = extract_members(fixture_tar, wanted | {"does/not/exist.png"}, tmp_path / "out")
    assert set(rep.written) == wanted and rep.not_found == ["does/not/exist.png"]
    files = [p for p in (tmp_path / "out").rglob("*") if p.is_file()]
    assert len(files) == 5 and all(p.parent.name == "images" for p in files)
    # second run never overwrites
    rep2 = extract_members(fixture_tar, wanted, tmp_path / "out")
    assert rep2.written == [] and set(rep2.skipped_existing) == wanted


def test_unsafe_members_are_rejected(tmp_path):
    tar = make_unsafe_tar(tmp_path / "bad.tar")
    with tarfile.open(tar) as tf:
        names = {m.name for m in tf}
    rep = extract_members(tar, names, tmp_path / "out")
    reasons = dict(rep.rejected)
    assert reasons["root/../../escape.png"] == "path_traversal"
    assert reasons["/abs/path.png"] == "absolute_path"
    assert reasons["root/images/link.png"] == "not_regular_file"
    assert reasons["root/images/hard.png"] == "not_regular_file"
    assert reasons["root/images/dup.png"] == "duplicate_member"
    assert sorted(rep.written) == ["root/images/dup.png", "root/images/ok.png"]
    assert (tmp_path / "out" / "images" / "dup.png").read_bytes() == b"first"
    written = [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix == ".png"]
    assert all((tmp_path / "out") in p.parents for p in written)
    assert inventory_tar(tar).duplicate.sum() == 1


def test_oversized_member_rejected(fixture_tar, tmp_path):
    with tarfile.open(fixture_tar) as tf:
        png = next(m.name for m in tf if m.name.endswith(".png"))
    rep = extract_members(fixture_tar, {png}, tmp_path / "o", max_member_bytes=10)
    assert rep.rejected == [(png, "too_large")]


def test_md5(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    assert md5_file(p) == "5d41402abc4b2a76b9719d911017c592"
