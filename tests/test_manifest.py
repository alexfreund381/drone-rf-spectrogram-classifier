import pytest

from drone_rf.annotations import parse_label_text
from drone_rf.manifest import (
    QUARANTINE,
    ParsedName,
    build_manifest,
    build_manifest_from_dir,
    derive_binary_label,
    parse_stem,
    read_manifest,
    summarize,
)
from drone_rf.synthetic import UNPARSEABLE_STEM


@pytest.mark.parametrize(
    "stem,expected",
    [
        # real names seen in the archive probe
        ("IQdata_sample5728_target3_snr8db", ParsedName(5728, "target3", 8)),
        ("IQdata_sample3693_target1_snr-2db", ParsedName(3693, "target1", -2)),
        ("IQdata_sample10880_labnoise_snr-30db", ParsedName(10880, "labnoise", -30)),
        ("IQdata_sample0_target0_snr0db", ParsedName(0, "target0", 0)),
        ("IQdata_sample7_target5_snr20db", ParsedName(7, "target5", 20)),
    ],
)
def test_parse_stem(stem, expected):
    assert parse_stem(stem) == expected


@pytest.mark.parametrize(
    "stem",
    [
        "IQdata_sample12_target6_snr0db",  # unknown tag
        "IQdata_sample12_target1_clean",  # pristine-style name we have not seen
        "IQdata_sampleX_target1_snr0db",
        "IQdata_sample12_target1_snr0db.png",
        "random",
        "",
    ],
)
def test_unexpected_names_return_none(stem):
    assert parse_stem(stem) is None


@pytest.mark.parametrize(
    "tag,text,label,reason",
    [
        ("target1", "1 0.5 0.5 0.1 0.1\n", "1", ""),
        ("target1", "1 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n", "1", ""),
        ("labnoise", "", "0", ""),
        ("target0", "", QUARANTINE, "target_zero_boxes"),
        ("labnoise", "0 0.5 0.5 0.1 0.1\n", QUARANTINE, "background_has_boxes"),
        ("target1", "3 0.5 0.5 0.1 0.1\n", QUARANTINE, "class_mismatch"),
        ("target2", "2 0.5 0.5 0.1 0.1\n4 0.2 0.2 0.1 0.1\n", QUARANTINE, "class_mismatch"),
        ("target3", "3 1.5 0.5 0.1 0.1\n", QUARANTINE, "invalid_label:coord_out_of_range"),
        ("labnoise", "garbage", QUARANTINE, "invalid_label:bad_column_count"),
    ],
)
def test_labeling_rules(tag, text, label, reason):
    assert derive_binary_label(tag, parse_label_text(text)) == (label, reason)


def test_missing_label_is_quarantined():
    assert derive_binary_label("target0", None) == (QUARANTINE, "missing_label")
    assert derive_binary_label("labnoise", None) == (QUARANTINE, "missing_label")


def test_clean_fixture_manifest(fixture_dir):
    m, problems = build_manifest_from_dir(fixture_dir)
    assert len(problems) == 0
    assert len(m) == (6 * 4 + 24) * 3
    assert set(m.binary_label) == {"0", "1"}
    assert (m[m.class_tag == "labnoise"].binary_label == "0").all()
    assert (m[m.class_tag != "labnoise"].binary_label == "1").all()
    assert sorted(m.snr_db.unique()) == [-30, 0, 20]


def test_edge_cases_are_quarantined_with_reasons(edge_dir):
    m, problems = build_manifest_from_dir(edge_dir)
    q = m[m.binary_label == QUARANTINE].set_index("source_sample_id").quarantine_reason.to_dict()
    assert q == {
        9000: "target_zero_boxes",
        9001: "background_has_boxes",
        9002: "class_mismatch",
        9003: "class_mismatch",
        9004: "invalid_label:coord_out_of_range",
        9005: "invalid_label:non_positive_size",
        9006: "invalid_label:bad_column_count",
        9007: "invalid_label:non_numeric",
        9008: "missing_label",
        9010: "invalid_label:bad_class_id",
    }
    probs = dict(zip(problems.filename_stem, problems.problem, strict=True))
    assert probs[UNPARSEABLE_STEM] == "unparseable_name"
    assert probs["IQdata_sample9009_target0_snr0db"] == "missing_image"
    assert len(probs) == 2


def test_label_only_mode_has_no_missing_image_problem():
    labels = {"IQdata_sample1_target0_snr0db": ("labels/a.txt", "0 0.5 0.5 0.1 0.1\n")}
    m, problems = build_manifest(None, labels)
    assert len(m) == 1 and len(problems) == 0 and m.image_member[0] == ""


def test_manifest_roundtrip_keeps_string_labels(manifest, tmp_path):
    path = tmp_path / "m.csv"
    manifest.to_csv(path, index=False)
    back = read_manifest(path)
    assert back.binary_label.tolist() == manifest.binary_label.tolist()
    assert back.quarantine_reason.eq("").all()


def test_manifest_is_deterministic(fixture_dir):
    a, _ = build_manifest_from_dir(fixture_dir)
    b, _ = build_manifest_from_dir(fixture_dir)
    assert a.equals(b)


def test_summary_counts(manifest, edge_dir):
    s = summarize(manifest)
    by_tag = s["samples_by_tag"].set_index("class_tag")
    assert by_tag.loc["labnoise", "n_source_samples"] == 24
    assert by_tag.loc["target0", "n_source_samples"] == 4
    assert len(s["sample_ids_under_multiple_tags"]) == 0
    assert (s["per_sample"].n_snr_levels == 3).all()
    edge, _ = build_manifest_from_dir(edge_dir)
    by_tag = summarize(edge)["samples_by_tag"].set_index("class_tag")
    assert by_tag.loc["target0", "samples_fully_quarantined"] == 3  # 9000, 9007, 9008
