import pytest

from drone_rf.annotations import Box, parse_label_text


def test_valid_rows_parse_exactly():
    # row copied from the real archive tail probe
    r = parse_label_text("1 0.233398 0.098633 0.019531 0.142578\n")
    assert r.error is None
    assert r.boxes == [Box(1, 0.233398, 0.098633, 0.019531, 0.142578)]


def test_multiple_rows_and_blank_lines():
    r = parse_label_text("5 0.1 0.2 0.3 0.4\n\n5 0.5 0.5 0.1 0.1\n")
    assert r.error is None and r.class_ids == [5, 5]


def test_empty_file_is_zero_boxes_not_error():
    for text in ("", "\n", "   \n"):
        r = parse_label_text(text)
        assert r.error is None and r.boxes == []


@pytest.mark.parametrize(
    "text,error",
    [
        ("1 0.5 0.5 0.1", "bad_column_count"),
        ("1 0.5 0.5 0.1 0.1 0.9", "bad_column_count"),
        ("1 abc 0.5 0.1 0.1", "non_numeric"),
        ("1 nan 0.5 0.1 0.1", "non_numeric"),
        ("6 0.5 0.5 0.1 0.1", "bad_class_id"),
        ("-1 0.5 0.5 0.1 0.1", "bad_class_id"),
        ("1.5 0.5 0.5 0.1 0.1", "bad_class_id"),
        ("1 1.2 0.5 0.1 0.1", "coord_out_of_range"),
        ("1 0.5 -0.1 0.1 0.1", "coord_out_of_range"),
        ("1 0.5 0.5 1.5 0.1", "coord_out_of_range"),
        ("1 0.5 0.5 0.0 0.1", "non_positive_size"),
        ("1 0.5 0.5 0.1 -0.2", "non_positive_size"),
    ],
)
def test_malformed_rows_give_specific_reason(text, error):
    assert parse_label_text(text).error == error


def test_one_bad_row_invalidates_whole_file():
    r = parse_label_text("1 0.5 0.5 0.1 0.1\n1 oops\n")
    assert r.error == "bad_column_count" and r.boxes == []
