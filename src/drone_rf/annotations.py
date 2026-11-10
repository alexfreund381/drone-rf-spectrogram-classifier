"""Parse and validate YOLO-format label files.

One row per box: ``class_id x_center y_center width height`` with geometry normalized to [0, 1].
An empty file means zero boxes. Any malformed row makes the whole file invalid (we never keep
half of a label file).
"""

from __future__ import annotations

from dataclasses import dataclass, field

N_DETECTOR_CLASSES = 6


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class LabelParse:
    """Result of parsing one label file. ``error`` is None when the file is valid."""

    boxes: list[Box] = field(default_factory=list)
    error: str | None = None

    @property
    def class_ids(self) -> list[int]:
        return [b.class_id for b in self.boxes]


def parse_label_text(text: str) -> LabelParse:
    """Parse label file contents.

    Error codes: ``bad_column_count``, ``non_numeric``, ``bad_class_id``,
    ``coord_out_of_range``, ``non_positive_size``.
    """
    boxes: list[Box] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            return LabelParse(error="bad_column_count")
        try:
            cls_float = float(parts[0])
            xc, yc, w, h = (float(p) for p in parts[1:])
        except ValueError:
            return LabelParse(error="non_numeric")
        if cls_float != int(cls_float) or not 0 <= int(cls_float) < N_DETECTOR_CLASSES:
            return LabelParse(error="bad_class_id")
        if any(v != v for v in (xc, yc, w, h)):  # NaN
            return LabelParse(error="non_numeric")
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and w <= 1.0 and h <= 1.0):
            return LabelParse(error="coord_out_of_range")
        if w <= 0.0 or h <= 0.0:
            return LabelParse(error="non_positive_size")
        boxes.append(Box(int(cls_float), xc, yc, w, h))
    return LabelParse(boxes=boxes)
