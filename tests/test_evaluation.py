import math

import numpy as np
import pandas as pd
import pytest

from drone_rf.evaluate import (
    aggregate_by_group,
    binary_metrics,
    bootstrap_by_group,
    confusion_counts,
    confusion_matrix,
    full_report,
    metrics_by,
    recall_by_tag,
)

# hand-made example: 10 images, TP=3 FN=1 FP=2 TN=4
Y_TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
Y_PRED = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]


def test_confusion_counts_by_hand():
    assert confusion_counts(Y_TRUE, Y_PRED) == {"tp": 3, "fp": 2, "fn": 1, "tn": 4}


def test_metric_formulas_by_hand():
    m = binary_metrics(Y_TRUE, Y_PRED)
    assert m["accuracy"] == pytest.approx(7 / 10)
    assert m["precision_drone"] == pytest.approx(3 / 5)
    assert m["recall_drone"] == pytest.approx(3 / 4)
    assert m["f1_drone"] == pytest.approx(2 * 0.6 * 0.75 / 1.35)
    assert m["precision_background"] == pytest.approx(4 / 5)
    assert m["recall_background"] == pytest.approx(4 / 6)
    assert m["false_positive_rate"] == pytest.approx(2 / 6)
    assert m["balanced_accuracy"] == pytest.approx((3 / 4 + 4 / 6) / 2)
    f1_b = 2 * 0.8 * (4 / 6) / (0.8 + 4 / 6)
    assert m["macro_f1"] == pytest.approx((m["f1_drone"] + f1_b) / 2)
    assert (m["support_drone"], m["support_background"], m["n"]) == (4, 6, 10)


def test_confusion_matrix_order_is_background_then_drone():
    assert confusion_matrix(Y_TRUE, Y_PRED).tolist() == [[4, 2], [1, 3]]


def test_undefined_ratios_are_nan_not_zero():
    m = binary_metrics([0, 0, 0], [0, 0, 0])  # no drone images, none predicted
    assert math.isnan(m["recall_drone"]) and math.isnan(m["precision_drone"])
    assert m["false_positive_rate"] == 0.0 and m["accuracy"] == 1.0


def _pred_frame():
    # group g1: drone, 3 SNR copies, wrong only at the lowest SNR; g2: background, always right
    # group g3: drone, always wrong
    return pd.DataFrame(
        {
            "split_group_id": ["g1"] * 3 + ["g2"] * 3 + ["g3"] * 3,
            "class_tag": ["target0"] * 3 + ["labnoise"] * 3 + ["target1"] * 3,
            "snr_db": [-30, 0, 20] * 3,
            "y_true": [1, 1, 1, 0, 0, 0, 1, 1, 1],
            "p_drone": [0.2, 0.9, 0.9, 0.1, 0.1, 0.1, 0.3, 0.4, 0.4],
            "y_pred": [0, 1, 1, 0, 0, 0, 0, 0, 0],
        }
    )


def test_per_snr_denominators():
    by = metrics_by(_pred_frame(), "snr_db").set_index("snr_db")
    assert by.n.tolist() == [3, 3, 3]
    assert by.loc[-30, "recall_drone"] == 0.0
    assert by.loc[20, "recall_drone"] == pytest.approx(0.5)
    assert by.loc[20, "support_drone"] == 2


def test_group_aggregation_counts_each_source_once():
    g = aggregate_by_group(_pred_frame()).set_index("split_group_id")
    assert len(g) == 3 and g.n_images.tolist() == [3, 3, 3]
    assert g.loc["g1", "y_pred"] == 1  # mean p = 0.667
    assert g.loc["g3", "y_pred"] == 0
    m = binary_metrics(g.y_true, g.y_pred)
    assert m["n"] == 3 and m["recall_drone"] == pytest.approx(0.5)


def test_recall_by_tag():
    t = recall_by_tag(_pred_frame()).set_index("class_tag")
    assert t.loc["target0", "correct_rate"] == pytest.approx(2 / 3)
    assert t.loc["target1", "correct_rate"] == 0.0
    assert t.loc["labnoise", "n_groups"] == 1


def test_bootstrap_resamples_groups_not_images():
    pred = _pred_frame()
    lo, hi = bootstrap_by_group(pred, "accuracy", n_boot=300, seed=0)
    # with whole groups resampled, accuracy can only be a mix of group accuracies 2/3, 1, 0
    assert 0.0 <= lo <= hi <= 1.0 and hi - lo > 0.3
    # all images in one group -> every resample is identical -> zero-width interval
    one = pred.assign(split_group_id="only")
    lo1, hi1 = bootstrap_by_group(one, "accuracy", n_boot=50, seed=0)
    assert lo1 == hi1 == pytest.approx(5 / 9)


def test_bootstrap_is_reproducible():
    pred = _pred_frame()
    assert bootstrap_by_group(pred, "accuracy", 100, seed=3) == bootstrap_by_group(pred, "accuracy", 100, seed=3)


def test_full_report_structure():
    r = full_report(_pred_frame(), n_boot=20)
    assert r["class_order"] == ["background_rf", "drone_rf"]
    assert r["n_images"] == 9 and r["n_groups"] == 3
    assert len(r["per_snr"]) == 3 and len(r["per_tag"]) == 3
    assert np.array(r["confusion_matrix_rows_true_cols_pred"]).sum() == 9
