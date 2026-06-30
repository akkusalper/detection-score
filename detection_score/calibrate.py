"""False-positive-rate calibration of the detection-score threshold.

The control (e.g. matched-normal) score distribution at the candidate positions is
overwhelmingly concentrated at zero, so a percentile is uninformative. We therefore
use a rank-based threshold: T is the k-th highest control score, with
k = round(target_fp * N). A call is made when the tumour score >= T, which bounds the
realised false-positive rate on the control at <= target_fp.
"""
from __future__ import annotations

import numpy as np


def calibrate_threshold(control_scores, target_fp: float = 1e-3) -> float:
    """Return the score threshold T that holds the control false-positive rate at
    <= target_fp.

    Parameters
    ----------
    control_scores : iterable of float
        Detection scores at the candidate positions in a control/normal sample.
    target_fp : float
        Target per-locus false-positive rate (default 1e-3).
    """
    s = np.asarray([x for x in control_scores if x is not None and np.isfinite(x)], dtype=float)
    if s.size == 0:
        return 0.0
    s.sort()
    s = s[::-1]
    k = max(1, int(round(target_fp * s.size)))
    return float(s[min(k - 1, s.size - 1)])


def label_detectable(scores, threshold: float):
    """Render the calibrated decision as human-readable text.

    Returns an array of "detectable" (detection score >= threshold) or
    "below the threshold" (score < threshold, or non-finite/uncovered). This is
    only the text rendering of the numeric decision detectable = (score >= T); it
    introduces no second cut-off, so its "detectable" count is identical to the
    0/1 column's sum.
    """
    s = np.asarray(scores, dtype=float)
    return np.where(s >= threshold, "detectable", "below the threshold")
