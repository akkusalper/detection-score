"""Detection-score: AVDS quality score + false-positive-rate-calibrated decision.

A variant is called *detected* when its detection score >= T, where T is either
supplied directly or calibrated on a control sample to a target false-positive rate.
This is the calibrated-score decision rule; it supersedes a fixed PASS cut-off.
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd

from .avds_calculator import AVDSConfig
from .avds_pipeline import AVDSPipeline
from .calibrate import calibrate_threshold


def score_vcf(vcf, bam, reference=None, config=None, threads=None) -> pd.DataFrame:
    """Score every variant in `vcf` against `bam`; returns a DataFrame with an
    `avds_score` column (the detection score)."""
    pipe = AVDSPipeline(bam, reference, config or AVDSConfig(), threads)
    tmp = tempfile.NamedTemporaryFile(suffix=".tsv", delete=False)
    tmp.close()
    try:
        df = pipe.process_vcf(vcf, tmp.name,
                              parallel=(threads is None or threads > 1), write_vcf=False)
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
    return df


def detect(vcf, tumor_bam, control_bam=None, threshold=None, target_fp=1e-3,
           reference=None, config=None, threads=None):
    """Score `vcf` on `tumor_bam` and make detection calls.

    The threshold T is taken from `threshold` if given, otherwise calibrated on
    `control_bam` to `target_fp`. Returns (DataFrame, T); the DataFrame gains
    `threshold`, `target_fp` and `detected` (0/1) columns.
    """
    tum = score_vcf(vcf, tumor_bam, reference, config, threads).copy()
    if threshold is None:
        if control_bam is None:
            raise ValueError("Provide either control_bam (to calibrate) or an explicit threshold")
        ctl = score_vcf(vcf, control_bam, reference, config, threads)
        scores = ctl["avds_score"] if "avds_score" in ctl else []
        threshold = calibrate_threshold(scores, target_fp)
    tum["threshold"] = threshold
    tum["target_fp"] = target_fp
    tum["detected"] = (tum.get("avds_score").fillna(-1) >= threshold).astype(int)
    return tum, float(threshold)
