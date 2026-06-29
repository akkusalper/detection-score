"""detection-score: calibrated somatic variant detection score.

A variant is detected when its AVDS quality score exceeds a threshold T that is
calibrated on a control sample to a target false-positive rate (default 1e-3).
"""
from .calibrate import calibrate_threshold
from .detect import detect, score_vcf

__all__ = ["detect", "score_vcf", "calibrate_threshold"]
__version__ = "0.1.0"
