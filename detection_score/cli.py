"""Command-line interface for detection-score."""
from __future__ import annotations

import argparse
import sys

from .avds_calculator import AVDSConfig
from .detect import detect, write_detectable_vcf


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="detection-score",
        description="Calibrated somatic detection score: AVDS quality score with a "
                    "false-positive-rate-calibrated decision threshold (call when score >= T).")
    ap.add_argument("-v", "--vcf", required=True, help="candidate variants (VCF/VCF.gz)")
    ap.add_argument("-b", "--bam", required=True, help="tumour BAM/CRAM")
    ap.add_argument("-c", "--control", help="control/normal BAM/CRAM used to calibrate the threshold")
    ap.add_argument("--threshold", type=float, help="use a fixed score threshold instead of calibrating")
    ap.add_argument("--fp", type=float, default=1e-3, help="target false-positive rate for calibration (default 1e-3)")
    ap.add_argument("-r", "--reference", help="reference FASTA (required for CRAM)")
    ap.add_argument("-t", "--threads", type=int, help="threads (default: CPU count - 1)")
    ap.add_argument("--theta-vaf", type=float, default=0.50, help="VAF normalisation threshold")
    ap.add_argument("--theta-depth", type=int, default=100, help="depth normalisation threshold")
    ap.add_argument("--min-coverage", type=int, default=5, help="minimum coverage to score a site")
    ap.add_argument("-o", "--output", required=True, help="output TSV of scores and detection calls")
    ap.add_argument("--vcf-out", help="also write an annotated VCF with the DETECTABLE flag")
    a = ap.parse_args(argv)

    if a.control is None and a.threshold is None:
        ap.error("provide either --control (to calibrate) or --threshold")

    cfg = AVDSConfig(theta_vaf=a.theta_vaf, theta_depth=a.theta_depth, min_coverage=a.min_coverage)
    df, T = detect(a.vcf, a.bam, control_bam=a.control, threshold=a.threshold,
                   target_fp=a.fp, reference=a.reference, config=cfg, threads=a.threads)

    cols = [c for c in ["chrom", "pos", "id", "ref", "alt", "avds_score", "vaf_raw",
                        "depth_total", "threshold", "target_fp", "detectable", "decision"]
            if c in df.columns]
    df.to_csv(a.output, sep="\t", index=False, columns=cols, float_format="%.4f")
    if a.vcf_out:
        write_detectable_vcf(a.vcf, a.vcf_out, df, T)

    # No end-of-run summary is printed: the only outputs are the TSV (and the VCF
    # when requested). Errors still propagate; routine progress/INFO is silenced.
    return 0


if __name__ == "__main__":
    sys.exit(main())
