"""
AVDS (Allele-Specific Variant Detection Score) Calculator
Version: 1.0
Date: February 2026

This module implements the complete AVDS mathematical formulation
for assessing genomic variant quality from BAM/CRAM alignments.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AVDSConfig:
    """Configuration parameters for AVDS calculation"""
    # VAF normalization thresholds
    theta_vaf: float = 0.50  # 0.50 (heterozygous), 0.30, 0.10, 0.05

    # Depth normalization threshold
    theta_depth: int = 100

    # Minimum strand ratio (25% rule)
    min_strand_ratio: float = 0.25

    # Component weights (must sum to 1.0)
    weight_vaf: float = 0.30
    weight_q_alt: float = 0.35
    weight_depth: float = 0.10
    weight_sb: float = 0.15
    weight_pb: float = 0.10

    # Read quality weights (must sum to 1.0)
    weight_mapq: float = 0.30
    weight_bq: float = 0.35
    weight_as: float = 0.20
    weight_nm: float = 0.10
    weight_cs: float = 0.05

    # Quality score normalization values
    max_mapq: int = 60
    max_bq: int = 40

    # Epsilon for zero protection
    epsilon: float = 1e-6

    # Minimum coverage to attempt calculation
    min_coverage: int = 5

    # Base quality threshold
    min_base_quality: int = 20

    # Mapping quality threshold
    min_mapping_quality: int = 20


@dataclass
class ReadInfo:
    """Information extracted from a single read"""
    is_alt: bool  # True if supports alternative allele
    is_forward: bool  # True if forward strand
    mapq: int  # Mapping quality
    base_quality: int  # Base quality at variant position
    alignment_score: int  # AS tag
    edit_distance: int  # NM tag
    read_length: int  # Read length
    query_position: int  # Position of variant in read (0-based)
    cigar_score: float  # CIGAR quality score


class AVDSCalculator:
    """
    Complete AVDS score calculator implementing the mathematical
    formulation from AVDS_Mathematical_Formulation_v2.
    """

    def __init__(self, config: Optional[AVDSConfig] = None):
        """
        Initialize AVDS calculator with configuration.

        Args:
            config: AVDSConfig object. If None, uses default parameters.
        """
        self.config = config if config is not None else AVDSConfig()
        self._validate_config()
        logger.info(f"AVDS Calculator initialized with theta_vaf={self.config.theta_vaf}")

    def _validate_config(self):
        """Validate configuration parameters"""
        # Check weights sum to 1.0
        component_weights = (
            self.config.weight_vaf +
            self.config.weight_q_alt +
            self.config.weight_depth +
            self.config.weight_sb +
            self.config.weight_pb
        )
        if not np.isclose(component_weights, 1.0):
            raise ValueError(f"Component weights must sum to 1.0, got {component_weights}")

        read_weights = (
            self.config.weight_mapq +
            self.config.weight_bq +
            self.config.weight_as +
            self.config.weight_nm +
            self.config.weight_cs
        )
        if not np.isclose(read_weights, 1.0):
            raise ValueError(f"Read quality weights must sum to 1.0, got {read_weights}")

    def calculate_cigar_score(self, cigar_tuples: List[Tuple[int, int]]) -> float:
        """
        Calculate CIGAR string quality score.

        CIGAR operations (from pysam):
        0 = M (match/mismatch)
        1 = I (insertion)
        2 = D (deletion)
        4 = S (soft clip)
        5 = H (hard clip)

        Scoring:
        - Perfect match (only M): 1.0
        - Contains I or D: 0.5
        - Contains S: 0.2
        - Contains H: 0.0

        Args:
            cigar_tuples: List of (operation, length) tuples

        Returns:
            CIGAR quality score (0.0-1.0)
        """
        if not cigar_tuples:
            return 0.0

        operations = set(op for op, length in cigar_tuples)

        # Hard clip = 0.0
        if 5 in operations:
            return 0.0

        # Soft clip = 0.2
        if 4 in operations:
            return 0.2

        # Insertion or deletion = 0.5
        if 1 in operations or 2 in operations:
            return 0.5

        # Perfect match (only M operations) = 1.0
        if operations == {0}:
            return 1.0

        # Default for other cases
        return 0.5

    def calculate_read_quality(self, read_info: ReadInfo) -> float:
        """
        Calculate per-read quality score Q_read.

        Q_read = w₁·(MAPQ/60) + w₂·(BQ/40) + w₃·(AS/L) + w₄·(1-NM/L) + w₅·CS

        Args:
            read_info: ReadInfo object with extracted read information

        Returns:
            Read quality score (0.0-1.0)
        """
        # Normalize MAPQ
        mapq_norm = min(read_info.mapq / self.config.max_mapq, 1.0)

        # Normalize base quality
        bq_norm = min(read_info.base_quality / self.config.max_bq, 1.0)

        # Normalize alignment score (AS / read_length)
        # AS is typically positive and scales with read length
        # For perfect match, AS ≈ read_length
        as_norm = min(read_info.alignment_score / read_info.read_length, 1.0) if read_info.read_length > 0 else 0.0

        # Normalize edit distance (1 - NM/L)
        nm_norm = max(0.0, 1.0 - (read_info.edit_distance / read_info.read_length)) if read_info.read_length > 0 else 0.0

        # CIGAR score
        cs_score = read_info.cigar_score

        # Weighted combination
        q_read = (
            self.config.weight_mapq * mapq_norm +
            self.config.weight_bq * bq_norm +
            self.config.weight_as * as_norm +
            self.config.weight_nm * nm_norm +
            self.config.weight_cs * cs_score
        )

        return max(0.0, min(1.0, q_read))

    def calculate_vaf_normalized(self, n_alt: int, n_ref: int) -> Tuple[float, float]:
        """
        Calculate normalized VAF (VAF*).

        VAF = n_alt / (n_alt + n_ref)
        VAF* = min(VAF / θ_VAF, 1.0)

        Args:
            n_alt: Number of alternative allele reads
            n_ref: Number of reference allele reads

        Returns:
            Tuple of (raw_vaf, normalized_vaf)
        """
        total = n_alt + n_ref
        if total == 0:
            return 0.0, 0.0

        raw_vaf = n_alt / total
        normalized_vaf = min(raw_vaf / self.config.theta_vaf, 1.0)

        return raw_vaf, normalized_vaf

    def calculate_depth_normalized(self, depth: int) -> float:
        """
        Calculate normalized depth (D*).

        D* = min(D / θ_depth, 1.0)

        Args:
            depth: Total sequencing depth

        Returns:
            Normalized depth score (0.0-1.0)
        """
        return min(depth / self.config.theta_depth, 1.0)

    def calculate_strand_bias_strict(self, n_forward: int, n_reverse: int, n_alt: int) -> Tuple[float, float, bool]:
        """
        Calculate strict strand bias score (SB_strict) with statistical confidence.

        Enhanced with binomial test and depth-aware penalties to account for
        depth-dependent uncertainty in strand bias assessment.

        This is the most critical component for artifact detection.

        Algorithm:
        1. Calculate strand ratios
        2. Perform binomial test for statistical significance
        3. Apply depth-dependent confidence penalty
        4. Calculate multi-factor strand bias score

        Args:
            n_forward: Alternative allele reads from forward strand
            n_reverse: Alternative allele reads from reverse strand
            n_alt: Total alternative allele reads

        Returns:
            Tuple of (sb_strict_score, min_ratio, failed_25_rule)
        """
        if n_alt == 0:
            return 0.0, 0.0, True

        # Step 1: Calculate strand ratios
        forward_ratio = n_forward / n_alt
        reverse_ratio = n_reverse / n_alt
        min_ratio = min(forward_ratio, reverse_ratio)

        # Step 2: Calculate base strand bias
        base_sb = 1.0 - abs(n_forward - n_reverse) / n_alt

        # Step 3: Check 25% minimum rule
        failed_25_rule = min_ratio < self.config.min_strand_ratio

        # === NEW: Statistical confidence assessment ===

        # Binomial test for statistical significance
        pvalue = self._binomial_test(n_forward, n_alt)
        is_significant = pvalue < 0.05

        # Depth-based confidence
        depth_conf = self._get_depth_confidence(n_alt)

        # === Multi-factor decision logic ===

        if n_alt >= 30 and is_significant:
            # High depth + statistically significant imbalance
            # → Strong evidence of artifact
            sb_strict = base_sb * 0.3
        elif n_alt < 10:
            # Low depth → High uncertainty
            if failed_25_rule:
                # Low depth + ratio failure → Very suspicious
                penalty_factor = min_ratio / self.config.min_strand_ratio
                sb_strict = base_sb * (penalty_factor ** 2) * depth_conf
            else:
                # Low depth but ratio OK → Still penalize uncertainty
                sb_strict = base_sb * depth_conf
        elif failed_25_rule:
            # Medium/high depth, ratio-based penalty (original method)
            penalty_factor = min_ratio / self.config.min_strand_ratio
            sb_strict = base_sb * (penalty_factor ** 2) * depth_conf
        else:
            # No issues detected
            sb_strict = base_sb * depth_conf

        return sb_strict, min_ratio, failed_25_rule

    def _binomial_test(self, n_forward: int, n_total: int) -> float:
        """
        Test if strand imbalance is statistically significant.

        H0: p_forward = 0.5 (balanced strands)

        Args:
            n_forward: Number of forward strand reads
            n_total: Total number of reads

        Returns:
            p-value from binomial test
        """
        from scipy.stats import binomtest

        if n_total == 0:
            return 1.0

        result = binomtest(n_forward, n_total, p=0.5, alternative='two-sided')
        return result.pvalue

    def _get_depth_confidence(self, n_alt: int) -> float:
        """
        Calculate confidence score based on read depth.

        Low depth = high uncertainty = low confidence

        Args:
            n_alt: Number of alternative allele reads

        Returns:
            Confidence score (0.0-1.0)
        """
        if n_alt < 10:
            return 0.3   # Very uncertain
        elif n_alt < 30:
            return 0.6   # Moderately uncertain
        elif n_alt < 100:
            return 0.85  # Somewhat confident
        else:
            return 1.0   # Highly confident

    def calculate_position_bias(self, query_positions: List[int], read_lengths: List[int]) -> Tuple[float, float]:
        """
        Calculate position bias score (PB).

        Variants near read ends are more likely to be errors.

        Algorithm:
        1. Normalize position for each read: (query_position / read_length) × 100
        2. Calculate average position P̄
        3. Calculate PB = 1 - |P̄ - 50| / 50

        Args:
            query_positions: List of variant positions in reads (0-based)
            read_lengths: List of corresponding read lengths

        Returns:
            Tuple of (pb_score, average_position_percent)
        """
        if not query_positions or not read_lengths:
            return 0.0, 0.0

        # Step 1: Normalize positions to percentage
        normalized_positions = []
        for pos, length in zip(query_positions, read_lengths):
            if length > 0:
                # Convert 0-based to percentage (0-100)
                norm_pos = (pos / length) * 100.0
                normalized_positions.append(norm_pos)

        if not normalized_positions:
            return 0.0, 0.0

        # Step 2: Calculate average position
        avg_position = np.mean(normalized_positions)

        # Step 3: Calculate position bias
        # PB = 1.0 when avg_position = 50 (centered)
        # PB decreases linearly toward 0.0 as position moves to ends
        pb_score = 1.0 - abs(avg_position - 50.0) / 50.0
        pb_score = max(0.0, min(1.0, pb_score))

        return pb_score, avg_position

    def calculate_avds(
        self,
        read_infos: List[ReadInfo],
        n_alt: int,
        n_ref: int
    ) -> Dict[str, float]:
        """
        Calculate complete AVDS score with all components.

        Main formula:
        AVDS = 100 × [VAF*^α₁ × Q̄_alt^α₂ × D*^α₃ × SB_strict^α₄ × PB^α₅]

        Args:
            read_infos: List of ReadInfo objects for alternative allele reads
            n_alt: Number of alternative allele reads
            n_ref: Number of reference allele reads

        Returns:
            Dictionary containing:
            - avds_score: Final AVDS score (0-100)
            - vaf_raw: Raw VAF
            - vaf_norm: Normalized VAF*
            - q_alt_mean: Average alternative allele quality
            - depth_total: Total depth
            - depth_norm: Normalized depth
            - sb_strict: Strict strand bias score
            - sb_min_ratio: Minimum strand ratio
            - sb_failed_25: Whether failed 25% rule
            - pb_score: Position bias score
            - pb_avg_pos: Average position percentage
            - n_forward: Forward strand count
            - n_reverse: Reverse strand count
        """
        total_depth = n_alt + n_ref

        # Initialize result dictionary
        result = {
            'avds_score': 0.0,
            'vaf_raw': 0.0,
            'vaf_norm': 0.0,
            'q_alt_mean': 0.0,
            'depth_total': total_depth,
            'depth_norm': 0.0,
            'sb_strict': 0.0,
            'sb_min_ratio': 0.0,
            'sb_failed_25': True,
            'pb_score': 0.0,
            'pb_avg_pos': 0.0,
            'n_alt': n_alt,
            'n_ref': n_ref,
            'n_forward': 0,
            'n_reverse': 0
        }

        # Check minimum coverage
        if total_depth < self.config.min_coverage:
            logger.debug(f"Insufficient coverage: {total_depth} < {self.config.min_coverage}")
            return result

        if n_alt == 0:
            logger.debug("No alternative allele reads found")
            return result

        # Component 1: VAF*
        vaf_raw, vaf_norm = self.calculate_vaf_normalized(n_alt, n_ref)
        result['vaf_raw'] = vaf_raw
        result['vaf_norm'] = vaf_norm

        # Component 2: Q̄_alt (Average alternative allele quality)
        alt_reads = [r for r in read_infos if r.is_alt]
        if not alt_reads:
            logger.debug("No alternative allele read information available")
            return result

        read_qualities = [self.calculate_read_quality(r) for r in alt_reads]
        q_alt_mean = np.mean(read_qualities) if read_qualities else 0.0
        result['q_alt_mean'] = q_alt_mean

        # Component 3: D* (Normalized depth)
        depth_norm = self.calculate_depth_normalized(total_depth)
        result['depth_norm'] = depth_norm

        # Component 4: SB_strict (Strict strand bias)
        n_forward = sum(1 for r in alt_reads if r.is_forward)
        n_reverse = sum(1 for r in alt_reads if not r.is_forward)
        result['n_forward'] = n_forward
        result['n_reverse'] = n_reverse

        sb_strict, min_ratio, failed_25 = self.calculate_strand_bias_strict(
            n_forward, n_reverse, n_alt
        )
        result['sb_strict'] = sb_strict
        result['sb_min_ratio'] = min_ratio
        result['sb_failed_25'] = failed_25

        # Component 5: PB (Position bias)
        query_positions = [r.query_position for r in alt_reads]
        read_lengths = [r.read_length for r in alt_reads]
        pb_score, avg_pos = self.calculate_position_bias(query_positions, read_lengths)
        result['pb_score'] = pb_score
        result['pb_avg_pos'] = avg_pos

        # Final AVDS calculation using weighted geometric mean
        # Apply epsilon protection to prevent log(0)
        epsilon = self.config.epsilon

        vaf_protected = max(vaf_norm, epsilon)
        q_alt_protected = max(q_alt_mean, epsilon)
        depth_protected = max(depth_norm, epsilon)
        sb_protected = max(sb_strict, epsilon)
        pb_protected = max(pb_score, epsilon)

        # Logarithmic method (numerically stable)
        log_sum = (
            self.config.weight_vaf * np.log(vaf_protected) +
            self.config.weight_q_alt * np.log(q_alt_protected) +
            self.config.weight_depth * np.log(depth_protected) +
            self.config.weight_sb * np.log(sb_protected) +
            self.config.weight_pb * np.log(pb_protected)
        )

        avds_score = 100.0 * np.exp(log_sum)

        # Ensure score is in valid range [0, 100]
        avds_score = max(0.0, min(100.0, avds_score))

        result['avds_score'] = avds_score

        return result

    def interpret_score(self, avds_score: float) -> str:
        """
        Interpret AVDS score into quality category.

        Args:
            avds_score: AVDS score (0-100)

        Returns:
            Quality category string
        """
        if avds_score >= 90:
            return "EXCELLENT"
        elif avds_score >= 70:
            return "HIGH_CONFIDENCE"
        elif avds_score >= 50:
            return "MODERATE_CONFIDENCE"
        elif avds_score >= 30:
            return "LOW_CONFIDENCE"
        else:
            return "ARTIFACT"

    def get_recommended_action(self, avds_score: float, application: str = "research") -> str:
        """
        Get recommended action based on AVDS score and application.

        Args:
            avds_score: AVDS score (0-100)
            application: Application type ("clinical", "research", "screening")

        Returns:
            Recommended action string
        """
        thresholds = {
            "clinical": 80,
            "research": 60,
            "screening": 40
        }

        threshold = thresholds.get(application, 60)

        if avds_score >= threshold:
            return "PASS"
        elif avds_score >= threshold - 20:
            return "MANUAL_REVIEW"
        else:
            return "FILTER_OUT"
