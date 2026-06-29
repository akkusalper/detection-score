"""
AVDS Pipeline for BAM/CRAM and VCF Processing
Version: 1.0
Date: February 2026

This module processes VCF files with BAM/CRAM alignments to calculate
AVDS scores for each variant, with parallel processing support.
"""

import pysam
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
from tqdm import tqdm
import pandas as pd

from .avds_calculator import AVDSCalculator, AVDSConfig, ReadInfo

logger = logging.getLogger(__name__)


class AVDSPipeline:
    """
    Pipeline for processing VCF and BAM/CRAM files to calculate AVDS scores.
    """

    def __init__(
        self,
        bam_path: str,
        reference_path: Optional[str] = None,
        config: Optional[AVDSConfig] = None,
        n_threads: Optional[int] = None
    ):
        """
        Initialize AVDS pipeline.

        Args:
            bam_path: Path to BAM or CRAM file
            reference_path: Path to reference FASTA (required for CRAM)
            config: AVDSConfig object
            n_threads: Number of parallel threads (default: CPU count - 1)
        """
        self.bam_path = Path(bam_path)
        self.reference_path = Path(reference_path) if reference_path else None
        self.config = config if config else AVDSConfig()
        self.calculator = AVDSCalculator(self.config)

        # Set number of threads
        if n_threads is None:
            self.n_threads = max(1, cpu_count() - 1)
        else:
            self.n_threads = max(1, n_threads)

        logger.info(f"AVDS Pipeline initialized")
        logger.info(f"BAM/CRAM: {self.bam_path}")
        logger.info(f"Reference: {self.reference_path}")
        logger.info(f"Threads: {self.n_threads}")

        # Validate files
        self._validate_files()

    def _validate_files(self):
        """Validate input files exist and are accessible"""
        if not self.bam_path.exists():
            raise FileNotFoundError(f"BAM/CRAM file not found: {self.bam_path}")

        # Check if CRAM requires reference
        if self.bam_path.suffix.lower() == '.cram':
            if not self.reference_path:
                raise ValueError("Reference FASTA required for CRAM files")
            if not self.reference_path.exists():
                raise FileNotFoundError(f"Reference file not found: {self.reference_path}")

        # Check for index file
        index_extensions = ['.bai', '.crai']
        index_exists = any(
            Path(str(self.bam_path) + ext).exists()
            for ext in index_extensions
        )
        if not index_exists:
            logger.warning(f"Index file not found for {self.bam_path}. This may cause slow performance.")

    def extract_read_info(
        self,
        read: pysam.AlignedSegment,
        chrom: str,
        pos: int,
        ref: str,
        alt: str
    ) -> Optional[ReadInfo]:
        """
        Extract ReadInfo from a pysam AlignedSegment.

        Args:
            read: pysam AlignedSegment object
            chrom: Chromosome name
            pos: 1-based variant position
            ref: Reference allele
            alt: Alternative allele

        Returns:
            ReadInfo object or None if read should be filtered
        """
        try:
            # Filter low quality reads
            if read.mapping_quality < self.config.min_mapping_quality:
                return None

            if read.is_unmapped or read.is_duplicate or read.is_qcfail:
                return None

            # Get 0-based position
            pos_0based = pos - 1

            # Check if variant position is covered by this read
            if not (read.reference_start <= pos_0based < read.reference_end):
                return None

            # Get query position (position in read sequence)
            # This handles insertions, deletions, and soft clips
            query_pos = None
            ref_pos = read.reference_start
            query_idx = 0

            for op, length in read.cigartuples:
                if op == 0:  # M - match/mismatch
                    if ref_pos <= pos_0based < ref_pos + length:
                        query_pos = query_idx + (pos_0based - ref_pos)
                        break
                    ref_pos += length
                    query_idx += length
                elif op == 1:  # I - insertion
                    query_idx += length
                elif op == 2:  # D - deletion
                    ref_pos += length
                elif op == 4:  # S - soft clip
                    query_idx += length
                elif op == 5:  # H - hard clip
                    pass  # No change to indices
                elif op == 7 or op == 8:  # = or X
                    if ref_pos <= pos_0based < ref_pos + length:
                        query_pos = query_idx + (pos_0based - ref_pos)
                        break
                    ref_pos += length
                    query_idx += length

            if query_pos is None:
                return None

            # Get base at variant position
            try:
                base = read.query_sequence[query_pos]
                base_quality = read.query_qualities[query_pos]
            except (IndexError, TypeError):
                return None

            # Filter low base quality
            if base_quality < self.config.min_base_quality:
                return None

            # Determine if read supports alt or ref
            is_alt = (base.upper() == alt.upper())
            is_ref = (base.upper() == ref.upper())

            if not (is_alt or is_ref):
                # Base doesn't match ref or alt (could be N or other variant)
                return None

            # Extract quality metrics
            mapq = read.mapping_quality

            # Get alignment score (AS tag)
            alignment_score = read.get_tag('AS') if read.has_tag('AS') else 0

            # Get edit distance (NM tag)
            edit_distance = read.get_tag('NM') if read.has_tag('NM') else 0

            # Read length
            read_length = read.query_length if read.query_length else 0

            # Strand (is_reverse = True means reverse strand)
            is_forward = not read.is_reverse

            # Calculate CIGAR score
            cigar_score = self.calculator.calculate_cigar_score(read.cigartuples)

            return ReadInfo(
                is_alt=is_alt,
                is_forward=is_forward,
                mapq=mapq,
                base_quality=base_quality,
                alignment_score=alignment_score,
                edit_distance=edit_distance,
                read_length=read_length,
                query_position=query_pos,
                cigar_score=cigar_score
            )

        except Exception as e:
            logger.debug(f"Error extracting read info: {e}")
            return None

    def process_variant(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        variant_id: str = "."
    ) -> Dict:
        """
        Process a single variant and calculate AVDS score.

        Args:
            chrom: Chromosome name
            pos: 1-based variant position
            ref: Reference allele
            alt: Alternative allele
            variant_id: Variant identifier

        Returns:
            Dictionary with variant info and AVDS results
        """
        result = {
            'chrom': chrom,
            'pos': pos,
            'id': variant_id,
            'ref': ref,
            'alt': alt,
            'error': None
        }

        try:
            # Open BAM/CRAM file
            if self.reference_path:
                samfile = pysam.AlignmentFile(
                    str(self.bam_path),
                    reference_filename=str(self.reference_path)
                )
            else:
                samfile = pysam.AlignmentFile(str(self.bam_path))

            # Fetch reads at variant position
            read_infos = []
            n_alt = 0
            n_ref = 0

            try:
                for read in samfile.fetch(chrom, pos - 1, pos):
                    read_info = self.extract_read_info(read, chrom, pos, ref, alt)
                    if read_info:
                        read_infos.append(read_info)
                        if read_info.is_alt:
                            n_alt += 1
                        else:
                            n_ref += 1
            except Exception as e:
                logger.warning(f"Error fetching reads at {chrom}:{pos}: {e}")
                result['error'] = str(e)
                return result
            finally:
                samfile.close()

            # Calculate AVDS
            avds_result = self.calculator.calculate_avds(read_infos, n_alt, n_ref)

            # Add interpretation
            avds_result['quality_category'] = self.calculator.interpret_score(avds_result['avds_score'])
            avds_result['recommended_action'] = self.calculator.get_recommended_action(avds_result['avds_score'])

            # Merge with variant info
            result.update(avds_result)

            return result

        except Exception as e:
            logger.error(f"Error processing variant {chrom}:{pos} {ref}>{alt}: {e}")
            result['error'] = str(e)
            return result

    def write_annotated_vcf(
        self,
        input_vcf_path: str,
        output_vcf_path: str,
        results_df: pd.DataFrame
    ):
        """
        Write VCF file annotated with AVDS scores.

        Args:
            input_vcf_path: Path to input VCF file
            output_vcf_path: Path to output VCF file
            results_df: DataFrame with AVDS results
        """
        logger.info(f"Writing annotated VCF: {output_vcf_path}")

        # Read input VCF
        vcf_in = pysam.VariantFile(str(input_vcf_path))

        # Add AVDS INFO fields to header using add_line method
        vcf_in.header.add_line('##INFO=<ID=AVDS_SCORE,Number=1,Type=Float,Description="AVDS quality score (0-100)">')
        vcf_in.header.add_line('##INFO=<ID=AVDS_QUALITY,Number=1,Type=String,Description="AVDS quality category (EXCELLENT, HIGH_CONFIDENCE, MODERATE_CONFIDENCE, LOW_CONFIDENCE, ARTIFACT)">')
        vcf_in.header.add_line('##INFO=<ID=AVDS_ACTION,Number=1,Type=String,Description="AVDS recommended action (PASS, MANUAL_REVIEW, FILTER_OUT)">')
        vcf_in.header.add_line('##INFO=<ID=AVDS_VAF,Number=1,Type=Float,Description="AVDS calculated variant allele frequency">')
        vcf_in.header.add_line('##INFO=<ID=AVDS_DEPTH,Number=1,Type=Integer,Description="AVDS total depth (alt + ref reads)">')
        vcf_in.header.add_line('##INFO=<ID=AVDS_SB_FAILED,Number=1,Type=String,Description="Failed AVDS strand bias 25% rule (True/False)">')

        # Open output VCF
        vcf_out = pysam.VariantFile(str(output_vcf_path), 'w', header=vcf_in.header)

        # Create lookup dictionary for quick access
        results_lookup = {}
        for _, row in results_df.iterrows():
            key = (row['chrom'], row['pos'], row['ref'], row['alt'])
            results_lookup[key] = row

        # Process each variant
        for record in vcf_in:
            # Handle multi-allelic sites
            for alt in record.alts:
                if alt is None:
                    continue

                key = (record.chrom, record.pos, record.ref, alt)

                if key in results_lookup:
                    result = results_lookup[key]

                    # Add AVDS INFO fields
                    record.info['AVDS_SCORE'] = float(result['avds_score'])
                    record.info['AVDS_QUALITY'] = str(result['quality_category'])
                    record.info['AVDS_ACTION'] = str(result['recommended_action'])
                    record.info['AVDS_VAF'] = float(result['vaf_raw'])
                    record.info['AVDS_DEPTH'] = int(result['depth_total'])
                    record.info['AVDS_SB_FAILED'] = str(result['sb_failed_25'])

            # Write record
            vcf_out.write(record)

        vcf_in.close()
        vcf_out.close()

        # Compress and index
        pysam.tabix_index(str(output_vcf_path), preset='vcf', force=True)
        logger.info(f"Created annotated VCF: {output_vcf_path}")
        logger.info(f"Created VCF index: {output_vcf_path}.tbi")

    def process_vcf(
        self,
        vcf_path: str,
        output_path: str,
        sample_name: Optional[str] = None,
        parallel: bool = True,
        write_vcf: bool = True
    ) -> pd.DataFrame:
        """
        Process entire VCF file and calculate AVDS for all variants.

        Args:
            vcf_path: Path to input VCF file
            output_path: Path to output TSV file
            sample_name: Sample name to process (default: first sample)
            parallel: Use parallel processing (default: True)
            write_vcf: Write annotated VCF output (default: True)

        Returns:
            DataFrame with results
        """
        vcf_path = Path(vcf_path)
        if not vcf_path.exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")

        logger.info(f"Processing VCF: {vcf_path}")

        # Read VCF and extract variants
        variants = []
        try:
            vcf = pysam.VariantFile(str(vcf_path))

            # Get sample name
            if sample_name is None:
                samples = list(vcf.header.samples)
                if not samples:
                    raise ValueError("No samples found in VCF")
                sample_name = samples[0]
                logger.info(f"Using sample: {sample_name}")

            # Extract variants
            for record in vcf:
                # Handle multi-allelic sites
                for alt in record.alts:
                    if alt is None:
                        continue

                    variant = {
                        'chrom': record.chrom,
                        'pos': record.pos,
                        'id': record.id if record.id else '.',
                        'ref': record.ref,
                        'alt': alt
                    }
                    variants.append(variant)

            vcf.close()

        except Exception as e:
            logger.error(f"Error reading VCF: {e}")
            raise

        logger.info(f"Found {len(variants)} variants to process")

        if not variants:
            logger.warning("No variants found in VCF")
            return pd.DataFrame()

        # Process variants
        results = []

        if parallel and self.n_threads > 1:
            logger.info(f"Processing variants in parallel with {self.n_threads} threads")

            # Create partial function with self bound
            process_func = partial(
                self._process_variant_wrapper,
                bam_path=str(self.bam_path),
                reference_path=str(self.reference_path) if self.reference_path else None,
                config=self.config
            )

            # Use multiprocessing pool
            with Pool(processes=self.n_threads) as pool:
                results = list(tqdm(
                    pool.imap(process_func, variants),
                    total=len(variants),
                    desc="Processing variants"
                ))
        else:
            logger.info("Processing variants sequentially")
            for variant in tqdm(variants, desc="Processing variants"):
                result = self.process_variant(
                    variant['chrom'],
                    variant['pos'],
                    variant['ref'],
                    variant['alt'],
                    variant['id']
                )
                results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save TSV file
        output_path = Path(output_path)
        df.to_csv(output_path, sep='\t', index=False, float_format='%.4f')
        logger.info(f"Results saved to: {output_path}")

        # Write annotated VCF if requested
        if write_vcf:
            # Generate VCF output path (replace .tsv with .vcf.gz or add .vcf.gz)
            if output_path.suffix == '.tsv':
                vcf_output_path = output_path.with_suffix('.vcf.gz')
            else:
                vcf_output_path = Path(str(output_path) + '.vcf.gz')

            try:
                self.write_annotated_vcf(vcf_path, vcf_output_path, df)
            except Exception as e:
                logger.warning(f"Failed to write annotated VCF: {e}")

        # Print summary statistics
        self._print_summary(df)

        return df

    @staticmethod
    def _process_variant_wrapper(variant: Dict, bam_path: str, reference_path: Optional[str], config: AVDSConfig) -> Dict:
        """
        Wrapper function for parallel processing.
        Creates a new pipeline instance for each worker.

        Args:
            variant: Variant dictionary
            bam_path: Path to BAM/CRAM
            reference_path: Path to reference
            config: AVDS configuration

        Returns:
            Result dictionary
        """
        pipeline = AVDSPipeline(bam_path, reference_path, config, n_threads=1)
        return pipeline.process_variant(
            variant['chrom'],
            variant['pos'],
            variant['ref'],
            variant['alt'],
            variant['id']
        )

    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics of AVDS scores"""
        logger.info("\n" + "="*80)
        logger.info("AVDS SCORE SUMMARY")
        logger.info("="*80)

        if 'avds_score' in df.columns:
            scores = df['avds_score']
            logger.info(f"Total variants: {len(df)}")
            logger.info(f"Mean AVDS: {scores.mean():.2f}")
            logger.info(f"Median AVDS: {scores.median():.2f}")
            logger.info(f"Std Dev: {scores.std():.2f}")
            logger.info(f"Min: {scores.min():.2f}")
            logger.info(f"Max: {scores.max():.2f}")

            # Category counts
            if 'quality_category' in df.columns:
                logger.info("\nQuality Categories:")
                category_counts = df['quality_category'].value_counts()
                for category, count in category_counts.items():
                    pct = 100 * count / len(df)
                    logger.info(f"  {category}: {count} ({pct:.1f}%)")

            # Recommended actions
            if 'recommended_action' in df.columns:
                logger.info("\nRecommended Actions (Research threshold):")
                action_counts = df['recommended_action'].value_counts()
                for action, count in action_counts.items():
                    pct = 100 * count / len(df)
                    logger.info(f"  {action}: {count} ({pct:.1f}%)")

            # Strand bias failures
            if 'sb_failed_25' in df.columns:
                failed_25 = df['sb_failed_25'].sum()
                pct = 100 * failed_25 / len(df)
                logger.info(f"\nStrand Bias 25% Rule Failures: {failed_25} ({pct:.1f}%)")

        # Errors
        if 'error' in df.columns:
            errors = df['error'].notna().sum()
            if errors > 0:
                logger.warning(f"\nVariants with errors: {errors}")

        logger.info("="*80 + "\n")


def main():
    """Main entry point for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description='AVDS Calculator - Allele-Specific Variant Detection Score',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process VCF with BAM file
  python avds_pipeline.py -v variants.vcf -b alignment.bam -o results.tsv

  # Process with CRAM (requires reference)
  python avds_pipeline.py -v variants.vcf -b alignment.cram -r reference.fa -o results.tsv

  # Use custom VAF threshold for low-frequency variants
  python avds_pipeline.py -v variants.vcf -b alignment.bam -o results.tsv --theta-vaf 0.10

  # Sequential processing (no parallelization)
  python avds_pipeline.py -v variants.vcf -b alignment.bam -o results.tsv --no-parallel

  # Use specific number of threads
  python avds_pipeline.py -v variants.vcf -b alignment.bam -o results.tsv -t 8
        """
    )

    parser.add_argument('-v', '--vcf', required=True, help='Input VCF file')
    parser.add_argument('-b', '--bam', required=True, help='Input BAM or CRAM file')
    parser.add_argument('-r', '--reference', help='Reference FASTA (required for CRAM)')
    parser.add_argument('-o', '--output', required=True, help='Output TSV file')
    parser.add_argument('-s', '--sample', help='Sample name (default: first sample in VCF)')
    parser.add_argument('-t', '--threads', type=int, help='Number of threads (default: CPU count - 1)')
    parser.add_argument('--theta-vaf', type=float, default=0.50,
                        help='VAF normalization threshold (default: 0.50)')
    parser.add_argument('--theta-depth', type=int, default=100,
                        help='Depth normalization threshold (default: 100)')
    parser.add_argument('--min-coverage', type=int, default=5,
                        help='Minimum coverage to attempt calculation (default: 5)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    parser.add_argument('--no-vcf', action='store_true',
                        help='Disable VCF output (TSV only)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Create configuration
    config = AVDSConfig(
        theta_vaf=args.theta_vaf,
        theta_depth=args.theta_depth,
        min_coverage=args.min_coverage
    )

    # Create pipeline
    pipeline = AVDSPipeline(
        bam_path=args.bam,
        reference_path=args.reference,
        config=config,
        n_threads=args.threads
    )

    # Process VCF
    try:
        pipeline.process_vcf(
            vcf_path=args.vcf,
            output_path=args.output,
            sample_name=args.sample,
            parallel=not args.no_parallel,
            write_vcf=not args.no_vcf
        )
        logger.info("Processing completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
