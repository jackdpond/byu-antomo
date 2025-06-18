#!/usr/bin/env python3
"""
Standalone script to compute contrast limits for all tomograms in a CSV file.
Designed to run on a supercomputer overnight.

Usage:
    python compute_contrast_limits_batch.py [--csv-file path/to/tomo_ids.csv] [--output-file path/to/output.csv]

This script will:
1. Read the tomo_ids.csv file
2. Compute contrast limits for all tomograms that don't have them
3. Save the results back to the CSV file
4. Provide progress reporting and error handling
"""

import argparse
import os
import sys
import time
import json
import pandas as pd
import numpy as np
import mrcfile
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('contrast_limits_computation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def adaptive_contrast_limits(image, fraction=0.005, bins=256):
    """
    Compute adaptive contrast limits using histogram analysis.
    """
    hist, bin_edges = np.histogram(image, bins=bins)
    hist_smooth = gaussian_filter1d(hist, sigma=2)
    threshold = hist_smooth.max() * fraction
    nonzero = np.where(hist_smooth > threshold)[0]
    
    if len(nonzero) == 0:
        return np.min(image), np.max(image)
    
    low = bin_edges[nonzero[0]]
    high = bin_edges[nonzero[-1]+1]  # +1 because bin_edges is len(hist)+1
    return low, high

def compute_tomogram_contrast_limits(filepath, fraction=0.005, bins=256):
    """
    Compute global contrast limits for an entire tomogram.
    """
    logger.info(f"Computing contrast limits for: {filepath}")
    start_time = time.time()
    
    try:
        with mrcfile.mmap(filepath, permissive=True) as mrc:
            data = mrc.data  # This is a numpy memmap array
        
        logger.info(f"Tomogram shape: {data.shape}")
        
        # Compute histogram for the entire tomogram
        hist, bin_edges = np.histogram(data, bins=bins)
        hist_smooth = gaussian_filter1d(hist, sigma=2)
        threshold = hist_smooth.max() * fraction
        nonzero = np.where(hist_smooth > threshold)[0]
        
        if len(nonzero) == 0:
            logger.warning("No bins above threshold, using min/max.")
            min_val, max_val = np.min(data), np.max(data)
        else:
            min_val = bin_edges[nonzero[0]]
            max_val = bin_edges[nonzero[-1]+1]  # +1 because bin_edges is len(hist)+1
        
        end_time = time.time()
        logger.info(f"Computed in {end_time - start_time:.2f} seconds - Min: {min_val:.4f}, Max: {max_val:.4f}")
        
        return min_val, max_val
        
    except Exception as e:
        logger.error(f"Error computing contrast limits for {filepath}: {e}")
        raise

def ensure_csv_columns(csv_path):
    """
    Ensure the CSV has the required min and max columns.
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        needs_update = False
        
        # Check if min column exists
        if 'min' not in df.columns:
            df['min'] = ''
            needs_update = True
            logger.info(f"Added 'min' column to {csv_path}")
        
        # Check if max column exists
        if 'max' not in df.columns:
            df['max'] = ''
            needs_update = True
            logger.info(f"Added 'max' column to {csv_path}")
        
        # Save if we made changes
        if needs_update:
            df.to_csv(csv_path, index=False)
            logger.info(f"Updated {csv_path} with new columns")
        
        return True
    except Exception as e:
        logger.error(f"Error ensuring CSV columns: {e}")
        return False

def compute_all_missing_contrast_limits(csv_path, output_path=None):
    """
    Compute contrast limits for all tomograms in the CSV that don't have them yet.
    """
    logger.info(f"Starting batch contrast limits computation")
    logger.info(f"Input CSV: {csv_path}")
    if output_path:
        logger.info(f"Output CSV: {output_path}")
    
    # Ensure the CSV has the required columns
    if not ensure_csv_columns(csv_path):
        return False
    
    try:
        # Read the CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} tomograms")
        
        # Find rows that need computation
        missing_mask = df['min'].isna() | (df['min'] == '') | df['max'].isna() | (df['max'] == '')
        missing_indices = missing_mask.to_numpy().nonzero()[0]
        
        if len(missing_indices) == 0:
            logger.info("✅ All tomograms already have contrast limits computed")
            return True
        
        logger.info(f"🔄 Computing contrast limits for {len(missing_indices)} tomograms...")
        
        # Track progress and errors
        successful = 0
        failed = 0
        errors = []
        
        for i, idx in enumerate(missing_indices):
            file_path = df.loc[idx, 'file_path']
            tomo_id = df.loc[idx, 'tomo_id']
            
            logger.info(f"Processing {i+1}/{len(missing_indices)}: {tomo_id}")
            
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    errors.append(f"{tomo_id}: File not found - {file_path}")
                    failed += 1
                    continue
                
                min_val, max_val = compute_tomogram_contrast_limits(file_path)
                df.loc[idx, 'min'] = min_val
                df.loc[idx, 'max'] = max_val
                
                # Save after each computation in case of interruption
                save_path = output_path if output_path else csv_path
                df.to_csv(save_path, index=False)
                logger.info(f"✅ Saved contrast limits for {tomo_id}")
                successful += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {tomo_id}: {e}")
                errors.append(f"{tomo_id}: {str(e)}")
                failed += 1
        
        # Final summary
        logger.info(f"🎉 Batch computation completed!")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        
        if errors:
            logger.info("Errors encountered:")
            for error in errors:
                logger.info(f"  - {error}")
        
        # Save final results
        save_path = output_path if output_path else csv_path
        df.to_csv(save_path, index=False)
        logger.info(f"Final results saved to: {save_path}")
        
        return failed == 0
        
    except Exception as e:
        logger.error(f"Failed to compute contrast limits: {e}")
        return False

def main():
    """
    Main function to handle command line arguments and run the batch computation.
    """
    parser = argparse.ArgumentParser(
        description="Compute contrast limits for all tomograms in a CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compute_contrast_limits_batch.py
  python compute_contrast_limits_batch.py --csv-file /path/to/tomo_ids.csv
  python compute_contrast_limits_batch.py --csv-file input.csv --output-file output.csv
        """
    )
    
    parser.add_argument(
        '--csv-file',
        type=str,
        default='tomo_ids.csv',
        help='Path to the CSV file containing tomogram information (default: tomo_ids.csv)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Path to save the output CSV file (default: overwrite input file)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be computed without actually doing it'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.csv_file):
        logger.error(f"Input CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # Log start information
    logger.info("=" * 60)
    logger.info("BATCH CONTRAST LIMITS COMPUTATION")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Input file: {args.csv_file}")
    if args.output_file:
        logger.info(f"Output file: {args.output_file}")
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual computation will be performed")
    logger.info("=" * 60)
    
    if args.dry_run:
        # Just show what would be computed
        try:
            df = pd.read_csv(args.csv_file)
            ensure_csv_columns(args.csv_file)
            df = pd.read_csv(args.csv_file)  # Re-read after ensuring columns
            
            missing_mask = df['min'].isna() | (df['min'] == '') | df['max'].isna() | (df['max'] == '')
            missing_indices = missing_mask.to_numpy().nonzero()[0]
            
            logger.info(f"Would compute contrast limits for {len(missing_indices)} tomograms:")
            for idx in missing_indices:
                tomo_id = df.loc[idx, 'tomo_id']
                file_path = df.loc[idx, 'file_path']
                logger.info(f"  - {tomo_id}: {file_path}")
            
            if len(missing_indices) == 0:
                logger.info("All tomograms already have contrast limits computed!")
            
        except Exception as e:
            logger.error(f"Error in dry run: {e}")
            sys.exit(1)
    else:
        # Actually compute the contrast limits
        success = compute_all_missing_contrast_limits(args.csv_file, args.output_file)
        
        logger.info("=" * 60)
        logger.info(f"End time: {datetime.now()}")
        if success:
            logger.info("✅ Batch computation completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Batch computation completed with errors!")
            sys.exit(1)

if __name__ == "__main__":
    main() 