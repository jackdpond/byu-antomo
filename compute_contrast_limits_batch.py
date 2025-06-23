#!/usr/bin/env python3
"""
GPU-accelerated script to compute contrast limits for all tomograms in a CSV file.
Designed to run on a supercomputer with GPU overnight.

Usage:
    python compute_contrast_limits_batch.py [--csv-file path/to/tomo_ids.csv] [--output-file path/to/output.csv]

This script will:
1. Read the tomo_ids.csv file
2. Compute contrast limits for all tomograms using GPU acceleration
3. Save the results back to the CSV file
4. Provide progress reporting and error handling

GPU acceleration provides significant speedup for large tomograms.
"""

import argparse
import os
import sys
import time
import json
import pandas as pd
import numpy as np
import mrcfile
from pathlib import Path
import logging
from datetime import datetime

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ GPU acceleration available with CuPy")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not available, falling back to CPU")
    from scipy.ndimage import gaussian_filter1d

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('contrast_limits_computation_gpu.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def validate_mrc_file(file_path):
    try:
        with mrcfile.open(file_path, permissive=True) as mrc:
            _ = mrc.data.shape  # Try to access data to ensure it's valid
        return True, ""
    except Exception as e:
        return False, str(e)

def adaptive_contrast_limits_gpu(image, fraction=0.005, bins=256):
    """
    Compute adaptive contrast limits using GPU-accelerated histogram analysis.
    """
    if GPU_AVAILABLE:
        # Move data to GPU
        image_gpu = cp.asarray(image)
        
        # Compute histogram on GPU
        hist_gpu, bin_edges_gpu = cp.histogram(image_gpu, bins=bins)
        
        # Apply Gaussian smoothing on GPU
        hist_smooth_gpu = cp.convolve(hist_gpu, cp.exp(-cp.arange(-5, 6)**2 / 8), mode='same')
        
        # Find threshold and nonzero indices on GPU
        threshold = hist_smooth_gpu.max() * fraction
        nonzero_gpu = cp.where(hist_smooth_gpu > threshold)[0]
        
        # Move results back to CPU
        hist_smooth = cp.asnumpy(hist_smooth_gpu)
        bin_edges = cp.asnumpy(bin_edges_gpu)
        nonzero = cp.asnumpy(nonzero_gpu)
        
    else:
        # Fallback to CPU
        hist, bin_edges = np.histogram(image, bins=bins)
        hist_smooth = gaussian_filter1d(hist, sigma=2)
        threshold = hist_smooth.max() * fraction
        nonzero = np.where(hist_smooth > threshold)[0]
    
    if len(nonzero) == 0:
        return np.min(image), np.max(image)
    
    low = bin_edges[nonzero[0]]
    high = bin_edges[nonzero[-1]+1]  # +1 because bin_edges is len(hist)+1
    return low, high

def compute_tomogram_contrast_limits_gpu(filepath, fraction=0.005, bins=256):
    """
    Compute global contrast limits for an entire tomogram using GPU acceleration.
    """
    logger.info(f"Computing contrast limits for: {filepath}")
    if GPU_AVAILABLE:
        logger.info("Using GPU acceleration")
    else:
        logger.info("Using CPU fallback")
    
    start_time = time.time()
    
    try:
        with mrcfile.mmap(filepath, permissive=True) as mrc:
            data = mrc.data  # This is a numpy memmap array
        
        logger.info(f"Tomogram shape: {data.shape}")
        
        # Compute contrast limits using GPU or CPU
        min_val, max_val = adaptive_contrast_limits_gpu(data, fraction, bins)
        
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

def compute_all_missing_contrast_limits_gpu(csv_path, output_path=None):
    """
    Compute contrast limits for all tomograms in the CSV using GPU acceleration.
    """
    logger.info(f"Starting GPU-accelerated batch contrast limits computation")
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
                
                min_val, max_val = compute_tomogram_contrast_limits_gpu(file_path)
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
        logger.info(f"🎉 GPU-accelerated batch computation completed!")
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
    Main function to handle command line arguments and run the GPU-accelerated batch computation.
    """
    parser = argparse.ArgumentParser(
        description="Compute contrast limits for all tomograms in a CSV file using GPU acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compute_contrast_limits_batch_gpu.py
  python compute_contrast_limits_batch_gpu.py --csv-file /path/to/tomo_ids.csv
  python compute_contrast_limits_batch_gpu.py --csv-file input.csv --output-file output.csv
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
        default='tomo_ids.csv',
        help='Path to save the output CSV file (default: overwrite input file)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be computed without actually doing it'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate MRC files without computing contrast limits'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.csv_file):
        logger.error(f"Input CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # Log start information
    logger.info("=" * 60)
    logger.info("GPU-ACCELERATED BATCH CONTRAST LIMITS COMPUTATION")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Input file: {args.csv_file}")
    if args.output_file:
        logger.info(f"Output file: {args.output_file}")
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual computation will be performed")
    if args.validate_only:
        logger.info("VALIDATE ONLY MODE - Only checking file validity")
    logger.info("=" * 60)
    
    if args.validate_only:
        # Only validate files
        try:
            df = pd.read_csv(args.csv_file)
            logger.info(f"Validating {len(df)} tomogram files...")
            
            valid_count = 0
            invalid_count = 0
            invalid_files = []
            
            for idx, row in df.iterrows():
                file_path = row['file_path']
                tomo_id = row['tomo_id']
                
                logger.info(f"Validating {idx+1}/{len(df)}: {tomo_id}")
                is_valid, error_message = validate_mrc_file(file_path)
                
                if is_valid:
                    logger.info(f"✅ {tomo_id}: Valid")
                    valid_count += 1
                else:
                    logger.error(f"❌ {tomo_id}: {error_message}")
                    invalid_count += 1
                    invalid_files.append((tomo_id, file_path, error_message))
            
            logger.info("=" * 60)
            logger.info("VALIDATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"✅ Valid files: {valid_count}")
            logger.info(f"❌ Invalid files: {invalid_count}")
            
            if invalid_files:
                logger.info("Invalid files:")
                for tomo_id, file_path, error in invalid_files:
                    logger.info(f"  - {tomo_id}: {error}")
                    logger.info(f"    Path: {file_path}")
            
            sys.exit(0 if invalid_count == 0 else 1)
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            sys.exit(1)
    
    elif args.dry_run:
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
        success = compute_all_missing_contrast_limits_gpu(args.csv_file, args.output_file)
        
        logger.info("=" * 60)
        logger.info(f"End time: {datetime.now()}")
        if success:
            logger.info("✅ GPU-accelerated batch computation completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ GPU-accelerated batch computation completed with errors!")
            sys.exit(1)

if __name__ == "__main__":
    main() 