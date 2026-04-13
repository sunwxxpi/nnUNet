import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Tuple
from tqdm import tqdm

# Label mapping table (old -> new)
LABEL_MAPPING: Dict[int, int] = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
    21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24, 27: 25, 28: 26,
    31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34,
    41: 35, 42: 36, 43: 37, 44: 38, 45: 39, 46: 40, 47: 41, 48: 42,
    103: 43, 104: 44, 105: 45,
    111: 46, 112: 47, 113: 48, 114: 49, 115: 50, 116: 51, 117: 52, 118: 53,
    121: 54, 122: 55, 123: 56, 124: 57, 125: 58, 126: 59, 127: 60, 128: 61,
    131: 62, 132: 63, 133: 64, 134: 65, 135: 66, 136: 67, 137: 68, 138: 69,
    141: 70, 142: 71, 143: 72, 144: 73, 145: 74, 146: 75, 147: 76, 148: 77
}

# --- Pre-computation for Optimization ---
# Create a lookup table for vectorized mapping to avoid re-computation in each process.
MAX_OLD_LABEL = max(LABEL_MAPPING.keys())
LOOKUP_TABLE = np.arange(MAX_OLD_LABEL + 1, dtype=np.int16)
for old_val, new_val in LABEL_MAPPING.items():
    LOOKUP_TABLE[old_val] = new_val

# --- Core Functions ---

def convert_single_file(label_file_path: Path) -> Tuple[str, str]:
    """
    Converts a single label file using a pre-computed lookup table.

    Args:
        label_file_path: Path to the NIfTI label file.

    Returns:
        A tuple containing the status ('success' or 'error') and a message.
    """
    try:
        nii = nib.load(label_file_path)
        original_data = nii.get_fdata().astype(np.int16)
        
        # Initialize new data array with zeros
        new_data = np.zeros_like(original_data, dtype=np.int16)
        
        # Create a mask for valid labels to apply the lookup table
        valid_mask = (original_data >= 0) & (original_data <= MAX_OLD_LABEL)
        
        # Apply the vectorized mapping
        new_data[valid_mask] = LOOKUP_TABLE[original_data[valid_mask]]
        
        # Create and save the new NIfTI file, overwriting the original
        new_nii = nib.Nifti1Image(new_data, nii.affine, nii.header)
        nib.save(new_nii, label_file_path)
        
        return "success", f"✓ Conversion successful: {label_file_path.name}"
        
    except Exception as e:
        return "error", f"✗ Conversion failed: {label_file_path.name} -> {e}"

def verify_single_file(label_file_path: Path) -> Dict[str, Any]:
    """
    Verifies that all label values in a single file are within the expected range [0, 77].

    Args:
        label_file_path: Path to the NIfTI label file.

    Returns:
        A dictionary containing verification results.
    """
    try:
        data = nib.load(label_file_path).get_fdata().astype(np.int16)
        unique_values = np.unique(data)
        
        # Check for any values outside the target range [0, 77]
        invalid_values = unique_values[(unique_values < 0) | (unique_values > 77)]
        is_valid = len(invalid_values) == 0
        
        return {
            'file_name': label_file_path.name,
            'is_valid': is_valid,
            'min': int(unique_values.min()),
            'max': int(unique_values.max()),
            'unique_count': len(unique_values),
            'unique_values': unique_values.tolist(),
            'invalid_values': invalid_values.tolist(),
            'error': None
        }
        
    except Exception as e:
        return {
            'file_name': label_file_path.name,
            'is_valid': False,
            'error': str(e)
        }

# --- Parallel Processing Wrappers ---

def run_parallel_conversion(labels_dir: Path) -> bool:
    """
    Converts all NIfTI files in a directory in parallel with a progress bar.

    Args:
        labels_dir: The directory containing label files.

    Returns:
        True if all conversions were successful, False otherwise.
    """
    label_files = sorted(list(labels_dir.glob('*.nii.gz')))
    num_files = len(label_files)
    
    if not num_files:
        print("No files found to convert.")
        return True
        
    print(f"Found {num_files} files to convert...")
    print(f"Using up to {cpu_count()} CPU cores for parallel processing.")
    
    results = []
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(convert_single_file, label_files), 
            total=num_files, 
            desc="Converting files"
        ))
    
    successes = [msg for status, msg in results if status == "success"]
    errors = [msg for status, msg in results if status == "error"]
    
    print("\n--- Conversion Summary ---")
    print(f"Successfully converted: {len(successes)} files")
    print(f"Failed to convert: {len(errors)} files")
    
    if errors:
        print("\n--- Failed Files ---")
        for error_msg in errors:
            print(error_msg)
            
    return not errors

def run_parallel_verification(labels_dir: Path) -> bool:
    """
    Verifies all NIfTI files in a directory in parallel with a progress bar.

    Args:
        labels_dir: The directory containing label files.

    Returns:
        True if all files are valid, False otherwise.
    """
    label_files = sorted(list(labels_dir.glob('*.nii.gz')))
    num_files = len(label_files)
    
    if not num_files:
        print("No files found to verify.")
        return True

    print(f"Found {num_files} files to verify...")
    print(f"Using up to {cpu_count()} CPU cores for parallel processing.")
    
    results = []
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(verify_single_file, label_files), 
            total=num_files, 
            desc="Verifying files"
        ))
        
    valid_files = [r for r in results if r['is_valid']]
    invalid_files = [r for r in results if not r['is_valid']]
    
    print("\n--- Verification Summary ---")
    print(f"Valid files: {len(valid_files)}")
    print(f"Invalid files: {len(invalid_files)}")

    # --- Aggregate Statistics ---
    if valid_files:
        all_unique_labels = set()
        for res in valid_files:
            all_unique_labels.update(res['unique_values'])
        
        sorted_labels = sorted(list(all_unique_labels))
        
        print("\n--- Overall Label Statistics (from valid files) ---")
        print(f"Global Min Label: {min(all_unique_labels)}")
        print(f"Global Max Label: {max(all_unique_labels)}")
        print(f"Total Unique Labels: {len(sorted_labels)}")
        print(f"Label Range: {sorted_labels[0]} to {sorted_labels[-1]}")
        
        expected_range = set(range(78)) # 0-77
        missing_labels = expected_range - all_unique_labels
        extra_labels = all_unique_labels - expected_range
        
        if missing_labels:
            print(f"Missing labels in expected range [0-77]: {sorted(list(missing_labels))}")
        if extra_labels:
            print(f"Labels found outside expected range [0-77]: {sorted(list(extra_labels))}")
        
        if not missing_labels and not extra_labels:
            print("✓ All labels are perfectly within the [0, 77] range.")
    
    # --- Detailed Report for Invalid Files ---
    if invalid_files:
        print("\n--- Invalid File Details ---")
        for res in invalid_files:
            if res['error']:
                print(f"{res['file_name']}: Processing Error -> {res['error']}")
            else:
                print(f"{res['file_name']}: Found invalid values -> {res['invalid_values']}")

    # --- Sample Report ---
    print("\n--- Label Distribution Samples (first 5 valid files) ---")
    for res in valid_files[:5]:
        print(f"{res['file_name']}: {res['min']}-{res['max']} (Total {res['unique_count']} classes)")
        
    return not invalid_files

# --- Main Execution ---

def main():
    """Main function to parse arguments and run the selected mode."""
    parser = argparse.ArgumentParser(
        description="An integrated script for NIfTI label conversion and verification.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode", 
        choices=["convert", "verify", "both"], 
        default="both",
        help="Execution mode:\n"
             "  convert: Perform label conversion only.\n"
             "  verify:  Perform label verification only.\n"
             "  both:    Perform conversion followed by verification (default)."
    )
    parser.add_argument(
        "--labels-dir", 
        type=Path, 
        required=True,
        help="Path to the directory containing the NIfTI label files (*.nii.gz)."
    )
    
    args = parser.parse_args()
    
    if not args.labels_dir.is_dir():
        print(f"❌ Error: The specified directory does not exist: {args.labels_dir}")
        return
        
    print(f"=== Starting Label Processing (Mode: {args.mode}) ===")
    print(f"Target Directory: {args.labels_dir}")
    
    conversion_ok = True
    verification_ok = True
    
    if args.mode in ["convert", "both"]:
        print("\n===== Stage 1: Label Conversion =====")
        conversion_ok = run_parallel_conversion(args.labels_dir)
    
    if args.mode in ["verify", "both"]:
        print("\n===== Stage 2: Label Verification =====")
        verification_ok = run_parallel_verification(args.labels_dir)
        
    # --- Final Conclusion ---
    print("\n\n===== Final Result =====")
    
    if args.mode in ["convert", "both"]:
        if conversion_ok:
            print("🎉 Label conversion completed successfully.")
        else:
            print("❌ Errors occurred during label conversion.")
    
    if args.mode in ["verify", "both"]:
        if verification_ok:
            print("🎉 All label files were successfully verified.")
        else:
            print("❌ Issues were found during label verification.")
            
    if conversion_ok and verification_ok:
        print("\n✨ All operations completed without any issues!")
    else:
        print("\n⚠️ Please review the logs above for details on the errors.")

if __name__ == "__main__":
    main()