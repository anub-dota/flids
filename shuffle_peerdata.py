import pandas as pd
import numpy as np
from pathlib import Path
import time

def shuffle_datapoints_file(input_file, output_file):
    """
    Read a datapoints file, shuffle its rows, and save to a new location
    """
    print(f"Processing {input_file.name}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Record original number of rows
    original_rows = len(df)
    
    # Shuffle the rows (this modifies df in-place)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save to the output location
    output_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_file, index=False)
    
    # Verify the file has the same number of rows
    shuffled_rows = len(pd.read_csv(output_file))
    
    print(f"  Shuffled {original_rows} rows -> {shuffled_rows} rows")
    print(f"  Saved to {output_file}")
    
    return original_rows, shuffled_rows

def main():
    """
    Process all datapoints files, shuffle them, and save to shuffled_data folder
    """
    start_time = time.time()
    
    # Define input and output directories
    input_dir = Path('data')
    output_dir = Path('shuffled_data')
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print("Error: 'data' directory not found. Please run generate_datapoints.py first.")
        return
    
    # Find all datapoints CSV files
    csv_files = list(input_dir.glob('*_datapts.csv'))
    
    if not csv_files:
        print("No datapoints CSV files found in the 'data' directory.")
        return
    
    print(f"Found {len(csv_files)} datapoints files.")
    print("Shuffling data...\n")
    
    # Process each file
    total_original_rows = 0
    total_shuffled_rows = 0
    
    for input_file in sorted(csv_files):
        # Create corresponding output file path with same name
        output_file = output_dir / input_file.name
        
        # Shuffle the file
        original_rows, shuffled_rows = shuffle_datapoints_file(input_file, output_file)
        
        # Update totals
        total_original_rows += original_rows
        total_shuffled_rows += shuffled_rows
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Processed {len(csv_files)} files")
    print(f"Total rows before shuffling: {total_original_rows}")
    print(f"Total rows after shuffling: {total_shuffled_rows}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
    
    # Validate that all data was preserved
    if total_original_rows == total_shuffled_rows:
        print("\nSUCCESS: All rows were preserved during shuffling.")
    else:
        print("\nWARNING: Row count mismatch. Some data may have been lost during shuffling.")

if __name__ == '__main__':
    main()
