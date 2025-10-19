import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def analyze_datapoints_file(file_path):
    """Analyze a single datapoints CSV file and return label statistics"""
    print(f"Analyzing {file_path.name}...")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Count labels
    label_counts = df['label'].value_counts().to_dict()
    total_points = len(df)
    
    # Get counts (defaulting to 0 if not present)
    normal_count = label_counts.get(0, 0)
    attack_count = label_counts.get(1, 0)
    
    # Calculate percentages
    normal_pct = (normal_count / total_points) * 100
    attack_pct = (attack_count / total_points) * 100
    
    print(f"  Total datapoints: {total_points}")
    print(f"  Normal (0): {normal_count} ({normal_pct:.2f}%)")
    print(f"  Attack (1): {attack_count} ({attack_pct:.2f}%)")
    print()
    
    return {
        'file': file_path.stem,
        'total': total_points,
        'normal': normal_count,
        'normal_pct': normal_pct,
        'attack': attack_count,
        'attack_pct': attack_pct
    }

def plot_label_distribution(stats):
    """Create visualizations for label distribution"""
    device_names = [s['file'] for s in stats]
    normal_pcts = [s['normal_pct'] for s in stats]
    attack_pcts = [s['attack_pct'] for s in stats]
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Create a bar chart
    x = np.arange(len(device_names))
    width = 0.35
    
    plt.bar(x - width/2, normal_pcts, width, label='Normal (0)')
    plt.bar(x + width/2, attack_pcts, width, label='Attack (1)')
    
    plt.xlabel('Device')
    plt.ylabel('Percentage')
    plt.title('Label Distribution by Device')
    plt.xticks(x, device_names, rotation=45, ha='right')
    plt.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('label_distribution.png')
    print(f"Plot saved as label_distribution.png")

def main():
    """Analyze all datapoints CSV files in the data directory"""
    data_dir = Path('data')
    
    if not data_dir.exists():
        print("Error: 'data' directory not found. Please run generate_datapoints.py first.")
        return
    
    # Find all CSV files
    csv_files = list(data_dir.glob('*_datapts.csv'))
    
    if not csv_files:
        print("No datapoints CSV files found in the 'data' directory.")
        return
    
    print(f"Found {len(csv_files)} datapoints files.")
    print("Analyzing label distributions...\n")
    
    # Analyze each file
    stats = []
    
    for file_path in sorted(csv_files):
        file_stats = analyze_datapoints_file(file_path)
        stats.append(file_stats)
    
    # Print summary table
    print("\n=== Summary Table ===")
    print(f"{'Device':<15} {'Total':<10} {'Normal':<12} {'Attack':<12} {'Normal %':<10} {'Attack %':<10}")
    print("-" * 80)
    
    for s in stats:
        print(f"{s['file']:<15} {s['total']:<10} {s['normal']:<12} {s['attack']:<12} {s['normal_pct']:<10.2f} {s['attack_pct']:<10.2f}")
    
    # Overall statistics
    total_normal = sum(s['normal'] for s in stats)
    total_attack = sum(s['attack'] for s in stats)
    total_all = total_normal + total_attack
    overall_normal_pct = (total_normal / total_all) * 100
    overall_attack_pct = (total_attack / total_all) * 100
    
    print("-" * 80)
    print(f"{'OVERALL':<15} {total_all:<10} {total_normal:<12} {total_attack:<12} {overall_normal_pct:<10.2f} {overall_attack_pct:<10.2f}")
    
    # Create visualizations
    try:
        plot_label_distribution(stats)
    except Exception as e:
        print(f"Warning: Could not create visualization. Error: {e}")
        print("Matplotlib may not be installed or configured correctly.")

if __name__ == '__main__':
    main()
