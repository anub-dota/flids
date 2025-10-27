"""
Training Results Visualization
This script creates comprehensive plots for federated learning training results
including loss, accuracy, precision, recall, and F1 scores
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_comprehensive_training_results(history_file='training_results.csv'):
    """Create comprehensive training visualization from saved results"""
    
    print("=" * 80)
    print("VISUALIZING TRAINING RESULTS")
    print("=" * 80)
    
    # Check if training results exist
    if not Path(history_file).exists():
        print(f"Error: {history_file} not found!")
        print("Please run the training first: python main.py")
        return
    
    # Load training history
    history = pd.read_csv(history_file)
    print(f"\n✓ Loaded training history: {len(history)} rounds")
    
    # Create output directory
    output_dir = Path('training_plots')
    output_dir.mkdir(exist_ok=True)
    
    # ==================== MAIN COMPREHENSIVE PLOT ====================
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Local Model Accuracy Over Time
    plt.subplot(4, 4, 1)
    plt.plot(history['round'], history['lightweight_accuracy'], 
             label='Lightweight (Local)', marker='o', linewidth=2, markersize=4)
    plt.plot(history['round'], history['heavyweight_accuracy'], 
             label='Heavyweight (Local)', marker='s', linewidth=2, markersize=4)
    plt.xlabel('Training Round', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.title('Local Model Accuracy', fontsize=12, weight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Plot 2: Global Model Accuracy Over Time
    plt.subplot(4, 4, 2)
    plt.plot(history['round'], history['global_lightweight_accuracy'], 
             label='Lightweight (Global)', marker='o', linewidth=2, markersize=4, color='#FF6B6B')
    plt.plot(history['round'], history['global_heavyweight_accuracy'], 
             label='Heavyweight (Global)', marker='s', linewidth=2, markersize=4, color='#4ECDC4')
    plt.xlabel('Training Round', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.title('Global Model Accuracy', fontsize=12, weight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Plot 3: Precision Over Time
    plt.subplot(4, 4, 3)
    plt.plot(history['round'], history['lightweight_precision'], 
             label='Lightweight', marker='o', linewidth=2, markersize=4, color='green')
    plt.plot(history['round'], history['heavyweight_precision'], 
             label='Heavyweight', marker='s', linewidth=2, markersize=4, color='purple')
    plt.xlabel('Training Round', fontsize=10)
    plt.ylabel('Precision', fontsize=10)
    plt.title('Model Precision', fontsize=12, weight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Plot 4: Recall Over Time
    plt.subplot(4, 4, 4)
    plt.plot(history['round'], history['lightweight_recall'], 
             label='Lightweight', marker='o', linewidth=2, markersize=4, color='orange')
    plt.plot(history['round'], history['heavyweight_recall'], 
             label='Heavyweight', marker='s', linewidth=2, markersize=4, color='brown')
    plt.xlabel('Training Round', fontsize=10)
    plt.ylabel('Recall', fontsize=10)
    plt.title('Model Recall', fontsize=12, weight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Plot 5: F1 Score Over Time
    plt.subplot(4, 4, 5)
    plt.plot(history['round'], history['lightweight_f1'], 
             label='Lightweight', marker='o', linewidth=2, markersize=4, color='teal')
    plt.plot(history['round'], history['heavyweight_f1'], 
             label='Heavyweight', marker='s', linewidth=2, markersize=4, color='navy')
    plt.xlabel('Training Round', fontsize=10)
    plt.ylabel('F1 Score', fontsize=10)
    plt.title('Model F1 Score', fontsize=12, weight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Plot 6: Implied Loss for Lightweight Models (1 - accuracy as proxy)
    plt.subplot(4, 4, 6)
    light_loss = 1 - history['lightweight_accuracy']
    light_global_loss = 1 - history['global_lightweight_accuracy']
    plt.plot(history['round'], light_loss, 
             label='Local Loss', marker='o', linewidth=2, markersize=4, color='red')
    plt.plot(history['round'], light_global_loss, 
             label='Global Loss', marker='s', linewidth=2, markersize=4, color='darkred')
    plt.xlabel('Training Round', fontsize=10)
    plt.ylabel('Loss (1 - Accuracy)', fontsize=10)
    plt.title('Lightweight Model Loss Proxy', fontsize=12, weight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Plot 7: Implied Loss for Heavyweight Models
    plt.subplot(4, 4, 7)
    heavy_loss = 1 - history['heavyweight_accuracy']
    heavy_global_loss = 1 - history['global_heavyweight_accuracy']
    plt.plot(history['round'], heavy_loss, 
             label='Local Loss', marker='o', linewidth=2, markersize=4, color='red')
    plt.plot(history['round'], heavy_global_loss, 
             label='Global Loss', marker='s', linewidth=2, markersize=4, color='darkred')
    plt.xlabel('Training Round', fontsize=10)
    plt.ylabel('Loss (1 - Accuracy)', fontsize=10)
    plt.title('Heavyweight Model Loss Proxy', fontsize=12, weight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Plot 8: Knowledge Transfer Effectiveness
    plt.subplot(4, 4, 8)
    plt.plot(history['round'], history['transfer_accuracy_light_to_heavy'], 
             label='Light → Heavy', marker='o', linewidth=2, markersize=4, color='#FF6B6B')
    plt.plot(history['round'], history['transfer_accuracy_heavy_to_light'], 
             label='Heavy → Light', marker='s', linewidth=2, markersize=4, color='#4ECDC4')
    plt.xlabel('Training Round', fontsize=10)
    plt.ylabel('Transfer Agreement', fontsize=10)
    plt.title('Knowledge Transfer Agreement', fontsize=12, weight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Plot 9: Training Samples Per Round
    plt.subplot(4, 4, 9)
    plt.plot(history['round'], history['num_lightweight_samples'], 
             label='Lightweight', marker='o', linewidth=2, markersize=4, color='skyblue')
    plt.plot(history['round'], history['num_heavyweight_samples'], 
             label='Heavyweight', marker='s', linewidth=2, markersize=4, color='steelblue')
    plt.xlabel('Training Round', fontsize=10)
    plt.ylabel('Number of Samples', fontsize=10)
    plt.title('Training Samples per Round', fontsize=12, weight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot 10: All Accuracies Comparison
    plt.subplot(4, 4, 10)
    plt.plot(history['round'], history['lightweight_accuracy'], 
             label='Local Light', marker='o', linewidth=1.5, markersize=3, alpha=0.8)
    plt.plot(history['round'], history['heavyweight_accuracy'], 
             label='Local Heavy', marker='s', linewidth=1.5, markersize=3, alpha=0.8)
    plt.plot(history['round'], history['global_lightweight_accuracy'], 
             label='Global Light', marker='^', linewidth=1.5, markersize=3, alpha=0.8)
    plt.plot(history['round'], history['global_heavyweight_accuracy'], 
             label='Global Heavy', marker='v', linewidth=1.5, markersize=3, alpha=0.8)
    plt.xlabel('Training Round', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.title('All Models Accuracy Comparison', fontsize=12, weight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Plot 11: Accuracy Improvement Rate (derivative)
    plt.subplot(4, 4, 11)
    light_acc_diff = np.diff(history['lightweight_accuracy'], prepend=history['lightweight_accuracy'].iloc[0])
    heavy_acc_diff = np.diff(history['heavyweight_accuracy'], prepend=history['heavyweight_accuracy'].iloc[0])
    plt.plot(history['round'], light_acc_diff, 
             label='Lightweight', linewidth=2, alpha=0.7, color='green')
    plt.plot(history['round'], heavy_acc_diff, 
             label='Heavyweight', linewidth=2, alpha=0.7, color='purple')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Training Round', fontsize=10)
    plt.ylabel('Accuracy Change', fontsize=10)
    plt.title('Accuracy Improvement Rate', fontsize=12, weight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot 12: Cumulative Performance
    plt.subplot(4, 4, 12)
    cum_light = history['lightweight_accuracy'].cumsum() / np.arange(1, len(history) + 1)
    cum_heavy = history['heavyweight_accuracy'].cumsum() / np.arange(1, len(history) + 1)
    plt.plot(history['round'], cum_light, 
             label='Lightweight', linewidth=2, color='green')
    plt.plot(history['round'], cum_heavy, 
             label='Heavyweight', linewidth=2, color='purple')
    plt.xlabel('Training Round', fontsize=10)
    plt.ylabel('Cumulative Average Accuracy', fontsize=10)
    plt.title('Cumulative Performance', fontsize=12, weight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Plot 13: Final Metrics Comparison (Bar Chart)
    plt.subplot(4, 4, 13)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    light_values = [
        history['lightweight_accuracy'].iloc[-1],
        history['lightweight_precision'].iloc[-1],
        history['lightweight_recall'].iloc[-1],
        history['lightweight_f1'].iloc[-1]
    ]
    heavy_values = [
        history['heavyweight_accuracy'].iloc[-1],
        history['heavyweight_precision'].iloc[-1],
        history['heavyweight_recall'].iloc[-1],
        history['heavyweight_f1'].iloc[-1]
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, light_values, width, label='Lightweight', 
            color='lightblue', edgecolor='black')
    plt.bar(x + width/2, heavy_values, width, label='Heavyweight', 
            color='lightcoral', edgecolor='black')
    plt.xlabel('Metric', fontsize=10)
    plt.ylabel('Score', fontsize=10)
    plt.title('Final Metrics Comparison', fontsize=12, weight='bold')
    plt.xticks(x, metrics, fontsize=9)
    plt.legend(fontsize=9)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Plot 14: Performance Summary Table
    plt.subplot(4, 4, 14)
    plt.axis('off')
    final_metrics = [
        ['Metric', 'Lightweight', 'Heavyweight'],
        ['Accuracy', f"{history['lightweight_accuracy'].iloc[-1]:.3f}", 
         f"{history['heavyweight_accuracy'].iloc[-1]:.3f}"],
        ['Precision', f"{history['lightweight_precision'].iloc[-1]:.3f}", 
         f"{history['heavyweight_precision'].iloc[-1]:.3f}"],
        ['Recall', f"{history['lightweight_recall'].iloc[-1]:.3f}", 
         f"{history['heavyweight_recall'].iloc[-1]:.3f}"],
        ['F1 Score', f"{history['lightweight_f1'].iloc[-1]:.3f}", 
         f"{history['heavyweight_f1'].iloc[-1]:.3f}"],
        ['', '', ''],
        ['Avg Samples', f"{history['num_lightweight_samples'].mean():.0f}", 
         f"{history['num_heavyweight_samples'].mean():.0f}"]
    ]
    table = plt.table(cellText=final_metrics, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    # Color header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold')
    plt.title('Final Performance Summary', pad=20, fontsize=12, weight='bold')
    
    # Plot 15: Error Distribution (Box Plot)
    plt.subplot(4, 4, 15)
    error_data = [
        1 - history['lightweight_accuracy'],
        1 - history['heavyweight_accuracy'],
        1 - history['global_lightweight_accuracy'],
        1 - history['global_heavyweight_accuracy']
    ]
    bp = plt.boxplot(error_data, labels=['Local\nLight', 'Local\nHeavy', 
                                          'Global\nLight', 'Global\nHeavy'],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']):
        patch.set_facecolor(color)
    plt.ylabel('Error Rate (1 - Accuracy)', fontsize=10)
    plt.title('Error Distribution', fontsize=12, weight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 16: Training Progress Indicator
    plt.subplot(4, 4, 16)
    plt.axis('off')
    progress_info = [
        ['Training Statistics', 'Value'],
        ['Total Rounds', f"{len(history)}"],
        ['Time Range', f"{history['time_start'].min():.0f}s - {history['time_end'].max():.0f}s"],
        ['', ''],
        ['Best Light Acc', f"{history['lightweight_accuracy'].max():.3f}"],
        ['Best Heavy Acc', f"{history['heavyweight_accuracy'].max():.3f}"],
        ['', ''],
        ['Final Light Acc', f"{history['lightweight_accuracy'].iloc[-1]:.3f}"],
        ['Final Heavy Acc', f"{history['heavyweight_accuracy'].iloc[-1]:.3f}"]
    ]
    table2 = plt.table(cellText=progress_info, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2)
    table2[(0, 0)].set_facecolor('#FF6B6B')
    table2[(0, 1)].set_facecolor('#FF6B6B')
    table2[(0, 0)].set_text_props(weight='bold')
    table2[(0, 1)].set_text_props(weight='bold')
    plt.title('Training Progress', pad=20, fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_training_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Comprehensive training plot saved to {output_dir / 'comprehensive_training_results.png'}")
    plt.close()
    
    # ==================== SEPARATE FOCUSED PLOTS ====================
    
    # Loss Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Lightweight Loss
    axes[0, 0].plot(history['round'], 1 - history['lightweight_accuracy'], 
                    label='Local', linewidth=2, marker='o', markersize=4, color='red')
    axes[0, 0].plot(history['round'], 1 - history['global_lightweight_accuracy'], 
                    label='Global', linewidth=2, marker='s', markersize=4, color='darkred')
    axes[0, 0].set_xlabel('Training Round', fontsize=11)
    axes[0, 0].set_ylabel('Loss (1 - Accuracy)', fontsize=11)
    axes[0, 0].set_title('Lightweight Model Loss', fontsize=13, weight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Heavyweight Loss
    axes[0, 1].plot(history['round'], 1 - history['heavyweight_accuracy'], 
                    label='Local', linewidth=2, marker='o', markersize=4, color='blue')
    axes[0, 1].plot(history['round'], 1 - history['global_heavyweight_accuracy'], 
                    label='Global', linewidth=2, marker='s', markersize=4, color='darkblue')
    axes[0, 1].set_xlabel('Training Round', fontsize=11)
    axes[0, 1].set_ylabel('Loss (1 - Accuracy)', fontsize=11)
    axes[0, 1].set_title('Heavyweight Model Loss', fontsize=13, weight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Combined Loss Comparison
    axes[1, 0].plot(history['round'], 1 - history['lightweight_accuracy'], 
                    label='Light Local', linewidth=2, marker='o', markersize=4, alpha=0.7)
    axes[1, 0].plot(history['round'], 1 - history['heavyweight_accuracy'], 
                    label='Heavy Local', linewidth=2, marker='s', markersize=4, alpha=0.7)
    axes[1, 0].plot(history['round'], 1 - history['global_lightweight_accuracy'], 
                    label='Light Global', linewidth=2, marker='^', markersize=4, alpha=0.7)
    axes[1, 0].plot(history['round'], 1 - history['global_heavyweight_accuracy'], 
                    label='Heavy Global', linewidth=2, marker='v', markersize=4, alpha=0.7)
    axes[1, 0].set_xlabel('Training Round', fontsize=11)
    axes[1, 0].set_ylabel('Loss (1 - Accuracy)', fontsize=11)
    axes[1, 0].set_title('All Models Loss Comparison', fontsize=13, weight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss Reduction Rate
    axes[1, 1].plot(history['round'], 
                    -np.diff(1 - history['lightweight_accuracy'], prepend=(1 - history['lightweight_accuracy'].iloc[0])),
                    label='Lightweight', linewidth=2, alpha=0.7)
    axes[1, 1].plot(history['round'], 
                    -np.diff(1 - history['heavyweight_accuracy'], prepend=(1 - history['heavyweight_accuracy'].iloc[0])),
                    label='Heavyweight', linewidth=2, alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Training Round', fontsize=11)
    axes[1, 1].set_ylabel('Loss Reduction Rate', fontsize=11)
    axes[1, 1].set_title('Loss Reduction Rate (Higher is Better)', fontsize=13, weight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Model Loss Analysis', fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Loss analysis plot saved to {output_dir / 'loss_analysis.png'}")
    plt.close()
    
    # Accuracy Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Local Models Accuracy
    axes[0, 0].plot(history['round'], history['lightweight_accuracy'], 
                    label='Lightweight', linewidth=2.5, marker='o', markersize=5, color='green')
    axes[0, 0].plot(history['round'], history['heavyweight_accuracy'], 
                    label='Heavyweight', linewidth=2.5, marker='s', markersize=5, color='purple')
    axes[0, 0].set_xlabel('Training Round', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('Local Models Accuracy', fontsize=13, weight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])
    
    # Global Models Accuracy
    axes[0, 1].plot(history['round'], history['global_lightweight_accuracy'], 
                    label='Lightweight', linewidth=2.5, marker='o', markersize=5, color='#FF6B6B')
    axes[0, 1].plot(history['round'], history['global_heavyweight_accuracy'], 
                    label='Heavyweight', linewidth=2.5, marker='s', markersize=5, color='#4ECDC4')
    axes[0, 1].set_xlabel('Training Round', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Global Models Accuracy', fontsize=13, weight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # Precision, Recall, F1 for Lightweight
    axes[1, 0].plot(history['round'], history['lightweight_accuracy'], 
                    label='Accuracy', linewidth=2, marker='o', markersize=4)
    axes[1, 0].plot(history['round'], history['lightweight_precision'], 
                    label='Precision', linewidth=2, marker='s', markersize=4)
    axes[1, 0].plot(history['round'], history['lightweight_recall'], 
                    label='Recall', linewidth=2, marker='^', markersize=4)
    axes[1, 0].plot(history['round'], history['lightweight_f1'], 
                    label='F1 Score', linewidth=2, marker='v', markersize=4)
    axes[1, 0].set_xlabel('Training Round', fontsize=11)
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].set_title('Lightweight Model Metrics', fontsize=13, weight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])
    
    # Precision, Recall, F1 for Heavyweight
    axes[1, 1].plot(history['round'], history['heavyweight_accuracy'], 
                    label='Accuracy', linewidth=2, marker='o', markersize=4)
    axes[1, 1].plot(history['round'], history['heavyweight_precision'], 
                    label='Precision', linewidth=2, marker='s', markersize=4)
    axes[1, 1].plot(history['round'], history['heavyweight_recall'], 
                    label='Recall', linewidth=2, marker='^', markersize=4)
    axes[1, 1].plot(history['round'], history['heavyweight_f1'], 
                    label='F1 Score', linewidth=2, marker='v', markersize=4)
    axes[1, 1].set_xlabel('Training Round', fontsize=11)
    axes[1, 1].set_ylabel('Score', fontsize=11)
    axes[1, 1].set_title('Heavyweight Model Metrics', fontsize=13, weight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])
    
    plt.suptitle('Model Accuracy and Performance Metrics', fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Accuracy metrics plot saved to {output_dir / 'accuracy_metrics.png'}")
    plt.close()
    
    # Generate text summary
    summary_path = output_dir / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("FEDERATED LEARNING TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Training Rounds: {len(history)}\n")
        f.write(f"Time Range: {history['time_start'].min():.0f}s - {history['time_end'].max():.0f}s\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("FINAL PERFORMANCE METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Lightweight Models:\n")
        f.write(f"  - Final Accuracy:  {history['lightweight_accuracy'].iloc[-1]:.4f}\n")
        f.write(f"  - Final Precision: {history['lightweight_precision'].iloc[-1]:.4f}\n")
        f.write(f"  - Final Recall:    {history['lightweight_recall'].iloc[-1]:.4f}\n")
        f.write(f"  - Final F1 Score:  {history['lightweight_f1'].iloc[-1]:.4f}\n")
        f.write(f"  - Peak Accuracy:   {history['lightweight_accuracy'].max():.4f} (Round {history['lightweight_accuracy'].idxmax() + 1})\n\n")
        
        f.write("Heavyweight Models:\n")
        f.write(f"  - Final Accuracy:  {history['heavyweight_accuracy'].iloc[-1]:.4f}\n")
        f.write(f"  - Final Precision: {history['heavyweight_precision'].iloc[-1]:.4f}\n")
        f.write(f"  - Final Recall:    {history['heavyweight_recall'].iloc[-1]:.4f}\n")
        f.write(f"  - Final F1 Score:  {history['heavyweight_f1'].iloc[-1]:.4f}\n")
        f.write(f"  - Peak Accuracy:   {history['heavyweight_accuracy'].max():.4f} (Round {history['heavyweight_accuracy'].idxmax() + 1})\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("TRAINING STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Average Lightweight Samples per Round: {history['num_lightweight_samples'].mean():.0f}\n")
        f.write(f"Average Heavyweight Samples per Round: {history['num_heavyweight_samples'].mean():.0f}\n")
        f.write(f"Total Lightweight Samples: {history['num_lightweight_samples'].sum():.0f}\n")
        f.write(f"Total Heavyweight Samples: {history['num_heavyweight_samples'].sum():.0f}\n\n")
        
        f.write("Knowledge Transfer:\n")
        f.write(f"  - Final Light→Heavy Agreement: {history['transfer_accuracy_light_to_heavy'].iloc[-1]:.4f}\n")
        f.write(f"  - Final Heavy→Light Agreement: {history['transfer_accuracy_heavy_to_light'].iloc[-1]:.4f}\n")
    
    print(f"✓ Training summary saved to {summary_path}")
    
    print("\n" + "=" * 80)
    print("✓ VISUALIZATION COMPLETE!")
    print(f"✓ All plots saved to '{output_dir}' directory")
    print("=" * 80)


def main():
    """Main execution"""
    plot_comprehensive_training_results()


if __name__ == "__main__":
    main()
