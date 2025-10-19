import os
import pandas as pd
import matplotlib.pyplot as plt
from federated_learning import FederatedIntrustionDetection

def verify_data_files():
    """Verify that all required datapoint files exist"""
    print("Verifying data files...")
    
    missing_files = []
    for i in range(1, 7):  # peer_1 to peer_6
        file_path = f'data/peer_{i}_datapts.csv'
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            # Check file structure
            df = pd.read_csv(file_path, nrows=5)
            print(f"  ✓ {file_path}: {len(pd.read_csv(file_path))} rows, {len(df.columns)} columns")
    
    if missing_files:
        print("\nMissing files:")
        for f in missing_files:
            print(f"  ✗ {f}")
        print("\nPlease run 'python generate_datapoints.py' first to generate the datapoint files.")
        return False
    
    print("All data files verified successfully!\n")
    return True

def plot_training_results(history):
    """Plot training results with comprehensive metrics"""
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Local Model Accuracy
    plt.subplot(3, 3, 1)
    plt.plot(history['round'], history['lightweight_accuracy'], label='Lightweight (Local Ensemble)', marker='o', linewidth=2)
    plt.plot(history['round'], history['heavyweight_accuracy'], label='Heavyweight (Local Ensemble)', marker='s', linewidth=2)
    plt.xlabel('Training Round')
    plt.ylabel('Accuracy')
    plt.title('Local Model Accuracy Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Global Model Accuracy
    plt.subplot(3, 3, 2)
    plt.plot(history['round'], history['global_lightweight_accuracy'], label='Global Lightweight', marker='o', linewidth=2)
    plt.plot(history['round'], history['global_heavyweight_accuracy'], label='Global Heavyweight', marker='s', linewidth=2)
    plt.xlabel('Training Round')
    plt.ylabel('Accuracy')
    plt.title('Global Model Accuracy Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Precision
    plt.subplot(3, 3, 3)
    plt.plot(history['round'], history['lightweight_precision'], label='Lightweight', marker='o', linewidth=2)
    plt.plot(history['round'], history['heavyweight_precision'], label='Heavyweight', marker='s', linewidth=2)
    plt.xlabel('Training Round')
    plt.ylabel('Precision')
    plt.title('Model Precision Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Recall
    plt.subplot(3, 3, 4)
    plt.plot(history['round'], history['lightweight_recall'], label='Lightweight', marker='o', linewidth=2)
    plt.plot(history['round'], history['heavyweight_recall'], label='Heavyweight', marker='s', linewidth=2)
    plt.xlabel('Training Round')
    plt.ylabel('Recall')
    plt.title('Model Recall Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: F1 Score
    plt.subplot(3, 3, 5)
    plt.plot(history['round'], history['lightweight_f1'], label='Lightweight', marker='o', linewidth=2)
    plt.plot(history['round'], history['heavyweight_f1'], label='Heavyweight', marker='s', linewidth=2)
    plt.xlabel('Training Round')
    plt.ylabel('F1 Score')
    plt.title('Model F1 Score Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Knowledge Transfer Effectiveness
    plt.subplot(3, 3, 6)
    plt.plot(history['round'], history['transfer_accuracy_light_to_heavy'], label='Light → Heavy', marker='o', linewidth=2)
    plt.plot(history['round'], history['transfer_accuracy_heavy_to_light'], label='Heavy → Light', marker='s', linewidth=2)
    plt.xlabel('Training Round')
    plt.ylabel('Transfer Agreement')
    plt.title('Knowledge Transfer Agreement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Training Samples
    plt.subplot(3, 3, 7)
    plt.plot(history['round'], history['num_lightweight_samples'], label='Lightweight', marker='o', linewidth=2)
    plt.plot(history['round'], history['num_heavyweight_samples'], label='Heavyweight', marker='s', linewidth=2)
    plt.xlabel('Training Round')
    plt.ylabel('Number of Samples')
    plt.title('Training Samples per Round')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Comparison - All Accuracies
    plt.subplot(3, 3, 8)
    plt.plot(history['round'], history['lightweight_accuracy'], label='Local Light', marker='o', linewidth=1.5, alpha=0.7)
    plt.plot(history['round'], history['heavyweight_accuracy'], label='Local Heavy', marker='s', linewidth=1.5, alpha=0.7)
    plt.plot(history['round'], history['global_lightweight_accuracy'], label='Global Light', marker='^', linewidth=1.5, alpha=0.7)
    plt.plot(history['round'], history['global_heavyweight_accuracy'], label='Global Heavy', marker='v', linewidth=1.5, alpha=0.7)
    plt.xlabel('Training Round')
    plt.ylabel('Accuracy')
    plt.title('All Models Accuracy Comparison')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Performance Summary Table
    plt.subplot(3, 3, 9)
    plt.axis('off')
    final_metrics = [
        ['Metric', 'Lightweight', 'Heavyweight'],
        ['Accuracy', f"{history['lightweight_accuracy'][-1]:.3f}", f"{history['heavyweight_accuracy'][-1]:.3f}"],
        ['Precision', f"{history['lightweight_precision'][-1]:.3f}", f"{history['heavyweight_precision'][-1]:.3f}"],
        ['Recall', f"{history['lightweight_recall'][-1]:.3f}", f"{history['heavyweight_recall'][-1]:.3f}"],
        ['F1 Score', f"{history['lightweight_f1'][-1]:.3f}", f"{history['heavyweight_f1'][-1]:.3f}"],
    ]
    table = plt.table(cellText=final_metrics, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Final Performance Metrics', pad=20)
    
    # Save results
    results_df = pd.DataFrame(history)
    results_df.to_csv('training_results.csv', index=False)
    print("\n" + "="*50)
    print("Training results saved to training_results.csv")
    print("="*50)
    print("\nFinal Performance Summary:")
    print(f"  Lightweight Models - Acc: {history['lightweight_accuracy'][-1]:.3f}, "
          f"Prec: {history['lightweight_precision'][-1]:.3f}, "
          f"Rec: {history['lightweight_recall'][-1]:.3f}, "
          f"F1: {history['lightweight_f1'][-1]:.3f}")
    print(f"  Heavyweight Models - Acc: {history['heavyweight_accuracy'][-1]:.3f}, "
          f"Prec: {history['heavyweight_precision'][-1]:.3f}, "
          f"Rec: {history['heavyweight_recall'][-1]:.3f}, "
          f"F1: {history['heavyweight_f1'][-1]:.3f}")
    print("="*50)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("Training plots saved to training_results.png")
    plt.show()

def main():
    """Main execution function"""
    print("Federated Learning for Intrusion Detection")
    print("=" * 50)
    
    # Verify data files exist
    if not verify_data_files():
        print("\nERROR: Required data files not found.")
        print("Please run: python generate_datapoints.py")
        return
    
    # Initialize and run federated learning system
    fed_system = FederatedIntrustionDetection()
    
    # Run simulation (you can adjust total_seconds based on your data)
    history = fed_system.run_simulation(total_seconds=6000)  # Start with 100 seconds for testing
    
    # Plot results
    if history['round']:  # Only plot if we have data
        plot_training_results(history)
    else:
        print("\nNo training data collected. Check if datapoint files have data in the time range.")
    
    print("\n✓ Simulation completed successfully!")
    print("✓ Check training_results.csv and training_results.png for detailed results.")

if __name__ == "__main__":
    main()
