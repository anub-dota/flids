import os
import pandas as pd
import matplotlib.pyplot as plt
from federated_learning import FederatedIntrustionDetection

def create_sample_data():
    """Create sample CSV files for testing"""
    print("Creating sample data files...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Feature columns
    base_features = ['pkts', 'avg_pkt_size', 'pkt_size_var', 'syn', 'ack', 'tcp', 'udp']
    time_windows = ['60s', '30s', '15s', '5s']
    
    feature_cols = ['timestamp']
    for feature in base_features:
        for window in time_windows:
            feature_cols.append(f'{feature}_{window}')
    feature_cols.extend(['queue_avg_5s', 'queue_var_5s', 'label'])
    
    # Create 6 device CSV files
    for device_id in range(6):
        data = []
        
        # Create data for 1000 seconds, every 0.5 seconds
        for t in range(0, 1000, 0.5):
            row = [t]  # timestamp
            
            # Generate synthetic network features
            for feature in base_features:
                for window in time_windows:
                    if 'pkts' in feature:
                        row.append(np.random.randint(10, 1000))
                    elif 'size' in feature:
                        row.append(np.random.normal(500, 100))
                    elif feature in ['syn', 'ack', 'tcp', 'udp']:
                        row.append(np.random.randint(1, 100))
            
            # Queue features
            row.append(np.random.uniform(0.1, 0.9))  # queue_avg_5s
            row.append(np.random.uniform(0.01, 0.1))  # queue_var_5s
            
            # Label (more attacks in later time periods)
            if t > 500:
                label = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% attack
            else:
                label = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% attack
            row.append(label)
            
            data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(data, columns=feature_cols)
        df.to_csv(f'data/device_{device_id + 1}.csv', index=False)
        print(f"Created device_{device_id + 1}.csv with {len(df)} rows")

def plot_training_results(history):
    """Plot training results"""
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy over time
    plt.subplot(2, 2, 1)
    plt.plot(history['round'], history['lightweight_accuracy'], label='Lightweight', marker='o')
    plt.plot(history['round'], history['heavyweight_accuracy'], label='Heavyweight', marker='s')
    plt.xlabel('Training Round')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.legend()
    plt.grid(True)
    
    # Save results
    results_df = pd.DataFrame(history)
    results_df.to_csv('training_results.csv', index=False)
    print("Training results saved to training_results.csv")
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    print("Federated Learning for Intrusion Detection")
    print("=" * 50)
    
    # Create sample data if it doesn't exist
    if not os.path.exists('data/device_1.csv'):
        create_sample_data()
    
    # Initialize and run federated learning system
    fed_system = FederatedIntrustionDetection()
    
    # Run simulation (you can reduce this for testing)
    history = fed_system.run_simulation(total_seconds=100)  # Start with 100 seconds for testing
    
    # Plot results
    plot_training_results(history)
    
    print("\nSimulation completed successfully!")
    print("Check training_results.csv and training_results.png for detailed results.")

if __name__ == "__main__":
    import numpy as np
    main()
