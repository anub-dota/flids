"""
Exploratory Data Analysis (EDA) for Device Logs and Pre-shuffled Data
This script analyzes the dataset and generates comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def analyze_device_logs():
    """Analyze raw device logs from the logs/ directory"""
    print("=" * 80)
    print("ANALYZING DEVICE LOGS")
    print("=" * 80)
    
    logs_dir = Path('logs')
    if not logs_dir.exists():
        print("Logs directory not found!")
        return
    
    # Create output directory
    output_dir = Path('eda_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Load all device logs
    device_logs = {}
    for log_file in logs_dir.glob('*_logs.csv'):
        device_name = log_file.stem.replace('_logs', '')
        try:
            df = pd.read_csv(log_file)
            device_logs[device_name] = df
            print(f"\n✓ Loaded {device_name}: {len(df)} log entries")
        except Exception as e:
            print(f"✗ Error loading {log_file}: {e}")
    
    if not device_logs:
        print("No device logs found!")
        return
    
    # Create comprehensive EDA plots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Total packets per device
    plt.subplot(3, 4, 1)
    packet_counts = {name: len(df) for name, df in device_logs.items()}
    plt.bar(packet_counts.keys(), packet_counts.values(), color='skyblue', edgecolor='black')
    plt.xlabel('Device')
    plt.ylabel('Total Packets')
    plt.title('Total Packets per Device')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Attack vs Normal traffic distribution
    plt.subplot(3, 4, 2)
    all_logs = pd.concat(device_logs.values(), ignore_index=True)
    attack_counts = all_logs['infected'].value_counts()
    plt.pie(attack_counts.values, labels=['Attack', 'Normal'], autopct='%1.1f%%', 
            colors=[ 'salmon','lightgreen'], startangle=90)
    plt.title('Attack vs Normal Traffic Distribution')
    
    # Plot 3: Packet types distribution
    plt.subplot(3, 4, 3)
    packet_types = all_logs['packet_type'].value_counts()
    plt.barh(packet_types.index, packet_types.values, color='coral', edgecolor='black')
    plt.xlabel('Count')
    plt.ylabel('Packet Type')
    plt.title('Packet Types Distribution')
    plt.grid(axis='x', alpha=0.3)
    
    # Plot 4: Average packet size by type
    plt.subplot(3, 4, 4)
    avg_sizes = all_logs.groupby('packet_type')['size'].mean().sort_values(ascending=False)
    plt.bar(avg_sizes.index, avg_sizes.values, color='mediumpurple', edgecolor='black')
    plt.xlabel('Packet Type')
    plt.ylabel('Average Size (bytes)')
    plt.title('Average Packet Size by Type')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 5: Traffic over time (sent vs received)
    plt.subplot(3, 4, 5)
    sent_received = all_logs['sent_or_received'].value_counts()
    plt.bar(sent_received.index, sent_received.values, color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
    plt.xlabel('Direction')
    plt.ylabel('Count')
    plt.title('Sent vs Received Packets')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 6: Queue utilization over time
    plt.subplot(3, 4, 6)
    for device_name, df in list(device_logs.items())[:3]:  # Show first 3 devices
        if 'queue_full_percentage' in df.columns:
            plt.plot(df['timestamp'], df['queue_full_percentage'], 
                    label=device_name, alpha=0.7, linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Queue Utilization (%)')
    plt.title('Queue Utilization Over Time')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    
    # Plot 7: Attack timeline
    plt.subplot(3, 4, 7)
    attack_timeline = all_logs.groupby('timestamp')['infected'].sum()
    plt.fill_between(attack_timeline.index, attack_timeline.values, 
                     color='red', alpha=0.3, label='Attack Periods')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Infected Packets')
    plt.title('Attack Timeline')
    plt.grid(alpha=0.3)
    
    # Plot 8: Packet size distribution
    plt.subplot(3, 4, 8)
    plt.hist(all_logs['size'], bins=50, color='teal', edgecolor='black', alpha=0.7)
    plt.xlabel('Packet Size (bytes)')
    plt.ylabel('Frequency')
    plt.title('Packet Size Distribution')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 9: Top source-destination pairs
    plt.subplot(3, 4, 9)
    all_logs['src_dst'] = all_logs['src'] + ' → ' + all_logs['dst']
    top_pairs = all_logs['src_dst'].value_counts().head(10)
    plt.barh(range(len(top_pairs)), top_pairs.values, color='orange', edgecolor='black')
    plt.yticks(range(len(top_pairs)), top_pairs.index, fontsize=8)
    plt.xlabel('Packet Count')
    plt.title('Top 10 Source-Destination Pairs')
    plt.grid(axis='x', alpha=0.3)
    
    # Plot 10: Attack intensity by device
    plt.subplot(3, 4, 10)
    attack_by_device = {name: df['infected'].sum() for name, df in device_logs.items()}
    plt.bar(attack_by_device.keys(), list(attack_by_device.values()), 
            color='crimson', edgecolor='black')
    plt.xlabel('Device')
    plt.ylabel('Infected Packets')
    plt.title('Attack Packets by Device')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 11: Port usage distribution
    plt.subplot(3, 4, 11)
    port_usage = all_logs['dst_port'].value_counts().head(10)
    plt.bar(port_usage.index.astype(str), port_usage.values, 
            color='steelblue', edgecolor='black')
    plt.xlabel('Destination Port')
    plt.ylabel('Count')
    plt.title('Top 10 Most Used Ports')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 12: Summary statistics table
    plt.subplot(3, 4, 12)
    plt.axis('off')
    summary_stats = [
        ['Metric', 'Value'],
        ['Total Packets', f"{len(all_logs):,}"],
        ['Total Devices', f"{len(device_logs)}"],
        ['Attack Packets', f"{all_logs['infected'].sum():,}"],
        ['Normal Packets', f"{(~all_logs['infected']).sum():,}"],
        ['Attack %', f"{(all_logs['infected'].sum() / len(all_logs) * 100):.2f}%"],
        ['Avg Packet Size', f"{all_logs['size'].mean():.2f} bytes"],
        ['Time Range', f"{all_logs['timestamp'].min():.1f}s - {all_logs['timestamp'].max():.1f}s"]
    ]
    table = plt.table(cellText=summary_stats, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    plt.title('Summary Statistics', pad=20, fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'device_logs_eda.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Device logs EDA saved to {output_dir / 'device_logs_eda.png'}")
    plt.close()
    
    # Generate summary report
    report_path = output_dir / 'device_logs_summary.txt'
    with open(report_path, 'w') as f:
        f.write("DEVICE LOGS ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Packets: {len(all_logs):,}\n")
        f.write(f"Total Devices: {len(device_logs)}\n")
        f.write(f"Time Range: {all_logs['timestamp'].min():.2f}s - {all_logs['timestamp'].max():.2f}s\n")
        f.write(f"Duration: {all_logs['timestamp'].max() - all_logs['timestamp'].min():.2f}s\n\n")
        
        f.write("Attack Statistics:\n")
        f.write(f"  - Attack Packets: {all_logs['infected'].sum():,}\n")
        f.write(f"  - Normal Packets: {(~all_logs['infected']).sum():,}\n")
        f.write(f"  - Attack Percentage: {(all_logs['infected'].sum() / len(all_logs) * 100):.2f}%\n\n")
        
        f.write("Packet Statistics:\n")
        f.write(f"  - Average Size: {all_logs['size'].mean():.2f} bytes\n")
        f.write(f"  - Min Size: {all_logs['size'].min()} bytes\n")
        f.write(f"  - Max Size: {all_logs['size'].max()} bytes\n")
        f.write(f"  - Std Dev: {all_logs['size'].std():.2f} bytes\n\n")
        
        f.write("Per-Device Statistics:\n")
        for device_name, df in device_logs.items():
            f.write(f"\n  {device_name}:\n")
            f.write(f"    - Total Packets: {len(df):,}\n")
            f.write(f"    - Attack Packets: {df['infected'].sum():,}\n")
            f.write(f"    - Sent: {(df['sent_or_received'] == 'sent').sum():,}\n")
            f.write(f"    - Received: {(df['sent_or_received'] == 'received').sum():,}\n")
    
    print(f"✓ Summary report saved to {report_path}")


def analyze_datapoints():
    """Analyze pre-shuffled datapoint files"""
    print("\n" + "=" * 80)
    print("ANALYZING DATAPOINTS")
    print("=" * 80)
    
    data_dir = Path('data')
    if not data_dir.exists():
        print("Shuffled data directory not found!")
        return
    
    output_dir = Path('eda_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Load all datapoint files
    datapoints = {}
    for data_file in data_dir.glob('peer_*_datapts.csv'):
        peer_name = data_file.stem.replace('_datapts', '')
        try:
            df = pd.read_csv(data_file)
            datapoints[peer_name] = df
            print(f"\n✓ Loaded {peer_name}: {len(df)} datapoints, {df['label'].sum()} attacks")
        except Exception as e:
            print(f"✗ Error loading {data_file}: {e}")
    
    if not datapoints:
        print("No datapoint files found!")
        return
    
    # Create comprehensive feature analysis
    fig = plt.figure(figsize=(20, 14))
    
    # Combine all data
    all_data = pd.concat(datapoints.values(), ignore_index=True)
    
    # Plot 1: Label distribution per device
    plt.subplot(4, 4, 1)
    label_counts = {name: [len(df) - df['label'].sum(), df['label'].sum()] 
                    for name, df in datapoints.items()}
    x = np.arange(len(label_counts))
    width = 0.35
    normal_counts = [counts[0] for counts in label_counts.values()]
    attack_counts = [counts[1] for counts in label_counts.values()]
    
    plt.bar(x, normal_counts, width, label='Normal', color='lightgreen', edgecolor='black')
    plt.bar(x, attack_counts, width, bottom=normal_counts, label='Attack', 
            color='salmon', edgecolor='black')
    plt.xlabel('Device')
    plt.ylabel('Count')
    plt.title('Normal vs Attack Datapoints per Device')
    plt.xticks(x, label_counts.keys(), rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Overall label distribution
    plt.subplot(4, 4, 2)
    labels = all_data['label'].value_counts()
    plt.pie(labels.values, labels=['Normal', 'Attack'], autopct='%1.1f%%',
            colors=['lightgreen', 'salmon'], startangle=90)
    plt.title('Overall Label Distribution')
    
    # Plot 3: Feature correlation heatmap (sample features)
    plt.subplot(4, 4, 3)
    feature_cols = [col for col in all_data.columns if col not in ['timestamp', 'label']]
    sample_features = feature_cols[:20]  # Take first 20 features for visualization
    corr_matrix = all_data[sample_features].corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8}, 
                xticklabels=False, yticklabels=False)
    plt.title('Feature Correlation Heatmap\n(Sample 20 features)')
    
    # Plot 4: Average feature values (Attack vs Normal)
    plt.subplot(4, 4, 4)
    attack_data = all_data[all_data['label'] == 1]
    normal_data = all_data[all_data['label'] == 0]
    
    # Select a few key features for comparison
    key_features = [f for f in feature_cols if 'pkts_30s' in f][:6]
    attack_means = attack_data[key_features].mean()
    normal_means = normal_data[key_features].mean()
    
    x = np.arange(len(key_features))
    width = 0.35
    plt.bar(x - width/2, normal_means, width, label='Normal', color='lightgreen', edgecolor='black')
    plt.bar(x + width/2, attack_means, width, label='Attack', color='salmon', edgecolor='black')
    plt.xlabel('Feature')
    plt.ylabel('Average Value')
    plt.title('Key Features: Attack vs Normal')
    plt.xticks(x, [f.split('_')[0] + '_' + f.split('_')[2] for f in key_features], 
               rotation=45, ha='right', fontsize=8)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 5-8: Feature distributions over time for different window sizes
    time_windows = ['30s', '10s', '5s']
    for idx, window in enumerate(time_windows):
        plt.subplot(4, 4, 5 + idx)
        feature = f'pkts_{window}_src0'
        if feature in all_data.columns:
            plt.plot(all_data['timestamp'], all_data[feature], 
                    alpha=0.5, linewidth=0.5, color='blue')
            plt.xlabel('Time (s)')
            plt.ylabel('Packet Count')
            plt.title(f'Packets Over Time ({window} window)')
            plt.grid(alpha=0.3)
    
    # Plot 8: Attack intensity timeline
    plt.subplot(4, 4, 8)
    attack_timeline = all_data.groupby('timestamp')['label'].sum()
    plt.fill_between(attack_timeline.index, attack_timeline.values,
                     color='red', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Attack Intensity')
    plt.title('Attack Intensity Over Time')
    plt.grid(alpha=0.3)
    
    # Plot 9: Queue statistics distribution
    plt.subplot(4, 4, 9)
    queue_features = [f for f in feature_cols if 'queue_avg' in f][:3]
    for feature in queue_features:
        if feature in all_data.columns:
            plt.hist(all_data[feature], bins=50, alpha=0.5, 
                    label=feature.split('_')[2], edgecolor='black')
    plt.xlabel('Queue Average')
    plt.ylabel('Frequency')
    plt.title('Queue Statistics Distribution')
    plt.legend(fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 10: TCP vs UDP distribution
    plt.subplot(4, 4, 10)
    tcp_feature = 'tcp_30s_src0'
    udp_feature = 'udp_30s_src0'
    if tcp_feature in all_data.columns and udp_feature in all_data.columns:
        protocol_data = pd.DataFrame({
            'TCP': all_data[tcp_feature].sum(),
            'UDP': all_data[udp_feature].sum()
        }, index=[0])
        protocol_data.T.plot(kind='bar', legend=False, color=['skyblue', 'coral'],
                            edgecolor='black', ax=plt.gca())
        plt.xlabel('Protocol')
        plt.ylabel('Total Count')
        plt.title('TCP vs UDP Traffic')
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
    
    # Plot 11: Packet size variance
    plt.subplot(4, 4, 11)
    pkt_size_var = 'pkt_size_var_30s_src0'
    if pkt_size_var in all_data.columns:
        normal_var = normal_data[pkt_size_var]
        attack_var = attack_data[pkt_size_var]
        plt.hist(normal_var, bins=50, alpha=0.5, label='Normal', 
                color='lightgreen', edgecolor='black')
        plt.hist(attack_var, bins=50, alpha=0.5, label='Attack',
                color='salmon', edgecolor='black')
        plt.xlabel('Packet Size Variance')
        plt.ylabel('Frequency')
        plt.title('Packet Size Variance Distribution')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
    
    # Plot 12: Data imbalance per device
    plt.subplot(4, 4, 12)
    imbalance_ratios = {name: df['label'].sum() / len(df) * 100 
                        for name, df in datapoints.items()}
    plt.bar(imbalance_ratios.keys(), imbalance_ratios.values(),
            color='mediumpurple', edgecolor='black')
    plt.xlabel('Device')
    plt.ylabel('Attack Percentage (%)')
    plt.title('Attack Percentage by Device')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 13-16: Feature statistics summary
    plt.subplot(4, 4, 13)
    plt.axis('off')
    feature_stats = [
        ['Statistic', 'Value'],
        ['Total Datapoints', f"{len(all_data):,}"],
        ['Total Features', f"{len(feature_cols)}"],
        ['Attack Datapoints', f"{all_data['label'].sum():,}"],
        ['Normal Datapoints', f"{(~all_data['label']).sum():,}"],
        ['Class Imbalance', f"{(all_data['label'].sum() / len(all_data) * 100):.2f}%"],
        ['Time Range', f"{all_data['timestamp'].min():.1f}s - {all_data['timestamp'].max():.1f}s"],
        ['Devices', f"{len(datapoints)}"]
    ]
    table = plt.table(cellText=feature_stats, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    plt.title('Datapoint Statistics', pad=20, fontsize=12, weight='bold')
    
    # Additional feature analysis plots
    plt.subplot(4, 4, 14)
    # Average packet size comparison
    avg_pkt_features = [f for f in feature_cols if 'avg_pkt_size' in f][:6]
    if avg_pkt_features:
        avg_values = all_data[avg_pkt_features].mean()
        plt.bar(range(len(avg_values)), avg_values.values,
                color='teal', edgecolor='black')
        plt.xlabel('Feature Index')
        plt.ylabel('Average Value')
        plt.title('Average Packet Size Features')
        plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(4, 4, 15)
    # Feature value ranges
    sample_features_range = feature_cols[:10]
    ranges = [(all_data[f].max() - all_data[f].min()) for f in sample_features_range]
    plt.barh(range(len(ranges)), ranges, color='orange', edgecolor='black')
    plt.yticks(range(len(ranges)), [f[:20] + '...' if len(f) > 20 else f 
                                     for f in sample_features_range], fontsize=7)
    plt.xlabel('Value Range')
    plt.title('Feature Value Ranges (Sample)')
    plt.grid(axis='x', alpha=0.3)
    
    plt.subplot(4, 4, 16)
    # Missing values analysis
    missing_counts = all_data[feature_cols].isnull().sum()
    if missing_counts.sum() > 0:
        top_missing = missing_counts.nlargest(10)
        plt.bar(range(len(top_missing)), top_missing.values,
                color='red', alpha=0.6, edgecolor='black')
        plt.xlabel('Feature Index')
        plt.ylabel('Missing Count')
        plt.title('Missing Values (Top 10 Features)')
        plt.grid(axis='y', alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Missing Values\n✓', 
                ha='center', va='center', fontsize=20, color='green',
                weight='bold', transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title('Missing Values Check')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'shuffled_datapoints_eda.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Shuffled datapoints EDA saved to {output_dir / 'shuffled_datapoints_eda.png'}")
    plt.close()
    
    # Generate detailed summary
    report_path = output_dir / 'datapoints_summary.txt'
    with open(report_path, 'w') as f:
        f.write("SHUFFLED DATAPOINTS ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Datapoints: {len(all_data):,}\n")
        f.write(f"Total Devices: {len(datapoints)}\n")
        f.write(f"Total Features: {len(feature_cols)}\n")
        f.write(f"Time Range: {all_data['timestamp'].min():.2f}s - {all_data['timestamp'].max():.2f}s\n\n")
        
        f.write("Label Distribution:\n")
        f.write(f"  - Normal: {(~all_data['label']).sum():,} ({(~all_data['label']).sum() / len(all_data) * 100:.2f}%)\n")
        f.write(f"  - Attack: {all_data['label'].sum():,} ({all_data['label'].sum() / len(all_data) * 100:.2f}%)\n\n")
        
        f.write("Per-Device Statistics:\n")
        for peer_name, df in datapoints.items():
            f.write(f"\n  {peer_name}:\n")
            f.write(f"    - Total Datapoints: {len(df):,}\n")
            f.write(f"    - Attack Datapoints: {df['label'].sum():,}\n")
            f.write(f"    - Attack %: {df['label'].sum() / len(df) * 100:.2f}%\n")
    
    print(f"✓ Datapoints summary saved to {report_path}")


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    # Analyze device logs
    analyze_device_logs()
    
    # Analyze datapoints
    analyze_datapoints()
    
    print("\n" + "=" * 80)
    print("✓ EDA COMPLETE!")
    print("✓ Check the 'eda_plots' directory for all visualizations")
    print("=" * 80)


if __name__ == "__main__":
    main()
