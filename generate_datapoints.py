import pandas as pd
import numpy as np
from pathlib import Path

def get_feature_columns(num_sources=6):
    """Return the feature column names"""
    base_features = ['pkts', 'avg_pkt_size', 'pkt_size_var', 'syn', 'ack', 'tcp', 'udp']
    time_windows = ['60s', '30s', '15s', '5s']
    
    feature_cols = []
    
    # Features aggregated per source
    for feature in base_features:
        for window in time_windows:
            for source_id in range(num_sources):
                feature_cols.append(f'{feature}_{window}_src{source_id}')
    
    # Add queue features for all time windows per source
    # Queue belongs to the device, so same value for all sources
    for window in time_windows:
        for source_id in range(num_sources):
            feature_cols.extend([f'queue_avg_{window}_src{source_id}', f'queue_var_{window}_src{source_id}'])
    
    return feature_cols


def identify_sources(df, device_name):
    """Identify unique sources that communicate with this device"""
    # Sources are devices that send packets to this device or receive from it
    sources = set()
    
    # Sources that send to this device (received packets)
    received = df[df['sent_or_received'] == 'received']['src'].unique()
    sources.update(received)
    
    # Destinations that receive from this device (sent packets)
    sent_to = df[df['sent_or_received'] == 'sent']['dst'].unique()
    sources.update(sent_to)
    
    # Remove the device itself
    sources.discard(device_name)
    
    # Convert to sorted list for consistent indexing
    sources = sorted(list(sources))
    
    return sources


def calculate_features_for_timestamp(df, current_time, sources, device_name, time_windows):
    """Calculate features for a specific timestamp"""
    features = {}
    
    # Map sources to indices (0-5)
    source_to_idx = {src: idx for idx, src in enumerate(sources)}
    
    for window_name, window_seconds in time_windows.items():
        # Cap the window start at 0 (don't go before the beginning)
        window_start = max(0, current_time - window_seconds)
        window_data = df[(df['timestamp'] >= window_start) & (df['timestamp'] < current_time)]
        
        # Initialize features for each source
        for src_idx in range(len(sources)):
            features[f'pkts_{window_name}_src{src_idx}'] = 0
            features[f'avg_pkt_size_{window_name}_src{src_idx}'] = 0
            features[f'pkt_size_var_{window_name}_src{src_idx}'] = 0
            features[f'syn_{window_name}_src{src_idx}'] = 0
            features[f'ack_{window_name}_src{src_idx}'] = 0
            features[f'tcp_{window_name}_src{src_idx}'] = 0
            features[f'udp_{window_name}_src{src_idx}'] = 0
        
        # Calculate features per source
        for source in sources:
            if source not in source_to_idx:
                continue
                
            src_idx = source_to_idx[source]
            
            # Filter packets related to this source
            # Packets received from this source OR sent to this source
            source_packets = window_data[
                ((window_data['src'] == source) & (window_data['sent_or_received'] == 'received')) |
                ((window_data['dst'] == source) & (window_data['sent_or_received'] == 'sent'))
            ]
            
            if len(source_packets) == 0:
                continue
            
            # Packet count
            features[f'pkts_{window_name}_src{src_idx}'] = len(source_packets)
            
            # Average packet size
            features[f'avg_pkt_size_{window_name}_src{src_idx}'] = source_packets['size'].mean()
            
            # Packet size variance
            features[f'pkt_size_var_{window_name}_src{src_idx}'] = source_packets['size'].var() if len(source_packets) > 1 else 0
            
            # SYN packets (TCP flags - checking packet_type or dst_port)
            # Note: The log doesn't have explicit SYN flags, using heuristic
            syn_count = 0  # Would need actual SYN flag in data
            features[f'syn_{window_name}_src{src_idx}'] = syn_count
            
            # ACK packets (TCP flags)
            ack_count = 0  # Would need actual ACK flag in data
            features[f'ack_{window_name}_src{src_idx}'] = ack_count
            
            # TCP packet count
            tcp_count = len(source_packets[source_packets['packet_type'] == 'TCP'])
            features[f'tcp_{window_name}_src{src_idx}'] = tcp_count
            
            # UDP packet count
            udp_count = len(source_packets[source_packets['packet_type'] == 'UDP'])
            features[f'udp_{window_name}_src{src_idx}'] = udp_count
    
    # Queue features for all time windows
    # Queue belongs to the device, so same value is filled for all sources
    for window_name, window_seconds in time_windows.items():
        # Cap the window start at 0 (don't go before the beginning)
        window_start = max(0, current_time - window_seconds)
        window_data = df[(df['timestamp'] >= window_start) & (df['timestamp'] < current_time)]
        
        # Calculate queue statistics for this time window (device-level, not per-source)
        if len(window_data) > 0:
            queue_avg = window_data['queue_full_percentage'].mean()
            queue_var = window_data['queue_full_percentage'].var() if len(window_data) > 1 else 0
        else:
            queue_avg = 0
            queue_var = 0
        
        # Fill the same value for all sources
        for src_idx in range(len(sources)):
            features[f'queue_avg_{window_name}_src{src_idx}'] = queue_avg
            features[f'queue_var_{window_name}_src{src_idx}'] = queue_var
    
    return features


def detect_attack(df, current_time):
    """Detect if there's an attack at the current timestamp
    
    Simple heuristic: Attack if infected=True or abnormal traffic patterns
    """
    # Check in a small window around current time
    window_start = current_time - 1
    window_data = df[(df['timestamp'] >= window_start) & (df['timestamp'] <= current_time)]
    
    # Check if device is infected
    if window_data['infected'].any():
        return 1
    
    # Check for abnormal patterns (high packet rate, large queue)
    if len(window_data) > 0:
        # High packet rate in 1 second
        if len(window_data) > 100:
            return 1
        
        # High queue percentage
        if window_data['queue_full_percentage'].max() > 40:
            return 1
    
    return 0


def process_log_file(log_file_path, num_sources=6):
    """Process a single log file and generate datapoints"""
    print(f"Processing {log_file_path}...")
    
    # Read log file
    df = pd.read_csv(log_file_path)
    
    # Get device name from first row
    device_name = df['device_name'].iloc[0]
    
    # Identify sources
    sources = identify_sources(df, device_name)
    print(f"  Found {len(sources)} sources: {sources}")
    
    # Ensure we have exactly num_sources
    if len(sources) < num_sources:
        # Pad with dummy sources
        for i in range(len(sources), num_sources):
            sources.append(f'dummy_src_{i}')
    elif len(sources) > num_sources:
        # Take only first num_sources
        sources = sources[:num_sources]
    
    # Define time windows in seconds
    time_windows = {
        '60s': 60,
        '30s': 30,
        '15s': 15,
        '5s': 5
    }
    
    # Generate timestamps every 0.5 seconds
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    
    # Start from 0.5 seconds (first datapoint after 0.5s has passed)
    # For early timestamps, we use all available data from 0 up to that point
    start_time = 0.5
    timestamps = np.arange(start_time, max_time, 0.5)
    
    # Get feature column names
    feature_cols = get_feature_columns(num_sources)
    
    # Initialize datapoints
    datapoints = []
    
    for ts in timestamps:
        # Calculate features
        features = calculate_features_for_timestamp(df, ts, sources, device_name, time_windows)
        
        # Detect attack
        label = detect_attack(df, ts)
        
        # Create datapoint
        datapoint = {
            'timestamp': ts,
            **features,
            'label': label
        }
        
        datapoints.append(datapoint)
    
    # Create DataFrame
    result_df = pd.DataFrame(datapoints)
    
    # Ensure all feature columns exist (fill missing with 0)
    for col in feature_cols:
        if col not in result_df.columns:
            result_df[col] = 0
    
    # Reorder columns
    column_order = ['timestamp'] + feature_cols + ['label']
    result_df = result_df[column_order]
    
    # Fill NaN with 0
    result_df = result_df.fillna(0)
    
    return result_df


def main():
    """Process all peer log files"""
    logs_dir = Path('logs')
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    num_sources = 6  # Total number of peers + server
    
    # Process each peer log file
    for i in range(1, 7):  # peer_1 to peer_6
        log_file = logs_dir / f'peer_{i}_logs.csv'
        
        if not log_file.exists():
            print(f"Warning: {log_file} not found, skipping...")
            continue
        
        # Process the log file
        datapoints_df = process_log_file(log_file, num_sources)
        
        # Save to CSV
        output_file = output_dir / f'peer_{i}_datapts.csv'
        datapoints_df.to_csv(output_file, index=False)
        
        print(f"  Saved {len(datapoints_df)} datapoints to {output_file}")
        print(f"  Shape: {datapoints_df.shape}")
        print()


if __name__ == '__main__':
    main()
