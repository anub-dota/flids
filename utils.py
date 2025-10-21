import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_device_data_for_interval(device_id, start_time, end_time):
    """Load data for a specific device and time interval based on row indices
    
    Args:
        device_id: Device ID (0-5 for peer_1 to peer_6)
        start_time: Start timestamp
        end_time: End timestamp
    """
    # Map device_id to peer file (device 0 -> peer_1, device 1 -> peer_2, etc.)
    df = pd.read_csv(f'shuffled_data/peer_{device_id + 1}_datapts.csv')

    # Calculate row indices based on start_time and end_time
    start_idx = int(start_time * 2)
    end_idx = int(end_time * 2)

    # Select rows based on calculated indices
    interval_data = df.iloc[start_idx:end_idx]

    return interval_data

def create_validation_dataset():
    """Create a constant validation dataset from all devices"""
    all_data = []
    
    for device_id in range(6):
        df = pd.read_csv(f'shuffled_data/peer_{device_id + 1}_datapts.csv')
        # Sample 10% of data from each device
        sampled_data = df.sample(frac=0.1, random_state=42)
        all_data.append(sampled_data)
    
    validation_data = pd.concat(all_data, ignore_index=True)
    validation_data.to_csv('shuffled_data/validation_data.csv', index=False)
    return validation_data

def get_feature_columns(num_sources=6):
    """Return the feature column names
    
    Args:
        num_sources: Number of sources to generate features for. 
                     If None, generates features without source aggregation (legacy behavior).
    """
    base_features = ['pkts', 'avg_pkt_size', 'pkt_size_var',  'tcp', 'udp']
    time_windows = [ '30s', '10s', '5s']
    
    feature_cols = []
    
    # New behavior: features aggregated per source
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

def preprocess_data(data):
    """Preprocess data for training"""
    feature_cols = get_feature_columns()
    
    X = data[feature_cols].fillna(0)  # Handle any missing values
    y = data['label']
    
    return X.values, y.values
