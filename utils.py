import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_device_data_for_interval(device_id, start_time, end_time):
    """Load data for a specific device and time interval"""
    df = pd.read_csv(f'data/device_{device_id + 1}.csv')
    
    # Filter for the 5-second interval (should get 10 rows - every 0.5s)
    interval_data = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]
    
    return interval_data

def create_validation_dataset():
    """Create a constant validation dataset from all devices"""
    all_data = []
    
    for device_id in range(6):
        df = pd.read_csv(f'data/device_{device_id + 1}.csv')
        # Sample 5% of data from each device
        sampled_data = df.sample(frac=0.05, random_state=42)
        all_data.append(sampled_data)
    
    validation_data = pd.concat(all_data, ignore_index=True)
    validation_data.to_csv('data/validation_data.csv', index=False)
    return validation_data

def get_feature_columns(num_sources=6):
    """Return the feature column names
    
    Args:
        num_sources: Number of sources to generate features for. 
                     If None, generates features without source aggregation (legacy behavior).
    """
    base_features = ['pkts', 'avg_pkt_size', 'pkt_size_var', 'syn', 'ack', 'tcp', 'udp']
    time_windows = ['60s', '30s', '15s', '5s']
    
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
