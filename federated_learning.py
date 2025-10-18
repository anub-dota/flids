import numpy as np
import pandas as pd
from utils import load_device_data_for_interval, preprocess_data, create_validation_dataset
from models import LightweightModel, HeavyweightModel, GlobalModel

class FederatedIntrustionDetection:
    """Main federated learning system"""
    
    def __init__(self):
        # Initialize local models
        self.lightweight_models = [LightweightModel(i) for i in range(3)]
        self.heavyweight_models = [HeavyweightModel(i) for i in range(3, 6)]
        
        # Initialize global models
        self.global_lightweight = GlobalModel('lightweight')
        self.global_heavyweight = GlobalModel('heavyweight')
        
        # Load validation data for knowledge transfer
        self.validation_data = self.load_or_create_validation_data()
        
        # Training history
        self.training_history = {
            'round': [],
            'lightweight_accuracy': [],
            'heavyweight_accuracy': [],
            'transfer_accuracy_light_to_heavy': [],
            'transfer_accuracy_heavy_to_light': []
        }
    
    def load_or_create_validation_data(self):
        """Load or create validation dataset"""
        try:
            validation_data = pd.read_csv('data/validation_data.csv')
            print("Loaded existing validation dataset")
        except FileNotFoundError:
            print("Creating validation dataset...")
            validation_data = create_validation_dataset()
        
        return validation_data
    
    def train_local_models(self, round_num):
        """Train local models for one round"""
        start_time = round_num * 5
        end_time = start_time + 5
        
        print(f"Training round {round_num + 1}: Time {start_time}-{end_time}s")
        
        # Train lightweight models (devices 0, 1, 2)
        for i, model in enumerate(self.lightweight_models):
            device_data = load_device_data_for_interval(i, start_time, end_time)
            if len(device_data) > 0:
                X, y = preprocess_data(device_data)
                model.partial_fit(X, y)
                print(f"  Lightweight device {i}: trained on {len(X)} samples")
        
        # Train heavyweight models (devices 3, 4, 5)
        for i, model in enumerate(self.heavyweight_models):
            device_data = load_device_data_for_interval(i + 3, start_time, end_time)
            if len(device_data) > 0:
                X, y = preprocess_data(device_data)
                model.partial_fit(X, y)
                print(f"  Heavyweight device {i + 3}: trained on {len(X)} samples")
    
    def federated_averaging(self):
        """Perform federated averaging"""
        print("  Performing federated averaging...")
        
        # Aggregate lightweight models
        lightweight_weights = [model.get_weights() for model in self.lightweight_models]
        self.global_lightweight.aggregate_weights(lightweight_weights)
        
        # Aggregate heavyweight models
        heavyweight_weights = [model.get_weights() for model in self.heavyweight_models]
        self.global_heavyweight.aggregate_weights(heavyweight_weights)
    
    def teacher_student_knowledge_transfer(self):
        """Perform bidirectional knowledge transfer"""
        print("  Performing knowledge transfer...")
        
        if not self.global_lightweight.is_fitted or not self.global_heavyweight.is_fitted:
            print("    Skipping knowledge transfer - models not ready")
            return
        
        X_val, y_val = preprocess_data(self.validation_data)
        
        # Light -> Heavy: Lightweight model teaches heavyweight model
        light_predictions = self.global_lightweight.predict_for_knowledge_transfer(X_val)
        # Heavy -> Light: Heavyweight model teaches lightweight model
        heavy_predictions = self.global_heavyweight.predict_for_knowledge_transfer(X_val)
        
        # Train simultaneously to avoid update conflicts
        self.global_heavyweight.fit_from_teacher(X_val, light_predictions)
        self.global_lightweight.fit_from_teacher(X_val, heavy_predictions)
    
    def update_local_models(self):
        """Update local models with global weights (70% local + 30% global)"""
        print("  Updating local models...")
        
        global_light_weights = self.global_lightweight.get_weights()
        global_heavy_weights = self.global_heavyweight.get_weights()
        
        # Update lightweight models
        for model in self.lightweight_models:
            local_weights = model.get_weights()
            if local_weights is not None and global_light_weights is not None:
                # 70% local + 30% global
                updated_weights = {}
                for key in local_weights:
                    updated_weights[key] = 0.7 * local_weights[key] + 0.3 * global_light_weights[key]
                model.set_weights(updated_weights)
        
        # Update heavyweight models  
        for model in self.heavyweight_models:
            local_weights = model.get_weights()
            if local_weights is not None and global_heavy_weights is not None:
                # 70% local + 30% global
                updated_weights = {}
                for key in local_weights:
                    if key in ['coefs', 'intercepts']:
                        updated_weights[key] = []
                        for layer_idx in range(len(local_weights[key])):
                            layer_update = 0.7 * local_weights[key][layer_idx] + 0.3 * global_heavy_weights[key][layer_idx]
                            updated_weights[key].append(layer_update)
                model.set_weights(updated_weights)
    
    def run_simulation(self, total_seconds=1000):
        """Run the complete federated learning simulation"""
        print(f"Starting federated learning simulation for {total_seconds} seconds...")
        print(f"Total rounds: {total_seconds // 5}")
        print("=" * 50)
        
        for round_num in range(total_seconds // 5):
            # 1. Train local models on new data
            self.train_local_models(round_num)
            
            # 2. Perform federated averaging
            self.federated_averaging()
            
            # 3. Knowledge transfer between global models
            self.teacher_student_knowledge_transfer()

            # 4. Update local models
            self.update_local_models()
            
            print("-" * 30)
        
        print("Simulation completed!")
        return self.training_history
