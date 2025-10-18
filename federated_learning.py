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
            'time_start': [],
            'time_end': [],
            'lightweight_accuracy': [],
            'heavyweight_accuracy': [],
            'global_lightweight_accuracy': [],
            'global_heavyweight_accuracy': [],
            'lightweight_precision': [],
            'heavyweight_precision': [],
            'lightweight_recall': [],
            'heavyweight_recall': [],
            'lightweight_f1': [],
            'heavyweight_f1': [],
            'transfer_accuracy_light_to_heavy': [],
            'transfer_accuracy_heavy_to_light': [],
            'num_lightweight_samples': [],
            'num_heavyweight_samples': []
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
        
        lightweight_samples = 0
        heavyweight_samples = 0
        
        # Train lightweight models (devices 0, 1, 2)
        for i, model in enumerate(self.lightweight_models):
            device_data = load_device_data_for_interval(i, start_time, end_time)
            if len(device_data) > 0:
                X, y = preprocess_data(device_data)
                if model.is_fitted:
                    model.partial_fit(X, y)
                else:
                    model.fit(X, y)
                lightweight_samples += len(X)
                print(f"  Lightweight device {i}: trained on {len(X)} samples")
        
        # Train heavyweight models (devices 3, 4, 5)
        for i, model in enumerate(self.heavyweight_models):
            device_data = load_device_data_for_interval(i + 3, start_time, end_time)
            if len(device_data) > 0:
                X, y = preprocess_data(device_data)
                if model.is_fitted:
                    model.partial_fit(X, y)
                else:
                    model.fit(X, y)
                heavyweight_samples += len(X)
                print(f"  Heavyweight device {i + 3}: trained on {len(X)} samples")
        
        return start_time, end_time, lightweight_samples, heavyweight_samples
    
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
    
    def evaluate_models(self, round_num):
        """Evaluate all models on validation data (does not affect training)"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        X_val, y_val = preprocess_data(self.validation_data)
        
        # Evaluate local lightweight models (average across devices)
        lightweight_preds = []
        for model in self.lightweight_models:
            if model.is_fitted:
                preds = model.predict(X_val)
                lightweight_preds.append(preds)
        
        if lightweight_preds:
            # Average predictions (ensemble)
            avg_light_preds = np.round(np.mean(lightweight_preds, axis=0)).astype(int)
            light_acc = accuracy_score(y_val, avg_light_preds)
            light_prec = precision_score(y_val, avg_light_preds, zero_division=0)
            light_rec = recall_score(y_val, avg_light_preds, zero_division=0)
            light_f1 = f1_score(y_val, avg_light_preds, zero_division=0)
        else:
            light_acc = light_prec = light_rec = light_f1 = 0.0
        
        # Evaluate local heavyweight models (average across devices)
        heavyweight_preds = []
        for model in self.heavyweight_models:
            if model.is_fitted:
                preds = model.predict(X_val)
                heavyweight_preds.append(preds)
        
        if heavyweight_preds:
            # Average predictions (ensemble)
            avg_heavy_preds = np.round(np.mean(heavyweight_preds, axis=0)).astype(int)
            heavy_acc = accuracy_score(y_val, avg_heavy_preds)
            heavy_prec = precision_score(y_val, avg_heavy_preds, zero_division=0)
            heavy_rec = recall_score(y_val, avg_heavy_preds, zero_division=0)
            heavy_f1 = f1_score(y_val, avg_heavy_preds, zero_division=0)
        else:
            heavy_acc = heavy_prec = heavy_rec = heavy_f1 = 0.0
        
        # Evaluate global models
        if self.global_lightweight.is_fitted:
            global_light_preds = self.global_lightweight.predict_for_knowledge_transfer(X_val)
            global_light_acc = accuracy_score(y_val, global_light_preds)
        else:
            global_light_acc = 0.0
        
        if self.global_heavyweight.is_fitted:
            global_heavy_preds = self.global_heavyweight.predict_for_knowledge_transfer(X_val)
            global_heavy_acc = accuracy_score(y_val, global_heavy_preds)
        else:
            global_heavy_acc = 0.0
        
        # Evaluate knowledge transfer effectiveness
        if self.global_lightweight.is_fitted and self.global_heavyweight.is_fitted:
            light_predictions = self.global_lightweight.predict_for_knowledge_transfer(X_val)
            heavy_predictions = self.global_heavyweight.predict_for_knowledge_transfer(X_val)
            
            # How well does light model's prediction match heavy model's expertise
            transfer_light_to_heavy = accuracy_score(heavy_predictions, light_predictions)
            # How well does heavy model's prediction match light model's expertise
            transfer_heavy_to_light = accuracy_score(light_predictions, heavy_predictions)
        else:
            transfer_light_to_heavy = 0.0
            transfer_heavy_to_light = 0.0
        
        return {
            'lightweight_accuracy': light_acc,
            'heavyweight_accuracy': heavy_acc,
            'global_lightweight_accuracy': global_light_acc,
            'global_heavyweight_accuracy': global_heavy_acc,
            'lightweight_precision': light_prec,
            'heavyweight_precision': heavy_prec,
            'lightweight_recall': light_rec,
            'heavyweight_recall': heavy_rec,
            'lightweight_f1': light_f1,
            'heavyweight_f1': heavy_f1,
            'transfer_accuracy_light_to_heavy': transfer_light_to_heavy,
            'transfer_accuracy_heavy_to_light': transfer_heavy_to_light
        }
    
    def record_history(self, round_num, start_time, end_time, num_light_samples, num_heavy_samples, metrics):
        """Record training history (does not affect training)"""
        self.training_history['round'].append(round_num + 1)
        self.training_history['time_start'].append(start_time)
        self.training_history['time_end'].append(end_time)
        self.training_history['num_lightweight_samples'].append(num_light_samples)
        self.training_history['num_heavyweight_samples'].append(num_heavy_samples)
        
        for key, value in metrics.items():
            if key in self.training_history:
                self.training_history[key].append(value)
    
    def run_simulation(self, total_seconds=1000):
        """Run the complete federated learning simulation"""
        print(f"Starting federated learning simulation for {total_seconds} seconds...")
        print(f"Total rounds: {total_seconds // 5}")
        print("=" * 50)
        
        for round_num in range(total_seconds // 5):
            # 1. Train local models on new data
            start_time, end_time, num_light_samples, num_heavy_samples = self.train_local_models(round_num)
            
            # 2. Perform federated averaging
            self.federated_averaging()
            
            # 3. Knowledge transfer between global models
            self.teacher_student_knowledge_transfer()

            # 4. Update local models
            self.update_local_models()
            
            # 5. Evaluate models (does NOT affect training)
            metrics = self.evaluate_models(round_num)
            
            # 6. Record history (does NOT affect training)
            self.record_history(round_num, start_time, end_time, num_light_samples, num_heavy_samples, metrics)
            
            # Print metrics
            print(f"  Metrics: Light Acc={metrics['lightweight_accuracy']:.3f}, Heavy Acc={metrics['heavyweight_accuracy']:.3f}")
            
            print("-" * 30)
        
        print("Simulation completed!")
        return self.training_history
