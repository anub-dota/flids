import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import copy

class LightweightModel:
    """SGDClassifier model for lightweight devices with true incremental learning"""
    
    def __init__(self, device_id):
        self.device_id = device_id
        # Use SGDClassifier with log loss for logistic regression-like behavior
        self.model = SGDClassifier(
            loss='log_loss',  # logistic regression loss
            penalty='l2',
            alpha=0.0001,
            random_state=42,
            max_iter=6000,
            tol=1e-3,
            warm_start=False  # We'll use partial_fit instead
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def partial_fit(self, X, y):
        """Train incrementally on new data - true online learning"""
        classes = np.array([0, 1])  # Binary classification
        
        if not self.is_fitted:
            # First training - fit scaler and initialize model
            X_scaled = self.scaler.fit_transform(X)
            self.model.partial_fit(X_scaled, y, classes=classes)
            self.is_fitted = True
        else:
            # True incremental training - model learns from new data without forgetting
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y, classes=classes)
    
    def fit(self, X, y):
        """Train model on initial data"""
        classes = np.array([0, 1])  # Binary classification
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.partial_fit(X_scaled, y,classes=classes)
        self.is_fitted = True

    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_weights(self):
        """Get model weights for federated averaging"""
        if not self.is_fitted:
            return None
        return {
            'coef': self.model.coef_.copy(),
            'intercept': self.model.intercept_.copy()
        }
    
    def set_weights(self, weights):
        """Set model weights from federated averaging"""
        if weights is not None and self.is_fitted:
            self.model.coef_ = weights['coef'].copy()
            self.model.intercept_ = weights['intercept'].copy()
        else:
            raise ValueError("Weights cannot be set because the model is not fitted or weights are None.")
        
class HeavyweightModel:
    """MLP model for heavyweight devices with true incremental learning"""
    
    def __init__(self, device_id):
        self.device_id = device_id
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            random_state=42,
            max_iter=200,
            warm_start=False,  # Enable incremental learning
            early_stopping=False,  # Disable early stopping for incremental learning
            learning_rate_init=0.001,
            alpha=0.0001  # L2 regularization
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def partial_fit(self, X, y):
        """Train incrementally on new data - true online learning"""
        classes = np.array([0, 1])
        if not self.is_fitted:
            # First training - fit scaler and initialize model
            X_scaled = self.scaler.fit_transform(X)
            self.model.partial_fit(X_scaled, y, classes=classes)
            self.is_fitted = True
        else:
            # True incremental training - warm_start allows model to continue learning
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y, classes=classes)
    
    def fit(self, X, y):
        """Train model on initial data"""
        classes = np.array([0, 1])  # Binary classification
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.partial_fit(X_scaled, y,classes=classes)
        self.is_fitted = True
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_weights(self):
        """Get model weights for federated averaging"""
        if not self.is_fitted:
            return None
        return {
            'coefs': [coef.copy() for coef in self.model.coefs_],
            'intercepts': [intercept.copy() for intercept in self.model.intercepts_]
        }
    
    def set_weights(self, weights):
        """Set model weights from federated averaging"""
        if weights is not None and self.is_fitted:
            self.model.coefs_ = [coef.copy() for coef in weights['coefs']]
            self.model.intercepts_ = [intercept.copy() for intercept in weights['intercepts']]

class GlobalModel:
    """Global model for federated averaging"""
    
    def __init__(self, model_type='lightweight'):
        self.model_type = model_type
        if model_type == 'lightweight':
            self.model = SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=0.0001,
                random_state=42,
                max_iter=6000,
                tol=1e-3,
                warm_start=False
            )
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                random_state=42,
                max_iter=200,
                warm_start=False
            )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def aggregate_weights(self, local_weights_list):
        """Perform federated averaging"""
        if not local_weights_list or all(w is None for w in local_weights_list):
            return
        
        # Filter out None weights
        valid_weights = [w for w in local_weights_list if w is not None]
        if not valid_weights:
            return
        
        if self.model_type == 'lightweight':
            # Average logistic regression weights
            avg_coef = np.mean([w['coef'] for w in valid_weights], axis=0)
            avg_intercept = np.mean([w['intercept'] for w in valid_weights], axis=0)
            
            if not self.is_fitted:
                # Initialize the model with dummy data first
                dummy_X = np.random.random((10, avg_coef.shape[1]))
                dummy_y = np.random.randint(0, 2, 10)
                self.scaler.fit(dummy_X)
                self.model.fit(self.scaler.transform(dummy_X), dummy_y)
                self.is_fitted = True
            
            self.model.coef_ = avg_coef
            self.model.intercept_ = avg_intercept
            
        else:
            # Average MLP weights
            avg_coefs = []
            avg_intercepts = []
            
            # Average each layer
            for layer_idx in range(len(valid_weights[0]['coefs'])):
                layer_coefs = [w['coefs'][layer_idx] for w in valid_weights]
                avg_coefs.append(np.mean(layer_coefs, axis=0))
            
            for layer_idx in range(len(valid_weights[0]['intercepts'])):
                layer_intercepts = [w['intercepts'][layer_idx] for w in valid_weights]
                avg_intercepts.append(np.mean(layer_intercepts, axis=0))
            
            if not self.is_fitted:
                # Initialize the model with dummy data first
                # This dummy training is needed to create the internal structure of the MLP
                dummy_X = np.random.random((10, avg_coefs[0].shape[0]))
                dummy_y = np.random.randint(0, 2, 10)
                self.scaler.fit(dummy_X)
                self.model.fit(self.scaler.transform(dummy_X), dummy_y)
                self.is_fitted = True
            
            self.model.coefs_ = avg_coefs
            self.model.intercepts_ = avg_intercepts
    
    def get_weights(self):
        """Get global model weights"""
        if not self.is_fitted:
            return None
        
        if self.model_type == 'lightweight':
            return {
                'coef': self.model.coef_.copy(),
                'intercept': self.model.intercept_.copy()
            }
        else:
            return {
                'coefs': [coef.copy() for coef in self.model.coefs_],
                'intercepts': [intercept.copy() for intercept in self.model.intercepts_]
            }
    
    def predict_for_knowledge_transfer(self, X):
        """Generate labels for knowledge transfer"""
        if not self.is_fitted:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def fit_from_teacher(self, X, teacher_labels):
        """Train this model using teacher model labels"""
        
        X_scaled = self.scaler.fit_transform(X) if not self.is_fitted else self.scaler.transform(X)
        
        if self.model_type == 'lightweight':
            # SGDClassifier supports partial_fit with classes parameter
            classes = np.array([0, 1])
            if not self.is_fitted:
                self.model.partial_fit(X_scaled, teacher_labels, classes=classes)
                self.is_fitted = True
            else:
                self.model.partial_fit(X_scaled, teacher_labels)
        else:
            # MLPClassifier with warm_start - use fit instead
            self.model.partial_fit(X_scaled, teacher_labels)
            self.is_fitted = True
