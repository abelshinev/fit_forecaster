"""
Model Architecture for Fit Forecaster
Implements ensemble approach with XGBoost, Prophet, and Neural Network
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import joblib
import os

# ML Libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import keras.optimizers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Time Series
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

logger = logging.getLogger(__name__)

class BaseModel:
    """Base class for all models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
        self.model = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model and return metrics"""
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError(f"{self.model_name} model is not trained")
        raise NotImplementedError
        
    def save(self, filepath: str):
        """Save the model"""
        if self.model is not None:
            joblib.dump(self.model, filepath)
            logger.info(f"Saved {self.model_name} to {filepath}")
            
    def load(self, filepath: str):
        """Load the model"""
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            self.is_trained = True
            logger.info(f"Loaded {self.model_name} from {filepath}")
        else:
            logger.warning(f"Model file {filepath} not found")

class RandomForestModel(BaseModel):
    def __init__(self, **params):
        super().__init__("RandomForest")
        self.params = {'n_estimators': 100, 'random_state': 42, **params}
        self.model = RandomForestRegressor(**self.params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        self.model.fit(X, y)
        self.is_trained = True
        y_pred = self.model.predict(X)
        return {
            'mae': float(mean_absolute_error(y, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'r2': float(r2_score(y, y_pred))
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class LinearRegressionModel(BaseModel):
    def __init__(self, **params):
        super().__init__("LinearRegression")
        self.model = LinearRegression(**params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        self.model.fit(X, y)
        self.is_trained = True
        y_pred = self.model.predict(X)
        return {
            'mae': float(mean_absolute_error(y, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'r2': float(r2_score(y, y_pred))
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

class XGBoostModel(BaseModel):
    """XGBoost model for gym attendance prediction"""
    
    def __init__(self, **params):
        super().__init__("XGBoost")
        self.params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            **params
        }
        self.model = xgb.XGBRegressor(**self.params)
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        # Handle categorical features
        if 'split_id' in X.columns:
            X_encoded = pd.get_dummies(X, columns=['split_id'])
        else:
            X_encoded = X
        self.model.fit(X_encoded, y)
        self.is_trained = True
        
        # Make predictions for metrics
        y_pred = self.model.predict(X_encoded)
        
        metrics = {
            'mae': float(mean_absolute_error(y, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'r2': float(r2_score(y, y_pred))
        }
        
        logger.info(f"XGBoost training complete. MAE: {metrics['mae']:.3f}")
        return metrics
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost"""
        if 'split_id' in X.columns:
            X_encoded = pd.get_dummies(X, columns=['split_id'])
        else:
            X_encoded = X
        return self.model.predict(X_encoded)

class ProphetModel(BaseModel):
    """Prophet model for time series forecasting"""
    
    def __init__(self, **params):
        super().__init__("Prophet")
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install with: pip install prophet")
            
        self.params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'multiplicative',
            **params
        }
        self.models = {}  # One model per split
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train Prophet models for each split"""
        logger.info("Training Prophet models...")
        
        # Prepare data for Prophet
        X['ds'] = pd.to_datetime(X['date']) + pd.to_timedelta(X['time_slot'] * 30, unit='minutes')
        X['y'] = y
        
        # Train separate model for each split
        split_metrics = {}
        for split_id in X['split_id'].unique():
            split_data = pd.DataFrame(X[X['split_id'] == split_id]).sort_values(['date', 'time_slot'])
            
            if len(split_data) > 10:  # Need minimum data for Prophet
                model = Prophet(**self.params)
                model.fit(split_data)
                self.models[split_id] = model
                
                # Calculate metrics for this split
                forecast = model.predict(split_data[['ds']])
                split_metrics[split_id] = {
                    'mae': float(mean_absolute_error(split_data['y'], forecast['yhat'])),
                    'rmse': float(np.sqrt(mean_squared_error(split_data['y'], forecast['yhat']))),
                    'r2': float(r2_score(split_data['y'], forecast['yhat']))
                }
        
        self.is_trained = len(self.models) > 0
        
        # Aggregate metrics
        if split_metrics:
            avg_metrics = {
                'mae': float(np.mean([m['mae'] for m in split_metrics.values()])),
                'rmse': float(np.mean([m['rmse'] for m in split_metrics.values()])),
                'r2': float(np.mean([m['r2'] for m in split_metrics.values()]))
            }
            logger.info(f"Prophet training complete. Average MAE: {avg_metrics['mae']:.3f}")
            return avg_metrics
        else:
            logger.warning("No Prophet models could be trained")
            return {'mae': float('inf'), 'rmse': float('inf'), 'r2': 0.0}
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Prophet"""
        predictions = []
        
        for _, row in X.iterrows():
            split_id = row['split_id']
            ds = pd.to_datetime(row['date']) + pd.to_timedelta(row['time_slot'] * 30, unit='minutes')
            
            if split_id in self.models:
                forecast = self.models[split_id].predict(pd.DataFrame({'ds': [ds]}))
                predictions.append(forecast['yhat'].iloc[0])
            else:
                predictions.append(0.0)  # Default for unknown splits
                
        return np.array(predictions)

class LSTMModel(BaseModel):
    """LSTM model for sequence prediction"""
    
    def __init__(self, sequence_length: int = 24, **params):
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.params = params
        
    def create_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        # Group by split_id and time_slot
        sequences_X = []
        sequences_y = []
        
        for split_id in X['split_id'].unique():
            split_data = pd.DataFrame(X[X['split_id'] == split_id]).sort_values(['date', 'time_slot'])
            split_y = pd.Series(y[X['split_id'] == split_id]).sort_index()
            
            # Create sequences
            for i in range(self.sequence_length, len(split_data)):
                seq_X = split_data.iloc[i-self.sequence_length:i][
                    ['visitor_count', 'avg_duration', 'day_of_week', 'month', 
                     'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
                ].values
                seq_y = split_y.iloc[i]
                
                sequences_X.append(seq_X)
                sequences_y.append(seq_y)
        
        return np.array(sequences_X), np.array(sequences_y)
        
    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001), #type: ignore
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train LSTM model"""
        logger.info("Training LSTM model...")
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        if len(X_seq) == 0:
            logger.warning("No sequences could be created for LSTM")
            return {'mae': float('inf'), 'rmse': float('inf'), 'r2': 0.0}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        
        # Build and train model
        self.model = self.build_model((X_seq.shape[1], X_seq.shape[2]))
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_scaled, y_seq,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0 #type: ignore
        )
        
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.model.predict(X_scaled)
        metrics = {
            'mae': float(mean_absolute_error(y_seq, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_seq, y_pred))),
            'r2': float(r2_score(y_seq, y_pred))
        }
        
        logger.info(f"LSTM training complete. MAE: {metrics['mae']:.3f}")
        return metrics
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM"""
        # This is a simplified prediction - in practice, you'd need to create sequences
        # For now, return zeros as placeholder
        return np.zeros(len(X))

class EnsembleModel:
    """Ensemble model combining multiple base models"""
    
    def __init__(self, models, weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.is_trained = False
        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
            
    def train(self, X, y) -> Dict[str, float]:
        """Train all models in the ensemble"""
        logger.info("Training ensemble model...")
        
        model_metrics = {}
        for model in self.models:
            try:
                metrics = model.train(X, y)
                model_metrics[model.model_name] = metrics
            except Exception as e:
                logger.error(f"Error training {getattr(model, 'model_name', str(model))}: {e}")
                model_metrics[getattr(model, 'model_name', str(model))] = {'mae': float('inf'), 'rmse': float('inf'), 'r2': 0.0}
        
        # Mark as trained if at least one model is trained
        self.is_trained = any(getattr(model, 'is_trained', False) for model in self.models)
        if not all(getattr(model, 'is_trained', False) for model in self.models):
            logger.warning("Not all models trained successfully, but at least one is available for prediction.")
        
        # Calculate ensemble metrics
        if self.is_trained:
            ensemble_predictions = self.predict(X)
            ensemble_metrics = {
                'mae': float(mean_absolute_error(y, ensemble_predictions)),
                'rmse': float(np.sqrt(mean_squared_error(y, ensemble_predictions))),
                'r2': float(r2_score(y, ensemble_predictions))
            }
            logger.info(f"Ensemble training complete. MAE: {ensemble_metrics['mae']:.3f}")
            return ensemble_metrics
        else:
            logger.warning("No models trained successfully")
            return {'mae': float('inf'), 'rmse': float('inf'), 'r2': 0.0}
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble model is not trained")
            
        predictions = []
        used_weights = []
        for model, weight in zip(self.models, self.weights):
            if getattr(model, 'is_trained', False):
                pred = model.predict(X)
                predictions.append(pred)
                used_weights.append(weight)
            else:
                logger.warning(f"{getattr(model, 'model_name', str(model))} is not trained, skipping")
                
        if not predictions:
            raise ValueError("No trained models available for prediction")
            
        # Weighted average of predictions
        weighted_predictions = np.zeros_like(predictions[0])
        total_weight = 0
        
        for pred, weight in zip(predictions, used_weights):
            weighted_predictions += pred * weight
            total_weight += weight
            
        return weighted_predictions / total_weight
        
    def save_ensemble(self, directory: str):
        """Save all models in the ensemble"""
        os.makedirs(directory, exist_ok=True)
        
        for model in self.models:
            if model.is_trained:
                model_path = os.path.join(directory, f"{model.model_name.lower()}.joblib")
                model.save(model_path)
                
        # Save ensemble metadata
        ensemble_metadata = {
            'weights': self.weights,
            'model_names': [model.model_name for model in self.models]
        }
        joblib.dump(ensemble_metadata, os.path.join(directory, 'ensemble_metadata.joblib'))
        
    def load_ensemble(self, directory: str):
        """Load all models in the ensemble"""
        # Load ensemble metadata
        metadata_path = os.path.join(directory, 'ensemble_metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.weights = metadata['weights']
            
            # Load individual models
            for model_name in metadata['model_names']:
                model_path = os.path.join(directory, f"{model_name.lower()}.joblib")
                for model in self.models:
                    if model.model_name == model_name:
                        model.load(model_path)
                        break
                        
            self.is_trained = any(getattr(model, 'is_trained', False) for model in self.models)

class SplitSpecificModels:
    """Manages separate models for each workout split"""
    
    def __init__(self):
        self.split_models = {}
        self.global_model = None
        
    def create_split_model(self, split_id: str) -> EnsembleModel:
        """Create ensemble model for a specific split"""
        models = [
            XGBoostModel(),
            RandomForestModel(),
            LinearRegressionModel()
        ]
        if PROPHET_AVAILABLE:
            models.append(ProphetModel())
        return EnsembleModel(models)
        
    def train_split_models(self, data, target_col: str = 'visitor_count'):
        """Train models for each split"""
        logger.info("Training split-specific models...")
        if 'split_id' not in data.columns:
            logger.warning("No 'split_id' column in data. Skipping training.")
            return
        for split_id in data['split_id'].unique():
            split_data = data[data['split_id'] == split_id].copy()
            if len(split_data) > 50:
                logger.info(f"Training model for split: {split_id}")
                feature_cols = [
                    'time_slot', 'day_of_week', 'month', 'is_weekend',
                    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                    'prev_day_attendance', 'prev_week_attendance',
                    'rolling_3day_avg', 'rolling_7day_avg'
                ]
                X = pd.DataFrame(split_data[feature_cols]).fillna(0)
                y = split_data[target_col]
                model = self.create_split_model(split_id)
                metrics = model.train(X, y)
                self.split_models[split_id] = {
                    'model': model,
                    'metrics': metrics,
                    'feature_cols': feature_cols
                }
                logger.info(f"Split {split_id} - MAE: {metrics['mae']:.3f}")
            else:
                logger.warning(f"Insufficient data for split {split_id}: {len(split_data)} records")
                
    def predict_split(self, split_id: str, X: pd.DataFrame) -> np.ndarray:
        """Predict for a specific split"""
        if split_id not in self.split_models:
            logger.warning(f"No model available for split {split_id}")
            return np.zeros(len(X))
            
        model_info = self.split_models[split_id]
        feature_cols = model_info['feature_cols']
        
        X_features = pd.DataFrame(X[feature_cols]).fillna(0)
        return model_info['model'].predict(X_features)
        
    def predict_all_splits(self, data) -> pd.DataFrame:
        """Predict for all splits in the data"""
        predictions = []
        
        for split_id in data['split_id'].unique():
            split_data = data[data['split_id'] == split_id].copy()
            split_preds = self.predict_split(split_id, split_data)
            
            split_data['predicted_visitors'] = split_preds
            predictions.append(split_data)
            
        return pd.concat(predictions, ignore_index=True) 