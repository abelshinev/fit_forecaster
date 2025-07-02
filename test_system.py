"""
Comprehensive Testing Script for Fit Forecaster
Demonstrates how to test all components of the overhauled system
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import DataProcessor
from src.models import SplitSpecificModels, XGBoostModel, EnsembleModel
from src.recommendation_engine import SmartRecommendationEngine, Recommendation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFitForecaster(unittest.TestCase):
    """Test suite for Fit Forecaster system"""
    
    def setUp(self):
        """Set up test environment"""
        self.data_processor = DataProcessor()
        self.split_models = SplitSpecificModels()
        self.recommendation_engine = SmartRecommendationEngine()
        
        # Create test data
        self.test_data = self.create_test_data()
    
    def create_test_data(self):
        """Create synthetic test data"""
        np.random.seed(42)
        
        # Generate test data
        dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')
        time_slots = list(range(48))
        splits = ['Chest', 'Back', 'Legs']
        
        data = []
        for date in dates:
            for time_slot in time_slots:
                for split in splits:
                    # Generate realistic attendance
                    base_attendance = 10
                    
                    # Time effects
                    if 16 <= time_slot <= 20:  # Evening peak
                        base_attendance += 5
                    
                    # Day effects
                    day_of_week = date.weekday()
                    if day_of_week == 1:  # Tuesday
                        base_attendance += 3
                    
                    # Split effects
                    if split == 'Chest':
                        base_attendance += 2
                    
                    attendance = max(0, int(base_attendance + np.random.normal(0, 2)))
                    
                    data.append({
                        'date': date,
                        'time_slot': time_slot,
                        'split_id': split,
                        'visitor_count': attendance,
                        'avg_duration': 60 + np.random.normal(0, 10),
                        'day_of_week': day_of_week,
                        'month': date.month,
                        'is_weekend': 1 if day_of_week >= 5 else 0,
                        'hour_sin': np.sin(2 * np.pi * time_slot / 48),
                        'hour_cos': np.cos(2 * np.pi * time_slot / 48),
                        'day_sin': np.sin(2 * np.pi * day_of_week / 7),
                        'day_cos': np.cos(2 * np.pi * day_of_week / 7),
                        'prev_day_attendance': 0,
                        'prev_week_attendance': 0,
                        'rolling_3day_avg': 0,
                        'rolling_7day_avg': 0
                    })
        
        return pd.DataFrame(data)
    
    def test_data_processor(self):
        """Test data processing functionality"""
        logger.info("Testing data processor...")
        
        # Test split parsing
        test_splits = [
            "chest, back, Core",
            "Cardio, Running",
            "leg, Arm",
            "",
            None
        ]
        
        expected_results = [
            ['Chest', 'Back', 'Core'],
            ['Cardio', 'Cardio'],  # Running maps to Cardio
            ['Legs', 'Arms'],
            ['General'],
            ['General']
        ]
        
        for test_split, expected in zip(test_splits, expected_results):
            result = self.data_processor.parse_workout_splits(test_split)
            self.assertEqual(set(result), set(expected))
        
        # Test time conversion
        test_times = ["08:30:00", "14:15", "22:45:30", None]
        expected_hours = [8.5, 14.25, 22.758, 12.0]
        
        for test_time, expected in zip(test_times, expected_hours):
            result = self.data_processor.convert_time_to_hours(test_time)
            self.assertAlmostEqual(result, expected, places=2)
        
        # Test data quality validation
        quality_report = self.data_processor.validate_data_quality(self.test_data)
        self.assertIn('total_records', quality_report)
        self.assertIn('missing_values', quality_report)
        self.assertIn('outliers', quality_report)
        
        logger.info("✅ Data processor tests passed")
    
    def test_model_training(self):
        """Test model training functionality"""
        logger.info("Testing model training...")
        
        # Train models
        self.split_models.train_split_models(self.test_data)
        
        # Check that models were trained
        self.assertGreater(len(self.split_models.split_models), 0)
        
        # Test predictions
        for split_id, model_info in self.split_models.split_models.items():
            # Check model is trained
            self.assertTrue(model_info['model'].is_trained)
            
            # Check metrics exist
            self.assertIn('mae', model_info['metrics'])
            self.assertIn('r2', model_info['metrics'])
            
            # Test prediction
            test_features = pd.DataFrame({
                'time_slot': [16],
                'day_of_week': [1],
                'month': [1],
                'is_weekend': [0],
                'hour_sin': [np.sin(2 * np.pi * 16 / 48)],
                'hour_cos': [np.cos(2 * np.pi * 16 / 48)],
                'day_sin': [np.sin(2 * np.pi * 1 / 7)],
                'day_cos': [np.cos(2 * np.pi * 1 / 7)],
                'prev_day_attendance': [0],
                'prev_week_attendance': [0],
                'rolling_3day_avg': [0],
                'rolling_7day_avg': [0]
            })
            
            prediction = model_info['model'].predict(test_features)
            self.assertIsInstance(prediction, np.ndarray)
            self.assertEqual(len(prediction), 1)
            self.assertGreaterEqual(prediction[0], 0)  # Non-negative predictions
        
        logger.info("✅ Model training tests passed")
    
    def test_recommendation_engine(self):
        """Test recommendation engine functionality"""
        logger.info("Testing recommendation engine...")
        
        # Initialize quotas
        time_slots = list(range(48))
        splits = ['Chest', 'Back', 'Legs']
        self.recommendation_engine.initialize_quotas(time_slots, splits)
        
        # Test quota initialization
        for split_id in splits:
            self.assertIn(split_id, self.recommendation_engine.recommendation_quotas)
            for time_slot in time_slots:
                self.assertIn(time_slot, self.recommendation_engine.recommendation_quotas[split_id])
        
        # Test congestion calculation
        congestion_score = self.recommendation_engine.calculate_congestion_score('Chest', 16, 10.0)
        self.assertGreaterEqual(congestion_score, 0)
        self.assertLessEqual(congestion_score, 1)
        
        # Test optimal time finding
        optimal_times = self.recommendation_engine.find_optimal_times(
            'Chest', [16, 17, 18], lambda x: np.array([10.0] * len(x))
        )
        self.assertIsInstance(optimal_times, list)
        self.assertGreater(len(optimal_times), 0)
        
        logger.info("✅ Recommendation engine tests passed")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        logger.info("Testing end-to-end workflow...")
        
        # 1. Train models
        self.split_models.train_split_models(self.test_data)
        
        # 2. Initialize recommendation engine
        time_slots = list(range(48))
        splits = list(self.split_models.split_models.keys())
        self.recommendation_engine.initialize_quotas(time_slots, splits)
        
        # 3. Get recommendation
        if 'Chest' in self.split_models.split_models:
            prediction_model = self.split_models.split_models['Chest']['model']
            
            recommendation = self.recommendation_engine.get_recommendation(
                user_id=1,
                split_id='Chest',
                preferred_times=[16, 17, 18],
                prediction_model=prediction_model
            )
            
            # Validate recommendation
            self.assertIsInstance(recommendation, Recommendation)
            self.assertEqual(recommendation.split_id, 'Chest')
            self.assertGreaterEqual(recommendation.recommended_time_slot, 0)
            self.assertLess(recommendation.recommended_time_slot, 48)
            self.assertGreaterEqual(recommendation.confidence_score, 0)
            self.assertLessEqual(recommendation.confidence_score, 1)
            self.assertIsInstance(recommendation.alternative_times, list)
            self.assertIsInstance(recommendation.reason, str)
        
        logger.info("✅ End-to-end workflow tests passed")
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        logger.info("Testing model persistence...")
        
        # Train a model
        self.split_models.train_split_models(self.test_data)
        
        # Save models
        test_output_dir = "test_models"
        os.makedirs(test_output_dir, exist_ok=True)
        
        for split_id, model_info in self.split_models.split_models.items():
            split_dir = os.path.join(test_output_dir, split_id.lower())
            model_info['model'].save_ensemble(split_dir)
            
            # Check files were created
            self.assertTrue(os.path.exists(split_dir))
            self.assertTrue(os.path.exists(os.path.join(split_dir, 'ensemble_metadata.joblib')))
        
        # Load models
        loaded_models = SplitSpecificModels()
        for split_id in self.split_models.split_models.keys():
            split_dir = os.path.join(test_output_dir, split_id.lower())
            if os.path.exists(split_dir):
                # Create ensemble and load
                models = [
                    XGBoostModel(),
                    # You need to create custom wrapper classes for RandomForest and LinearRegression
                    # or just use XGBoostModel and ProphetModel for now
                ]
                ensemble = EnsembleModel(models)
                ensemble.load_ensemble(split_dir)
                
                loaded_models.split_models[split_id] = {
                    'model': ensemble,
                    'metrics': {'mae': 0, 'rmse': 0, 'r2': 0},
                    'feature_cols': ['time_slot', 'day_of_week', 'month', 'is_weekend']
                }
        
        # Test loaded models
        self.assertEqual(len(loaded_models.split_models), len(self.split_models.split_models))
        
        # Clean up
        import shutil
        shutil.rmtree(test_output_dir)
        
        logger.info("✅ Model persistence tests passed")
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        logger.info("Testing error handling...")
        
        # Test with empty data
        empty_data = pd.DataFrame()
        self.split_models.train_split_models(empty_data)
        self.assertEqual(len(self.split_models.split_models), 0)
        
        # Test with invalid split
        with self.assertRaises(KeyError):
            self.split_models.predict_split('InvalidSplit', pd.DataFrame())
        
        # Test recommendation with no models
        empty_engine = SmartRecommendationEngine()
        with self.assertRaises(ValueError):
            empty_engine.get_recommendation(1, 'Chest', [16], lambda x: np.array([0]))
        
        logger.info("✅ Error handling tests passed")

def run_performance_benchmark():
    """Run performance benchmarks"""
    logger.info("Running performance benchmarks...")
    
    # Create larger test dataset
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    time_slots = list(range(48))
    splits = ['Chest', 'Back', 'Legs', 'Core', 'Cardio']
    
    benchmark_data = []
    for date in dates:
        for time_slot in time_slots:
            for split in splits:
                attendance = max(0, int(10 + np.random.normal(0, 3)))
                benchmark_data.append({
                    'date': date,
                    'time_slot': time_slot,
                    'split_id': split,
                    'visitor_count': attendance,
                    'avg_duration': 60 + np.random.normal(0, 10),
                    'day_of_week': date.weekday(),
                    'month': date.month,
                    'is_weekend': 1 if date.weekday() >= 5 else 0,
                    'hour_sin': np.sin(2 * np.pi * time_slot / 48),
                    'hour_cos': np.cos(2 * np.pi * time_slot / 48),
                    'day_sin': np.sin(2 * np.pi * date.weekday() / 7),
                    'day_cos': np.cos(2 * np.pi * date.weekday() / 7),
                    'prev_day_attendance': 0,
                    'prev_week_attendance': 0,
                    'rolling_3day_avg': 0,
                    'rolling_7day_avg': 0
                })
    
    benchmark_df = pd.DataFrame(benchmark_data)
    
    # Benchmark training time
    import time
    start_time = time.time()
    
    split_models = SplitSpecificModels()
    split_models.train_split_models(benchmark_df)
    
    training_time = time.time() - start_time
    
    # Benchmark prediction time
    start_time = time.time()
    
    test_features = pd.DataFrame({
        'time_slot': [16] * 1000,
        'day_of_week': [1] * 1000,
        'month': [1] * 1000,
        'is_weekend': [0] * 1000,
        'hour_sin': [np.sin(2 * np.pi * 16 / 48)] * 1000,
        'hour_cos': [np.cos(2 * np.pi * 16 / 48)] * 1000,
        'day_sin': [np.sin(2 * np.pi * 1 / 7)] * 1000,
        'day_cos': [np.cos(2 * np.pi * 1 / 7)] * 1000,
        'prev_day_attendance': [0] * 1000,
        'prev_week_attendance': [0] * 1000,
        'rolling_3day_avg': [0] * 1000,
        'rolling_7day_avg': [0] * 1000
    })
    
    for split_id, model_info in split_models.split_models.items():
        predictions = model_info['model'].predict(test_features)
    
    prediction_time = time.time() - start_time
    
    logger.info(f"Performance Benchmark Results:")
    logger.info(f"  Training time: {training_time:.2f} seconds")
    logger.info(f"  Prediction time (1000 samples): {prediction_time:.4f} seconds")
    logger.info(f"  Records processed: {len(benchmark_df)}")
    logger.info(f"  Models trained: {len(split_models.split_models)}")

def main():
    """Main testing function"""
    logger.info("Starting Fit Forecaster comprehensive tests...")
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmark
    run_performance_benchmark()
    
    logger.info("All tests completed!")

if __name__ == "__main__":
    main() 