"""
Real-time Learning Module for Fit Forecaster
Handles user feedback collection, data storage, and incremental model retraining
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path

# Import existing modules
from data_processor import DataProcessor
from models import SplitSpecificModels, EnsembleModel

logger = logging.getLogger(__name__)

class RealTimeLearning:
    """Handles real-time learning by collecting user feedback and retraining models"""
    
    def __init__(self, data_dir: str = "week_data", models_dir: str = "models", 
                 feedback_dir: str = "user_feedback"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.feedback_dir = feedback_dir
        self.data_processor = DataProcessor()
        
        # Create feedback directory if it doesn't exist
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Track feedback for batching
        self.pending_feedback = []
        self.feedback_batch_size = 10  # Retrain after 10 new feedback entries
        
    def collect_user_feedback(self, user_selection: Dict, actual_attendance: Optional[int] = None) -> str:
        """
        Collect user feedback when they select a time/split and optionally report actual attendance
        
        Args:
            user_selection: Dict containing user's selection
                - date: datetime object
                - split_id: str (e.g., 'Chest', 'Back')
                - time_slot: int (0-47, representing 30-min intervals)
                - predicted_attendance: float
                - user_id: int (optional)
            actual_attendance: Optional actual attendance count (if user reports it)
            
        Returns:
            feedback_id: Unique identifier for this feedback entry
        """
        feedback_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        feedback_entry = {
            'feedback_id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'user_selection': user_selection,
            'actual_attendance': actual_attendance,
            'feedback_type': 'user_selection' if actual_attendance is None else 'attendance_report'
        }
        
        # Store feedback
        feedback_file = os.path.join(self.feedback_dir, f"{feedback_id}.json")
        with open(feedback_file, 'w') as f:
            json.dump(feedback_entry, f, indent=2, default=str)
        
        self.pending_feedback.append(feedback_entry)
        logger.info(f"Collected feedback {feedback_id} for {user_selection['split_id']} at {user_selection['time_slot']}")
        
        # Check if we should trigger retraining
        if len(self.pending_feedback) >= self.feedback_batch_size:
            self._trigger_retraining()
        
        return feedback_id
    
    def _trigger_retraining(self):
        """Trigger model retraining when enough feedback is collected"""
        try:
            logger.info(f"Triggering retraining with {len(self.pending_feedback)} new feedback entries")
            
            # Process feedback into training data
            new_data = self._process_feedback_to_data()
            
            if new_data is not None and not new_data.empty:
                # Update existing datasets
                self._update_datasets(new_data)
                
                # Retrain models
                self._retrain_models()
                
                # Clear pending feedback
                self.pending_feedback = []
                
                logger.info("Retraining completed successfully")
            else:
                logger.warning("No valid data from feedback for retraining")
                
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
    
    def _process_feedback_to_data(self) -> Optional[pd.DataFrame]:
        """Convert feedback entries into training data format"""
        if not self.pending_feedback:
            return None
        
        new_entries = []
        
        for feedback in self.pending_feedback:
            selection = feedback['user_selection']
            actual_attendance = feedback.get('actual_attendance')
            
            # Skip if no actual attendance reported
            if actual_attendance is None:
                continue
            
            # Convert to the format expected by data processor
            date = pd.to_datetime(selection['date'])
            time_str = f"{selection['time_slot'] // 2:02d}:{(selection['time_slot'] % 2) * 30:02d}:00"
            
            entry = {
                'id': f"feedback_{feedback['feedback_id']}",
                'first_name': f"User_{selection.get('user_id', 'unknown')}",
                'gender': 'Unknown',
                'Age': 25,  # Default age
                'visit_per_week': 1,
                'fav_group_lesson': selection['split_id'],
                'avg_time_check_in': time_str,
                'avg_time_check_out': time_str,  # Will be adjusted by data processor
                'avg_time_in_gym': 60,  # Default 1 hour
                'Mon': date.weekday() == 0,
                'Tue': date.weekday() == 1,
                'Wed': date.weekday() == 2,
                'Thu': date.weekday() == 3,
                'Fri': date.weekday() == 4,
                'Sat': date.weekday() == 5,
                'Sun': date.weekday() == 6,
                'actual_attendance': actual_attendance,
                'date': date.strftime('%Y-%m-%d')
            }
            
            new_entries.append(entry)
        
        if new_entries:
            return pd.DataFrame(new_entries)
        return None
    
    def _update_datasets(self, new_data: pd.DataFrame):
        """Update existing CSV files with new data"""
        # Group by day of week
        for day_idx in range(7):
            day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_idx]
            day_data = new_data[new_data[day_name] == True]
            
            if day_data.empty:
                continue
            
            # Get corresponding CSV file
            csv_files = {
                0: 'mongym.csv', 1: 'tuegym.csv', 2: 'wedgym.csv', 
                3: 'thugym.csv', 4: 'frigym.csv', 5: 'satgym.csv', 6: 'sungym.csv'
            }
            
            csv_file = os.path.join(self.data_dir, csv_files[day_idx])
            
            if os.path.exists(csv_file):
                # Read existing data
                existing_data = pd.read_csv(csv_file)
                
                # Prepare new data (remove day columns, keep only relevant ones)
                columns_to_keep = [col for col in existing_data.columns if col not in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']]
                new_data_subset = day_data[columns_to_keep].copy()
                
                # Append new data
                updated_data = pd.concat([existing_data, new_data_subset], ignore_index=True)
                
                # Save updated data
                updated_data.to_csv(csv_file, index=False)
                logger.info(f"Updated {csv_file} with {len(new_data_subset)} new entries")
            else:
                logger.warning(f"CSV file {csv_file} not found")
    
    def _retrain_models(self):
        """Retrain all models with updated data"""
        try:
            # Load updated data
            file_paths = {name[:-7]: os.path.join(self.data_dir, fname) for name, fname in zip(
                ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                ['mongym.csv', 'tuegym.csv', 'wedgym.csv', 'thugym.csv', 'frigym.csv', 'satgym.csv', 'sungym.csv'])
                if os.path.exists(os.path.join(self.data_dir, fname))}
            
            updated_data = self.data_processor.process_historical_data(file_paths)
            
            # Initialize models
            split_models = SplitSpecificModels()
            
            # Train models for each split
            split_models.train_split_models(updated_data, target_col='visitor_count')
            
            # Save updated models
            for split_id, model_info in split_models.split_models.items():
                if model_info['model'] is not None:
                    model_dir = os.path.join(self.models_dir, split_id.lower())
                    os.makedirs(model_dir, exist_ok=True)
                    model_info['model'].save_ensemble(model_dir)
                    logger.info(f"Saved updated model for {split_id}")
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            raise
    
    def get_feedback_summary(self) -> Dict:
        """Get summary of collected feedback"""
        feedback_files = [f for f in os.listdir(self.feedback_dir) if f.endswith('.json')]
        
        summary = {
            'total_feedback': len(feedback_files),
            'pending_feedback': len(self.pending_feedback),
            'feedback_types': {},
            'splits_feedback': {},
            'recent_feedback': []
        }
        
        for feedback_file in feedback_files[-10:]:  # Last 10 feedback entries
            try:
                with open(os.path.join(self.feedback_dir, feedback_file), 'r') as f:
                    feedback = json.load(f)
                    
                feedback_type = feedback.get('feedback_type', 'unknown')
                summary['feedback_types'][feedback_type] = summary['feedback_types'].get(feedback_type, 0) + 1
                
                split_id = feedback['user_selection']['split_id']
                summary['splits_feedback'][split_id] = summary['splits_feedback'].get(split_id, 0) + 1
                
                summary['recent_feedback'].append({
                    'timestamp': feedback['timestamp'],
                    'split': split_id,
                    'time_slot': feedback['user_selection']['time_slot'],
                    'actual_attendance': feedback.get('actual_attendance')
                })
                    
            except Exception as e:
                logger.warning(f"Error reading feedback file {feedback_file}: {e}")
        
        return summary
    
    def force_retraining(self):
        """Force immediate retraining regardless of batch size"""
        if self.pending_feedback:
            self._trigger_retraining()
        else:
            logger.info("No pending feedback to retrain with")
    
    def clear_feedback(self):
        """Clear all pending feedback (use with caution)"""
        self.pending_feedback = []
        logger.info("Cleared all pending feedback")

class FeedbackCollector:
    """Streamlit-specific feedback collection interface"""
    
    def __init__(self, realtime_learner: RealTimeLearning):
        self.learner = realtime_learner
    
    def collect_prediction_feedback(self, date, split_id, time_slot, predicted_attendance, user_id=None):
        """Collect feedback when user selects a prediction"""
        user_selection = {
            'date': date,
            'split_id': split_id,
            'time_slot': time_slot,
            'predicted_attendance': predicted_attendance,
            'user_id': user_id
        }
        
        return self.learner.collect_user_feedback(user_selection)
    
    def collect_attendance_feedback(self, feedback_id, actual_attendance):
        """Collect actual attendance feedback"""
        # Update the original feedback file with actual attendance
        feedback_file = os.path.join(self.learner.feedback_dir, f"{feedback_id}.json")
        
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback = json.load(f)
            
            feedback['actual_attendance'] = actual_attendance
            feedback['feedback_type'] = 'attendance_report'
            
            with open(feedback_file, 'w') as f:
                json.dump(feedback, f, indent=2, default=str)
            
            logger.info(f"Updated feedback {feedback_id} with actual attendance: {actual_attendance}")
            return True
        else:
            logger.warning(f"Feedback file {feedback_id} not found")
            return False 