"""
Data Processing Module for Fit Forecaster
Handles data cleaning, split parsing, and feature engineering
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data processing and feature engineering for gym attendance forecasting"""
    
    def __init__(self):
        # Standardized split mapping
        self.split_mapping = {
            'chest': 'Chest',
            'back': 'Back', 
            'leg': 'Legs',
            'legs': 'Legs',
            'core': 'Core',
            'Core': 'Core',
            'cardio': 'Cardio',
            'Cardio': 'Cardio',
            'running': 'Cardio',
            'Running': 'Cardio',
            'arm': 'Arms',
            'Arm': 'Arms',
            'arms': 'Arms'
        }
        
        # Time slots for binning
        self.time_slots = np.arange(0, 24.5, 0.5)  # 30-minute intervals
        
    def parse_workout_splits(self, fav_group_lesson) -> List[str]:
        """
        Parse compound workout preferences like 'chest, back, Core'
        
        Args:
            fav_group_lesson: Raw workout preference string
            
        Returns:
            List of standardized split names
        """
        if pd.isna(fav_group_lesson) or fav_group_lesson == '':
            return ['General']
        
        # Split by comma and clean
        splits = [s.strip().lower() for s in str(fav_group_lesson).split(',')]
        
        # Map to standardized names
        standardized_splits = []
        for split in splits:
            if split in self.split_mapping:
                standardized_splits.append(self.split_mapping[split])
            else:
                # Handle unknown splits
                standardized_splits.append(split.title())
        
        return list(set(standardized_splits))  # Remove duplicates
    
    def create_user_split_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-split preference matrix for recommendation system
        
        Args:
            df: Raw user data DataFrame
            
        Returns:
            DataFrame with user-split preferences
        """
        user_splits = []
        
        for _, row in df.iterrows():
            splits = self.parse_workout_splits(row.get('fav_group_lesson', ''))
            for split in splits:
                user_splits.append({
                    'age': row.get('Age', np.nan),
                    'visit_per_week': row.get('visit_per_week', 1),
                    'split_name': split,
                    'avg_time_check_in': row.get('avg_time_check_in', '12:00'),
                    'avg_time_check_out': row.get('avg_time_check_out', '13:00'),
                    'avg_time_in_gym': row.get('avg_time_in_gym', 60),
                    'tue_visitor': row.get('Tue', False)
                })
        
        return pd.DataFrame(user_splits)
    
    def convert_time_to_hours(self, time_str) -> float:
        """
        Convert time string to decimal hours
        
        Args:
            time_str: Time in format 'HH:MM:SS' or 'HH:MM'
            
        Returns:
            Time in decimal hours
        """
        try:
            if pd.isna(time_str):
                return 12.0  # Default to noon
            
            # Handle different time formats
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                hours = int(parts[0])
                minutes = int(parts[1]) if len(parts) > 1 else 0
                return hours + (minutes / 60.0)
            else:
                return float(time_str)
        except:
            return 12.0  # Default fallback
    
    def generate_realistic_visit_patterns(self, user_data: pd.DataFrame, 
                                        days_to_generate: int = 30) -> pd.DataFrame:
        """
        Generate realistic daily visit patterns from user averages
        
        Args:
            user_data: User preference data
            days_to_generate: Number of days to generate
            
        Returns:
            DataFrame with synthetic visit records
        """
        visit_records = []
        start_date = datetime.now() - timedelta(days=days_to_generate)
        
        for _, user in user_data.iterrows():
            # Ensure visits_per_week is a number
            visits_per_week = user.get('visit_per_week', 1)
            if visits_per_week is None or pd.isna(visits_per_week):
                visits_per_week = 1
            daily_visit_prob = float(visits_per_week) / 7.0
            
            # Generate visits for each day
            for day_offset in range(days_to_generate):
                current_date = start_date + timedelta(days=day_offset)
                day_of_week = current_date.weekday()
                
                # Adjust probability based on day of week
                if day_of_week == 1:  # Tuesday
                    adjusted_prob = daily_visit_prob * 1.5  # Higher for Tuesday
                else:
                    adjusted_prob = daily_visit_prob * 0.8  # Lower for other days
                
                # Determine if user visits today
                if np.random.random() < adjusted_prob:
                    # Generate check-in time with some variation
                    base_check_in = self.convert_time_to_hours(user.get('avg_time_check_in', '12:00'))
                    check_in_variation = np.random.normal(0, 1.0)  # ±1 hour variation
                    check_in_time = max(6, min(22, base_check_in + check_in_variation))
                    
                    # Generate check-out time
                    avg_time_in_gym = user.get('avg_time_in_gym', 60)
                    if avg_time_in_gym is None or pd.isna(avg_time_in_gym):
                        avg_time_in_gym = 60
                    avg_duration = float(avg_time_in_gym) / 60.0  # Convert to hours
                    duration_variation = np.random.normal(0, 0.5)  # ±30 min variation
                    duration = max(0.5, min(3.0, float(avg_duration + duration_variation)))
                    check_out_time = check_in_time + duration
                    
                    # Parse user's preferred splits
                    splits = self.parse_workout_splits(user.get('fav_group_lesson', ''))
                    chosen_split = np.random.choice(splits)
                    
                    visit_records.append({
                        'visit_id': len(visit_records) + 1,
                        'user_id': user.get('id', 0),
                        'split_id': chosen_split,
                        'check_in_time': check_in_time,
                        'check_out_time': check_out_time,
                        'visit_date': current_date.date(),
                        'duration_minutes': duration * 60,
                        'day_of_week': day_of_week,
                        'time_slot': int(check_in_time * 2)  # 30-minute slots
                    })
        
        return pd.DataFrame(visit_records)
    
    def create_traffic_features(self, visit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create traffic prediction features from visit data
        
        Args:
            visit_data: Visit records DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Group by date, time slot, and split
        traffic_data = visit_data.groupby([
            'visit_date', 'time_slot', 'split_id'
        ]).agg({
            'user_id': 'count',
            'duration_minutes': 'mean'
        }).reset_index()
        
        traffic_data.columns = ['date', 'time_slot', 'split_id', 'visitor_count', 'avg_duration']
        
        # Add temporal features
        traffic_data['date'] = pd.to_datetime(traffic_data['date'])
        traffic_data['day_of_week'] = traffic_data['date'].dt.dayofweek
        traffic_data['month'] = traffic_data['date'].dt.month
        traffic_data['is_weekend'] = traffic_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Add cyclical time features
        traffic_data['hour_sin'] = np.sin(2 * np.pi * traffic_data['time_slot'] / 48)
        traffic_data['hour_cos'] = np.cos(2 * np.pi * traffic_data['time_slot'] / 48)
        traffic_data['day_sin'] = np.sin(2 * np.pi * traffic_data['day_of_week'] / 7)
        traffic_data['day_cos'] = np.cos(2 * np.pi * traffic_data['day_of_week'] / 7)
        
        # Add lag features (previous day attendance)
        traffic_data = traffic_data.sort_values(['split_id', 'date', 'time_slot'])
        traffic_data['prev_day_attendance'] = traffic_data.groupby(['split_id', 'time_slot'])['visitor_count'].shift(1)
        traffic_data['prev_week_attendance'] = traffic_data.groupby(['split_id', 'time_slot'])['visitor_count'].shift(7)
        
        # Add rolling statistics
        traffic_data['rolling_3day_avg'] = traffic_data.groupby(['split_id', 'time_slot'])['visitor_count'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        traffic_data['rolling_7day_avg'] = traffic_data.groupby(['split_id', 'time_slot'])['visitor_count'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        
        # Fill NaN values
        traffic_data = traffic_data.fillna(0)
        
        return traffic_data
    
    def process_historical_data(self, file_paths: Dict[str, str]) -> pd.DataFrame:
        """
        Process all historical data files and create unified dataset
        
        Args:
            file_paths: Dictionary mapping day names to file paths
            
        Returns:
            Combined and processed DataFrame
        """
        all_data = []
        
        for day_name, file_path in file_paths.items():
            try:
                df = pd.read_csv(file_path)
                df['source_day'] = day_name
                # Drop columns that are all True (for bool dtype) or named 'gender'
                drop_cols = []
                for col in df.columns:
                    if col.lower() == 'gender':
                        drop_cols.append(col)
                    elif df[col].dtype == bool:
                        # Only drop if the column is all True and not a Series
                        if bool((df[col] == True).all()):
                            drop_cols.append(col)
                if drop_cols:
                    df = df.drop(columns=drop_cols)
                all_data.append(df)
                logger.info(f"Loaded {len(df)} records from {day_name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not all_data:
            raise ValueError("No data files could be loaded")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Process time columns
        combined_df['avg_time_check_in'] = combined_df['avg_time_check_in'].apply(self.convert_time_to_hours)
        combined_df['avg_time_check_out'] = combined_df['avg_time_check_out'].apply(self.convert_time_to_hours)
        
        # Create user-split matrix
        user_split_matrix = self.create_user_split_matrix(combined_df)
        
        # Generate realistic visit patterns
        visit_patterns = self.generate_realistic_visit_patterns(combined_df)
        
        # Create traffic features
        traffic_features = self.create_traffic_features(visit_patterns)
        
        return traffic_features
    
    def validate_data_quality(self, df: pd.DataFrame):
        """
        Validate data quality and return quality metrics
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'outliers': {}
        }
        
        # Check for outliers in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            quality_report['outliers'][col] = len(outliers)
        
        return quality_report 