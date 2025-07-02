"""
Smart Recommendation Engine for Fit Forecaster
Prevents recommendation-induced congestion and provides optimal workout times
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Recommendation:
    """Data class for workout recommendations"""
    split_id: str
    recommended_time_slot: int
    predicted_attendance: float
    confidence_score: float
    alternative_times: List[int]
    reason: str

class SmartRecommendationEngine:
    """Smart recommendation engine with congestion prevention"""
    
    def __init__(self, max_capacity: int = 50, congestion_threshold: float = 0.8):
        self.max_capacity = max_capacity
        self.congestion_threshold = congestion_threshold
        self.recommendation_quotas = {}  # Track allocations per time slot
        self.recent_recommendations = []  # Track recent recommendations
        self.split_preferences = {}  # User split preferences
        
    def initialize_quotas(self, time_slots: List[int], splits: List[str]):
        """Initialize quota tracking for all time slots and splits"""
        for split_id in splits:
            self.recommendation_quotas[split_id] = {}
            for time_slot in time_slots:
                self.recommendation_quotas[split_id][time_slot] = {
                    'allocated': 0,
                    'max_quota': int(self.max_capacity * 0.3),  # 30% of capacity
                    'last_updated': datetime.now()
                }
    
    def get_traffic_prediction(self, split_id: str, time_slot: int, 
                             prediction_model) -> float:
        """Get traffic prediction for a specific split and time slot"""
        try:
            # Create feature vector for prediction
            current_date = datetime.now()
            features = pd.DataFrame({
                'split_id': [split_id],
                'time_slot': [time_slot],
                'day_of_week': [current_date.weekday()],
                'month': [current_date.month],
                'is_weekend': [1 if current_date.weekday() >= 5 else 0],
                'hour_sin': [np.sin(2 * np.pi * time_slot / 48)],
                'hour_cos': [np.cos(2 * np.pi * time_slot / 48)],
                'day_sin': [np.sin(2 * np.pi * current_date.weekday() / 7)],
                'day_cos': [np.cos(2 * np.pi * current_date.weekday() / 7)],
                'prev_day_attendance': [0],  # Would need historical data
                'prev_week_attendance': [0],  # Would need historical data
                'rolling_3day_avg': [0],  # Would need historical data
                'rolling_7day_avg': [0]   # Would need historical data
            })
            
            prediction = prediction_model.predict(features)[0]
            return max(0, prediction)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error getting traffic prediction: {e}")
            return 0.0
    
    def calculate_congestion_score(self, split_id: str, time_slot: int, 
                                 base_traffic: float) -> float:
        """Calculate congestion score for a time slot"""
        # Get current quota usage
        quota_info = self.recommendation_quotas.get(split_id, {}).get(time_slot, {})
        allocated = quota_info.get('allocated', 0)
        
        # Calculate total expected attendance
        total_expected = base_traffic + allocated
        
        # Calculate congestion score (0 = empty, 1 = full)
        congestion_score = total_expected / self.max_capacity
        
        return min(1.0, congestion_score)
    
    def find_optimal_times(self, split_id: str, preferred_times: List[int], 
                          prediction_model, max_alternatives: int = 3) -> List[Tuple[int, float]]:
        """Find optimal workout times with congestion consideration"""
        all_time_slots = list(range(48))  # 30-minute slots for 24 hours
        
        time_scores = []
        
        for time_slot in all_time_slots:
            # Get base traffic prediction
            base_traffic = self.get_traffic_prediction(split_id, time_slot, prediction_model)
            
            # Calculate congestion score
            congestion_score = self.calculate_congestion_score(split_id, time_slot, base_traffic)
            
            # Calculate preference bonus
            preference_bonus = 0.0
            if time_slot in preferred_times:
                preference_bonus = 0.3  # 30% bonus for preferred times
            
            # Calculate final score (lower is better for congestion)
            final_score = congestion_score - preference_bonus
            
            # Only consider times with acceptable congestion
            if congestion_score < self.congestion_threshold:
                time_scores.append((time_slot, final_score))
        
        # Sort by score (best first) and return top alternatives
        time_scores.sort(key=lambda x: x[1])
        return time_scores[:max_alternatives]
    
    def get_recommendation(self, user_id: int, split_id: str, 
                          preferred_times: List[int], prediction_model) -> Recommendation:
        """Get smart recommendation for a user"""
        logger.info(f"Getting recommendation for user {user_id}, split {split_id}")
        
        # Find optimal times
        optimal_times = self.find_optimal_times(split_id, preferred_times, prediction_model)
        
        if not optimal_times:
            # Fallback: recommend any available time
            logger.warning(f"No optimal times found for {split_id}, using fallback")
            optimal_times = [(12, 0.5)]  # Default to noon
        
        # Select best time
        best_time_slot, best_score = optimal_times[0]
        
        # Get prediction for best time
        base_traffic = self.get_traffic_prediction(split_id, best_time_slot, prediction_model)
        congestion_score = self.calculate_congestion_score(split_id, best_time_slot, base_traffic)
        
        # Calculate confidence score
        confidence_score = max(0.1, 1.0 - congestion_score)
        
        # Get alternative times
        alternative_times = [time_slot for time_slot, _ in optimal_times[1:]]
        
        # Update quota
        self.update_quota(split_id, best_time_slot)
        
        # Log recommendation
        self.log_recommendation(user_id, split_id, best_time_slot, base_traffic)
        
        # Determine reason for recommendation
        reason = self.get_recommendation_reason(best_time_slot, preferred_times, congestion_score)
        
        return Recommendation(
            split_id=split_id,
            recommended_time_slot=best_time_slot,
            predicted_attendance=base_traffic,
            confidence_score=confidence_score,
            alternative_times=alternative_times,
            reason=reason
        )
    
    def update_quota(self, split_id: str, time_slot: int):
        """Update quota allocation for a time slot"""
        if split_id not in self.recommendation_quotas:
            self.recommendation_quotas[split_id] = {}
        
        if time_slot not in self.recommendation_quotas[split_id]:
            self.recommendation_quotas[split_id][time_slot] = {
                'allocated': 0,
                'max_quota': int(self.max_capacity * 0.3),
                'last_updated': datetime.now()
            }
        
        quota_info = self.recommendation_quotas[split_id][time_slot]
        quota_info['allocated'] += 1
        quota_info['last_updated'] = datetime.now()
        
        logger.info(f"Updated quota for {split_id} at {time_slot}: {quota_info['allocated']}")
    
    def log_recommendation(self, user_id: int, split_id: str, time_slot: int, 
                          predicted_traffic: float):
        """Log recommendation for tracking"""
        recommendation_log = {
            'user_id': user_id,
            'split_id': split_id,
            'time_slot': time_slot,
            'predicted_traffic': predicted_traffic,
            'timestamp': datetime.now(),
            'quota_used': self.recommendation_quotas.get(split_id, {}).get(time_slot, {}).get('allocated', 0)
        }
        
        self.recent_recommendations.append(recommendation_log)
        
        # Keep only last 1000 recommendations
        if len(self.recent_recommendations) > 1000:
            self.recent_recommendations = self.recent_recommendations[-1000:]
    
    def get_recommendation_reason(self, time_slot: int, preferred_times: List[int], 
                                congestion_score: float) -> str:
        """Generate human-readable reason for recommendation"""
        time_str = f"{time_slot // 2:02d}:{(time_slot % 2) * 30:02d}"
        
        if time_slot in preferred_times:
            return f"Recommended {time_str} - matches your preferred time and has low congestion ({congestion_score:.1%})"
        elif congestion_score < 0.3:
            return f"Recommended {time_str} - very low congestion ({congestion_score:.1%})"
        elif congestion_score < 0.6:
            return f"Recommended {time_str} - moderate congestion ({congestion_score:.1%})"
        else:
            return f"Recommended {time_str} - best available time with {congestion_score:.1%} congestion"
    
    def get_weekly_recommendations(self, user_id: int, split_id: str, 
                                 prediction_model) -> Dict[int, Recommendation]:
        """Get recommendations for the entire week"""
        weekly_recommendations = {}
        
        # Define preferred times for each day (example)
        daily_preferences = {
            0: [16, 17, 18],  # Monday: 8-9 PM
            1: [14, 15, 16],  # Tuesday: 7-8 PM
            2: [16, 17, 18],  # Wednesday: 8-9 PM
            3: [14, 15, 16],  # Thursday: 7-8 PM
            4: [15, 16, 17],  # Friday: 7:30-8:30 PM
            5: [10, 11, 12],  # Saturday: 5-6 PM
            6: [10, 11, 12]   # Sunday: 5-6 PM
        }
        
        for day_of_week in range(7):
            preferred_times = daily_preferences.get(day_of_week, [12, 13, 14])
            
            # Get recommendation for this day
            recommendation = self.get_recommendation(
                user_id, split_id, preferred_times, prediction_model
            )
            
            weekly_recommendations[day_of_week] = recommendation
        
        return weekly_recommendations
    
    def get_quota_status(self) -> Dict[str, Dict]:
        """Get current quota status for monitoring"""
        return self.recommendation_quotas
    
    def reset_quotas(self):
        """Reset all quotas (call daily)"""
        for split_id in self.recommendation_quotas:
            for time_slot in self.recommendation_quotas[split_id]:
                self.recommendation_quotas[split_id][time_slot]['allocated'] = 0
                self.recommendation_quotas[split_id][time_slot]['last_updated'] = datetime.now()
        
        logger.info("Reset all recommendation quotas")
    
    def get_recommendation_analytics(self):
        """Get analytics about recent recommendations"""
        if not self.recent_recommendations:
            return {}
        
        df = pd.DataFrame(self.recent_recommendations)
        
        analytics = {
            'total_recommendations': len(df),
            'unique_users': df['user_id'].nunique(),
            'split_distribution': df['split_id'].value_counts().to_dict(),
            'time_slot_distribution': df['time_slot'].value_counts().to_dict(),
            'avg_predicted_traffic': df['predicted_traffic'].mean(),
            'quota_utilization': df['quota_used'].mean() / (self.max_capacity * 0.3) 
        }
        
        return analytics

class HybridRecommendationSystem:
    """Hybrid system combining content-based and collaborative filtering"""
    
    def __init__(self):
        self.content_based = ContentBasedRecommender()
        self.collaborative = CollaborativeFilter()
        self.time_series = TimeSeriesPredictor()
        
    def recommend(self, split_type: str, user_preferences: Dict, 
                 has_sufficient_data: bool = True) -> Recommendation:
        """Get hybrid recommendation"""
        if has_sufficient_data:
            return self.collaborative.predict(split_type, user_preferences)
        else:
            return self.content_based.predict(split_type, user_preferences)

class ContentBasedRecommender:
    """Content-based recommendation system for new splits"""
    
    def predict(self, split_type: str, user_preferences: Dict) -> Recommendation:
        """Make content-based prediction"""
        # Simplified implementation
        return Recommendation(
            split_id=split_type,
            recommended_time_slot=12,  # Default to noon
            predicted_attendance=10.0,
            confidence_score=0.5,
            alternative_times=[13, 14, 15],
            reason="Content-based recommendation for new split"
        )

class CollaborativeFilter:
    """Collaborative filtering for popular splits"""
    
    def predict(self, split_type: str, user_preferences: Dict) -> Recommendation:
        """Make collaborative prediction"""
        # Simplified implementation
        return Recommendation(
            split_id=split_type,
            recommended_time_slot=16,  # Default to 8 PM
            predicted_attendance=25.0,
            confidence_score=0.8,
            alternative_times=[17, 18, 19],
            reason="Collaborative filtering recommendation"
        )

class TimeSeriesPredictor:
    """Time series predictor for traffic patterns"""
    
    def predict(self, split_type: str, time_slot: int) -> float:
        """Predict traffic for specific time slot"""
        # Simplified implementation
        return 15.0  # Default prediction 