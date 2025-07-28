"""
Real-time Learning Management Utility
Provides tools for managing the real-time learning system
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import shutil
import logging
from pathlib import Path

# Import real-time learning components
import sys
sys.path.append('src')
from realtime_learning import RealTimeLearning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeLearningManager:
    """Manages the real-time learning system with backup, validation, and monitoring"""
    
    def __init__(self, data_dir="week_data", models_dir="models", feedback_dir="user_feedback"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.feedback_dir = feedback_dir
        self.realtime_learner = RealTimeLearning(data_dir, models_dir, feedback_dir)
        
        # Create backup directory
        self.backup_dir = "backups"
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_backup(self, backup_name=None):
        """Create a backup of current data and models"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = os.path.join(self.backup_dir, backup_name)
        os.makedirs(backup_path, exist_ok=True)
        
        # Backup data files
        data_backup_path = os.path.join(backup_path, "data")
        os.makedirs(data_backup_path, exist_ok=True)
        
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                src = os.path.join(self.data_dir, file)
                dst = os.path.join(data_backup_path, file)
                shutil.copy2(src, dst)
        
        # Backup models
        models_backup_path = os.path.join(backup_path, "models")
        if os.path.exists(self.models_dir):
            shutil.copytree(self.models_dir, models_backup_path, dirs_exist_ok=True)
        
        # Backup feedback
        feedback_backup_path = os.path.join(backup_path, "feedback")
        if os.path.exists(self.feedback_dir):
            shutil.copytree(self.feedback_dir, feedback_backup_path, dirs_exist_ok=True)
        
        # Create backup metadata
        metadata = {
            'backup_name': backup_name,
            'timestamp': datetime.now().isoformat(),
            'data_files': len([f for f in os.listdir(data_backup_path) if f.endswith('.csv')]),
            'model_dirs': len([d for d in os.listdir(models_backup_path) if os.path.isdir(os.path.join(models_backup_path, d))]) if os.path.exists(models_backup_path) else 0,
            'feedback_files': len([f for f in os.listdir(feedback_backup_path) if f.endswith('.json')]) if os.path.exists(feedback_backup_path) else 0
        }
        
        with open(os.path.join(backup_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Backup created: {backup_path}")
        return backup_path
    
    def restore_backup(self, backup_name):
        """Restore from a backup"""
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        if not os.path.exists(backup_path):
            raise ValueError(f"Backup {backup_name} not found")
        
        # Restore data files
        data_backup_path = os.path.join(backup_path, "data")
        if os.path.exists(data_backup_path):
            for file in os.listdir(data_backup_path):
                if file.endswith('.csv'):
                    src = os.path.join(data_backup_path, file)
                    dst = os.path.join(self.data_dir, file)
                    shutil.copy2(src, dst)
        
        # Restore models
        models_backup_path = os.path.join(backup_path, "models")
        if os.path.exists(models_backup_path):
            if os.path.exists(self.models_dir):
                shutil.rmtree(self.models_dir)
            shutil.copytree(models_backup_path, self.models_dir)
        
        # Restore feedback
        feedback_backup_path = os.path.join(backup_path, "feedback")
        if os.path.exists(feedback_backup_path):
            if os.path.exists(self.feedback_dir):
                shutil.rmtree(self.feedback_dir)
            shutil.copytree(feedback_backup_path, self.feedback_dir)
        
        logger.info(f"Restored from backup: {backup_name}")
    
    def list_backups(self):
        """List all available backups"""
        if not os.path.exists(self.backup_dir):
            return []
        
        backups = []
        for backup_name in os.listdir(self.backup_dir):
            backup_path = os.path.join(self.backup_dir, backup_name)
            if os.path.isdir(backup_path):
                metadata_file = os.path.join(backup_path, 'metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    backups.append(metadata)
                else:
                    backups.append({
                        'backup_name': backup_name,
                        'timestamp': 'Unknown',
                        'data_files': 0,
                        'model_dirs': 0,
                        'feedback_files': 0
                    })
        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
    
    def validate_system(self):
        """Validate the real-time learning system"""
        issues = []
        
        # Check data files
        if not os.path.exists(self.data_dir):
            issues.append("Data directory not found")
        else:
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            if len(csv_files) == 0:
                issues.append("No CSV data files found")
        
        # Check models
        if not os.path.exists(self.models_dir):
            issues.append("Models directory not found")
        else:
            model_dirs = [d for d in os.listdir(self.models_dir) if os.path.isdir(os.path.join(self.models_dir, d))]
            if len(model_dirs) == 0:
                issues.append("No model directories found")
        
        # Check feedback directory
        if not os.path.exists(self.feedback_dir):
            issues.append("Feedback directory not found")
        
        # Check model files
        for model_dir in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, model_dir)
            if os.path.isdir(model_path):
                required_files = ['ensemble_metadata.joblib', 'linearregression.joblib', 
                               'randomforest.joblib', 'xgboost.joblib']
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
                if missing_files:
                    issues.append(f"Model {model_dir} missing files: {missing_files}")
        
        return issues
    
    def get_system_stats(self):
        """Get system statistics"""
        stats = {
            'data_files': 0,
            'model_dirs': 0,
            'feedback_files': 0,
            'total_feedback': 0,
            'pending_feedback': 0,
            'last_backup': None
        }
        
        # Count data files
        if os.path.exists(self.data_dir):
            stats['data_files'] = len([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
        
        # Count model directories
        if os.path.exists(self.models_dir):
            stats['model_dirs'] = len([d for d in os.listdir(self.models_dir) if os.path.isdir(os.path.join(self.models_dir, d))])
        
        # Count feedback files
        if os.path.exists(self.feedback_dir):
            stats['feedback_files'] = len([f for f in os.listdir(self.feedback_dir) if f.endswith('.json')])
        
        # Get feedback summary
        summary = self.realtime_learner.get_feedback_summary()
        stats['total_feedback'] = summary['total_feedback']
        stats['pending_feedback'] = summary['pending_feedback']
        
        # Get last backup
        backups = self.list_backups()
        if backups:
            stats['last_backup'] = backups[0]['timestamp']
        
        return stats
    
    def cleanup_old_feedback(self, days_to_keep=30):
        """Clean up old feedback files"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        
        if os.path.exists(self.feedback_dir):
            for file in os.listdir(self.feedback_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(self.feedback_dir, file)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    
                    if file_time < cutoff_date:
                        os.remove(file_path)
                        deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old feedback files")
        return deleted_count
    
    def force_retraining(self):
        """Force immediate retraining"""
        try:
            self.realtime_learner.force_retraining()
            logger.info("Forced retraining completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during forced retraining: {e}")
            return False

def main():
    """Main function for command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Real-time Learning System")
    parser.add_argument('action', choices=['backup', 'restore', 'list-backups', 'validate', 'stats', 'cleanup', 'retrain'],
                       help='Action to perform')
    parser.add_argument('--backup-name', help='Backup name for restore action')
    parser.add_argument('--days', type=int, default=30, help='Days to keep for cleanup')
    
    args = parser.parse_args()
    
    manager = RealTimeLearningManager()
    
    if args.action == 'backup':
        backup_path = manager.create_backup()
        print(f"Backup created: {backup_path}")
    
    elif args.action == 'restore':
        if not args.backup_name:
            print("Error: --backup-name required for restore action")
            return
        manager.restore_backup(args.backup_name)
        print(f"Restored from backup: {args.backup_name}")
    
    elif args.action == 'list-backups':
        backups = manager.list_backups()
        if backups:
            print("Available backups:")
            for backup in backups:
                print(f"  {backup['backup_name']} - {backup['timestamp']}")
        else:
            print("No backups found")
    
    elif args.action == 'validate':
        issues = manager.validate_system()
        if issues:
            print("System validation issues:")
            for issue in issues:
                print(f"  ❌ {issue}")
        else:
            print("✅ System validation passed")
    
    elif args.action == 'stats':
        stats = manager.get_system_stats()
        print("System Statistics:")
        print(f"  Data files: {stats['data_files']}")
        print(f"  Model directories: {stats['model_dirs']}")
        print(f"  Feedback files: {stats['feedback_files']}")
        print(f"  Total feedback: {stats['total_feedback']}")
        print(f"  Pending feedback: {stats['pending_feedback']}")
        print(f"  Last backup: {stats['last_backup']}")
    
    elif args.action == 'cleanup':
        deleted = manager.cleanup_old_feedback(args.days)
        print(f"Cleaned up {deleted} old feedback files")
    
    elif args.action == 'retrain':
        success = manager.force_retraining()
        if success:
            print("✅ Forced retraining completed successfully")
        else:
            print("❌ Forced retraining failed")

if __name__ == "__main__":
    main() 