"""
StreamLit Application for Fit Forecaster
Provides user-friendly interface for gym attendance forecasting and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from models import SplitSpecificModels
from recommendation_engine import SmartRecommendationEngine, Recommendation

# Page configuration
st.set_page_config(
    page_title="Fit Forecaster",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load trained models"""
    try:
        models_dir = "models"
        if not os.path.exists(models_dir):
            st.error("Models not found. Please run the training script first.")
            return None
        
        split_models = SplitSpecificModels()
        
        # Load models for each split
        for split_dir in os.listdir(models_dir):
            split_path = os.path.join(models_dir, split_dir)
            if os.path.isdir(split_path):
                # Load ensemble metadata
                metadata_path = os.path.join(split_path, 'ensemble_metadata.joblib')
                if os.path.exists(metadata_path):
                    import joblib
                    metadata = joblib.load(metadata_path)
                    
                    # Create ensemble model and load it
                    from models import EnsembleModel, XGBoostModel
                    models = [XGBoostModel()]
                    ensemble = EnsembleModel(models)
                    ensemble.load_ensemble(split_path)
                    
                    # Load metrics
                    metrics_path = os.path.join(split_path, 'metrics.json')
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)
                    else:
                        metrics = {'mae': 0, 'rmse': 0, 'r2': 0}
                    
                    split_models.split_models[split_dir.title()] = {
                        'model': ensemble,
                        'metrics': metrics,
                        'feature_cols': [
                            'time_slot', 'day_of_week', 'month', 'is_weekend',
                            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                            'prev_day_attendance', 'prev_week_attendance',
                            'rolling_3day_avg', 'rolling_7day_avg'
                        ]
                    }
        
        return split_models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    try:
        data_processor = DataProcessor()
        
        # Load sample data
        file_paths = {
            'monday': 'week_data/mongym.csv',
            'tuesday': 'week_data/tuegym.csv',
            'wednesday': 'week_data/wedgym.csv',
            'friday': 'week_data/frigym.csv'
        }
        
        # Check if files exist
        existing_files = {k: v for k, v in file_paths.items() if os.path.exists(v)}
        
        if not existing_files:
            st.warning("Sample data files not found. Using synthetic data.")
            return create_synthetic_data()
        
        return data_processor.process_historical_data(existing_files)
    except Exception as e:
        st.warning(f"Error loading data: {e}. Using synthetic data.")
        return create_synthetic_data()

def create_synthetic_data():
    """Create synthetic data for demonstration"""
    np.random.seed(42)
    
    # Generate synthetic traffic data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    time_slots = list(range(48))  # 30-minute slots
    splits = ['Chest', 'Back', 'Legs', 'Core', 'Cardio', 'Arms']
    
    data = []
    for date in dates:
        for time_slot in time_slots:
            for split in splits:
                # Generate realistic attendance patterns
                base_attendance = 10
                
                # Time of day effect
                if 6 <= time_slot <= 10:  # Morning
                    base_attendance += 5
                elif 16 <= time_slot <= 20:  # Evening
                    base_attendance += 8
                
                # Day of week effect
                day_of_week = date.weekday()
                if day_of_week == 1:  # Tuesday
                    base_attendance += 3
                elif day_of_week >= 5:  # Weekend
                    base_attendance -= 2
                
                # Split-specific effects
                if split == 'Cardio':
                    base_attendance += 2
                elif split == 'Chest':
                    base_attendance += 1
                
                # Add noise
                attendance = max(0, int(base_attendance + np.random.normal(0, 3)))
                
                data.append({
                    'date': date,
                    'time_slot': time_slot,
                    'split_id': split,
                    'visitor_count': attendance,
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

def main(): # Ref wh
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏋️ Fit Forecaster</h1>', unsafe_allow_html=True)
    st.markdown("### Smart Gym Attendance Forecasting & Recommendations")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Dashboard", "🎯 Get Recommendations", "📈 Traffic Analysis", "⚙️ Model Performance"]
    )
    
    # Load data and models
    with st.spinner("Loading models and data..."):
        models = load_models()
        data = load_sample_data()
    
    if page == "📊 Dashboard":
        show_dashboard(data, models)
    elif page == "🎯 Get Recommendations":
        show_recommendations(data, models)
    elif page == "📈 Traffic Analysis":
        show_traffic_analysis(data)
    elif page == "⚙️ Model Performance":
        show_model_performance(models)

def show_dashboard(data, models):
    """Show main dashboard"""
    st.header("📊 Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Records</h3>
            <h2>{}</h2>
        </div>
        """.format(len(data)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Workout Splits</h3>
            <h2>{}</h2>
        </div>
        """.format(data['split_id'].nunique()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Daily Visitors</h3>
            <h2>{:.1f}</h2>
        </div>
        """.format(data['visitor_count'].mean()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Trained Models</h3>
            <h2>{}</h2>
        </div>
        """.format(len(models.split_models) if models else 0), unsafe_allow_html=True)
    
    # Traffic overview
    st.subheader("Traffic Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily traffic
        daily_traffic = data.groupby('date')['visitor_count'].sum().reset_index()
        fig = px.line(daily_traffic, x='date', y='visitor_count', 
                     title='Daily Total Attendance')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Split distribution
        split_traffic = data.groupby('split_id')['visitor_count'].sum().reset_index()
        fig = px.pie(split_traffic, values='visitor_count', names='split_id',
                    title='Attendance by Workout Split')
        st.plotly_chart(fig, use_container_width=True)
    
    # Time-based patterns
    st.subheader("Time-based Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly pattern
        hourly_traffic = data.groupby('time_slot')['visitor_count'].mean().reset_index()
        hourly_traffic['time'] = hourly_traffic['time_slot'].apply(
            lambda x: f"{x//2:02d}:{(x%2)*30:02d}"
        )
        fig = px.bar(hourly_traffic, x='time', y='visitor_count',
                    title='Average Attendance by Time of Day')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Day of week pattern
        day_traffic = data.groupby('day_of_week')['visitor_count'].mean().reset_index()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_traffic['day_name'] = day_traffic['day_of_week'].apply(lambda x: day_names[x])
        fig = px.bar(day_traffic, x='day_name', y='visitor_count',
                    title='Average Attendance by Day of Week')
        st.plotly_chart(fig, use_container_width=True)

def show_recommendations(data, models):
    """Show recommendation interface"""
    st.header("🎯 Get Workout Recommendations")
    
    if not models:
        st.error("Models not loaded. Please train the models first.")
        return
    
    # User input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Preferences")
        
        # Split selection
        available_splits = list(models.split_models.keys())
        selected_split = st.selectbox("Choose your workout split:", available_splits)
        
        # Time preferences
        st.write("Preferred workout times:")
        morning = st.checkbox("Morning (6 AM - 10 AM)")
        afternoon = st.checkbox("Afternoon (12 PM - 4 PM)")
        evening = st.checkbox("Evening (6 PM - 10 PM)")
        
        # Convert preferences to time slots
        preferred_times = []
        if morning:
            preferred_times.extend([12, 13, 14, 15, 16, 17, 18, 19])  # 6-10 AM
        if afternoon:
            preferred_times.extend([24, 25, 26, 27, 28, 29, 30, 31])  # 12-4 PM
        if evening:
            preferred_times.extend([36, 37, 38, 39, 40, 41, 42, 43])  # 6-10 PM
        
        if not preferred_times:
            preferred_times = [16, 17, 18, 19]  # Default to evening
    
    with col2:
        st.subheader("Recommendation Settings")
        # Day of week selector
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        selected_day_name = st.selectbox("Select Day of Week:", day_names)
        selected_day_of_week = day_names.index(selected_day_name)
        # User ID
        user_id = st.number_input("User ID:", min_value=1, value=1)
        # Confidence threshold
        confidence_threshold = st.slider("Minimum confidence:", 0.0, 1.0, 0.5)
        # Get recommendation button
        if st.button("🎯 Get Recommendation", type="primary"):
            if models and selected_split in models.split_models:
                # Initialize recommendation engine
                recommendation_engine = SmartRecommendationEngine()
                time_slots = list(range(48))
                splits = list(models.split_models.keys())
                recommendation_engine.initialize_quotas(time_slots, splits)
                # Get prediction model
                prediction_model = models.split_models[selected_split]['model']
                # Get recommendation
                recommendation = recommendation_engine.get_recommendation(
                    user_id, selected_split, preferred_times, prediction_model
                )
                # Display recommendation
                st.success("✅ Recommendation Generated!")
                # Recommendation card
                time_str = f"{recommendation.recommended_time_slot // 2:02d}:{(recommendation.recommended_time_slot % 2) * 30:02d}"
                st.markdown(f"""
                <div class="recommendation-card">
                    <h3>🏋️ {selected_split} Workout</h3>
                    <p><strong>Recommended Time:</strong> {time_str}</p>
                    <p><strong>Predicted Attendance:</strong> {recommendation.predicted_attendance:.1f} people</p>
                    <p><strong>Confidence:</strong> {recommendation.confidence_score:.2f}</p>
                    <p><strong>Reason:</strong> {recommendation.reason}</p>
                </div>
                """, unsafe_allow_html=True)
                # Alternative times
                if recommendation.alternative_times:
                    st.subheader("Alternative Times:")
                    alt_times = []
                    for alt_time in recommendation.alternative_times:
                        alt_time_str = f"{alt_time // 2:02d}:{(alt_time % 2) * 30:02d}"
                        alt_times.append(alt_time_str)
                    st.write(", ".join(alt_times))
                # Traffic prediction chart
                st.subheader("Traffic Prediction")
                # Generate predictions for all time slots
                all_predictions = []
                for time_slot in range(48):
                    features = pd.DataFrame({
                        'split_id': [selected_split],
                        'time_slot': [time_slot],
                        'day_of_week': [selected_day_of_week],
                        'month': [datetime.now().month],
                        'is_weekend': [1 if selected_day_of_week >= 5 else 0],
                        'hour_sin': [np.sin(2 * np.pi * time_slot / 48)],
                        'hour_cos': [np.cos(2 * np.pi * time_slot / 48)],
                        'day_sin': [np.sin(2 * np.pi * selected_day_of_week / 7)],
                        'day_cos': [np.cos(2 * np.pi * selected_day_of_week / 7)],
                        'prev_day_attendance': [0],
                        'prev_week_attendance': [0],
                        'rolling_3day_avg': [0],
                        'rolling_7day_avg': [0]
                    })
                    try:
                        pred = prediction_model.predict(features)[0]
                        all_predictions.append(pred)
                    except:
                        all_predictions.append(0)
                # Create prediction chart
                time_labels = [f"{i//2:02d}:{(i%2)*30:02d}" for i in range(48)]
                fig = px.line(x=time_labels, y=all_predictions, 
                            title=f'Predicted {selected_split} Attendance by Time')
                # Mark preferred time slots
                for slot in preferred_times:
                    fig.add_scatter(x=[time_labels[slot]], y=[all_predictions[slot]],
                                   mode='markers', marker=dict(color='blue', size=12, symbol='circle'),
                                   name='Preferred Time')
                # Mark recommended (minima) time slot
                fig.add_scatter(x=[time_labels[recommendation.recommended_time_slot]],
                               y=[all_predictions[recommendation.recommended_time_slot]],
                               mode='markers', marker=dict(color='red', size=16, symbol='star'),
                               name='Recommended (Min)')
                fig.add_vline(x=recommendation.recommended_time_slot, 
                            line_dash="dash", line_color="red",
                            annotation_text="Recommended")
                fig.update_xaxes(tickangle=45)
                fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
                st.plotly_chart(fig, use_container_width=True)

def show_traffic_analysis(data):
    """Show detailed traffic analysis"""
    st.header("📈 Traffic Analysis")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_split = st.selectbox("Select Split:", ['All'] + list(data['split_id'].unique()))
    
    with col2:
        selected_day = st.selectbox("Select Day:", ['All'] + ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    with col3:
        date_range = st.date_input("Date Range:", 
                                  value=(data['date'].min(), data['date'].max()),
                                  min_value=data['date'].min(),
                                  max_value=data['date'].max())
    
    # Filter data
    filtered_data = data.copy()
    
    if selected_split != 'All':
        filtered_data = filtered_data[filtered_data['split_id'] == selected_split]
    
    if selected_day != 'All':
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                  'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        filtered_data = filtered_data[filtered_data['day_of_week'] == day_map[selected_day]]
    
    if len(date_range) == 2:
        filtered_data = filtered_data[
            (filtered_data['date'] >= pd.to_datetime(date_range[0])) &
            (filtered_data['date'] <= pd.to_datetime(date_range[1]))
        ]
    
    # Analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Time series
        if len(filtered_data) > 0:
            daily_traffic = filtered_data.groupby('date')['visitor_count'].sum().reset_index()
            fig = px.line(daily_traffic, x='date', y='visitor_count',
                         title='Daily Attendance Trend')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Heatmap
        if len(filtered_data) > 0:
            heatmap_data = filtered_data.pivot_table(
                values='visitor_count',
                index='time_slot',
                columns='day_of_week',
                aggfunc='mean'
            ).fillna(0)
            
            fig = px.imshow(heatmap_data, 
                           title='Attendance Heatmap: Time vs Day',
                           labels=dict(x="Day of Week", y="Time Slot", color="Average Visitors"))
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    if len(filtered_data) > 0:
        st.subheader("Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Visitors", f"{filtered_data['visitor_count'].sum():,}")
        
        with col2:
            st.metric("Average per Time Slot", f"{filtered_data['visitor_count'].mean():.1f}")
        
        with col3:
            st.metric("Peak Attendance", f"{filtered_data['visitor_count'].max():.0f}")
        
        with col4:
            st.metric("Standard Deviation", f"{filtered_data['visitor_count'].std():.1f}")

def show_model_performance(models):
    """Show model performance metrics"""
    st.header("⚙️ Model Performance")
    
    if not models or not models.split_models:
        st.error("No trained models available.")
        return
    
    # Model metrics
    st.subheader("Model Performance Metrics")
    
    metrics_data = []
    for split_id, model_info in models.split_models.items():
        metrics = model_info['metrics']
        metrics_data.append({
            'Split': split_id,
            'MAE': metrics.get('mae', 0),
            'RMSE': metrics.get('rmse', 0),
            'R²': metrics.get('r2', 0)
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display metrics table
    st.dataframe(metrics_df, use_container_width=True)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(metrics_df, x='Split', y='MAE',
                    title='Mean Absolute Error by Split')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(metrics_df, x='Split', y='R²',
                    title='R² Score by Split')
        st.plotly_chart(fig, use_container_width=True)
    
    # Model details
    st.subheader("Model Details")
    
    for split_id, model_info in models.split_models.items():
        with st.expander(f"📊 {split_id} Model Details"):
            st.write(f"**Model Type:** Ensemble (XGBoost + Random Forest + Linear Regression)")
            st.write(f"**Features:** {len(model_info['feature_cols'])}")
            st.write(f"**Training Status:** {'✅ Trained' if model_info['model'].is_trained else '❌ Not Trained'}")
            
            # Feature importance (if available)
            if hasattr(model_info['model'], 'models') and len(model_info['model'].models) > 0:
                xgb_model = model_info['model'].models[0]
                if hasattr(xgb_model, 'model') and hasattr(xgb_model.model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'Feature': model_info['feature_cols'],
                        'Importance': xgb_model.model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.write("**Top Features:**")
                    st.dataframe(feature_importance.head(10))

if __name__ == "__main__":
    main() 