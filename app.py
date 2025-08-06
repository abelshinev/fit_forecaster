"""
StreamLit Application for Fit Forecaster
Provides user-friendly interface for gym attendance forecasting and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_processor import DataProcessor
from models import SplitSpecificModels, EnsembleModel
from realtime_learning import RealTimeLearning, FeedbackCollector

st.set_page_config(page_title="Fit Forecaster", page_icon="↗️", layout="wide")
st.title("🏋️ Fit Forecaster - Gym Attendance Prediction")

# Initialize real-time learning
@st.cache_resource
def initialize_realtime_learning():
    return RealTimeLearning()

@st.cache_resource
def initialize_feedback_collector():
    realtime_learner = initialize_realtime_learning()
    return FeedbackCollector(realtime_learner)

realtime_learner = initialize_realtime_learning()
feedback_collector = initialize_feedback_collector()

# --- Load Data and Models ---
@st.cache_data
def load_processed_data():
    data_processor = DataProcessor()
    file_paths = {name[:-7]: os.path.join('week_data', fname) for name, fname in zip(
        ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
        ['mongym.csv', 'tuegym.csv', 'wedgym.csv', 'thugym.csv', 'frigym.csv', 'satgym.csv', 'sungym.csv'])
        if os.path.exists(os.path.join('week_data', fname))}
    return data_processor.process_historical_data(file_paths)

@st.cache_data
def load_models():
    models_dir = "models"
    split_models = SplitSpecificModels()
    for split_dir in os.listdir(models_dir):
        split_path = os.path.join(models_dir, split_dir)
        if os.path.isdir(split_path):
            split_models.split_models[split_dir.title()] = {
                'model': None,
                'feature_cols': [
                    'time_slot', 'day_of_week', 'month', 'is_weekend',
                    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                    'prev_day_attendance', 'prev_week_attendance',
                    'rolling_3day_avg', 'rolling_7day_avg'
                ]
            }
            # Load ensemble (Prophet removed for optimal performance)
            from models import XGBoostModel, RandomForestModel, LinearRegressionModel, EnsembleModel
            models = [XGBoostModel(), RandomForestModel(), LinearRegressionModel()]
            ensemble = EnsembleModel(models)
            ensemble.load_ensemble(split_path)
            split_models.split_models[split_dir.title()]['model'] = ensemble
    return split_models

data = load_processed_data()
models = load_models()

# --- Sidebar for Settings and Feedback ---
st.sidebar.header("Settings & Feedback")

# User ID for tracking
user_id = st.sidebar.number_input("User ID (for feedback tracking)", min_value=1, value=1)

# Feedback section
st.sidebar.subheader("📊 Learning Feedback")
feedback_tab1, feedback_tab2 = st.sidebar.tabs(["Report Attendance", "Learning Status"])

with feedback_tab1:
    st.write("Help improve predictions by reporting actual attendance!")
    
    # Date and time selection for feedback
    feedback_date = st.date_input("Date of visit", value=datetime.today())
    feedback_split = st.selectbox("Workout split", sorted(list(models.split_models.keys())))
    feedback_time = st.selectbox("Time slot", 
                                [f"{i//2:02d}:{(i%2)*30:02d}" for i in range(48)],
                                index=16)  # Default to 8:00
    
    # Convert time string to slot
    time_parts = feedback_time.split(':')
    feedback_time_slot = int(time_parts[0]) * 2 + (int(time_parts[1]) // 30)
    
    actual_attendance = st.number_input("Actual attendance count", min_value=0, value=10)
    
    if st.button("📈 Submit Attendance Report"):
        # Collect feedback
        feedback_id = feedback_collector.collect_prediction_feedback(
            feedback_date, feedback_split, feedback_time_slot, 0, user_id
        )
        
        # Update with actual attendance
        if feedback_collector.collect_attendance_feedback(feedback_id, actual_attendance):
            st.success(f"✅ Attendance report submitted! Feedback ID: {feedback_id}")
        else:
            st.error("❌ Failed to submit attendance report")

with feedback_tab2:
    # Show learning status
    summary = realtime_learner.get_feedback_summary()
    
    st.metric("Total Feedback", summary['total_feedback'])
    st.metric("Pending Feedback", summary['pending_feedback'])
    
    if summary['recent_feedback']:
        st.write("**Recent Feedback:**")
        for feedback in summary['recent_feedback'][-5:]:  # Show last 5
            st.write(f"• {feedback['split']} at {feedback['time_slot']//2:02d}:{(feedback['time_slot']%2)*30:02d} - {feedback.get('actual_attendance', 'N/A')} people")
    
    if st.button("🔄 Force Retraining"):
        realtime_learner.force_retraining()
        st.success("✅ Models retrained with latest feedback!")

# --- Main Prediction Interface ---
st.header("🎯 Gym Attendance Prediction")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input("Select Date", value=datetime.today())
    selected_day_of_week = pd.to_datetime(selected_date).weekday()
    selected_month = pd.to_datetime(selected_date).month

    splits = sorted(list(models.split_models.keys()))
    selected_split = st.selectbox("Select Workout Split", splits)

with col2:
    range_label = st.radio(
        "Select Time Range:",
        ["Morning (6 AM - 10 AM)", "Afternoon (12 PM - 4 PM)", "Evening (6 PM - 10 PM)"],
        index=2
    )
    if range_label == "Morning (6 AM - 10 AM)":
        preferred_times = list(range(12, 20))  # 6:00 to 10:00
    elif range_label == "Afternoon (12 PM - 4 PM)":
        preferred_times = list(range(24, 32))  # 12:00 to 16:00
    else:
        preferred_times = list(range(36, 44))  # 18:00 to 22:00

# --- Prediction ---
model_info = models.split_models[selected_split]
prediction_model = model_info['model']
feature_cols = model_info['feature_cols']

all_predictions = []
time_labels = [f"{i//2:02d}:{(i%2)*30:02d}" for i in range(48)]

for time_slot in range(48):
    # Use most recent available rolling features for this split and slot
    row = data[(data['split_id'] == selected_split) & (data['time_slot'] == time_slot)].sort_values('date').tail(1)
    if not row.empty:
        prev_day_attendance = row.iloc[0].get('prev_day_attendance', 0)
        prev_week_attendance = row.iloc[0].get('prev_week_attendance', 0)
        rolling_3day_avg = row.iloc[0].get('rolling_3day_avg', 0)
        rolling_7day_avg = row.iloc[0].get('rolling_7day_avg', 0)
    else:
        prev_day_attendance = 0
        prev_week_attendance = 0
        rolling_3day_avg = 0
        rolling_7day_avg = 0
    
    features = pd.DataFrame({
        'time_slot': [time_slot],
        'day_of_week': [selected_day_of_week],
        'month': [selected_month],
        'is_weekend': [1 if selected_day_of_week >= 5 else 0],
        'hour_sin': [np.sin(2 * np.pi * time_slot / 48)],
        'hour_cos': [np.cos(2 * np.pi * time_slot / 48)],
        'day_sin': [np.sin(2 * np.pi * selected_day_of_week / 7)],
        'day_cos': [np.cos(2 * np.pi * selected_day_of_week / 7)],
        'prev_day_attendance': [prev_day_attendance],
        'prev_week_attendance': [prev_week_attendance],
        'rolling_3day_avg': [rolling_3day_avg],
        'rolling_7day_avg': [rolling_7day_avg]
    })
    features = features[feature_cols]
    
    try:
        pred = prediction_model.predict(features)[0]
        pred = max(0, pred)
        all_predictions.append(pred)
    except Exception as e:
        all_predictions.append(0)

# --- Find Local Minima in Preferred Range ---
min_pred = min([all_predictions[slot] for slot in preferred_times])
best_slots = [slot for slot in preferred_times if all_predictions[slot] == min_pred]
best_slot = best_slots[0]  # Earliest if tie
best_time_str = time_labels[best_slot]

# --- Plot ---
st.subheader(f"Predicted {selected_split} Attendance on {selected_date.strftime('%A, %d %B %Y')}")
fig = px.line(x=time_labels, y=all_predictions, labels={'x': 'Time', 'y': 'Predicted Attendance'},
              title=f'Predicted {selected_split} Attendance by Time')

for slot in preferred_times:
    fig.add_scatter(x=[time_labels[slot]], y=[all_predictions[slot]],
                   mode='markers', marker=dict(color='blue', size=12, symbol='circle'),
                   name='Preferred Time')

fig.add_scatter(x=[time_labels[best_slot]], y=[all_predictions[best_slot]],
               mode='markers', marker=dict(color='red', size=16, symbol='star'),
               name='Recommended (Min)')

fig.update_xaxes(tickangle=45)
fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
fig.update_yaxes(range=[0, None])
st.plotly_chart(fig, use_container_width=True)

# --- Recommendation Output ---
st.success(f"**Recommended time:** {best_time_str} (Predicted attendance: {min_pred:.1f})")

# Create list of all times in the preferred range with their predictions
preferred_time_options = []
for slot in sorted(preferred_times):
    time_str = time_labels[slot]
    pred_value = all_predictions[slot]
    preferred_time_options.append({
        'slot': slot,
        'time': time_str,
        'prediction': pred_value,
        'display': f"{time_str} (Predicted: {pred_value:.1f})"
    })

# --- Time Selection Interface ---
st.subheader("🕐 Select Your Preferred Time")
st.info(f"Choose any time from your selected {range_label.lower()} range:")

# Create selectbox with all available times in range
selected_time_option = st.selectbox(
    "Choose your workout time:",
    options=preferred_time_options,
    format_func=lambda x: x['display'],
    index=next((i for i, opt in enumerate(preferred_time_options) if opt['slot'] == best_slot), 0),
    key="time_selector"
)

selected_user_slot = selected_time_option['slot']
selected_user_time = selected_time_option['time']
selected_user_prediction = selected_time_option['prediction']

# Show selected time info
if selected_user_slot == best_slot:
    st.success(f"✨ Great choice! You've selected our recommended time: **{selected_user_time}**")
else:
    st.info(f"📝 You've selected: **{selected_user_time}** (Predicted attendance: {selected_user_prediction:.1f})")

# --- Quick Feedback Collection ---
st.subheader("💡 Confirm Your Visit")
col3, col4 = st.columns(2)

with col3:
    if st.button(f"✅ I'll visit at {selected_user_time}"):
        # Use the user-selected time instead of recommended time
        feedback_id = feedback_collector.collect_prediction_feedback(
            selected_date, selected_split, selected_user_slot, selected_user_prediction, user_id
        )
        st.success(f"✅ Visit confirmed for {selected_user_time}! Feedback ID: {feedback_id}")

with col4:
    if st.button("🔄 I will select a different time"):
        st.session_state.show_alternative_time_selector = True

# --- Alternative Time Selection (if user wants different time) ---
if st.session_state.get('show_alternative_time_selector', False):
    st.subheader("🎯 Select Alternative Time")
    st.info(f"Choose any time from your original {range_label.lower()} range:")
    
    # Create selectbox with all times in the ORIGINAL preferred range
    alternative_time_option = st.selectbox(
        "Choose your alternative workout time:",
        options=preferred_time_options,
        format_func=lambda x: x['display'],
        index=0,  # Start with first option
        key="alternative_time_selector"
    )
    
    alternative_slot = alternative_time_option['slot']
    alternative_time = alternative_time_option['time']
    alternative_prediction = alternative_time_option['prediction']
    
    st.info(f"📝 Alternative time selected: **{alternative_time}** (Predicted attendance: {alternative_prediction:.1f})")
    
    col5, col6 = st.columns(2)
    
    with col5:
        if st.button(f"✅ Confirm visit at {alternative_time}", key="confirm_alternative"):
            # Record the alternative time selection
            feedback_id = feedback_collector.collect_prediction_feedback(
                selected_date, selected_split, alternative_slot, alternative_prediction, user_id
            )
            st.success(f"✅ Alternative visit confirmed for {alternative_time}! Feedback ID: {feedback_id}")
            st.session_state.show_alternative_time_selector = False
            st.rerun()
    
    with col6:
        if st.button("❌ Cancel", key="cancel_alternative"):
            st.session_state.show_alternative_time_selector = False
            st.rerun()

# --- Model Performance Info ---
st.sidebar.subheader("📈 Model Performance")
if st.sidebar.button("🔄 Refresh Models"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Show model info
st.sidebar.write("**Active Models:**")
for split_name in splits:
    if split_name in models.split_models:
        st.sidebar.write(f"• {split_name} ✓")
    else:
        st.sidebar.write(f"• {split_name} ❌") 