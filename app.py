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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_processor import DataProcessor
from models import SplitSpecificModels, EnsembleModel

st.set_page_config(page_title="Fit Forecaster", page_icon="🏋️", layout="wide")
st.title("🏋️ Fit Forecaster - Gym Attendance Prediction")

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
            # Load ensemble
            from models import XGBoostModel, RandomForestModel, LinearRegressionModel, ProphetModel, EnsembleModel
            models = [XGBoostModel(), RandomForestModel(), LinearRegressionModel()]
            try:
                models.append(ProphetModel())
            except:
                pass
            ensemble = EnsembleModel(models)
            ensemble.load_ensemble(split_path)
            split_models.split_models[split_dir.title()]['model'] = ensemble
    return split_models

data = load_processed_data()
models = load_models()

# --- User Inputs ---
st.sidebar.header("Prediction Settings")
selected_date = st.sidebar.date_input("Select Date", value=datetime.today())
selected_day_of_week = pd.to_datetime(selected_date).weekday()
selected_month = pd.to_datetime(selected_date).month

splits = sorted(list(models.split_models.keys()))
selected_split = st.sidebar.selectbox("Select Workout Split", splits)

range_label = st.sidebar.radio(
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
st.success(f"Best time to visit the gym for {selected_split} on {selected_date.strftime('%A')} in your selected range is: {best_time_str} (Predicted attendance: {min_pred:.1f})") 