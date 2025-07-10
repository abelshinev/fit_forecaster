# Fit Forecaster

A gym attendance prediction system built with Python and machine learning. Uses ensemble learning to predict how many people will be at the gym at any given time.

## What it does

- Predicts gym attendance for different workout splits (Arms, Back, Cardio, Chest, Core, General, Legs)
- Gives smart recommendations for the best times to work out
- Shows you when the gym will be crowded or quiet
- Works with 92.6% accuracy across all workout types

## Modules

### 1. **`src/data_processor.py`**

Handles all the data loading and preprocessing. Takes CSV files from different days of the week and turns them into features the models can use.

- Loads gym attendance data from CSV files
- Creates features like day of week, time slots, rolling averages
- Handles missing data and outliers
- Prepares data for training

### 2. **`src/models.py`**

Contains all the machine learning models. Uses an ensemble approach with multiple algorithms working together.

- **XGBoost**: Main prediction model for complex patterns
- **RandomForest**: Backup model for robustness
- **LinearRegression**: Simple baseline model
- **Prophet**: Time series forecasting for trends
- **LSTM**: Deep learning for sequence patterns

### 3. **`src/recommendation_engine.py`**

The smart recommendation system that tells you the best times to work out.

- Calculates gym congestion levels
- Suggests optimal workout times
- Considers your preferred times
- Manages quotas to avoid overcrowding

### 4. **`app.py`**

The web interface built with Streamlit. Where you actually use the system.

- Interactive dashboard for making predictions
- Real-time attendance forecasts
- Smart workout recommendations
- Performance charts and analytics

## How it works

1. **Data Processing**: Takes historical gym data and creates features
2. **Model Training**: Trains separate models for each workout split
3. **Predictions**: Uses ensemble of models to predict attendance
4. **Recommendations**: Suggests best times based on congestion
5. **Web Interface**: Shows everything in a nice dashboard
