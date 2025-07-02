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

## Performance

The models are pretty good at predicting gym attendance:

| Workout Type | Accuracy (R²) | Error (MAE) |
|--------------|---------------|-------------|
| Arms | 88.2% | 1.2 people |
| Back | 93.1% | 1.9 people |
| Cardio | 92.2% | 1.8 people |
| Chest | 92.1% | 1.7 people |
| Core | 91.2% | 1.6 people |
| General | 97.3% | 5.2 people |
| Legs | 94.7% | 1.7 people |

**Average accuracy: 92.6%** - which is pretty solid for predicting gym attendance!

## Setup

### Quick Start

```bash
# Install stuff
pip install -r requirements.txt

# Train the models (first time only)
python train_and_test.py

# Start the web app
python run_production.py
```

Then open http://localhost:8501 in your browser.

### What you need

- Python 3.8+
- Some RAM (4GB+ recommended)
- The CSV files in the `week_data/` folder

## Usage

1. **Make a prediction**: Pick a workout type, date, and time
2. **Get recommendations**: Tell it your preferred times, get smart suggestions
3. **Check the charts**: See performance and attendance patterns
4. **Avoid crowds**: Use the recommendations to find quiet times

## Current Status

**Production Ready** ✅

- All models trained and working
- Web interface functional
- Smart recommendations working
- Error handling in place
- Documentation complete

## What's Working

- ✅ Attendance prediction for all workout splits
- ✅ Smart workout time recommendations
- ✅ Web interface with real-time predictions
- ✅ Performance visualizations
- ✅ Production deployment scripts
- ✅ Error handling and logging

## Future Plans

- Add more gym locations
- Mobile app version
- Integration with gym management systems
- Real-time data feeds
- More workout types
- Advanced analytics dashboard

## Technical Stuff

The system uses ensemble learning, which means it combines multiple machine learning algorithms to get better predictions. It's like having multiple experts vote on the answer instead of just one.

The models are trained on historical gym attendance data and can predict:
- How many people will be at the gym
- When it will be crowded
- The best times for different workouts
- Confidence levels for predictions

## Files Overview

- `train_and_test.py` - Trains all the models
- `run_production.py` - Starts the production server
- `app.py` - The web interface
- `src/` - All the core modules
- `models/` - Saved trained models
- `outputs/` - Performance charts
- `week_data/` - Training data CSV files

## Testing

The system has been tested with:
- 6,739 records of gym attendance data
- 7 different workout splits
- Multiple time periods and days
- Various attendance patterns

Results show it's pretty accurate at predicting gym attendance, which is useful for both gym owners (staffing) and members (avoiding crowds).

---

*Built for gym-goers who want to avoid the crowds and gym owners who want to optimize staffing* 🏋️‍♂️
