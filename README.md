# 🚴‍♂️ Florida Bike Rental Demand Prediction

A comprehensive machine learning project analyzing hourly bike rental patterns to forecast demand using weather and temporal features.

## 📊 Project Overview

This project analyzes 8,760 hourly bike rental records from Florida to identify patterns and build predictive models for demand forecasting. The analysis reveals strong temporal patterns and weather dependencies that can inform operational decisions.

## 🎯 Key Results

### Model Performance
- **Best Model:** Polynomial Lasso with Cross-Validation
- **Test MAE:** 237 bikes/hour
- **Test R²:** 0.21 (explaining 21% of variance)
- **Most Balanced Model:** ElasticNet (R² = 0.59, minimal overfitting)

### Business Insights
- **Peak Hours:** 8 AM and 6 PM (commuter patterns)
- **Weather Impact:** Rain reduces demand by ~80%
- **Seasonal Variation:** Summer demand 5x higher than winter
- **Optimal Conditions:** 20-25°C with moderate humidity

## 📁 Project Structure

```
BikeRentals/
├── Data/
│   ├── FloridaBikeRentals.csv          # Raw data (8,760 observations)
│   └── bike_rental_features.csv        # Engineered features
├── Notebooks/
│   ├── BikeRentalAnalysis_Final.ipynb  # Main analysis notebook
│   └── analysis.ipynb                  # Exploratory work
├── models/
│   └── best_bike_rental_model.pkl      # Trained Poly Lasso model
├── .gitignore
├── README.md
└── requirements.txt
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Cait-Dris/BikeRentals.git
cd BikeRentals
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start
```python
# Run the complete analysis
python professional_analysis.py

# Or open the notebook
jupyter notebook Notebooks/BikeRentalAnalysis_Final.ipynb
```

### Load Saved Model
```python
import joblib
import pandas as pd

# Load model
model_data = joblib.load('models/best_bike_rental_model.pkl')
model = model_data['model']
scaler = model_data['scaler']

# Make predictions
# (requires feature engineering pipeline)
```

## 📈 Analysis Pipeline

### 1. Data Preprocessing
- Temporal feature extraction (hour, day, month, season)
- Cyclical encoding for time features
- Weather data cleaning and validation

### 2. Feature Engineering
- **Interaction Features:** Hour×Temperature, Temperature×Humidity
- **Comfort Index:** Composite metric combining temperature, humidity, and wind
- **Binary Indicators:** Rush hour, weekend, bad weather, extreme temperatures
- **Cyclical Encodings:** Sin/cos transformations for temporal features

### 3. Models Evaluated
| Model | Test MAE | Test R² | Training Time |
|-------|----------|---------|---------------|
| Polynomial Lasso | 237.03 | 0.21 | 6.56s |
| ElasticNet | 290.26 | 0.59 | 1.46s |
| Polynomial ElasticNet | 323.63 | 0.41 | 8.86s |
| Linear Regression | 757.99 | -0.98 | 0.00s |

### 4. Key Features (by importance)
1. Hour_Temperature interaction
2. Comfort_Index
3. Temperature
4. Hour of day
5. Summer_Temperature interaction

## 🔍 Key Findings

### Temporal Patterns
- **Bimodal daily pattern:** Morning (8 AM) and evening (6 PM) peaks
- **Weekday vs Weekend:** 15% higher demand on weekdays
- **Seasonal:** Peak in June (~1200 rentals), lowest in January (~200)

### Weather Impact
- **Temperature:** Optimal range 20-25°C
- **Rain:** Reduces rentals by 80%
- **Humidity:** Negative correlation, optimal 40-60%
- **Wind:** High winds (>5 m/s) decrease rentals by 25%

### Model Insights
- Non-linear relationships require polynomial features
- Regularization (Lasso/ElasticNet) prevents overfitting
- Time-weather interactions are crucial predictors

## 💡 Business Recommendations

1. **Fleet Management**
   - Increase availability 30% during rush hours
   - Reduce fleet 15% during rain forecasts
   - Seasonal adjustments: 5x capacity for summer

2. **Pricing Strategy**
   - Surge pricing (10-15%) during peak hours
   - Weather-based discounts (20% during rain)
   - Seasonal pricing adjustments

3. **Operations**
   - Schedule 70% maintenance 2-5 AM (lowest demand)
   - Pre-position bikes before rush hours
   - Weather-responsive redistribution

## 🔧 Technical Details

### Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- See `requirements.txt` for complete list

### Data Requirements
- Input: Hourly rental counts with weather data
- Features: 14 original + 30+ engineered features
- Target: Rented Bike Count (hourly)

## 📊 Visualizations

The analysis includes:
- Distribution analysis of rentals
- Temporal pattern visualizations
- Weather impact analysis
- Feature correlation heatmaps
- Model performance comparisons
- Residual analysis plots

## 🚧 Future Improvements

1. **Advanced Models**
   - Random Forest / XGBoost for non-linear patterns
   - LSTM for time series forecasting
   - Ensemble methods

2. **Additional Features**
   - Local events calendar
   - Public transport schedules
   - Competition/alternative transport data

3. **Real-time Implementation**
   - API for live predictions
   - Dashboard for monitoring
   - Automated retraining pipeline

## 👤 Author

**Caitlin Driscoll**
- GitHub: [My GitHub](https://github.com/Cait-Dris)
- Project Date: May 2025

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Data source: Fullstack Academy Provided Dataset
- Inspired by bike-sharing demand forecasting research
- Built with scikit-learn and pandas

---

*For questions or collaboration, please open an issue or contact the author.*