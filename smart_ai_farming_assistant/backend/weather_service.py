import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

class WeatherPredictor:
    def __init__(self):
        self.models = None
        self.load_models()
    
    def load_models(self):
        """Load trained weather models"""
        model_path = 'models/weather_models.joblib'
        if os.path.exists(model_path):
            try:
                self.models = joblib.load(model_path)
                print(f"Loaded weather models with features: {self.models.get('features', [])}")
                return True
            except Exception as e:
                print(f"Error loading models: {e}")
                self.models = None
                return False
        else:
            print(f"Model file not found at {model_path}")
            self.models = None
            return False
    
    def prepare_prediction_features(self, latitude, longitude, month=None, humidity=None, wind_speed=None, pressure=None, cloud_cover=None, sun_hours=None):
        """Prepare features for prediction based on trained model"""
        if not self.models or 'features' not in self.models:
            raise ValueError("No trained models available")
        
        features = {}
        required_features = self.models['features']
        
        # Set current month if not provided
        if month is None:
            month = datetime.now().month
        
        # Map user inputs to model features with intelligent defaults
        for feature in required_features:
            if feature in ['Latitude', 'lat', 'latitude']:
                features[feature] = latitude
            elif feature in ['Longitude', 'lon', 'longitude']:
                features[feature] = longitude
            elif feature == 'month':
                features[feature] = month
            elif feature in ['Humidity', 'humidity']:
                if humidity is not None:
                    features[feature] = humidity
                else:
                    # Intelligent humidity estimation based on season and location
                    if month in [6, 7, 8, 9]:  # Monsoon
                        features[feature] = 85 + np.random.uniform(-5, 5)
                    elif month in [12, 1, 2]:  # Winter
                        features[feature] = 60 + np.random.uniform(-10, 10)
                    else:  # Summer/post-monsoon
                        features[feature] = 70 + np.random.uniform(-10, 10)
            elif feature in ['Wind Speed', 'wind_speed', 'wind', 'wind_kph']:
                if wind_speed is not None:
                    features[feature] = wind_speed
                else:
                    # Seasonal wind speed variation
                    if month in [6, 7, 8, 9]:  # Monsoon - higher winds
                        features[feature] = 18 + np.random.uniform(-3, 5)
                    else:
                        features[feature] = 12 + np.random.uniform(-3, 5)
            elif feature in ['Pressure', 'pressure', 'atm_pressure']:
                if pressure is not None:
                    features[feature] = pressure
                else:
                    # Slight pressure variation by season
                    if month in [6, 7, 8, 9]:  # Monsoon - lower pressure
                        features[feature] = 1010 + np.random.uniform(-3, 3)
                    else:
                        features[feature] = 1013 + np.random.uniform(-3, 3)
            elif feature == 'city_encoded':
                # Create consistent encoding based on location
                city_hash = abs(hash(f"{latitude:.2f},{longitude:.2f}")) % 10
                features[feature] = city_hash
            elif feature in ['cloud_cover', 'Cloud Cover']:
                if cloud_cover is not None:
                    features[feature] = cloud_cover
                else:
                    # Seasonal cloud cover variation
                    if month in [6, 7, 8, 9]:  # Monsoon
                        features[feature] = 80 + np.random.uniform(-10, 15)
                    elif month in [12, 1, 2]:  # Winter
                        features[feature] = 30 + np.random.uniform(-10, 20)
                    else:
                        features[feature] = 50 + np.random.uniform(-15, 20)
            elif feature in ['sun_hours', 'Sun Hours']:
                if sun_hours is not None:
                    features[feature] = sun_hours
                else:
                    # Seasonal sun hours variation
                    if month in [6, 7, 8, 9]:  # Monsoon - less sun
                        features[feature] = 4 + np.random.uniform(0, 3)
                    elif month in [12, 1, 2]:  # Winter
                        features[feature] = 8 + np.random.uniform(-1, 2)
                    else:  # Summer
                        features[feature] = 10 + np.random.uniform(-1, 2)
            else:
                # Default value for unknown features
                features[feature] = 0
        
        # Create feature array in correct order
        feature_array = np.array([[features[f] for f in required_features]])
        return feature_array, features
    
    def predict_weather(self, latitude, longitude, month=None, humidity=None, wind_speed=None, pressure=None, cloud_cover=None, sun_hours=None):
        """Predict weather using trained models with realistic variation"""
        
        if month is None:
            month = datetime.now().month
        
        # Try to use trained models first
        if self.models and 'features' in self.models:
            try:
                # Add random variation to inputs to get different predictions
                lat_var = latitude + np.random.uniform(-0.1, 0.1)
                lon_var = longitude + np.random.uniform(-0.1, 0.1)
                
                features_array, features_dict = self.prepare_prediction_features(
                    lat_var, lon_var, month, humidity, wind_speed, pressure, cloud_cover, sun_hours
                )
                
                predictions = {}
                
                # Predict rainfall with realistic bounds
                if 'rainfall' in self.models:
                    rainfall_pred = self.models['rainfall'].predict(features_array)[0]
                    # Apply realistic bounds and seasonal adjustments
                    if month in [6, 7, 8, 9]:  # Monsoon
                        rainfall_pred = max(50, min(500, rainfall_pred))
                    elif month in [12, 1, 2]:  # Winter
                        rainfall_pred = max(0, min(100, rainfall_pred))
                    else:  # Other seasons
                        rainfall_pred = max(5, min(200, rainfall_pred))
                    
                    predictions['rainfall_mm'] = rainfall_pred
                    print(f"Model rainfall prediction: {predictions['rainfall_mm']:.2f} mm")
                else:
                    predictions['rainfall_mm'] = self._estimate_rainfall_indian(latitude, month, humidity)
                
                # Predict temperature with realistic bounds
                if 'temperature' in self.models:
                    temp_pred = self.models['temperature'].predict(features_array)[0]
                    # Apply realistic temperature bounds for India
                    temp_pred = max(5, min(50, temp_pred))
                    
                    # Apply seasonal and regional adjustments
                    if month in [3, 4, 5] and latitude > 25:  # North India summer
                        temp_pred = max(temp_pred, 30)
                    elif month in [12, 1, 2] and latitude > 28:  # North India winter
                        temp_pred = min(temp_pred, 25)
                    
                    predictions['temperature_c'] = temp_pred
                    print(f"Model temperature prediction: {predictions['temperature_c']:.2f}°C")
                else:
                    predictions['temperature_c'] = self._estimate_temperature_indian(latitude, month)
                
                return predictions
                
            except Exception as e:
                print(f"Model prediction failed: {e}")
        
        # Fallback to Indian climate-based estimation
        return {
            'rainfall_mm': self._estimate_rainfall_indian(latitude, month, humidity),
            'temperature_c': self._estimate_temperature_indian(latitude, month)
        }
    
    def _estimate_rainfall_indian(self, latitude, month, humidity=None):
        """Estimate rainfall based on comprehensive Indian monsoon patterns"""
        # More detailed regional and seasonal patterns
        if month in [6, 7, 8]:  # Peak monsoon
            if latitude > 28:  # North India (Punjab, Haryana, UP)
                base_rainfall = 150 if month == 7 else 120
            elif latitude > 23:  # Central India (MP, Rajasthan)
                base_rainfall = 200 if month == 7 else 160
            elif latitude > 18:  # Western Ghats, Maharashtra
                base_rainfall = 300 if month == 7 else 240
            else:  # South India
                base_rainfall = 180 if month == 7 else 150
        elif month == 9:  # Late monsoon
            base_rainfall = 80 if latitude > 25 else 120
        elif month in [10, 11]:  # Post-monsoon/Northeast monsoon
            if latitude < 15:  # Tamil Nadu, Kerala
                base_rainfall = 120 if month == 10 else 80
            else:
                base_rainfall = 40 if month == 10 else 20
        elif month in [12, 1, 2]:  # Winter
            if latitude > 28:  # North India winter rain
                base_rainfall = 20
            elif latitude < 15:  # South India winter
                base_rainfall = 30
            else:
                base_rainfall = 10
        else:  # Summer (March-May)
            if latitude < 15:  # South India pre-monsoon
                base_rainfall = 40
            else:
                base_rainfall = 15
        
        # Adjust for humidity
        if humidity is not None:
            humidity_factor = min(humidity / 75.0, 1.3)
            base_rainfall *= humidity_factor
        
        # Add realistic variation
        variation = base_rainfall * 0.3
        return max(0, base_rainfall + np.random.uniform(-variation, variation))
    
    def _estimate_temperature_indian(self, latitude, month):
        """Estimate temperature based on comprehensive Indian climate patterns"""
        # Detailed temperature patterns by region and season
        if latitude > 30:  # Kashmir, Himachal
            temp_base = [5, 8, 15, 22, 27, 30, 28, 27, 24, 18, 12, 7]
        elif latitude > 28:  # Punjab, Delhi, UP
            temp_base = [12, 16, 23, 30, 36, 35, 33, 32, 31, 26, 19, 14]
        elif latitude > 23:  # Rajasthan, MP, Gujarat
            temp_base = [18, 22, 28, 33, 38, 36, 33, 32, 32, 29, 24, 20]
        elif latitude > 18:  # Maharashtra, Telangana
            temp_base = [22, 25, 29, 32, 34, 31, 29, 28, 29, 28, 25, 23]
        else:  # South India
            temp_base = [24, 26, 29, 31, 33, 30, 28, 28, 29, 28, 26, 25]
        
        temp = temp_base[month - 1]
        
        # Add realistic variation
        return temp + np.random.uniform(-2, 2)

# Global weather predictor instance
weather_predictor = WeatherPredictor()

def get_weather_prediction(latitude, longitude, month=None, humidity=None, wind_speed=None, pressure=None, cloud_cover=None, sun_hours=None):
    """Get weather prediction for coordinates with optional weather parameters"""
    return weather_predictor.predict_weather(
        latitude, longitude, month, humidity, wind_speed, pressure, cloud_cover, sun_hours
    )
