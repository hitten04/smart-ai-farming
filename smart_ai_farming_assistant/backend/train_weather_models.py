#!/usr/bin/env python3
"""
Train tiny weather prediction models using synthetic data.
Creates models/weather_models.joblib for offline predictions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def generate_synthetic_weather_data(n_samples=5000):
    """Generate synthetic weather training data"""
    np.random.seed(42)
    
    # Features: lat, lon, month, humidity, cloud_cover, wind_kph, sun_hours
    data = {
        'lat': np.random.uniform(-60, 60, n_samples),
        'lon': np.random.uniform(-180, 180, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'humidity': np.random.uniform(20, 95, n_samples),
        'cloud_cover': np.random.uniform(0, 100, n_samples),
        'wind_kph': np.random.uniform(0, 50, n_samples),
        'sun_hours': np.random.uniform(0, 14, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Simple synthetic relationships
    # Rainfall: higher in monsoon months, tropical regions, high humidity/clouds
    monsoon_factor = np.where(df['month'].isin([6,7,8,9]), 2.0, 0.5)
    tropical_factor = 1 + np.exp(-np.abs(df['lat'])/20)
    df['rainfall_mm'] = (
        monsoon_factor * tropical_factor * 
        (df['humidity']/100) * (df['cloud_cover']/100) * 50 +
        np.random.normal(0, 10, n_samples)
    ).clip(0, 200)
    
    # Temperature: varies by latitude, season, sun hours
    seasonal_temp = 15 * np.sin(2 * np.pi * (df['month'] - 3) / 12)
    latitude_temp = 30 - np.abs(df['lat']) * 0.5
    df['temperature_c'] = (
        latitude_temp + seasonal_temp + df['sun_hours'] * 2 +
        np.random.normal(0, 3, n_samples)
    ).clip(-20, 50)
    
    return df

def train_models():
    """Train and save weather prediction models"""
    print("Generating synthetic training data...")
    df = generate_synthetic_weather_data()
    
    features = ['lat', 'lon', 'month', 'humidity', 'cloud_cover', 'wind_kph', 'sun_hours']
    X = df[features]
    
    # Train rainfall model
    print("Training rainfall model...")
    y_rain = df['rainfall_mm']
    X_train, X_test, y_train, y_test = train_test_split(X, y_rain, test_size=0.2, random_state=42)
    
    rain_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rain_model.fit(X_train, y_train)
    rain_score = rain_model.score(X_test, y_test)
    print(f"Rainfall model R² score: {rain_score:.3f}")
    
    # Train temperature model
    print("Training temperature model...")
    y_temp = df['temperature_c']
    X_train, X_test, y_train, y_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    
    temp_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    temp_model.fit(X_train, y_train)
    temp_score = temp_model.score(X_test, y_test)
    print(f"Temperature model R² score: {temp_score:.3f}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    models = {
        'rainfall': rain_model,
        'temperature': temp_model,
        'features': features
    }
    
    joblib.dump(models, 'models/weather_models.joblib')
    print("Models saved to models/weather_models.joblib")

if __name__ == "__main__":
    train_models()