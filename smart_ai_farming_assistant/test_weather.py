#!/usr/bin/env python3
"""
Test script to demonstrate weather prediction system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
import joblib
from pathlib import Path

def test_weather_prediction():
    """Test the weather prediction system"""
    
    print("🌤️ Weather Prediction System Test")
    print("=" * 50)
    
    # Load the trained models
    try:
        models_path = Path(__file__).parent / 'models/weather_models.joblib'
        weather_models = joblib.load(models_path)
        print("✅ Models loaded successfully")
        print(f"   Features: {weather_models['features']}")
        print(f"   Rainfall model: {type(weather_models['rainfall']).__name__}")
        print(f"   Temperature model: {type(weather_models['temperature']).__name__}")
    except FileNotFoundError:
        print("❌ Models not found. Run train_weather_models.py first")
        return
    
    print("\n🧪 Testing Different Locations:")
    print("-" * 30)
    
    # Test cases: [lat, lon, month, description]
    test_cases = [
        [19.0760, 72.8777, 7, "Mumbai, July (Monsoon)"],
        [28.6139, 77.2090, 1, "Delhi, January (Winter)"],
        [13.0827, 80.2707, 4, "Chennai, April (Summer)"],
        [22.5726, 88.3639, 8, "Kolkata, August (Monsoon)"],
        [12.9716, 77.5946, 10, "Bangalore, October (Post-monsoon)"]
    ]
    
    # Default values for missing features
    defaults = {
        'humidity': 65,
        'cloud_cover': 40,
        'wind_kph': 15,
        'sun_hours': 8
    }
    
    for lat, lon, month, description in test_cases:
        print(f"\n📍 {description}")
        
        # Prepare features
        features = {
            'lat': lat,
            'lon': lon,
            'month': month,
            'humidity': defaults['humidity'],
            'cloud_cover': defaults['cloud_cover'],
            'wind_kph': defaults['wind_kph'],
            'sun_hours': defaults['sun_hours']
        }
        
        # Create input array in correct order
        X = np.array([[features[f] for f in weather_models['features']]])
        
        # Predict
        rain_mm = max(0, weather_models['rainfall'].predict(X)[0])
        temp_c = weather_models['temperature'].predict(X)[0]
        
        print(f"   🌧️  Rainfall: {rain_mm:.1f} mm")
        print(f"   🌡️  Temperature: {temp_c:.1f}°C")
        
        # Show reasoning
        if month in [6, 7, 8, 9]:
            season = "Monsoon season"
        elif month in [12, 1, 2]:
            season = "Winter season"
        elif month in [3, 4, 5]:
            season = "Summer season"
        else:
            season = "Post-monsoon season"
            
        print(f"   💭 Reasoning: {season}, Latitude {lat}°")
    
    print("\n🔍 How the Model Works:")
    print("-" * 25)
    print("1. Takes 7 input features: lat, lon, month, humidity, cloud_cover, wind_kph, sun_hours")
    print("2. Uses Random Forest algorithm trained on 5000 synthetic weather records")
    print("3. Predicts rainfall based on monsoon patterns and tropical factors")
    print("4. Predicts temperature based on latitude and seasonal variations")
    print("5. Results are approximate - based on simplified weather patterns")
    
    print("\n⚠️  Limitations:")
    print("-" * 15)
    print("• Uses synthetic (fake) training data, not real weather history")
    print("• Simplified relationships may not match real meteorology")
    print("• Missing important factors like pressure, elevation, etc.")
    print("• Best used as rough estimates, not precise forecasts")

if __name__ == "__main__":
    test_weather_prediction()