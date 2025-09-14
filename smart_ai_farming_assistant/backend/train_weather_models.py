#!/usr/bin/env python3
"""
Train weather prediction models using Indian cities weather data.
Dataset: https://www.kaggle.com/datasets/mukeshdevrath007/indian-5000-cities-weather-data
Creates models/weather_models.joblib for offline predictions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_indian_weather_data(csv_path='data/indian_weather_data.csv'):
    """Load and preprocess Indian weather dataset"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} weather records from Indian cities dataset")
        
        # Display column names to understand the structure
        print("Dataset columns:", df.columns.tolist())
        print("Dataset shape:", df.shape)
        print("Sample data:")
        print(df.head())
        
        # Basic data cleaning - be less aggressive with dropna
        original_len = len(df)
        df = df.dropna(thresh=len(df.columns)*0.7)  # Keep rows with at least 70% non-null values
        print(f"After cleaning: {len(df)} records (removed {original_len - len(df)} records)")
        
        # Extract month from date if date column exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['month'] = df['date'].dt.month
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['month'] = df['Date'].dt.month
        
        # Encode categorical variables if they exist
        le_city = LabelEncoder()
        if 'city' in df.columns:
            df['city_encoded'] = le_city.fit_transform(df['city'].astype(str))
        elif 'City' in df.columns:
            df['city_encoded'] = le_city.fit_transform(df['City'].astype(str))
        
        return df
        
    except FileNotFoundError:
        print(f"Dataset not found at {csv_path}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/mukeshdevrath007/indian-5000-cities-weather-data")
        print("And place it in the data/ directory")
        
        # Create a comprehensive sample CSV with expected structure
        print("Creating sample CSV structure...")
        
        # Generate more comprehensive sample data for better model training
        np.random.seed(42)
        n_samples = 1000
        
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
        months = list(range(1, 13))
        
        sample_data = {
            'City': np.random.choice(cities, n_samples),
            'Latitude': np.random.uniform(8, 35, n_samples),  # India's latitude range
            'Longitude': np.random.uniform(68, 97, n_samples),  # India's longitude range
            'month': np.random.choice(months, n_samples),
        }
        
        # Generate realistic weather data based on Indian patterns
        sample_df = pd.DataFrame(sample_data)
        
        # Temperature: varies by latitude and season
        seasonal_temp = 15 * np.sin(2 * np.pi * (sample_df['month'] - 3) / 12)
        latitude_temp = 35 - (sample_df['Latitude'] - 8) * 0.8
        sample_df['Temperature'] = (latitude_temp + seasonal_temp + np.random.normal(0, 3, n_samples)).clip(5, 50)
        
        # Humidity: higher in coastal areas and monsoon
        coastal_humidity = 60 + np.random.uniform(0, 20, n_samples)
        monsoon_humidity = np.where(sample_df['month'].isin([6, 7, 8, 9]), 15, 0)
        sample_df['Humidity'] = (coastal_humidity + monsoon_humidity + np.random.normal(0, 5, n_samples)).clip(30, 95)
        
        # Rainfall: higher during monsoon
        monsoon_factor = np.where(sample_df['month'].isin([6, 7, 8, 9]), 3.0, 0.3)
        base_rainfall = np.random.uniform(10, 100, n_samples)
        sample_df['Rainfall'] = (base_rainfall * monsoon_factor).clip(0, 300)
        
        # Wind Speed
        sample_df['Wind Speed'] = np.random.uniform(5, 25, n_samples)
        
        # Pressure
        sample_df['Pressure'] = np.random.uniform(1008, 1018, n_samples)
        
        # Encode cities
        le_city = LabelEncoder()
        sample_df['city_encoded'] = le_city.fit_transform(sample_df['City'])
        
        os.makedirs('data', exist_ok=True)
        sample_df.to_csv('data/sample_weather_data.csv', index=False)
        print(f"Created sample_weather_data.csv with {len(sample_df)} records")
        
        return sample_df

def prepare_features(df):
    """Prepare features based on available columns in the dataset"""
    # Common possible column names in Indian weather datasets
    column_mapping = {
        'temperature': ['temp', 'temperature', 'Temperature', 'temp_c'],
        'humidity': ['humidity', 'Humidity', 'humid'],
        'rainfall': ['rainfall', 'Rainfall', 'rain', 'precipitation'],
        'wind_speed': ['wind_speed', 'Wind Speed', 'wind', 'wind_kph'],
        'pressure': ['pressure', 'Pressure', 'atm_pressure'],
        'latitude': ['lat', 'latitude', 'Latitude'],
        'longitude': ['lon', 'longitude', 'Longitude']
    }
    
    # Find actual column names
    feature_cols = []
    target_cols = {}
    
    for standard_name, possible_names in column_mapping.items():
        for col in possible_names:
            if col in df.columns:
                if standard_name in ['temperature', 'rainfall']:
                    target_cols[standard_name] = col
                else:
                    feature_cols.append(col)
                break
    
    # Add engineered features
    if 'month' in df.columns:
        feature_cols.append('month')
    if 'city_encoded' in df.columns:
        feature_cols.append('city_encoded')
    
    print(f"Using features: {feature_cols}")
    print(f"Target variables: {target_cols}")
    
    return feature_cols, target_cols

def train_models():
    """Train and save weather prediction models using Indian dataset"""
    print("Loading Indian weather dataset...")
    
    # Try to load from the provided CSV file
    csv_path = 'data/sample_weather_data.csv'
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from: {csv_path}")
        print(f"Dataset shape: {df.shape}")
        print("Dataset columns:", df.columns.tolist())
        print("Sample data:")
        print(df.head())
        
    except FileNotFoundError:
        print(f"CSV file not found. Creating comprehensive training dataset...")
        
        # Create larger, more realistic dataset
        np.random.seed(42)
        n_samples = 2000
        
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
        months = list(range(1, 13))
        
        # Generate coordinate data for major Indian cities
        city_coords = {
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.7041, 77.1025),
            'Bangalore': (12.9716, 77.5946),
            'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639),
            'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567),
            'Ahmedabad': (23.0225, 72.5714),
            'Jaipur': (26.9124, 75.7873),
            'Lucknow': (26.8467, 80.9462)
        }
        
        sample_data = []
        for i in range(n_samples):
            city = np.random.choice(cities)
            lat, lon = city_coords[city]
            month = np.random.choice(months)
            
            # Add realistic variations to coordinates
            lat += np.random.uniform(-2, 2)
            lon += np.random.uniform(-2, 2)
            
            sample_data.append({
                'City': city,
                'Latitude': lat,
                'Longitude': lon,
                'month': month
            })
        
        df = pd.DataFrame(sample_data)
        
        # Generate realistic weather data based on Indian patterns
        for idx, row in df.iterrows():
            lat, lon, month = row['Latitude'], row['Longitude'], row['month']
            
            # Temperature generation with regional patterns
            if lat > 28:  # North India
                temp_base = [15, 18, 25, 32, 38, 35, 32, 31, 30, 26, 20, 16]
            elif lat > 23:  # Central India
                temp_base = [20, 23, 28, 33, 37, 33, 30, 29, 30, 28, 24, 21]
            else:  # South India
                temp_base = [24, 26, 29, 31, 33, 30, 28, 28, 29, 28, 26, 25]
            
            df.at[idx, 'Temperature'] = temp_base[month-1] + np.random.normal(0, 2)
            
            # Humidity generation
            if month in [6, 7, 8, 9]:  # Monsoon
                humidity = np.random.uniform(75, 95)
            elif month in [12, 1, 2]:  # Winter
                humidity = np.random.uniform(50, 75)
            else:  # Summer/post-monsoon
                humidity = np.random.uniform(60, 80)
            df.at[idx, 'Humidity'] = humidity
            
            # Rainfall generation with monsoon patterns
            if month in [6, 7, 8]:  # Peak monsoon
                if lat > 25:  # North
                    base_rain = 120 if month == 7 else 80
                else:  # South/Central
                    base_rain = 180 if month == 7 else 140
                rainfall = max(0, np.random.normal(base_rain, base_rain * 0.4))
            elif month == 9:  # Late monsoon
                rainfall = max(0, np.random.normal(60, 30))
            elif month in [10, 11]:  # Post-monsoon
                if lat < 15:  # South India northeast monsoon
                    rainfall = max(0, np.random.normal(100, 40))
                else:
                    rainfall = max(0, np.random.normal(25, 15))
            elif month in [12, 1, 2]:  # Winter
                rainfall = max(0, np.random.normal(10, 8))
            else:  # Summer
                rainfall = max(0, np.random.normal(15, 12))
            
            df.at[idx, 'Rainfall'] = rainfall
            
            # Wind speed
            wind_base = 15 if month in [6, 7, 8, 9] else 10
            df.at[idx, 'Wind Speed'] = max(5, np.random.normal(wind_base, 5))
            
            # Pressure
            pressure_base = 1010 if month in [6, 7, 8, 9] else 1013
            df.at[idx, 'Pressure'] = np.random.normal(pressure_base, 3)
        
        # Encode cities
        le_city = LabelEncoder()
        df['city_encoded'] = le_city.fit_transform(df['City'])
        
        # Save the generated dataset
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/sample_weather_data.csv', index=False)
        print(f"Created comprehensive dataset with {len(df)} records")
    
    if df is None or len(df) == 0:
        print("No weather dataset available. Cannot proceed with training.")
        return
    
    # Prepare features for training
    feature_cols = ['Latitude', 'Longitude', 'month', 'Humidity', 'Wind Speed', 'Pressure', 'city_encoded']
    target_cols = {'rainfall': 'Rainfall', 'temperature': 'Temperature'}
    
    # Ensure all required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return
    
    X = df[feature_cols].fillna(0)
    models = {'features': feature_cols}
    
    print(f"Training with {len(X)} samples and {len(feature_cols)} features")
    print(f"Feature columns: {feature_cols}")
    
    # Train rainfall model
    if 'Rainfall' in df.columns:
        print("Training rainfall model...")
        y_rain = df['Rainfall'].fillna(0)
        
        if y_rain.std() > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y_rain, test_size=0.2, random_state=42)
            
            rain_model = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42, min_samples_split=5)
            rain_model.fit(X_train, y_train)
            rain_score = rain_model.score(X_test, y_test)
            print(f"Rainfall model R² score: {rain_score:.3f}")
            models['rainfall'] = rain_model
            
            # Print feature importance
            feature_importance = list(zip(feature_cols, rain_model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            print("Rainfall model feature importance:")
            for feat, imp in feature_importance:
                print(f"  {feat}: {imp:.3f}")
        else:
            print("No variation in rainfall data - skipping rainfall model")
    
    # Train temperature model
    if 'Temperature' in df.columns:
        print("Training temperature model...")
        y_temp = df['Temperature'].fillna(df['Temperature'].mean())
        
        if y_temp.std() > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
            
            temp_model = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42, min_samples_split=5)
            temp_model.fit(X_train, y_train)
            temp_score = temp_model.score(X_test, y_test)
            print(f"Temperature model R² score: {temp_score:.3f}")
            models['temperature'] = temp_model
            
            # Print feature importance
            feature_importance = list(zip(feature_cols, temp_model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            print("Temperature model feature importance:")
            for feat, imp in feature_importance:
                print(f"  {feat}: {imp:.3f}")
        else:
            print("No variation in temperature data - skipping temperature model")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(models, 'models/weather_models.joblib')
    print("Models saved to models/weather_models.joblib")
    print(f"Training completed with {len(df)} samples")
    
    # Test the models with sample predictions for different locations and seasons
    print("\nTesting models with sample predictions:")
    test_locations = [
        (19.0760, 72.8777, 7, "Mumbai, July"),    # Mumbai in monsoon
        (28.7041, 77.1025, 1, "Delhi, January"),  # Delhi in winter
        (13.0827, 80.2707, 4, "Chennai, April"),  # Chennai in summer
        (22.5726, 88.3639, 10, "Kolkata, October") # Kolkata post-monsoon
    ]
    
    for lat, lon, month, description in test_locations:
        try:
            if 'rainfall' in models and 'temperature' in models:
                # Create sample features
                sample_features = np.array([[
                    lat, lon, month, 
                    75, 15, 1012,  # Default humidity, wind, pressure
                    abs(hash(f"{lat:.2f},{lon:.2f}")) % 10  # city_encoded
                ]])
                
                rain_pred = models['rainfall'].predict(sample_features)[0]
                temp_pred = models['temperature'].predict(sample_features)[0]
                print(f"{description}: Rainfall={rain_pred:.1f}mm, Temperature={temp_pred:.1f}°C")
        except Exception as e:
            print(f"Error testing {description}: {e}")

if __name__ == "__main__":
    train_models()