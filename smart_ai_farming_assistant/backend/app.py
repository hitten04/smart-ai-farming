#!/usr/bin/env python3
"""
Smart AI Farming Assistant - FastAPI Backend
Serves API endpoints and static frontend files.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import yaml
import json
import joblib
import requests
import os
import numpy as np
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

app = FastAPI(title="Smart AI Farming Assistant", version="1.0.0")

# Load environment variables
load_dotenv()

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    gemini_model = None
    print("Warning: GEMINI_KEY not found in environment variables")

# Load data and models
def load_data():
    """Load YAML data files"""
    base_dir = Path(__file__).parent.parent
    with open(base_dir / 'data/soil_crops.yaml', 'r', encoding='utf-8') as f:
        soil_data = yaml.safe_load(f)
    with open(base_dir / 'data/schemes.yaml', 'r', encoding='utf-8') as f:
        schemes_data = yaml.safe_load(f)
    with open(base_dir / 'offline_cache/sample_weather.json', 'r', encoding='utf-8') as f:
        weather_cache = json.load(f)
    return soil_data, schemes_data, weather_cache

def load_models():
    """Load trained weather models"""
    try:
        base_dir = Path(__file__).parent.parent
        return joblib.load(base_dir / 'models/weather_models.joblib')
    except FileNotFoundError:
        return None

soil_data, schemes_data, weather_cache = load_data()
weather_models = load_models()

# Request/Response models
class WeatherRequest(BaseModel):
    lat: float
    lon: float
    month: int
    humidity: Optional[float] = None
    cloud_cover: Optional[float] = None
    wind_kph: Optional[float] = None
    sun_hours: Optional[float] = None
    offline_only: bool = True

class CropRequest(BaseModel):
    soil_type: str
    season: str

class SchemeRequest(BaseModel):
    query: Optional[str] = None
    state: Optional[str] = None

class DiseaseRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    question: str
    lang: str = 'en'
    context: Optional[str] = None

# API Endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "models_loaded": weather_models is not None,
        "gemini_available": gemini_model is not None and GEMINI_API_KEY is not None
    }

@app.post("/predict/weather")
async def predict_weather(request: WeatherRequest):
    """Predict rainfall and temperature"""
    
    # Use defaults for missing features
    defaults = weather_cache["default_features"]
    features = {
        'lat': request.lat,
        'lon': request.lon,
        'month': request.month,
        'humidity': request.humidity or defaults["humidity"],
        'cloud_cover': request.cloud_cover or defaults["cloud_cover"],
        'wind_kph': request.wind_kph or defaults["wind_kph"],
        'sun_hours': request.sun_hours or defaults["sun_hours"]
    }
    
    # Try Open-Meteo API if not offline_only
    api_data = None
    if not request.offline_only:
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={request.lat}&longitude={request.lon}&current=temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m&timezone=auto"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                api_data = response.json()
                current = api_data.get('current', {})
                features.update({
                    'humidity': current.get('relative_humidity_2m', features['humidity']),
                    'cloud_cover': current.get('cloud_cover', features['cloud_cover']),
                    'wind_kph': current.get('wind_speed_10m', 0) * 3.6  # m/s to km/h
                })
        except:
            pass  # Fall back to offline mode
    
    # Predict using models or fallback
    if weather_models:
        X = np.array([[features[f] for f in weather_models['features']]])
        rain_mm = max(0, weather_models['rainfall'].predict(X)[0])
        temp_c = weather_models['temperature'].predict(X)[0]
        why = "ML model prediction"
    else:
        # Simple fallback logic
        monsoon_months = [6, 7, 8, 9]
        if request.month in monsoon_months:
            rain_mm = features['humidity'] * 2.5
        else:
            rain_mm = features['humidity'] * 0.3
        
        # Temperature based on latitude and season
        seasonal_adj = 10 * np.sin(2 * np.pi * (request.month - 3) / 12)
        temp_c = 25 - abs(request.lat) * 0.4 + seasonal_adj
        why = "Simple heuristic (models not loaded)"
    
    return {
        "rain_mm": round(rain_mm, 1),
        "temp_c": round(temp_c, 1),
        "features": features,
        "why": why,
        "api_used": api_data is not None
    }

@app.post("/recommend/crops")
async def recommend_crops(request: CropRequest):
    """Recommend crops based on soil type and season"""
    
    soil_types = soil_data.get('soil_types', {})
    if request.soil_type not in soil_types:
        raise HTTPException(status_code=400, detail="Invalid soil type")
    
    soil_info = soil_types[request.soil_type]
    crops = soil_info.get('crops', {}).get(request.season, [])
    
    if not crops:
        raise HTTPException(status_code=400, detail="No crops found for this combination")
    
    return {
        "crops": crops,
        "soil_name": soil_info['name'],
        "characteristics": soil_info['characteristics'],
        "why": f"These crops are suitable for {soil_info['name']} during {request.season} season"
    }

@app.post("/schemes/search")
async def search_schemes(request: SchemeRequest):
    """Search government schemes"""
    
    schemes = schemes_data.get('schemes', [])
    results = []
    
    for scheme in schemes:
        # Filter by state
        if request.state and scheme['state'] != 'all' and scheme['state'] != request.state.lower():
            continue
        
        # Filter by query
        if request.query:
            query_lower = request.query.lower()
            if (query_lower not in scheme['name'].lower() and 
                query_lower not in scheme['description'].lower() and
                query_lower not in scheme['category'].lower()):
                continue
        
        results.append(scheme)
    
    return {"results": results[:10]}  # Limit to 10 results

@app.post("/disease/quick-diagnose")
async def quick_diagnose(request: DiseaseRequest):
    """Quick disease diagnosis based on symptoms"""
    
    text_lower = request.text.lower()
    
    # Simple rule-based diagnosis
    if any(word in text_lower for word in ['yellow', 'yellowing', 'chlorosis']):
        return {
            "likely": "Nutrient Deficiency (Nitrogen/Iron)",
            "advice": "Apply balanced fertilizer, check soil pH, ensure proper drainage",
            "why": "Yellowing leaves often indicate nutrient deficiency"
        }
    elif any(word in text_lower for word in ['spots', 'brown', 'black', 'fungus']):
        return {
            "likely": "Fungal Disease",
            "advice": "Remove affected parts, apply fungicide, improve air circulation",
            "why": "Spots and discoloration suggest fungal infection"
        }
    elif any(word in text_lower for word in ['wilt', 'wilting', 'drooping']):
        return {
            "likely": "Water Stress or Root Disease",
            "advice": "Check soil moisture, inspect roots, adjust watering schedule",
            "why": "Wilting indicates water or root problems"
        }
    elif any(word in text_lower for word in ['holes', 'eaten', 'chewed', 'pest']):
        return {
            "likely": "Pest Infestation",
            "advice": "Identify pest type, use appropriate pesticide or biological control",
            "why": "Physical damage suggests pest activity"
        }
    else:
        return {
            "likely": "General Plant Stress",
            "advice": "Check water, light, nutrients, and temperature conditions",
            "why": "Multiple factors could be causing the issue"
        }

@app.post("/chat")
async def chat(request: ChatRequest):
    """AI-powered farming chatbot using Gemini Pro"""
    
    # Check if Gemini is available
    if gemini_model and GEMINI_API_KEY:
        try:
            # Create farming-specific prompt
            system_prompt = f"""
You are an expert AI farming assistant. Provide helpful, accurate advice about:
- Weather and climate for farming
- Crop selection and recommendations
- Soil management and fertilizers
- Plant diseases and pest control
- Government schemes for farmers
- Sustainable farming practices
- Agricultural technology

Respond in {request.lang} language ({'English' if request.lang == 'en' else 'Hindi' if request.lang == 'hi' else 'Gujarati'}).
Keep responses practical, concise, and farmer-friendly.

User question: {request.question}
"""
            
            # Generate response using Gemini Pro
            response = gemini_model.generate_content(system_prompt)
            
            if response and response.text:
                return {"response": response.text.strip()}
                
        except Exception as e:
            print(f"Gemini API error: {e}")
    
    # Fallback to simple responses
    question_lower = request.question.lower()
    
    if any(word in question_lower for word in ['weather', 'rain', 'temperature']):
        responses = {
            'en': "Check the Weather & Crop tab for rainfall and temperature predictions based on your location.",
            'hi': "मौसम और फसल टैब में अपने स्थान के आधार पर बारिश और तापमान की भविष्यवाणी देखें।",
            'gu': "તમારા સ્થાન પર આધારિત વરસાદ અને તાપમાનની આગાહી માટે હવામાન અને પાક ટેબ જુઓ।"
        }
    elif any(word in question_lower for word in ['crop', 'farming', 'soil']):
        responses = {
            'en': "Use the crop recommendation feature to find suitable crops for your soil type and season.",
            'hi': "अपनी मिट्टी के प्रकार और मौसम के लिए उपयुक्त फसलों को खोजने के लिए फसल सिफारिश सुविधा का उपयोग करें।",
            'gu': "તમારી માટીના પ્રકાર અને મોસમ માટે યોગ્ય પાકો શોધવા માટે પાક ભલામણ સુવિધાનો ઉપયોગ કરો।"
        }
    elif any(word in question_lower for word in ['scheme', 'government', 'subsidy']):
        responses = {
            'en': "Check the Government Schemes tab to find relevant schemes and subsidies for farmers.",
            'hi': "किसानों के लिए प्रासंगिक योजनाओं और सब्सिडी खोजने के लिए सरकारी योजनाएं टैब देखें।",
            'gu': "ખેડૂતો માટે સંબંધિત યોજનાઓ અને સબસિડી શોધવા માટે સરકારી યોજનાઓ ટેબ જુઓ।"
        }
    else:
        responses = {
            'en': "I can help with weather predictions, crop recommendations, disease diagnosis, and government schemes. What would you like to know?",
            'hi': "मैं मौसम की भविष्यवाणी, फसल की सिफारिश, रोग निदान और सरकारी योजनाओं में मदद कर सकता हूं। आप क्या जानना चाहते हैं?",
            'gu': "હું હવામાનની આગાહી, પાકની ભલામણ, રોગ નિદાન અને સરકારી યોજનાઓમાં મદદ કરી શકું છું. તમે શું જાણવા માંગો છો?"
        }
    
    return {"response": responses.get(request.lang, responses['en'])}

# Serve frontend
base_dir = Path(__file__).parent.parent
app.mount("/", StaticFiles(directory=str(base_dir / "frontend"), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)