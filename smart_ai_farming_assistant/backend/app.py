#!/usr/bin/env python3
"""
Smart AI Farming Assistant - FastAPI Backend
Serves API endpoints and static frontend files.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
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
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smart AI Farming Assistant", version="1.0.0")

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add custom OPTIONS handler to fix preflight issues
@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str) -> Response:
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

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
    """Load YAML data files with fallback creation"""
    base_dir = Path(__file__).parent.parent
    
    # Default soil data
    default_soil_data = {
        'soil_types': {
            'clay': {
                'name': 'Clay Soil',
                'characteristics': 'Heavy, retains water, rich in nutrients',
                'crops': {
                    'kharif': ['Rice', 'Cotton', 'Sugarcane'],
                    'rabi': ['Wheat', 'Barley', 'Gram'],
                    'zaid': ['Fodder crops', 'Green vegetables']
                }
            },
            'sandy': {
                'name': 'Sandy Soil',
                'characteristics': 'Light, well-drained, easy to work',
                'crops': {
                    'kharif': ['Millet', 'Groundnut', 'Cotton'],
                    'rabi': ['Wheat', 'Mustard', 'Gram'],
                    'zaid': ['Watermelon', 'Fodder crops']
                }
            },
            'loamy': {
                'name': 'Loamy Soil',
                'characteristics': 'Balanced mixture, ideal for most crops',
                'crops': {
                    'kharif': ['Rice', 'Maize', 'Sugarcane', 'Cotton'],
                    'rabi': ['Wheat', 'Barley', 'Peas', 'Mustard'],
                    'zaid': ['Vegetables', 'Fodder crops']
                }
            },
            'black': {
                'name': 'Black Soil',
                'characteristics': 'Rich in lime, iron, magnesia and alumina',
                'crops': {
                    'kharif': ['Cotton', 'Sugarcane', 'Jowar'],
                    'rabi': ['Wheat', 'Gram', 'Linseed'],
                    'zaid': ['Fodder crops']
                }
            },
            'red': {
                'name': 'Red Soil',
                'characteristics': 'Iron-rich, good drainage',
                'crops': {
                    'kharif': ['Rice', 'Ragi', 'Groundnut'],
                    'rabi': ['Wheat', 'Cotton', 'Pulses'],
                    'zaid': ['Fodder crops', 'Vegetables']
                }
            }
        }
    }
    
    # Default schemes data
    default_schemes_data = {
        'schemes': [
            {
                'name': 'PM-KISAN',
                'description': 'Direct income support to farmers',
                'category': 'income_support',
                'eligibility': 'All landholding farmers',
                'state': 'all'
            },
            {
                'name': 'Kisan Credit Card',
                'description': 'Credit support for agriculture and allied activities',
                'category': 'credit',
                'eligibility': 'Farmers with land records',
                'state': 'all'
            },
            {
                'name': 'Pradhan Mantri Fasal Bima Yojana',
                'description': 'Crop insurance scheme',
                'category': 'insurance',
                'eligibility': 'All farmers',
                'state': 'all'
            }
        ]
    }
    
    # Default weather cache
    default_weather_cache = {
        'default_features': {
            'humidity': 70,
            'cloud_cover': 50,
            'wind_kph': 10,
            'sun_hours': 8
        }
    }
    
    # Try to load files, create defaults if not found
    os.makedirs(base_dir / 'data', exist_ok=True)
    os.makedirs(base_dir / 'offline_cache', exist_ok=True)
    
    try:
        with open(base_dir / 'data/soil_crops.yaml', 'r', encoding='utf-8') as f:
            soil_data = yaml.safe_load(f)
    except FileNotFoundError:
        soil_data = default_soil_data
        with open(base_dir / 'data/soil_crops.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(default_soil_data, f)
    
    try:
        with open(base_dir / 'data/schemes.yaml', 'r', encoding='utf-8') as f:
            schemes_data = yaml.safe_load(f)
    except FileNotFoundError:
        schemes_data = default_schemes_data
        with open(base_dir / 'data/schemes.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(default_schemes_data, f)
    
    try:
        with open(base_dir / 'offline_cache/sample_weather.json', 'r', encoding='utf-8') as f:
            weather_cache = json.load(f)
    except FileNotFoundError:
        weather_cache = default_weather_cache
        with open(base_dir / 'offline_cache/sample_weather.json', 'w', encoding='utf-8') as f:
            json.dump(default_weather_cache, f, indent=2)
    
    return soil_data, schemes_data, weather_cache

def load_models():
    """Load trained weather models"""
    try:
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / 'models/weather_models.joblib'
        if model_path.exists():
            return joblib.load(model_path)
        else:
            print("Weather models not found")
            return None
    except Exception as e:
        print(f"Error loading models: {e}")
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
    """Enhanced weather prediction using trained RandomForest models"""
    
    print(f"Weather prediction request: lat={request.lat}, lon={request.lon}, month={request.month}")
    
    try:
        # Import the weather service
        from weather_service import get_weather_prediction
        
        # Use the trained model prediction
        result = get_weather_prediction(
            latitude=request.lat,
            longitude=request.lon,
            month=request.month,
            humidity=request.humidity,
            wind_speed=request.wind_kph,
            pressure=None,  # Will use default
            cloud_cover=request.cloud_cover,
            sun_hours=request.sun_hours
        )
        
        # Determine climate zone for explanation
        if request.lat > 30:
            zone = "Northern Mountain"
        elif request.lat > 25:
            zone = "Northern Plains"
        elif request.lat > 20:
            zone = "Central India"
        elif request.lat > 15:
            zone = "Deccan Plateau"
        else:
            zone = "Southern Peninsula"
        
        # Seasonal explanation
        season_names = {
            1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
            6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
            10: "Post-Monsoon", 11: "Post-Monsoon", 12: "Winter"
        }
        
        season = season_names[request.month]
        
        return {
            "rain_mm": round(result['rainfall_mm'], 1),
            "temp_c": round(result['temperature_c'], 1),
            "features": {
                'lat': request.lat,
                'lon': request.lon,
                'month': request.month,
                'humidity': request.humidity or 70,
                'cloud_cover': request.cloud_cover or 50,
                'wind_kph': request.wind_kph or 10,
                'sun_hours': request.sun_hours or 8
            },
            "why": f"Prediction during {season} season using trained RandomForest model based on Indian weather data patterns",
            "api_used": False
        }
        
    except Exception as e:
        print(f"Model prediction failed: {e}")
        
        # Enhanced fallback with realistic Indian weather patterns
        monsoon_months = [6, 7, 8, 9]
        winter_months = [12, 1, 2]
        summer_months = [3, 4, 5]
        post_monsoon_months = [10, 11]
        
        # More realistic rainfall calculation based on Indian patterns
        if request.month in monsoon_months:
            if request.lat > 25:  # Northern India
                base_rain = 120 if request.month in [7, 8] else 80
            else:  # Southern/Central India
                base_rain = 180 if request.month in [7, 8] else 120
            
            # Adjust for humidity
            if request.humidity:
                humidity_factor = (request.humidity / 80)
                base_rain *= humidity_factor
            
            # Coastal areas get more rain
            is_coastal = (request.lon < 75 and request.lat < 25) or (request.lon > 85 and request.lat < 22)
            if is_coastal:
                base_rain *= 1.4
                
            rainfall = base_rain + np.random.uniform(-30, 30)
            
        elif request.month in winter_months:
            if request.lat > 28:  # North India gets some winter rain
                rainfall = 15 + np.random.uniform(-5, 15)
            elif request.lat < 15:  # South India northeast monsoon
                rainfall = 40 + np.random.uniform(-15, 25)
            else:
                rainfall = 8 + np.random.uniform(-3, 12)
                
        elif request.month in summer_months:
            if request.lat < 15:  # South India pre-monsoon
                rainfall = 25 + np.random.uniform(-10, 20)
            else:
                rainfall = 12 + np.random.uniform(-5, 15)
                
        else:  # post-monsoon
            if request.lat < 15:  # South India retreat monsoon
                rainfall = 90 + np.random.uniform(-25, 35)
            else:
                rainfall = 35 + np.random.uniform(-15, 25)
        
        # More realistic temperature calculation for Indian climate
        if request.month in summer_months:
            if request.lat > 25:  # North India
                base_temp = 38 - (request.lat - 25) * 0.8
            else:  # South India
                base_temp = 34 - (request.lat - 8) * 0.3
        elif request.month in monsoon_months:
            if request.lat > 25:
                base_temp = 32 - (request.lat - 25) * 0.6
            else:
                base_temp = 29 - (request.lat - 8) * 0.2
        elif request.month in winter_months:
            if request.lat > 25:
                base_temp = 18 - (request.lat - 25) * 1.2
            else:
                base_temp = 26 - (request.lat - 8) * 0.4
        else:  # post-monsoon
            if request.lat > 25:
                base_temp = 25 - (request.lat - 25) * 0.5
            else:
                base_temp = 28 - (request.lat - 8) * 0.3
        
        # Add coastal moderation
        is_coastal = (request.lon < 75 and request.lat < 25) or (request.lon > 85 and request.lat < 22)
        if is_coastal:
            if request.month in summer_months:
                base_temp -= 3
            elif request.month in winter_months:
                base_temp += 2
        
        # Add realistic variation
        temperature = base_temp + np.random.uniform(-2, 2)
        
        # Determine zone for explanation
        if request.lat > 30:
            zone = "Northern Mountain"
        elif request.lat > 25:
            zone = "Northern Plains"
        elif request.lat > 20:
            zone = "Central India"
        elif request.lat > 15:
            zone = "Deccan Plateau"
        else:
            zone = "Southern Peninsula"
        
        season = season_names[request.month]
        
        return {
            "rain_mm": round(max(0, rainfall), 1),
            "temp_c": round(temperature, 1),
            "features": {
                'lat': request.lat,
                'lon': request.lon,
                'month': request.month,
                'humidity': request.humidity or 70,
                'cloud_cover': request.cloud_cover or 50,
                'wind_kph': request.wind_kph or 10,
                'sun_hours': request.sun_hours or 8
            },
            "why": f"Enhanced fallback prediction for {zone} region during {season} season based on Indian climate patterns (trained models not available)",
            "api_used": False
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

# Handle missing enhanced-styles.css
@app.get("/enhanced-styles.css")
async def get_enhanced_styles():
    enhanced_css = """
/* Enhanced styles for Smart AI Farming Assistant */
.input-error {
    color: #f44336;
    font-size: 0.8rem;
    margin-top: 0.25rem;
    font-weight: 500;
}

.form-group input.error,
.form-group select.error,
.form-group textarea.error {
    border-color: #f44336 !important;
    box-shadow: 0 0 0 3px rgba(244, 67, 54, 0.1) !important;
}

.form-group input.success,
.form-group select.success,
.form-group textarea.success {
    border-color: #4caf50 !important;
    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1) !important;
}

.geo-button {
    margin-top: 0.75rem;
    padding: 0.75rem 1rem;
    background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
    color: white;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
    width: 100%;
}

.geo-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
}

.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    color: white;
    font-weight: 600;
    z-index: 1000;
    transform: translateX(100%);
    transition: transform 0.3s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    max-width: 300px;
}

.notification.show {
    transform: translateX(0);
}

.notification.success { 
    background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
}
.notification.error { 
    background: linear-gradient(135deg, #f44336 0%, #e57373 100%);
}
.notification.warning { 
    background: linear-gradient(135deg, #ff9800 0%, #ffb74d 100%);
}
.notification.info { 
    background: linear-gradient(135deg, #2196f3 0%, #64b5f6 100%);
}

.prediction-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 1.5rem 0;
}

.prediction-item {
    background: linear-gradient(135deg, white 0%, #f8f9fa 100%);
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    border: 2px solid #e8f5e9;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.prediction-item:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(76, 175, 80, 0.2);
    border-color: #4caf50;
}

.prediction-item .icon {
    font-size: 2.5rem;
    display: block;
    margin-bottom: 1rem;
}

.prediction-item .label {
    display: block;
    font-size: 0.9rem;
    font-weight: 600;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.prediction-item .value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #2e7d32;
    display: block;
}
"""
    return Response(content=enhanced_css, media_type="text/css")

# Serve frontend
base_dir = Path(__file__).parent.parent
app.mount("/", StaticFiles(directory=str(base_dir / "frontend"), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
