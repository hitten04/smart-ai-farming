#!/usr/bin/env python3
"""
Test suite for Smart AI Farming Assistant API endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import app

client = TestClient(app)

class TestHealthEndpoint:
    def test_health_check(self):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

class TestWeatherPrediction:
    def test_weather_prediction_offline(self):
        """Test weather prediction in offline mode"""
        payload = {
            "lat": 19.0760,
            "lon": 72.8777,
            "month": 7,
            "offline_only": True
        }
        response = client.post("/predict/weather", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "rain_mm" in data
        assert "temp_c" in data
        assert "features" in data
        assert "why" in data
        assert isinstance(data["rain_mm"], (int, float))
        assert isinstance(data["temp_c"], (int, float))

    def test_weather_prediction_with_features(self):
        """Test weather prediction with all features"""
        payload = {
            "lat": 28.6139,
            "lon": 77.2090,
            "month": 12,
            "humidity": 65,
            "cloud_cover": 40,
            "wind_kph": 15,
            "sun_hours": 8,
            "offline_only": True
        }
        response = client.post("/predict/weather", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["features"]["humidity"] == 65
        assert data["features"]["cloud_cover"] == 40

    def test_weather_prediction_invalid_month(self):
        """Test weather prediction with invalid month"""
        payload = {
            "lat": 19.0760,
            "lon": 72.8777,
            "month": 13,  # Invalid month
            "offline_only": True
        }
        response = client.post("/predict/weather", json=payload)
        # Should still work but with clamped values
        assert response.status_code == 200

class TestCropRecommendation:
    def test_crop_recommendation_valid(self):
        """Test crop recommendation with valid inputs"""
        payload = {
            "soil_type": "clay",
            "season": "kharif"
        }
        response = client.post("/recommend/crops", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "crops" in data
        assert "soil_name" in data
        assert "characteristics" in data
        assert "why" in data
        assert isinstance(data["crops"], list)
        assert len(data["crops"]) > 0

    def test_crop_recommendation_all_soil_types(self):
        """Test crop recommendation for all soil types"""
        soil_types = ["clay", "sandy", "loamy", "black", "red"]
        seasons = ["kharif", "rabi", "zaid"]
        
        for soil_type in soil_types:
            for season in seasons:
                payload = {
                    "soil_type": soil_type,
                    "season": season
                }
                response = client.post("/recommend/crops", json=payload)
                assert response.status_code == 200
                data = response.json()
                assert len(data["crops"]) > 0

    def test_crop_recommendation_invalid_soil(self):
        """Test crop recommendation with invalid soil type"""
        payload = {
            "soil_type": "invalid_soil",
            "season": "kharif"
        }
        response = client.post("/recommend/crops", json=payload)
        assert response.status_code == 400

class TestSchemeSearch:
    def test_scheme_search_no_filters(self):
        """Test scheme search without filters"""
        payload = {}
        response = client.post("/schemes/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_scheme_search_with_query(self):
        """Test scheme search with query"""
        payload = {
            "query": "insurance"
        }
        response = client.post("/schemes/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        # Should find insurance-related schemes
        insurance_schemes = [s for s in data["results"] if "insurance" in s["name"].lower() or "insurance" in s["description"].lower()]
        assert len(insurance_schemes) > 0

    def test_scheme_search_with_state(self):
        """Test scheme search with state filter"""
        payload = {
            "state": "gujarat"
        }
        response = client.post("/schemes/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        # Should include all-state schemes and gujarat-specific schemes
        for scheme in data["results"]:
            assert scheme["state"] in ["all", "gujarat"]

class TestDiseaseDiganosis:
    def test_disease_diagnosis_yellowing(self):
        """Test disease diagnosis for yellowing symptoms"""
        payload = {
            "text": "My plants have yellow leaves"
        }
        response = client.post("/disease/quick-diagnose", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "likely" in data
        assert "advice" in data
        assert "why" in data
        assert "nutrient" in data["likely"].lower()

    def test_disease_diagnosis_spots(self):
        """Test disease diagnosis for spot symptoms"""
        payload = {
            "text": "Brown spots on leaves"
        }
        response = client.post("/disease/quick-diagnose", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "fungal" in data["likely"].lower()

    def test_disease_diagnosis_wilting(self):
        """Test disease diagnosis for wilting symptoms"""
        payload = {
            "text": "Plants are wilting and drooping"
        }
        response = client.post("/disease/quick-diagnose", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "water" in data["likely"].lower() or "root" in data["likely"].lower()

class TestChatbot:
    def test_chat_english(self):
        """Test chatbot in English"""
        payload = {
            "question": "How to grow rice?",
            "lang": "en"
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0

    def test_chat_hindi(self):
        """Test chatbot in Hindi"""
        payload = {
            "question": "मौसम कैसा होगा?",
            "lang": "hi"
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        # Should contain Hindi text
        assert any(ord(char) > 127 for char in data["response"])

    def test_chat_gujarati(self):
        """Test chatbot in Gujarati"""
        payload = {
            "question": "હવામાન કેવું છે?",
            "lang": "gu"
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        # Should contain Gujarati text
        assert any(ord(char) > 127 for char in data["response"])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])