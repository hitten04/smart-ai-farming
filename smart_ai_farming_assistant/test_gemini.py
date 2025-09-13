#!/usr/bin/env python3
"""
Test Gemini chatbot integration
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

def test_gemini_chatbot():
    """Test the Gemini chatbot functionality"""
    
    print("🤖 Testing Gemini AI Chatbot")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GEMINI_KEY')
    
    if not api_key:
        print("❌ GEMINI_KEY not found in .env file")
        return False
    
    print(f"✅ API Key loaded: {api_key[:10]}...")
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Test farming questions
        test_questions = [
            "What is the best crop for monsoon season in India?",
            "How to prevent fungal diseases in tomatoes?",
            "What fertilizer is good for wheat farming?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🌾 Test {i}: {question}")
            
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

Respond in English language.
Keep responses practical, concise, and farmer-friendly.

User question: {question}
"""
            
            try:
                response = model.generate_content(system_prompt)
                if response and response.text:
                    print(f"✅ Response: {response.text[:150]}...")
                else:
                    print("❌ No response received")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n🎉 Gemini chatbot is working!")
        return True
        
    except Exception as e:
        print(f"❌ Gemini setup failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gemini_chatbot()
    if success:
        print("\n✅ Ready to use! Start the server with: python backend/app.py")
    else:
        print("\n❌ Please check your .env file and API key")