import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
print(f"API Key loaded: {api_key[:20]}..." if api_key else "No API key found")

genai.configure(api_key=api_key)

print("\nListing models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ {m.name}")
except Exception as e:
    print(f"Error: {e}")

print("\nTrying different model names...")
test_models = [
    'gemini-1.5-flash-latest',
    'gemini-1.5-pro-latest', 
    'gemini-pro',
    'models/gemini-1.5-flash-latest',
    'models/gemini-1.5-pro-latest',
    'models/gemini-pro'
]

for model_name in test_models:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say hi")
        print(f"✅ WORKS: {model_name}")
        print(f"   Response: {response.text[:50]}")
        break
    except Exception as e:
        print(f"❌ FAILED: {model_name} - {str(e)[:60]}")
