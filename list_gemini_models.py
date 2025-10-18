"""
List available Gemini models
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

print("=" * 60)
print("AVAILABLE GEMINI MODELS")
print("=" * 60)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"\n✅ {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Description: {model.description}")
        print(f"   Supported: {', '.join(model.supported_generation_methods)}")

print("\n" + "=" * 60)
