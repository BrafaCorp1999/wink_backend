# utils/gemini_client.py
from google import genai
from google.genai import types
import os

# Inicializa el cliente Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Falta la variable de entorno GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)
