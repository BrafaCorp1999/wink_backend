# main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import services
from utils.gemini_service import init_gemini
from utils.openai_service import init_openai

# Routers
from routers import analyze_body_with_face
from routers import generate_outfit_demo

app = FastAPI(
    title="AI Outfit Backend",
    version="1.0",
    description="Backend for body analysis + outfit generation using Gemini + OpenAI."
)

# === Initialize external AI services ===
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if GEMINI_KEY:
    print("🔹 Gemini API detected → Initializing...")
    init_gemini(GEMINI_KEY)
else:
    print("⚠️ GEMINI_API_KEY missing → Gemini disabled")

if OPENAI_KEY:
    print("🔹 OpenAI API detected → Initializing...")
    init_openai(OPENAI_KEY)
else:
    print("⚠️ OPENAI_API_KEY missing → OpenAI disabled")

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Register routes ===
app.include_router(analyze_body_with_face.router, prefix="/api")
app.include_router(generate_outfit_demo.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "✅ Backend running successfully"}
