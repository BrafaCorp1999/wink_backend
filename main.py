# main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Routers ---
from routers import analyze_body_with_face
from routers import generate_outfit_demo

# =========================
# Crear app
# =========================
app = FastAPI(
    title="AI Outfit Backend",
    version="1.0",
    description="Backend for body analysis + outfit generation using Gemini + Replicate."
)

# =========================
# Leer llaves de entorno
# =========================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if GEMINI_KEY:
    print("🔹 Gemini API detected → Ready")
else:
    print("⚠️ GEMINI_API_KEY missing → Gemini disabled")

if REPLICATE_API_KEY:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY
    print("🔹 Replicate API detected → Ready")
else:
    print("⚠️ REPLICATE_API_KEY missing → Replicate disabled")

# =========================
# Middleware CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Routers
# =========================
app.include_router(analyze_body_with_face.router, prefix="/api")
app.include_router(generate_outfit_demo.router, prefix="/api")

# =========================
# Health check
# =========================
@app.get("/")
def root():
    return {"message": "✅ Backend running successfully"}
