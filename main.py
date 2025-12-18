# main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Crear app (PRIMERO SIEMPRE)
# =========================
app = FastAPI(
    title="AI Outfit Backend",
    version="1.0",
    description="Backend for body analysis + outfit generation using OpenAI"
)

# =========================
# Routers (DESPUÉS)
# =========================
from routers import analyze_body_with_face
from routers import generate_outfit_demo
from routers import register_generate_base_images

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
# Routers include
# =========================
app.include_router(analyze_body_with_face.router, prefix="/api")
app.include_router(generate_outfit_demo.router, prefix="/api")
app.include_router(register_generate_base_images.router, prefix="/api")

# =========================
# Health check
# =========================
@app.get("/")
def root():
    return {"message": "✅ Backend running successfully"}
