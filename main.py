# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers
from routers import tryon
from routers import generate_image
from routers import generate_body_model
from routers import analyze_body_with_face
from routers import generate_outfit_demo  # <-- NUEVO

app = FastAPI(title="AI Outfit Backend", version="1.0")

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar si quieres restringir dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Registrar endpoints ===
app.include_router(tryon.router, prefix="/api")
app.include_router(generate_image.router, prefix="/api")
app.include_router(generate_body_model.router, prefix="/api")
app.include_router(analyze_body_with_face.router, prefix="/api")
app.include_router(generate_outfit_demo.router, prefix="/api")  # <-- Incluir correctamente

@app.get("/")
def root():
    return {"message": "✅ Backend running successfully"}
