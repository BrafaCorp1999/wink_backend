from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 🔹 Importa todos tus routers aquí
from routers import tryon
from routers import generate_image
from routers import generate_body_model
from routers import analyze_body_with_face  # 👈 ESTE FALTABA

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Registrar routers
app.include_router(tryon.router, prefix="/api")
app.include_router(generate_image.router, prefix="/api")
app.include_router(generate_body_model.router, prefix="/api")
app.include_router(analyze_body_with_face.router, prefix="/api")
