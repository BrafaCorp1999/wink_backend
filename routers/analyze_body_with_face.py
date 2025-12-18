# analyze_body_with_face.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np

router = APIRouter()

class AnalyzeBodyRequest(BaseModel):
    image_base64: str
    gender_hint: str  # "male" o "female"

# --- Función auxiliar para simular extracción de medidas ---
def extract_body_features(image: Image.Image, gender: str) -> dict:
    """
    Aquí iría la lógica real de análisis. 
    Para demo, retornamos valores simulados basados en la imagen.
    """
    width, height = image.size
    aspect_ratio = height / width

    # Contextura simulada según proporción (solo demo)
    if aspect_ratio > 2.2:
        body_type = "slim"
    elif aspect_ratio < 1.6:
        body_type = "plus"
    else:
        body_type = "average"

    # Altura y peso estimados (solo demo, ajustar según modelo real)
    if gender.lower() == "male":
        height_cm = 175
        weight_kg = 70
    else:
        height_cm = 165
        weight_kg = 60

    # Tipo de cabello estimado (solo demo)
    hair_type = "medium length, straight"

    return {
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "body_type": body_type,
        "hair_type": hair_type
    }

# =========================
# Endpoint para analizar cuerpo + rostro
# =========================
@router.post("/analyze-body-with-face")
def analyze_body_with_face(req: AnalyzeBodyRequest):
    try:
        # Convertimos base64 a imagen
        image_bytes = base64.b64decode(req.image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Extraemos rasgos del cuerpo
        features = extract_body_features(image, req.gender_hint)

        return features

    except Exception as e:
        raise HTTPException(500, f"Error analyzing body: {str(e)}")
