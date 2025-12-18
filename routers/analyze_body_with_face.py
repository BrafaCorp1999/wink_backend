from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from PIL import Image
from io import BytesIO
import base64
import json

router = APIRouter()

# =========================
# Función demo para extraer rasgos
# =========================
def extract_body_features(image: Image.Image, gender: str) -> dict:
    width, height = image.size
    aspect_ratio = height / width

    if aspect_ratio > 2.2:
        body_type = "slim"
    elif aspect_ratio < 1.6:
        body_type = "plus"
    else:
        body_type = "average"

    if gender.lower() == "male":
        height_cm = 175
        weight_kg = 70
    else:
        height_cm = 165
        weight_kg = 60

    hair_type = "medium length, straight"

    return {
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "body_type": body_type,
        "hair_type": hair_type
    }

# =========================
# Endpoint: analizar cuerpo + rostro
# =========================
@router.post("/analyze-body-with-face")
async def analyze_body_with_face(
    gender_hint: str = Form(...),
    image_file: UploadFile = File(...)
):
    try:
        image_bytes = await image_file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        features = extract_body_features(image, gender_hint)
        return {"status": "ok", "traits": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing body: {str(e)}")
