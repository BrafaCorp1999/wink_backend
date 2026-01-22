from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import base64
from typing import Dict

router = APIRouter()

# =========================
# Request model
# =========================
class AnalyzeBodyWebRequest(BaseModel):
    gender_hint: str
    image_base64: str

# =========================
# Helpers (reutilizados)
# =========================
def normalize_gender(value: str) -> str:
    value = value.lower().strip()
    if value in ("male", "man", "hombre"):
        return "male"
    if value in ("female", "woman", "mujer"):
        return "female"
    return "female"

def extract_body_features(image: Image.Image, gender: str) -> Dict:
    width, height = image.size
    aspect_ratio = round(height / width, 2)

    if aspect_ratio > 1.7:
        body_type = "slim"
    elif aspect_ratio > 1.5:
        body_type = "average"
    else:
        body_type = "curvy"

    if gender == "male":
        shoulder_ratio = 1.25
        hip_ratio = 1.0
    else:
        shoulder_ratio = 1.1
        hip_ratio = 1.2

    return {
        "gender": gender,
        "body_type": body_type,
        "estimated_measurements": {
            "shoulders": shoulder_ratio,
            "waist": 1.0,
            "hips": hip_ratio
        },
        "image_stats": {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio
        }
    }

# =========================
# Endpoint Web
# =========================
@router.post("/analyze-body-with-face-web")
async def analyze_body_web(request: AnalyzeBodyWebRequest):
    try:
        gender = normalize_gender(request.gender_hint)
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        MAX_SIZE = 1024
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE))

        traits = extract_body_features(image, gender)

        return {
            "status": "ok",
            "traits": traits,
            "source": "body_photo_web"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing body (web): {str(e)}")
