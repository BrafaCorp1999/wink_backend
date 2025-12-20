from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from PIL import Image
from io import BytesIO
import json

router = APIRouter()

# =========================
# Demo extractor (mock)
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

    return {
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "body_type": body_type,
        "hair_type": "medium length, straight"
    }

# =========================
# Endpoint: SOLO body photo
# =========================
@router.post("/analyze-body-with-face")
async def analyze_body_with_face(
    gender_hint: str = Form(...),
    image_file: UploadFile = File(...)
):
    # -------------------------
    # Validar mimetype
    # -------------------------
    if image_file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/webp",
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format"
        )

    try:
        image_bytes = await image_file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        traits = extract_body_features(image, gender_hint)

        return {
            "status": "ok",
            "traits": traits,
            "source": "body_photo"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing body: {str(e)}"
        )
