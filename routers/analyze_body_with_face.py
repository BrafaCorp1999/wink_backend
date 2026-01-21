from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from PIL import Image
from io import BytesIO
from typing import Dict

router = APIRouter()


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize_gender(value: str) -> str:
    value = value.lower().strip()
    if value in ("male", "man", "hombre"):
        return "male"
    if value in ("female", "woman", "mujer"):
        return "female"
    return "female"


def extract_body_features(image: Image.Image, gender: str) -> Dict:
    """
    Demo body analyzer.
    No IA, no ML, no dependencias externas.
    Devuelve valores aproximados y consistentes.
    """

    width, height = image.size
    aspect_ratio = round(height / width, 2)

    # Heurísticas simples solo para demo
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


# -------------------------------------------------
# Endpoint
# -------------------------------------------------
@router.post("/analyze-body-with-face")
async def analyze_body_with_face(
    gender_hint: str = Form(...),
    image_file: UploadFile = File(...)
):
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
        gender = normalize_gender(gender_hint)

        image_bytes = await image_file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        MAX_SIZE = 1024
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE))

        traits = extract_body_features(image, gender)

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
