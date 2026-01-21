from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from PIL import Image
from io import BytesIO

from utils.body_analysis_service import extract_body_features

router = APIRouter()

def normalize_gender(value: str) -> str:
    value = value.lower().strip()
    if value in ("male", "man", "hombre"):
        return "male"
    if value in ("female", "woman", "mujer"):
        return "female"
    return "female"

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
