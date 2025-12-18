from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from openai import OpenAI

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Request model
# =========================
class RegisterRequest(BaseModel):
    mode: str  # "selfie_manual" | "photo_body"
    gender: str

    # comunes
    body_type: Optional[str] = None

    # selfie + manual
    height_cm: Optional[int] = None
    weight_kg: Optional[int] = None

    # imágenes (NO se usan para IA aún)
    selfie_base64: Optional[str] = None
    body_image_base64: Optional[str] = None


# =========================
# PROMPTS
# =========================
def prompt_selfie_manual(data: RegisterRequest) -> str:
    return f"""
Ultra-realistic full-body fashion photography.

Person description:
- Gender: {data.gender}
- Height: {data.height_cm or 'average'} cm
- Weight: {data.weight_kg or 'average'} kg
- Body type: {data.body_type or 'average'}

Rules:
- Natural human proportions
- Realistic anatomy
- Consistent identity across images
- Fashion-model quality realism

Outfit:
Modern casual outfit, neutral tones, clean style.

Camera:
Full body shot, studio lighting, high realism.
"""


def prompt_photo_body(data: RegisterRequest) -> str:
    return f"""
Ultra-realistic full-body fashion photography.

Person description:
- Gender: {data.gender}
- Body type: {data.body_type or 'average'}

Rules:
- Maintain realistic body proportions
- Same identity across images
- Natural posture and anatomy

Outfit:
Modern casual outfit, neutral tones.

Camera:
Full body shot, studio lighting, realistic fashion catalog.
"""


# =========================
# ENDPOINT
# =========================
@router.post("/register_generate_base_images")
def register_generate_base_images(data: RegisterRequest):

    # -------- validation --------
    if data.mode not in ["selfie_manual", "photo_body"]:
        raise HTTPException(400, "Invalid registration mode")

    if data.mode == "selfie_manual":
        if not data.selfie_base64:
            raise HTTPException(400, "Selfie image required for selfie_manual mode")
        prompt = prompt_selfie_manual(data)

    elif data.mode == "photo_body":
        if not data.body_image_base64:
            raise HTTPException(400, "Body image required for photo_body mode")
        prompt = prompt_photo_body(data)

    # -------- image generation --------
    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            n=2
        )
    except Exception as e:
        raise HTTPException(500, f"Image generation failed: {str(e)}")

    return {
        "status": "ok",
        "mode": data.mode,
        "images": [img.b64_json for img in result.data],
    }
