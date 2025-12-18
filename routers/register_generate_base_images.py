from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import base64
import os
from openai import OpenAI

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RegisterRequest(BaseModel):
    mode: str
    gender: str
    selfie_base64: str
    body_image_base64: str | None = None
    height_cm: int | None = None
    weight_kg: int | None = None
    body_type: str | None = None

def build_prompt_selfie_manual(data: RegisterRequest) -> str:
    return f"""
Ultra realistic full body fashion photograph.

IMPORTANT RULES:
- Preserve the same facial identity from the reference image
- Do NOT change facial features
- Natural human anatomy

Person details:
Gender: {data.gender}
Height: {data.height_cm} cm
Body type: {data.body_type}

Generate a realistic full body representation consistent with these proportions.

Outfit:
Casual neutral outfit, modern style.

Camera:
Full body shot, studio lighting.
"""

def build_prompt_photo_body(data: RegisterRequest) -> str:
    return """
Ultra realistic full body fashion photograph.

IMPORTANT RULES:
- Preserve the same facial identity
- Do NOT change body structure
- Only change clothing

Use the provided full body image as visual reference.

Outfit:
Casual neutral outfit, modern style.

Camera:
Same pose and angle as reference image.
"""

@router.post("/register_generate_base_images")
def register_generate_images(data: RegisterRequest):

    if data.mode == "selfie_manual":
        prompt = build_prompt_selfie_manual(data)
        images = [data.selfie_base64]

    elif data.mode == "photo_body":
        if not data.body_image_base64:
            raise HTTPException(400, "Body image required")
        prompt = build_prompt_photo_body(data)
        images = [data.selfie_base64, data.body_image_base64]

    else:
        raise HTTPException(400, "Invalid mode")

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        image=images,
        n=2
    )

    return {
        "status": "ok",
        "images": [img.b64_json for img in result.data]
    }
