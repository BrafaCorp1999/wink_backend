from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import os
import json

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Helpers
# =========================
def prepare_image(file: UploadFile, size=1024) -> BytesIO:
    image = Image.open(file.file).convert("RGB")
    image = image.resize((size, size))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.name = "input.png"
    buffer.seek(0)
    return buffer


@router.post("/ai/generate-outfit-from-form")
async def generate_outfit_from_form(
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: str = Form(...),
    occasion: str = Form(...),
    climate: str = Form(...),
    colors: str = Form(...),
    base_image_file: UploadFile = File(...)
):
    try:
        image_file = prepare_image(base_image_file)

        # 1️⃣ TEXTO PRIMERO (fuente de verdad)
        text_prompt = f"""
You are a professional fashion stylist.

User profile:
- Gender: {gender}
- Style: {style}
- Occasion: {occasion}
- Climate: {climate}
- Preferred colors: {colors}

Describe ONE complete outfit in a concise, fashion-oriented way.
"""

        text_result = client.completions.create(
            model="gpt-4.1-mini",
            prompt=text_prompt,
            max_tokens=150
        )
        recommendation = text_result.choices[0].text.strip()

        # 2️⃣ IMAGEN BASADA EN EL TEXTO
        image_prompt = f"""
Apply the following outfit to the person in the image:

{recommendation}

Preserve face, body, pose and proportions.
Realistic fashion photography.
"""

        image_result = client.images.edit(
            model="gpt-image-1-mini",
            image=image_file,
            prompt=image_prompt,
            size="1024x1024"
        )

        generated_image = image_result.data[0].b64_json

        return {
            "status": "ok",
            "image": generated_image,
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
