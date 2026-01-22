from fastapi import APIRouter, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import json
import os
import uuid
import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO)

# ------------------------- OpenAI client -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if not client:
    raise RuntimeError("OPENAI_API_KEY not set")

# ========================= Helper: decode base64 =========================
def decode_base64_image(b64_string: str) -> BytesIO:
    try:
        if b64_string.startswith("data:image"):
            b64_string = b64_string.split(",")[-1]
        image_bytes = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        MAX_SIZE = 1024
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

# ========================= Prompts =========================
IMAGE_TO_IMAGE_PROMPT = """
Use the provided image as the SAME person reference.

IDENTITY LOCK (STRICT):
- Preserve face, hairstyle, skin tone, body proportions.
- Do NOT change age, ethnicity, body shape, or posture.

CLOTHING CHANGE ONLY:
- Style: {style}
- Occasion: {occasion}
- Climate: {climate}
- Preferred colors: {colors}
- Apply clothing naturally over the body.
- Respect folds, gravity, and fabric behavior.

RENDER RULES:
- Full-body (head to feet)
- Neutral standing pose
- Photorealistic
- Clean fashion photo, no illustration or CGI
"""

TEXT_PROMPT_TEMPLATE = """
Describe the outfit in the following image (base64 format):
{image_b64}

Use the following parameters as guidance:
- Style: {style}
- Occasion: {occasion}
- Climate: {climate}
- Preferred colors: {colors}

The description should be short, coherent, and reflect exactly what is shown in the image, including main clothing category and colors.
For example: 'Elegant red dress with black details, ideal for a party in cold weather.'
"""

# ========================= Endpoint Web =========================
@router.post("/ai/generate-outfit-from-form-web")
async def generate_outfit_from_form_web(
    gender: str = Form(...),
    style: str = Form("casual"),
    occasion: str = Form("daily"),
    climate: str = Form("temperate"),
    colors: str = Form("neutral"),
    base_image_b64: str = Form(...)
):
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    request_id = str(uuid.uuid4())
    logging.info(f"[AI-FORM-WEB] Request {request_id} started")

    # ------------------- Colors
    try:
        colors_list = json.loads(colors)
        colors_str = ", ".join(colors_list) if colors_list else "neutral tones"
    except Exception:
        colors_str = "neutral tones"

    # ------------------- Base image
    base_image = decode_base64_image(base_image_b64)

    # ------------------- Prompt para generar imagen
    image_prompt = IMAGE_TO_IMAGE_PROMPT.format(
        style=style,
        occasion=occasion,
        climate=climate,
        colors=colors_str
    )

    try:
        # ------------------- Generar imagen
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=base_image,
            prompt=image_prompt,
            n=1,
            size="auto"
        )

        if not response.data or not response.data[0].b64_json:
            raise Exception("Empty image response")

        generated_b64 = response.data[0].b64_json

        # ------------------- Generar descripción coherente con la imagen
        text_prompt = TEXT_PROMPT_TEMPLATE.format(
            image_b64=generated_b64,
            style=style,
            occasion=occasion,
            climate=climate,
            colors=colors_str
        )

        text_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": text_prompt}],
            temperature=0.7
        )

        recommendation = text_response.choices[0].message.content.strip()

        logging.info(f"[AI-FORM-WEB] Request {request_id} SUCCESS")

        return {
            "status": "ok",
            "request_id": request_id,
            "image": generated_b64,
            "recommendation": recommendation
        }

    except Exception as e:
        logging.error(f"[AI-FORM-WEB] Request {request_id} FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail="Outfit generation failed")
