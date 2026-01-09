from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import json
import os
import uuid
import logging

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO)

# =========================
# PROMPT BLOQUEADO (IDENTIDAD)
# =========================
IMAGE_TO_IMAGE_PROMPT = """
You MUST use the provided image as reference for the SAME real person.

IDENTITY LOCK (STRICT):
- Preserve face, hairstyle, skin tone and body proportions.
- Do NOT change age, ethnicity or body shape.

CLOTHING CHANGE ONLY:
- Style: {style}
- Occasion: {occasion}
- Climate: {climate}
- Preferred colors: {colors}

BODY REFERENCE:
- Height: {height_cm} cm
- Weight: {weight_kg} kg
- Body type: {body_type}

RENDER RULES:
- Full-body (head to feet)
- Natural standing pose
- Photorealistic
- High quality fashion photo
- NO illustration, NO CGI
"""

# =========================
# NORMALIZAR TRAITS
# =========================
def normalize_traits(traits: dict, gender: str) -> dict:
    return {
        "height_cm": traits.get("height_cm") or (175 if gender == "male" else 165),
        "weight_kg": traits.get("weight_kg") or (70 if gender == "male" else 60),
        "body_type": traits.get("body_type") or "average",
    }

# =========================
# ASEGURAR PNG
# =========================
def ensure_png_upload(upload: UploadFile) -> BytesIO:
    try:
        image_bytes = upload.file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = "base.png"
        return buffer
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

# =========================
# ENDPOINT SEGURO
# =========================
@router.post("/ai/generate-outfit-from-form")
async def generate_outfit_from_form(
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: str = Form("casual"),
    occasion: str = Form("daily"),
    climate: str = Form("temperate"),
    colors: str = Form("neutral"),
    base_image_file: UploadFile = File(...)
):
    request_id = str(uuid.uuid4())
    logging.info(f"[AI-FORM] Request {request_id} started")

    # -------- Parse traits
    try:
        raw_traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    traits = normalize_traits(raw_traits, gender)

    # -------- Colors
    try:
        colors_list = json.loads(colors)
        colors_str = ", ".join(colors_list) if colors_list else "neutral tones"
    except Exception:
        colors_str = "neutral tones"

    base_image = ensure_png_upload(base_image_file)

    prompt = IMAGE_TO_IMAGE_PROMPT.format(
        style=style,
        occasion=occasion,
        climate=climate,
        colors=colors_str,
        height_cm=traits["height_cm"],
        weight_kg=traits["weight_kg"],
        body_type=traits["body_type"],
    )

    try:
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=base_image,
            prompt=prompt,
            n=1,                     # 🔒 HARD LIMIT
            size="512x512"
        )

        if not response.data or not response.data[0].b64_json:
            raise Exception("Empty image response")

        logging.info(f"[AI-FORM] Request {request_id} SUCCESS")

        return {
            "status": "ok",
            "request_id": request_id,
            "image": response.data[0].b64_json,  # 👈 SOLO UNA
            "traits_used": traits
        }

    except Exception as e:
        logging.error(f"[AI-FORM] Request {request_id} FAILED: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Outfit generation failed"
        )
