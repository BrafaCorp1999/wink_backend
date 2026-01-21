from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import json
import os
import uuid
import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO)

# -------------------------
# OpenAI client (SAFE INIT)
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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

LIGHTING AND PLACE:
- Put the person in a relax place, not with more things but with some related with her/his outfit

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
        "height_cm": traits.get("height_cm") or (175 if gender.lower() == "male" else 165),
        "weight_kg": traits.get("weight_kg") or (70 if gender.lower() == "male" else 60),
        "body_type": traits.get("body_type") or "average",
    }

# =========================
# ASEGURAR PNG (ASYNC SAFE)
# =========================
async def ensure_png_upload(upload: UploadFile) -> BytesIO:
    try:
        image_bytes = await upload.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        MAX_SIZE = 1024
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE))

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = upload.filename or "base.png"
        return buffer

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

# =========================
# ENDPOINT GENERATE OUTFIT
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
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

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

    # -------- Prepare image
    base_image = await ensure_png_upload(base_image_file)

    # -------- Prompt (tu original intacto)
    prompt = IMAGE_TO_IMAGE_PROMPT.format(
        style=style,
        occasion=occasion,
        climate=climate,
        colors=colors_str,
        height_cm=traits["height_cm"],
        weight_kg=traits["weight_kg"],
        body_type=traits["body_type"],
    )

    # -------- Llamada OpenAI
    try:
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=base_image,
            prompt=prompt,
            n=1,                     # SOLO UNA IMAGEN
            size="auto"
        )

        if not response.data or not response.data[0].b64_json:
            raise Exception("Empty image response")

        logging.info(f"[AI-FORM] Request {request_id} SUCCESS")

        return {
            "status": "ok",
            "request_id": request_id,
            "image": response.data[0].b64_json,
            "traits_used": traits
        }

    except Exception as e:
        logging.error(f"[AI-FORM] Request {request_id} FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail="Outfit generation failed")
