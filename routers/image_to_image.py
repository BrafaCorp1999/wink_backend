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
# OpenAI client
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if not client:
    raise RuntimeError("OPENAI_API_KEY not set")

# =========================
# Helper: asegurar PNG
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
# PROMPT BASE
# =========================
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

# =========================
# ENDPOINT MOBILE
# =========================
@router.post("/ai/generate-outfit-from-form")
async def generate_outfit_from_form(
    gender: str = Form(...),
    style: str = Form("casual"),
    occasion: str = Form("daily"),
    climate: str = Form("temperate"),
    colors: str = Form("neutral"),
    base_image_file: UploadFile = File(...)
):
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    request_id = str(uuid.uuid4())
    logging.info(f"[AI-FORM-MOBILE] Request {request_id} started")

    # ------------------- Colors
    try:
        colors_list = json.loads(colors)
        colors_str = ", ".join(colors_list) if colors_list else "neutral tones"
    except Exception:
        colors_str = "neutral tones"

    # ------------------- Base image
    base_image = await ensure_png_upload(base_image_file)

    # ------------------- Prompt
    prompt = IMAGE_TO_IMAGE_PROMPT.format(
        style=style,
        occasion=occasion,
        climate=climate,
        colors=colors_str
    )

    try:
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=base_image,
            prompt=prompt,
            n=1,
            size="auto"
        )

        if not response.data or not response.data[0].b64_json:
            raise Exception("Empty image response")

        logging.info(f"[AI-FORM-MOBILE] Request {request_id} SUCCESS")

        return {
            "status": "ok",
            "request_id": request_id,
            "image": response.data[0].b64_json
        }

    except Exception as e:
        logging.error(f"[AI-FORM-MOBILE] Request {request_id} FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail="Outfit generation failed")
