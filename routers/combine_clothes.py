from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
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
# Helper: asegurar PNG
# =========================
def ensure_png_upload(upload: UploadFile) -> BytesIO:
    try:
        image_bytes = upload.file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = upload.filename or "input.png"
        return buffer
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# =========================
# ENDPOINT: COMBINAR PRENDAS
# =========================
@router.post("/ai/combine-clothes")
async def combine_clothes(
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: str = Form("casual"),
    base_image_file: UploadFile = File(...),
    clothes_files: List[UploadFile] = File(...),
    clothes_categories: str = Form(...)
):
    request_id = str(uuid.uuid4())
    logging.info(f"[COMBINE] Request {request_id} started")

    # -------------------------
    # Validar traits
    # -------------------------
    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    # -------------------------
    # Validar categorías
    # -------------------------
    try:
        categories = json.loads(clothes_categories)
        if not isinstance(categories, list):
            raise Exception()
        if len(categories) != len(clothes_files):
            raise HTTPException(
                status_code=400,
                detail="Categories count must match clothes files"
            )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid clothes_categories JSON")

    # -------------------------
    # Preparar imágenes
    # -------------------------
    base_image = ensure_png_upload(base_image_file)
    clothes_images = [ensure_png_upload(f) for f in clothes_files]

    # -------------------------
    # Prompt bloqueado
    # -------------------------
    items_text = "\n".join(
        [f"- {cat} (use uploaded image exactly)" for cat in categories]
    )

    prompt = f"""
Use the FIRST image as reference for the SAME person.

IDENTITY LOCK (STRICT):
- Preserve face, hairstyle, skin tone, body proportions.
- Do NOT change age or ethnicity.

CLOTHING COMBINATION (STRICT):
- Use ONLY the uploaded clothing images listed below.
- Do NOT invent new clothes.
- Fit the clothes naturally on the body.
- Style target: {style}

UPLOADED ITEMS:
{items_text}

OUTPUT RULES:
- Full body (head to feet)
- Natural standing pose
- Photorealistic fashion photo
- High quality
"""

    # -------------------------
    # Llamada OpenAI (MULTI-IMAGE)
    # -------------------------
    try:
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=[base_image, *clothes_images],  # 👈 CLAVE
            prompt=prompt,
            n=1,
            size="512x512"
        )

        if not response.data or not response.data[0].b64_json:
            raise Exception("Empty image response")

        logging.info(f"[COMBINE] Request {request_id} SUCCESS")

        return {
            "status": "ok",
            "request_id": request_id,
            "image": response.data[0].b64_json,
            "categories_used": categories,
            "traits_used": traits
        }

    except Exception as e:
        logging.error(f"[COMBINE] Request {request_id} FAILED: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Combine clothes generation failed"
        )
