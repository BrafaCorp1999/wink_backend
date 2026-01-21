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

# -------------------------
# OpenAI client (safe init)
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)

# =========================
# Helper: asegurar PNG (ASYNC SAFE)
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
        buffer.name = upload.filename or "input.png"
        return buffer

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )

# =========================
# Helper: validar traits
# =========================
def parse_traits(traits_json: str) -> dict:
    try:
        traits = json.loads(traits_json)
        if not isinstance(traits, dict):
            raise Exception()
        return traits
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid body_traits JSON"
        )

# =========================
# Helper: validar categorías
# =========================
def parse_categories(
    categories_json: str,
    clothes_files: List[UploadFile]
) -> List[str]:
    try:
        categories = json.loads(categories_json)
        if not isinstance(categories, list):
            raise Exception()

        if len(categories) != len(clothes_files):
            raise HTTPException(
                status_code=400,
                detail="Categories count must match clothes files"
            )

        return categories

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid clothes_categories JSON"
        )

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

    if not clothes_files:
        raise HTTPException(
            status_code=400,
            detail="At least one clothing image is required"
        )

    # -------------------------
    # Parse traits y categorías
    # -------------------------
    traits = parse_traits(body_traits)
    categories = parse_categories(clothes_categories, clothes_files)

    logging.info(f"[COMBINE] Traits: {traits}")
    logging.info(f"[COMBINE] Categories: {categories}")

    # -------------------------
    # Preparar imágenes
    # -------------------------
    base_image = await ensure_png_upload(base_image_file)
    clothes_images = [
        await ensure_png_upload(f) for f in clothes_files
    ]

    # -------------------------
    # Prompt (NO MODIFICADO)
    # -------------------------
    items_text = "\n".join(
        [f"- {cat} (use uploaded image exactly)" for cat in categories]
    )

    prompt = f"""
Use the FIRST image as the SAME person reference.

IDENTITY & BODY LOCK (STRICT, NON-NEGOTIABLE):
- Preserve the exact face, hairstyle, skin tone, body shape, body size and proportions.
- Do NOT change body volume, curves, height, weight, muscles or posture.
- Do NOT slim, enlarge or stylize the body.
- The person must look exactly the same as the base image.

CLOTHING REPLACEMENT ONLY:
- ONLY change the clothes.
- Fit the clothes naturally over the existing body.
- Respect natural folds, gravity and fabric behavior.

CLOTHING RULES (STRICT):
- Use ONLY the uploaded clothing images.
- Do NOT invent clothes, colors or textures.
- Do NOT add accessories.
- Do NOT remove underwear visibility if originally hidden.

STYLE TARGET:
- {style}
- Clean, realistic fashion photography.

SCENE & LIGHTING:
- Neutral background
- Soft natural lighting
- No strong shadows
- No transparent or fantasy environments

POSE & FRAMING:
- Full body (head to feet)
- Neutral standing pose
- Camera at human eye level

OUTPUT:
- One single final image
- Ultra realistic
- No illustration, no CGI, no 3D, no painting
"""

    # -------------------------
    # Llamada OpenAI (MULTI-IMAGE)
    # -------------------------
    try:
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=[base_image, *clothes_images],
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
