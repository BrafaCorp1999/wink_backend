from fastapi import APIRouter, HTTPException, Form
from typing import List, Optional
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import json
import os
import uuid
import logging

router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)
logging.basicConfig(level=logging.INFO)

# =========================
# Helper: decode base64 to PNG
# =========================
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

# =========================
# Helper: parse categories
# =========================
def parse_categories(categories_json: str) -> List[str]:
    try:
        categories = json.loads(categories_json)
        if not isinstance(categories, list):
            raise Exception()
        if not (1 <= len(categories) <= 3):
            raise HTTPException(
                status_code=400,
                detail="You can only replace 1 to 3 clothing items"
            )
        return categories
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid clothes_categories JSON")

# =========================
# ENDPOINT WEB: COMBINAR PRENDAS
# =========================
@router.post("/ai/combine-clothes-web")
async def combine_clothes_web(
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: str = Form("casual"),
    base_image_b64: str = Form(...),
    clothes_images_b64: str = Form(...),  # JSON array de strings base64
    clothes_categories: str = Form(...)
):
    request_id = str(uuid.uuid4())
    logging.info(f"[COMBINE-WEB] Request {request_id} started")

    try:
        clothes_list = json.loads(clothes_images_b64)
        if not isinstance(clothes_list, list) or not clothes_list:
            raise HTTPException(status_code=400, detail="clothes_images_b64 must be a non-empty JSON list")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid clothes_images_b64 format")

    categories = parse_categories(clothes_categories)

    if len(categories) != len(clothes_list):
        raise HTTPException(
            status_code=400,
            detail="Number of categories must match number of clothing images"
        )

    # Decodificar imágenes
    base_image = decode_base64_image(base_image_b64)
    clothes_images = [decode_base64_image(b64) for b64 in clothes_list]

    # Prompt dinámico
    items_text = "\n".join([f"- {cat} (use uploaded image exactly)" for cat in categories])
    prompt = f"""
Use the FIRST image as the SAME person reference.

IDENTITY & BODY LOCK (STRICT, NON-NEGOTIABLE):
- Preserve the exact face, hairstyle, skin tone, body shape, body size and proportions.
- Do NOT change body volume, curves, height, weight, muscles or posture.
- Do NOT slim, enlarge or stylize the body.
- The person must look exactly the same as the base image.

CLOTHING REPLACEMENT ONLY:
- Replace ONLY the following clothing items:
{items_text}
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
    logging.info(f"[COMBINE-WEB] Prompt length: {len(prompt)}")

    try:
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=[base_image, *clothes_images],
            prompt=prompt,
            n=1,
            size="auto"
        )

        if not response.data or not response.data[0].b64_json:
            raise Exception("Empty image response")

        logging.info(f"[COMBINE-WEB] Request {request_id} SUCCESS")
        return {
            "status": "ok",
            "request_id": request_id,
            "image": response.data[0].b64_json,
            "categories_used": categories,
            "traits_used": json.loads(body_traits)
        }

    except Exception as e:
        logging.error(f"[COMBINE-WEB] Request {request_id} FAILED: {repr(e)}")
        raise HTTPException(status_code=500, detail="Combine clothes generation failed")
