from fastapi import APIRouter, HTTPException, Form
from typing import List
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
# Helper: BytesIO → OpenAI file
# =========================
def bytesio_to_file(image: BytesIO, filename: str):
    image.seek(0)
    return (filename, image.read(), "image/png")

# =========================
# Category helpers
# =========================
CATEGORY_MAP = {
    "zapatos": "shoes",
    "blusas": "blouse",
    "chamarras": "jacket",
    "vestidos": "dress",
    "camisas": "shirt",
    "pantalones": "pants",
    "poleras": "t-shirt",
    "accesorios": "accessories",
}

def translate_categories(categories: List[str]) -> List[str]:
    return [CATEGORY_MAP.get(c.lower(), c.lower()) for c in categories]

def parse_categories(categories_json: str) -> List[str]:
    try:
        categories = json.loads(categories_json)
        if not isinstance(categories, list):
            raise Exception()
        if len(categories) > 2:
            raise HTTPException(
                status_code=400,
                detail="You can only replace up to 2 clothing items"
            )
        return categories
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid clothes_categories JSON")

# =========================
# ENDPOINT WEB
# =========================
@router.post("/ai/combine-clothes-web")
async def combine_clothes_web(
    gender: str = Form(...),
    style: str = Form("casual"),
    base_image_b64: str = Form(...),
    clothes_images_b64: str = Form(...),
    clothes_categories: str = Form(...)
):
    request_id = str(uuid.uuid4())
    logging.info(f"[COMBINE-WEB] Request {request_id} started")

    try:
        clothes_list = json.loads(clothes_images_b64)
        if not isinstance(clothes_list, list) or not clothes_list:
            raise Exception()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid clothes_images_b64")

    categories = parse_categories(clothes_categories)
    categories_en = translate_categories(categories)

    if len(categories) != len(clothes_list):
        raise HTTPException(status_code=400, detail="Categories count mismatch")

    # Decodificar imágenes
    base_image = decode_base64_image(base_image_b64)
    clothes_images = [decode_base64_image(b64) for b64 in clothes_list]

    # Construir texto de prendas
    items_text = "\n".join([f"- {cat} (use uploaded image exactly)" for cat in categories_en])

    # Prompt actualizado para preservar rostro, piel, y prendas no seleccionadas
    prompt = f"""
Use the FIRST image as the SAME person reference.

IDENTITY & BODY LOCK (STRICT):
- Preserve face, hairstyle, skin tone, natural imperfections, freckles, and any unique marks.
- Do NOT change body size, posture, or volume.
- Do NOT smooth, whiten, or alter facial or skin features.
- Keep all proportions exactly as in the original image.

CLOTHING REPLACEMENT ONLY:
- Replace ONLY the following clothing items:
{items_text}
- DO NOT alter any other clothing not listed above.
- Fit clothes naturally over the existing body.
- Respect natural folds, gravity, and fabric behavior.
- Use ONLY the uploaded clothing images exactly as provided.
- Do NOT invent colors, textures, or accessories for other clothing.

STYLE TARGET:
- {style}
- Clean, realistic fashion photography.

SCENE & LIGHTING:
- Neutral background
- Soft natural lighting
- No strong shadows
- No transparent or fantasy environments

POSE & FRAMING:
- Full body (head to feet) fully visible
- Neutral standing pose
- Camera at human eye level

OUTPUT:
- One ultra realistic final image
- No illustration, no CGI, no 3D, no painting
"""

    try:
        current_image = base_image

        for idx, _ in enumerate(clothes_images):
            response = client.images.edit(
                model="gpt-image-1-mini",
                image=bytesio_to_file(current_image, f"base_{idx}.png"),
                prompt=prompt,
                n=1,
                size="1024x1024"
            )

            if not response.data or not response.data[0].b64_json:
                raise Exception("Empty image response")

            current_image = BytesIO(
                base64.b64decode(response.data[0].b64_json)
            )

        final_b64 = base64.b64encode(current_image.getvalue()).decode()

        logging.info(f"[COMBINE-WEB] Request {request_id} SUCCESS")
        return {
            "status": "ok",
            "request_id": request_id,
            "image": final_b64,
            "categories_used": categories
        }

    except Exception as e:
        logging.error(f"[COMBINE-WEB] FAILED: {repr(e)}")
        raise HTTPException(status_code=500, detail="Combine clothes generation failed")
