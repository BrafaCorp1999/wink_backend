from fastapi import APIRouter, HTTPException, Form, UploadFile, File
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
# Helper: decode uploaded file
# =========================
def decode_file(file: UploadFile) -> BytesIO:
    try:
        content = file.file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
        MAX_SIZE = 1024
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# =========================
# Helper: translate categories
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

# =========================
# Helper: parse categories
# =========================
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
# ENDPOINT MOBILE: COMBINAR PRENDAS
# =========================
@router.post("/ai/combine-clothes")
async def combine_clothes_mobile(
    gender: str = Form(...),
    style: str = Form("casual"),
    base_image_file: UploadFile = File(...),
    clothes_files: List[UploadFile] = File(...),
    clothes_categories: str = Form(...)
):
    request_id = str(uuid.uuid4())
    logging.info(f"[COMBINE-MOBILE] Request {request_id} started")

    categories = parse_categories(clothes_categories)
    categories_en = translate_categories(categories)

    if len(categories) != len(clothes_files):
        raise HTTPException(
            status_code=400,
            detail="Number of categories must match number of clothing images"
        )

    # Decodificar imágenes
    base_image = decode_file(base_image_file)
    clothes_images = [decode_file(f) for f in clothes_files]

    # Construir prompt
    items_text = "\n".join([f"- {cat} (use uploaded image exactly)" for cat in categories_en])
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
    logging.info(f"[COMBINE-MOBILE] Prompt length: {len(prompt)}")

    try:
        # Editar cada prenda encadenada
        current_image = base_image
        for img in clothes_images:
            response = client.images.edit(
                model="gpt-image-1-mini",
                image=current_image,
                mask=None,
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            if not response.data or not response.data[0].b64_json:
                raise Exception("Empty image response")
            current_image = BytesIO(base64.b64decode(response.data[0].b64_json))

        final_b64 = base64.b64encode(current_image.getvalue()).decode()

        logging.info(f"[COMBINE-MOBILE] Request {request_id} SUCCESS")
        return {
            "status": "ok",
            "request_id": request_id,
            "image": final_b64,
            "categories_used": categories
        }

    except Exception as e:
        logging.error(f"[COMBINE-MOBILE] Request {request_id} FAILED: {repr(e)}")
        raise HTTPException(status_code=500, detail="Combine clothes generation failed")
