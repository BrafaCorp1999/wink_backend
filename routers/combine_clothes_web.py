from fastapi import APIRouter, HTTPException, Form
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Helpers
# =========================
def decode_base64_image(b64: str) -> BytesIO:
    try:
        image_bytes = base64.b64decode(b64)
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

def bytesio_to_file(image: BytesIO, filename: str):
    image.seek(0)
    return (filename, image.read(), "image/png")

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

def translate_category(cat: str) -> str:
    return CATEGORY_MAP.get(cat.lower(), cat.lower())

# =========================
# ENDPOINT WEB – 1 PRENDA
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
    logging.info(f"[COMBINE-WEB-1ITEM] Request {request_id} started")

    try:
        # =========================
        # Parse inputs
        # =========================
        clothes_list = json.loads(clothes_images_b64)
        categories = json.loads(clothes_categories)

        if not isinstance(clothes_list, list) or not isinstance(categories, list):
            raise HTTPException(status_code=400, detail="Invalid clothes data")

        if len(clothes_list) != 1 or len(categories) != 1:
            raise HTTPException(
                status_code=400,
                detail="This endpoint supports EXACTLY 1 clothing item"
            )

        category_en = translate_category(categories[0])

        # =========================
        # Decode images
        # =========================
        base_image = decode_base64_image(base_image_b64)

        # NOTA:
        # La prenda NO se pasa aún como imagen de referencia.
        # Primero validamos comportamiento del prompt.
        _ = decode_base64_image(clothes_list[0])

        # =========================
        # PROMPT SIMPLE Y DIRECTO
        # =========================
        prompt = f"""
TASK:
Replace ONE clothing item on the person using the uploaded clothing image.

REFERENCE PERSON (IMAGE 1):
- This is the SAME person.
- Keep the face EXACTLY the same.
- Do NOT change facial features, skin tone, lighting, or expression.
- Do NOT beautify, smooth, whiten, or stylize the face.
- Keep body shape, height, proportions, torso, legs, and posture unchanged.

CLOTHING REPLACEMENT:
- Replace ONLY the {category_en}.
- Use the uploaded clothing image EXACTLY as provided.
- Do NOT invent, redesign, or approximate the clothing.
- Do NOT change or remove any other clothing items.
- Fit the clothing naturally to the body.

DO NOT:
- Do NOT crop the image.
- Do NOT zoom in.
- Do NOT cut hair, head, hands, or feet.
- Do NOT add new clothes or accessories.

SCENE:
- Keep the original background exactly the same.
- Keep original lighting and environment.

FRAMING:
- Full body visible from head to feet.

OUTPUT:
- One realistic photo.
- No illustration, no CGI, no 3D, no painting.
"""

        # =========================
        # Image generation
        # =========================
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=bytesio_to_file(base_image, "base.png"),
            prompt=prompt,
            n=1,
            size="1024x1024"
        )

        if not response.data or not response.data[0].b64_json:
            raise Exception("Empty image response")

        final_b64 = response.data[0].b64_json

        logging.info(f"[COMBINE-WEB-1ITEM] Request {request_id} SUCCESS")

        return {
            "status": "ok",
            "request_id": request_id,
            "image": final_b64,
            "category_used": categories[0]
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[COMBINE-WEB-1ITEM] FAILED: {repr(e)}")
        raise HTTPException(status_code=500, detail="Combine clothes generation failed")
