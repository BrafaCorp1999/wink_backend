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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO)

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

def decode_file(file: UploadFile) -> BytesIO:
    content = file.file.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024))
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf

def bytesio_to_file(image: BytesIO, filename: str):
    image.seek(0)
    return (filename, image.read(), "image/png")

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

    categories = json.loads(clothes_categories)
    categories_en = translate_categories(categories)

    if len(categories) != len(clothes_files):
        raise HTTPException(status_code=400, detail="Categories count mismatch")

    base_image = decode_file(base_image_file)
    clothes_images = [decode_file(f) for f in clothes_files]

    for idx, cat in enumerate(categories_en):
        prompt = f"""
Use the first image as the exact same person reference.

IDENTITY & BODY LOCK:
- Preserve face, hairstyle, skin tone, natural marks, and proportions.
- Do not smooth, whiten, or stylize skin or facial features, preserve body traits.

CLOTHING REPLACEMENT:
- Replace ONLY the {cat} using the uploaded image exactly (image {idx+1}).
- Do NOT alter any other clothing.
- Fit naturally over the body respecting folds and gravity.

STYLE & SCENE:
- {style}, clean, realistic fashion photography.
- Preserve original background (garden, plaza, etc.)
- Soft natural lighting, no shadows, no CGI.

POSE & FRAMING:
- Full body, head to feet visible.
- Hair, hands, and feet fully visible.
- Camera at eye level.

OUTPUT:
- One ultra-realistic image, no illustration, no painting, no 3D.
"""
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=bytesio_to_file(base_image, f"base_{idx}.png"),
            prompt=prompt,
            n=1,
            size="1024x1024"
        )

        if not response.data or not response.data[0].b64_json:
            raise Exception("Empty image response")
        base_image = BytesIO(base64.b64decode(response.data[0].b64_json))

    final_b64 = base64.b64encode(base_image.getvalue()).decode()
    return {"status": "ok", "request_id": request_id, "image": final_b64, "categories_used": categories}
