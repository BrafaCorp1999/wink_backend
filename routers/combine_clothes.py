from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
from openai import OpenAI
from io import BytesIO
from PIL import Image
import json
import os

router = APIRouter()

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
# Endpoint: combinar prendas manteniendo identidad
# =========================
@router.post("/generate-outfit/combine-clothes-flex")
async def generate_outfit_combine_clothes_flex(
    gender: str = Form(...),
    body_traits: str = Form(...),  # JSON con medidas
    style: str = Form("casual"),
    base_image_file: UploadFile = File(...),
    clothes_files: List[UploadFile] = File(...),
    clothes_categories: str = Form(...)  # JSON array: ["top", "bottom", "shoes"]
):
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
        if len(categories) != len(clothes_files):
            raise HTTPException(status_code=400, detail="Mismatch between categories and files")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid clothes_categories JSON")

    # -------------------------
    # Preparar imágenes
    # -------------------------
    base_image = ensure_png_upload(base_image_file)
    clothes_images = [ensure_png_upload(f) for f in clothes_files]

    # -------------------------
    # Construir prompt dinámico
    # -------------------------
    items_text = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])
    prompt = f"""
Use the provided base image as reference for the SAME person.

IDENTITY LOCK:
- Preserve facial features and body proportions.

CLOTHING COMBINATION:
- Combine the following uploaded clothing items into a coherent outfit:
{items_text}
- Outfit should match the indicated style: {style}.
- Ensure clothes fit naturally on the body.
- Optional subtle accessories allowed.
"""

    # -------------------------
    # OpenAI Client
    # -------------------------
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        # -------------------------
        # Usar .edit para mantener identidad
        # -------------------------
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=base_image,
            prompt=prompt,
            n=1,
            size="512x512"
            # mask=None -> opcional si quieres reemplazar solo ropa
        )

        if not response.data:
            raise Exception("Empty image response")

        images_b64 = [img.b64_json for img in response.data]

        return {
            "status": "ok",
            "images": images_b64,
            "traits_used": traits,
            "categories_used": categories
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combine clothes failed: {str(e)}")
