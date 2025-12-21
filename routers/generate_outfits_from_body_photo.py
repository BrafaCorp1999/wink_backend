from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import json
import os

router = APIRouter()

# =========================
# PROMPT – BODY PHOTO
# =========================

BODY_PHOTO_PROMPT = """
Use the uploaded full-body image strictly as a visual reference for the SAME real person.

IDENTITY LOCK:
- Preserve facial features (face shape, eyes, nose, lips, skin tone).
- Preserve hairstyle and hair color.
- Maintain body proportions, height, and body type.
- Do NOT alter identity or pose.

CLOTHING REPLACEMENT:
- Replace the original outfit entirely.
- Create a new {style} outfit.
- Include top, bottoms, shoes.
- Add subtle accessories if appropriate.
- Fabrics must look realistic with natural folds.
- Colors harmonious and modern.

POSE & COMPOSITION:
- Full-body, head to toe, including legs and feet.
- Natural standing pose.
- Eye-level camera.
- Shoes/tennis must match outfit style.
- Do NOT crop or distort anatomy.

LIGHTING & QUALITY:
- Clean studio or neutral background.
- Soft natural lighting.
- DSLR-quality, photorealistic.
- No illustration, no CGI.

OUTPUT:
- Generate 1 outfit per request. Make multiple requests for variations.
"""

# =========================
# UTIL: asegurar PNG válido
# =========================

def ensure_png_upload(upload: UploadFile) -> BytesIO:
    """
    Convierte cualquier imagen recibida a PNG válido
    con filename y mimetype aceptados por OpenAI.
    """
    try:
        image_bytes = upload.file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = "input.png"

        return buffer

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )

# =========================
# ENDPOINT
# =========================

@router.post("/generate-outfits/body-photo")
async def generate_outfits_from_body_photo(
    gender: str = Form(...),
    body_traits: str = Form(...),   # JSON string desde analyze-body-with-face
    style: str = Form("casual"),
    image_file: UploadFile = File(...)
):
    # -------------------------
    # Validar traits
    # -------------------------
    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    # -------------------------
    # Preparar imagen base
    # -------------------------
    base_image = ensure_png_upload(image_file)

    # -------------------------
    # OpenAI Client
    # -------------------------
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    images_b64: list[str] = []

    try:
        # 🔁 LOOP INTERNO para generar 2 outfits distintos
        for i in range(2):
            # Construir prompt incluyendo traits, cabello, edad si están
            variation_prompt = (
                BODY_PHOTO_PROMPT.format(style=style)
                + "\n\nUSER BODY TRAITS:\n"
                f"- Gender: {gender}\n"
                f"- Height: {traits.get('height_cm', 'unknown')} cm\n"
                f"- Weight: {traits.get('weight_kg', 'unknown')} kg\n"
                f"- Waist: {traits.get('waist_cm', 'unknown')} cm\n"
                f"- Hips: {traits.get('hips_cm', 'unknown')} cm\n"
                f"- Shoulders: {traits.get('shoulders_cm', 'unknown')} cm\n"
                f"- Neck: {traits.get('neck_cm', 'unknown')} cm\n"
                f"- Body type: {traits.get('body_type', 'average')}\n"
            )

            # Opcional: edad y tipo de cabello si vienen en traits
            if 'age' in traits:
                variation_prompt += f"- Age: {traits['age']}\n"
            if 'hair_length' in traits:
                variation_prompt += f"- Hair length: {traits['hair_length']}\n"
            if 'hair_type' in traits:
                variation_prompt += f"- Hair type: {traits['hair_type']}\n"

            variation_prompt += (
                f"\nOUTFIT VARIATION #{i+1}: Make this outfit clearly different from the previous one."
            )

            response = client.images.edit(
                model="gpt-image-1.5",
                image=base_image,
                prompt=variation_prompt,
                n=1,            # Siempre 1 por loop
                size="512x512"  # Menor tamaño = menos créditos
            )

            if not response.data:
                raise Exception("Empty image response")

            images_b64.append(response.data[0].b64_json)

        # -------------------------
        # Retornar resultado
        # -------------------------
        return {
            "status": "ok",
            "mode": "body_photo",
            "images": images_b64,
            "traits_used": traits
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )
