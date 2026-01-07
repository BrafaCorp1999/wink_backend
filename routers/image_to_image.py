from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import json
import os

router = APIRouter()

# =========================
# PROMPT – IMAGE TO IMAGE (OPTIMIZADO)
# =========================
IMAGE_TO_IMAGE_PROMPT = """
Use the provided image as reference for the SAME real person.

IDENTITY LOCK:
- Preserve face, hairstyle, skin tone and body proportions.

CLOTHING:
- Change ONLY the outfit.
- Style: {style}
- Occasion: {occasion}
- Climate: {climate}
- Preferred colors: {colors}
- Clothing must fit naturally and look realistic.

BODY:
- Height: {height_cm} cm
- Weight: {weight_kg} kg
- Body type: {body_type}

POSE & QUALITY:
- Full-body, head to feet.
- Natural standing pose.
- Photorealistic.
- No illustration or CGI.
"""

# =========================
# Normalizar traits
# =========================
def normalize_traits(traits: dict, gender: str) -> dict:
    return {
        "height_cm": traits.get("height_cm") or (175 if gender == "male" else 165),
        "weight_kg": traits.get("weight_kg") or (70 if gender == "male" else 60),
        "body_type": traits.get("body_type", "average"),
    }

# =========================
# Asegurar PNG
# =========================
def ensure_png_upload(upload: UploadFile) -> BytesIO:
    try:
        image_bytes = upload.file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = "base.png"
        return buffer
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# =========================
# ENDPOINT
# =========================
@router.post("/generate-outfit/image-to-image")
async def generate_outfit_image_to_image(
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: str = Form("casual"),
    occasion: str = Form("daily"),
    climate: str = Form("temperate"),
    colors: str = Form("neutral"),
    base_image_file: UploadFile = File(...)
):
    try:
        raw_traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    traits = normalize_traits(raw_traits, gender)

    # Colores
    try:
        colors_list = json.loads(colors)
        colors_str = ", ".join(colors_list) if colors_list else "neutral tones"
    except Exception:
        colors_str = "neutral tones"

    base_image = ensure_png_upload(base_image_file)

    prompt = IMAGE_TO_IMAGE_PROMPT.format(
        style=style,
        occasion=occasion,
        climate=climate,
        colors=colors_str,
        height_cm=traits["height_cm"],
        weight_kg=traits["weight_kg"],
        body_type=traits["body_type"],
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=base_image,
            prompt=prompt,
            n=1,
            size="512x512"
        )

        if not response.data:
            raise Exception("Empty image response")

        return {
            "status": "ok",
            "mode": "image_to_image",
            "images": [response.data[0].b64_json],
            "traits_used": traits
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image-to-image failed: {str(e)}")
