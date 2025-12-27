from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import json
import os
import firebase_admin
from firebase_admin import credentials, firestore

# =========================
# Firestore setup
# =========================
if not firebase_admin._apps:
    cred = credentials.Certificate("path/to/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

router = APIRouter()

# =========================
# Prompt template dinámico
# =========================
IMAGE_TO_IMAGE_PROMPT = """
Use the provided base image strictly as reference for the SAME person.

IDENTITY LOCK:
- Preserve facial features, hair, skin tone, body proportions.

CLOTHING:
- Generate a modern, realistic outfit according to the following:
  - Style: {style}
  - Occasion: {occasion}
  - Climate: {climate}
  - Preferred colors: {colors}

BODY TRAITS:
- Height: {height_cm} cm
- Weight: {weight_kg} kg
- Waist: {waist_cm} cm
- Hips: {hips_cm} cm
- Shoulders: {shoulders_cm} cm
- Neck: {neck_cm} cm
- Body type: {body_type}

ADDITIONAL INSTRUCTIONS:
{additional_instructions}
"""

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
        buffer.name = "input.png"
        return buffer
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# =========================
# Endpoint
# =========================
@router.post("/generate-outfit/image-to-image")
async def generate_outfit_image_to_image(
    uid: str = Form(...),
    gender: str = Form(...),
    body_traits: str = Form(...),  # JSON string con medidas
    style: str = Form("casual"),
    occasion: str = Form("daily"),
    climate: str = Form("temperate"),
    colors: str = Form("neutral"),  # lista en JSON
    additional_instructions: str = Form(""),
    base_image_file: UploadFile = File(...)
):
    # -------------------------
    # Control de límite
    # -------------------------
    user_doc_ref = db.collection("users").document(uid)
    user_doc = user_doc_ref.get().to_dict() or {}
    used_count = user_doc.get("image_to_image_count", 0)
    if used_count >= 2:
        return {"status": "error", "message": "Límite alcanzado. Suscríbete a Wink Pro"}

    # -------------------------
    # Validar traits
    # -------------------------
    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    # -------------------------
    # Validar colores
    # -------------------------
    try:
        colors_list = json.loads(colors)
        colors_str = ", ".join(colors_list) if colors_list else "neutral tones"
    except Exception:
        colors_str = "neutral tones"

    # -------------------------
    # Preparar imagen base
    # -------------------------
    base_image = ensure_png_upload(base_image_file)

    # -------------------------
    # Generar prompt dinámico
    # -------------------------
    prompt = IMAGE_TO_IMAGE_PROMPT.format(
        style=style,
        occasion=occasion,
        climate=climate,
        colors=colors_str,
        height_cm=traits.get("height_cm", "unknown"),
        weight_kg=traits.get("weight_kg", "unknown"),
        waist_cm=traits.get("waist_cm", "unknown"),
        hips_cm=traits.get("hips_cm", "unknown"),
        shoulders_cm=traits.get("shoulders_cm", "unknown"),
        neck_cm=traits.get("neck_cm", "unknown"),
        body_type=traits.get("body_type", "average"),
        additional_instructions=additional_instructions
    )

    # -------------------------
    # OpenAI Client
    # -------------------------
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

        images_b64 = [img.b64_json for img in response.data]

        # -------------------------
        # Actualizar Firestore
        # -------------------------
        user_doc_ref.set({
            "image_to_image_count": used_count + 1,
            "last_generation": {
                "traits": traits,
                "colors": colors_list,
                "style": style,
                "occasion": occasion,
                "climate": climate
            }
        }, merge=True)

        return {
            "status": "ok",
            "images": images_b64,
            "traits_used": traits,
            "prompt_used": prompt
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image-to-Image failed: {str(e)}")
