from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import base64
import requests
from openai import OpenAI

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RegisterRequest(BaseModel):
    mode: str  # "selfie_manual" o "photo_body"
    gender: str
    selfie_base64: str
    body_image_base64: str | None = None
    height_cm: int | None = None
    weight_kg: int | None = None
    body_type: str | None = None
    hair_type: str | None = None

# --- Prompt selfie + medidas manuales ---
def build_prompt_selfie_manual(data: RegisterRequest) -> str:
    return f"""
Generate an ultra-realistic full body fashion photograph.

IMPORTANT:
- Preserve exact facial identity from the selfie reference (90% similarity)
- Hair fully visible and styled naturally
- Full body including feet and shoes appropriate for outfit
- Respect body proportions:
    - Height: {data.height_cm or 'unknown'} cm
    - Weight: {data.weight_kg or 'unknown'} kg
    - Body type: {data.body_type or 'average'}
    - Hair type: {data.hair_type or 'natural'}
- Outfit style:
    - Image 1: casual, modern, neutral colors, sneakers
    - Image 2: formal, elegant, different color palette, formal shoes
- Camera: full body shot, studio lighting, natural pose
- Ensure the face is clearly visible and unchanged
"""

# --- Prompt foto de cuerpo completo ---
def build_prompt_photo_body(extracted_features: dict, selfie_base64: str) -> str:
    return f"""
Generate an ultra-realistic full body fashion photograph.

IMPORTANT:
- Preserve exact facial identity from the reference image (90% similarity)
- Hair fully visible and styled naturally
- Full body including feet and shoes appropriate for outfit
- Respect extracted body measurements:
    - Height: {extracted_features.get('height_cm', 'unknown')} cm
    - Weight: {extracted_features.get('weight_kg', 'unknown')} kg
    - Body type: {extracted_features.get('body_type', 'average')}
    - Hair type: {extracted_features.get('hair_type', 'natural')}
- Outfit style:
    - Image 1: casual, modern, neutral colors, sneakers
    - Image 2: formal, elegant, different color palette, formal shoes
- Camera: full body shot, studio lighting, natural pose
- Ensure the face is clearly visible and unchanged
"""

# --- Analiza cuerpo usando endpoint externo y loguea rasgos ---
def analyze_body(body_base64: str, gender: str) -> dict:
    url = "https://wink-backend-1jao.onrender.com/api/analyze-body-with-face/"
    response = requests.post(url, json={
        "image_base64": body_base64,
        "gender_hint": gender
    }, timeout=120)

    if response.status_code != 200:
        raise HTTPException(500, f"Error analyzing body: {response.text}")

    decoded = response.json()
    # Log para verificar medidas en backend/render
    print("📊 [ANALYZE BODY] Rasgos obtenidos:", decoded)
    return decoded

# --- Endpoint principal ---
@router.post("/register_generate_base_images")
def register_generate_images(data: RegisterRequest):
    try:
        if data.mode == "selfie_manual":
            if not data.selfie_base64:
                raise HTTPException(400, "Selfie base64 required")
            prompt = build_prompt_selfie_manual(data)
            reference_images = [data.selfie_base64]

            # Log de medidas enviadas
            print("📊 [SELFIE MANUAL] Enviando rasgos:", {
                "height_cm": data.height_cm,
                "weight_kg": data.weight_kg,
                "body_type": data.body_type,
                "hair_type": data.hair_type
            })

        elif data.mode == "photo_body":
            if not data.body_image_base64:
                raise HTTPException(400, "Body image required")
            features = analyze_body(data.body_image_base64, data.gender)
            prompt = build_prompt_photo_body(features, data.selfie_base64)
            reference_images = [data.selfie_base64, data.body_image_base64]

        else:
            raise HTTPException(400, "Invalid mode")

        # --- Generación de 2 imágenes (casual y formal) ---
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            image=reference_images,
            n=2
        )

        # --- Convertir a base64 limpio ---
        images_b64 = [img.b64_json for img in result.data]

        print("📩 [GENERATE OUTFITS] Se generaron", len(images_b64), "imágenes")
        return {"status": "ok", "images": images_b64}

    except Exception as e:
        raise HTTPException(500, f"Image generation failed: {str(e)}")
