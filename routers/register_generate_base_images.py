from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import base64
import requests
from openai import OpenAI

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RegisterRequest(BaseModel):
    mode: str
    gender: str
    selfie_base64: str
    body_image_base64: str | None = None
    height_cm: int | None = None
    weight_kg: int | None = None
    body_type: str | None = None
    hair_type: str | None = None

# --- Prompt para selfie + datos manuales ---
def build_prompt_selfie_manual(data: RegisterRequest) -> str:
    return f"""
Generate an ultra-realistic full body fashion photograph.

IMPORTANT:
- Preserve exact facial identity from the selfie reference
- Do not alter facial features
- Respect body proportions:
    - Height: {data.height_cm} cm
    - Weight: {data.weight_kg} kg
    - Body type: {data.body_type}
    - Hair type: {data.hair_type or 'natural'}
- Outfit style: casual, modern, neutral colors
- Camera: full body shot, studio lighting
"""

# --- Prompt para foto de cuerpo completo ---
def build_prompt_photo_body(extracted_features: dict, selfie_base64: str) -> str:
    # extracted_features = resultado del endpoint analyze_body_with_face
    return f"""
Generate an ultra-realistic full body fashion photograph.

IMPORTANT:
- Preserve exact facial identity from the reference image
- Do not alter facial features or body structure
- Respect body measurements extracted:
    - Height: {extracted_features.get('height_cm', 'unknown')} cm
    - Weight: {extracted_features.get('weight_kg', 'unknown')} kg
    - Body type: {extracted_features.get('body_type', 'average')}
    - Hair type: {extracted_features.get('hair_type', 'natural')}
- Outfit style: casual, modern, neutral colors
- Camera: full body shot, studio lighting
"""

# --- Función auxiliar para extraer rasgos del cuerpo completo ---
def analyze_body(body_base64: str, gender: str) -> dict:
    # Llamada al endpoint analyze_body_with_face
    url = "https://wink-backend-1jao.onrender.com/api/analyze-body-with-face/"
    response = requests.post(url, json={
        "image_base64": body_base64,
        "gender_hint": gender
    }, timeout=120)
    if response.status_code != 200:
        raise HTTPException(500, "Error analyzing body")
    return response.json()  # debe devolver altura, peso, body_type, hair_type, etc.

@router.post("/register_generate_base_images")
def register_generate_images(data: RegisterRequest):
    try:
        if data.mode == "selfie_manual":
            prompt = build_prompt_selfie_manual(data)
            reference_images = [data.selfie_base64]

        elif data.mode == "photo_body":
            if not data.body_image_base64:
                raise HTTPException(400, "Body image required")
            # Extraemos rasgos usando el endpoint de análisis
            features = analyze_body(data.body_image_base64, data.gender)
            prompt = build_prompt_photo_body(features, data.selfie_base64)
            reference_images = [data.selfie_base64, data.body_image_base64]

        else:
            raise HTTPException(400, "Invalid mode")

        # Generación de imagen
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            image=reference_images,
            n=2  # Genera 2 imágenes
        )

        return {
            "status": "ok",
            "images": [img.b64_json for img in result.data]
        }

    except Exception as e:
        raise HTTPException(500, f"Image generation failed: {str(e)}")
