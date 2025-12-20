from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
import json
import os

router = APIRouter()

# =========================
# PROMPT – SELFIE + MANUAL
# =========================

SELFIE_PROMPT = """
Generate a photorealistic full-body image of a real person.

IDENTITY:
- Face must match the provided selfie.
- Preserve facial proportions, skin tone, hairstyle.

BODY SHAPE:
- Height: {height_cm} cm
- Weight: {weight_kg} kg
- Body type: {body_type}

CLOTHING:
- Create a modern {style} outfit.
- Realistic fit according to body type.
- Include shoes and optional subtle accessories.
- Natural fabric textures and folds.

POSE & COMPOSITION:
- Full-body, head to toe.
- Neutral standing pose.
- Eye-level camera.

LIGHTING & QUALITY:
- Clean neutral background.
- Soft natural lighting.
- DSLR-quality photorealism.
- No CGI, no illustration.

OUTPUT:
- Generate 2 distinct outfit variations.
"""

# =========================
# ENDPOINT
# =========================

@router.post("/generate-outfits/selfie")
async def generate_outfits_from_selfie(
    gender: str = Form(...),
    body_traits: str = Form(...),  # JSON manual desde Flutter
    style: str = Form("casual"),
    image_file: UploadFile = File(...)
):
    # -------------------------
    # Traits manuales
    # -------------------------
    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    height_cm = traits.get("height_cm", 170)
    weight_kg = traits.get("weight_kg", 65)
    body_type = traits.get("body_type", "average")

    # -------------------------
    # Leer selfie (no se envía al modelo)
    # -------------------------
    selfie_bytes = await image_file.read()
    if not selfie_bytes:
        raise HTTPException(status_code=400, detail="Empty selfie image")

    # -------------------------
    # Prompt final
    # -------------------------
    final_prompt = SELFIE_PROMPT.format(
        height_cm=height_cm,
        weight_kg=weight_kg,
        body_type=body_type,
        style=style
    )

    # -------------------------
    # OpenAI
    # -------------------------
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.images.generate(
            model="gpt-image-1.5",
            prompt=final_prompt,
            n=2,
            size="1024x1024"
        )

        images_b64 = [img.b64_json for img in response.data]

        return {
            "status": "ok",
            "images": images_b64,
            "traits_used": traits,
            "mode": "selfie_manual"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )
