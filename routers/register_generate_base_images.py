from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import base64
import json

router = APIRouter()

# =========================
# Función para construir prompt largo
# =========================
def build_long_prompt(traits: dict, gender: str, style: str, prompt_type: str) -> str:
    contextura = traits.get("body_type", "average")
    altura = traits.get("height_cm", 170)
    peso = traits.get("weight_kg", 65)
    hair_type = traits.get("hair_type", "medium length, straight")

    location = "a sunny outdoor park" if style == "casual" else "a modern indoor lounge"
    outfit_style = "casual streetwear" if style == "casual" else "formal chic outfit"
    pose = "standing casually with one hand in pocket, looking to the side" if style == "casual" else "sitting at a designer table with a calm confident expression"

    # Elegir prompt según tipo de registro
    if prompt_type == "photo_body":
        prompt_header = "Generate a full body image from reference photo."
    else:  # selfie_manual
        prompt_header = "Generate a full body image using selfie + provided measurements."

    prompt = f"""
{prompt_header}
Use 100% same face and hairstyle from the reference image.
Highly realistic 4K image of a {contextura} {gender} with height {altura} cm and weight {peso} kg.
Hair: {hair_type}. Pose: {pose}.
Outfit: {outfit_style}, with shoes matching the outfit.
Location: {location}, realistic lighting, shadows, depth of field.
Ensure full body visible including hair and shoes.
Negative prompt: no face change, no hairstyle modification, no body distortion, no watermark, no text overlay.
Render in ultra-realistic digital photography style.
"""
    return prompt

# =========================
# Endpoint: generar imágenes base
# =========================
@router.post("/register_generate_base_images")
async def register_generate_base_images(
    mode: str = Form(...),  # photo_body o selfie_manual
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: str = Form("casual"),
    image_file: UploadFile = File(...)
):
    try:
        image_bytes = await image_file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        traits = json.loads(body_traits)
        if not traits:
            # fallback demo
            traits = {
                "height_cm": 170,
                "weight_kg": 65,
                "body_type": "average",
                "hair_type": "medium length, straight"
            }

        # Construimos prompts largos para casual y formal
        prompts = []
        for st in ["casual", "formal"]:
            prompts.append(build_long_prompt(traits, gender, st, mode))

        # Demo: devolvemos la misma imagen 2 veces codificada
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return JSONResponse({
            "status": "ok",
            "images": [image_base64, image_base64],
            "prompts": prompts,
            "traits": traits
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
