# routers/generate_outfit_demo.py
import os
import base64
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import replicate
import httpx

router = APIRouter()
logger = logging.getLogger("generate_outfit_demo")
logging.basicConfig(level=logging.INFO)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_KEY")
if not REPLICATE_API_TOKEN:
    logger.warning("⚠️ REPLICATE_API_KEY missing — Replicate disabled")
else:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

def build_prompt(gender: str) -> str:
    return (
        f"A photorealistic full-body wardrobe outfit for a {gender} person, "
        "keeping facial features, skin tone, and body proportions true to the original reference. "
        "High-quality, realistic lighting and textures, no cartoon or game style."
    )

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "person")
    image_base64 = payload.get("image_base64")

    if not image_base64:
        return JSONResponse({"status": "error", "message": "Missing image_base64"}, status_code=400)

    # --- Replicate free model (e.g., ideogram‑ai/ideogram‑v3‑turbo) ---
    try:
        logger.info("➡️ Running Replicate free text‑to‑image model")

        # Usar versión de un modelo gratuito que esté en ./collections/text‑to‑image
        # Ejemplo: ideogram‑ai/ideogram‑v3‑turbo
        model_id = "ideogram‑ai/ideogram‑v3‑turbo"

        # Construir inputs — este modelo puede manejar texto y a veces imágenes,  
        # pero en modo free se usa mejor solo prompt textual y el dataset del modelo.
        output = replicate.run(
            model_id,
            input={
                "prompt": build_prompt(gender),
                "num_outputs": 1
            },
        )

        logger.info(f"Replicate output: {output}")

        if not output or len(output) == 0:
            raise Exception("No image returned by Replicate free model")

        # Descarga la imagen final desde URL
        img_url = output[0]
        async with httpx.AsyncClient() as client:
            img_resp = await client.get(img_url)
        if img_resp.status_code != 200:
            raise Exception("Failed to download image from Replicate")

        img_bytes = img_resp.content
        img_b64 = base64.b64encode(img_bytes).decode("utf‑8")
        return JSONResponse({"status": "ok", "image": img_b64})

    except Exception as e:
        logger.error(f"❌ Replicate free model failed: {e}")

    return JSONResponse({"status": "error", "message": "Image generation failed (free model)"}, status_code=500)
