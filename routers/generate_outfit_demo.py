# generate_outfit_demo.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import os
import base64
import logging
import requests

import replicate
import openai

router = APIRouter()

REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Endpoint para generar outfits
@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "unknown")
    images_b64 = []

    # -------------------------
    # 1️⃣ Intentar generar con OpenAI (DALL·E)
    # -------------------------
    if OPENAI_API_KEY:
        try:
            logging.info("➡️ Intentando generar imagen con OpenAI DALL·E...")
            response = openai.images.generate(
                model="gpt-image-1",
                prompt=f"A full-body outfit for a {gender} person, realistic, natural pose.",
                size="512x768",
                n=1
            )
            for data in response.data:
                img_b64 = data.b64_json
                images_b64.append("data:image/png;base64," + img_b64)
            logging.info(f"✅ OpenAI generó {len(images_b64)} imágenes")
        except Exception as e:
            logging.warning(f"⚠️ Error OpenAI: {e}")

    # -------------------------
    # 2️⃣ Intentar generar con Replicate
    # -------------------------
    if REPLICATE_API_KEY:
        try:
            logging.info("➡️ Intentando generar imagen con Replicate...")
            client = replicate.Client(api_token=REPLICATE_API_KEY)
            model = client.models.get("stability-ai/stable-diffusion")
            output = model.predict(
                prompt=f"A full-body outfit for a {gender} person, realistic, natural pose.",
                width=512, height=768
            )
            for url in output:
                resp = requests.get(url)
                if resp.status_code == 200:
                    images_b64.append("data:image/png;base64," + base64.b64encode(resp.content).decode("utf-8"))
                else:
                    logging.warning(f"⚠️ Failed to download Replicate image from {url}")
            logging.info(f"✅ Replicate generó {len(output)} imágenes")
        except Exception as e:
            logging.warning(f"⚠️ Error Replicate: {e}")

    # -------------------------
    # 3️⃣ Fallback local
    # -------------------------
    if not images_b64:
        logging.warning("⚠️ Ninguna IA generó imágenes, usando fallback local...")
        try:
            fallback_path = "assets/demo_outfit_fallback.png"  # Debe estar en Flutter assets
            with open(fallback_path, "rb") as f:
                images_b64.append("data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8"))
            logging.info("✅ Fallback local cargado correctamente")
        except Exception as e:
            logging.error(f"❌ Falló fallback local: {e}")
            return JSONResponse({
                "status": "error",
                "message": "No se pudo generar ninguna imagen. Revisa los logs."
            })

    return JSONResponse({
        "status": "ok",
        "demo_outfits": images_b64
    })
