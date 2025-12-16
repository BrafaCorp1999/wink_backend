# routers/generate_outfit_demo.py
import os
import base64
import logging
import requests

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import replicate
from google import genai      # SDK de Gemini v1
from google.genai import types

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_outfit_demo")

# Cargar llaves
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not set — Gemini disabled")
if not REPLICATE_API_KEY:
    logger.warning("⚠️ REPLICATE_API_KEY not set — Replicate disabled")

# Inicializar cliente Gemini (Google genai)
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "unknown")
    prompt = f"A full-body outfit for a {gender} person, realistic, natural pose."

    images_b64 = []

    # --- 1️⃣ Gemini API (si disponible) ---
    if gemini_client:
        try:
            logger.info("➡️ Generating with Gemini")
            response = gemini_client.models.generate_content(
                model="imagen-3.0-generate-002",  # modelo actual disponible
                input=types.GenerateContentRequest(
                    prompt=prompt,
                    response_modalities=["Image"],
                ),
            )
            # iterar imágenes generadas
            for img_obj in response.images:
                # cada imagen tiene .data (bytes)
                img_bytes = img_obj.data
                images_b64.append(
                    "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")
                )
            logger.info("✅ Gemini generated images")
        except Exception as e:
            logger.warning(f"⚠️ Gemini failed: {e}")

    # --- 2️⃣ Replicate ---
    if REPLICATE_API_KEY:
        try:
            logger.info("➡️ Generating with Replicate")
            output_urls = replicate.run(
                "stability-ai/stable-diffusion-3",
                input={"prompt": prompt, "width": 512, "height": 768}
            )
            for url in output_urls:
                resp = requests.get(url)
                if resp.status_code == 200:
                    images_b64.append(
                        "data:image/png;base64," + base64.b64encode(resp.content).decode("utf-8")
                    )
            logger.info("✅ Replicate generated images")
        except Exception as e:
            logger.warning(f"⚠️ Replicate failed: {e}")

    # --- 3️⃣ Si ninguna generó ---
    if not images_b64:
        logger.error("❌ Neither Gemini nor Replicate produced images")
        return JSONResponse({
            "status": "error",
            "message": "No images generated. Check logs for details"
        }, status_code=200)

    return JSONResponse({"status": "ok", "demo_outfits": images_b64})
