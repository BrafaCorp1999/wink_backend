# routers/generate_outfit_demo.py
import os
import base64
import logging
import requests

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import replicate
import google.generativeai as genai  # SDK de Gemini v1
from google.generativeai import types

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_outfit_demo")

# =========================
# Cargar llaves de entorno
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not set — Gemini disabled")
if not REPLICATE_API_KEY:
    logger.warning("⚠️ REPLICATE_API_KEY not set — Replicate disabled")

# =========================
# Inicializar cliente Gemini
# =========================
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("🔹 Gemini client initialized")

# =========================
# Endpoint principal
# =========================
@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "unknown")
    outfits_b64 = []

    # -------------------------
    # 1️⃣ Generar con Gemini
    # -------------------------
    if gemini_client:
        try:
            logger.info("➡️ Generating image with Gemini")
            response = gemini_client.generate_image(
                model="image-alpha-001",
                prompt=f"A full-body outfit for a {gender} person, realistic, natural pose.",
                size="512x768"
            )
            image_bytes = requests.get(response.url).content
            outfits_b64.append("data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8"))
            logger.info("✅ Gemini image generated")
        except Exception as e:
            logger.warning(f"⚠️ Error Gemini: {e}")

    # -------------------------
    # 2️⃣ Generar con Replicate
    # -------------------------
    if REPLICATE_API_KEY:
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY
        try:
            logger.info("➡️ Generating image with Replicate")
            output_urls = replicate.run(
                "stability-ai/stable-diffusion:latest",
                input={
                    "prompt": f"A full-body outfit for a {gender} person, realistic, natural pose.",
                    "width": 512,
                    "height": 768
                }
            )
            if isinstance(output_urls, list):
                for url in output_urls:
                    r = requests.get(url)
                    if r.status_code == 200:
                        outfits_b64.append("data:image/png;base64," + base64.b64encode(r.content).decode("utf-8"))
                logger.info("✅ Replicate images generated")
        except Exception as e:
            logger.warning(f"⚠️ Error Replicate: {e}")

    # -------------------------
    # 3️⃣ Fallback local
    # -------------------------
    if not outfits_b64:
        fallback_path = "./demo_outfit_fallback.png"
        try:
            with open(fallback_path, "rb") as f:
                outfits_b64.append("data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8"))
            logger.warning("⚠️ Using local fallback image")
        except Exception as e:
            logger.error(f"❌ Falló fallback local: {e}")
            return JSONResponse({
                "status": "error",
                "message": "No se pudo generar ninguna imagen. Revisa los logs."
            })

    return JSONResponse({"status": "ok", "demo_outfits": outfits_b64})
