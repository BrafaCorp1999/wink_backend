# routers/generate_outfit_demo.py
import os
import io
import base64
import logging
import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

import openai
import replicate

router = APIRouter()

# Inicializar llaves de entorno
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar APIs
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY
    logger.info("🔹 OpenAI API Key detected")
else:
    logger.warning("⚠️ OPENAI_API_KEY missing → OpenAI disabled")

if REPLICATE_API_KEY:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY
    logger.info("🔹 Replicate API Key detected")
else:
    logger.warning("⚠️ REPLICATE_API_KEY missing → Replicate disabled")

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "unknown")
    outfits_b64 = []

    # -------------------------
    # 1️⃣ Intentar OpenAI
    # -------------------------
    if OPENAI_KEY:
        try:
            logger.info("➡️ Generating image with OpenAI")
            response = openai.images.generate(
                model="gpt-image-1",
                prompt=f"A full-body outfit for a {gender} person, realistic, natural pose.",
                size="512x768",
                n=1
            )
            if "data" in response and len(response["data"]) > 0:
                url = response["data"][0]["url"]
                r = requests.get(url)
                if r.status_code == 200:
                    outfits_b64.append("data:image/png;base64," + base64.b64encode(r.content).decode("utf-8"))
                    logger.info("✅ OpenAI image generated")
        except Exception as e:
            logger.warning(f"⚠️ Error OpenAI: {e}")

    # -------------------------
    # 2️⃣ Intentar Replicate
    # -------------------------
    if REPLICATE_API_KEY:
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
            return JSONResponse({"status": "error", "message": "No se pudo generar ninguna imagen. Revisa los logs."})

    return JSONResponse({"status": "ok", "demo_outfits": outfits_b64})
