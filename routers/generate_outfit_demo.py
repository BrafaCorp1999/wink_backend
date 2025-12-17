# routers/generate_outfit_demo.py
import os   # <--- Agregar esta línea
import base64
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import httpx

router = APIRouter()
logger = logging.getLogger("generate_outfit_demo")
logging.basicConfig(level=logging.INFO)

# --- Stable Horde API config (no key required para uso básico) ---
HORDE_API_KEY = "0000000000"  # clave pública para demo (modo anónimo)

# --- DeepAI fallback key (requiere registro en https://deepai.org/) ---
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY")

def build_prompt(gender: str) -> str:
    return (
        f"A photorealistic full body outfit for a {gender} person. "
        "Keep the face, body proportions and skin tone natural. "
        "High resolution, realistic clothing."
    )

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "person")
    prompt = build_prompt(gender)

    ### 1️⃣ Stable Horde (Principal)
    try:
        logger.info("➡️ Trying Stable Horde free generation")
        horde_data = {
            "prompt": prompt,
            "params": {
                "sampler_name": "k_euler", 
                "steps": 25,
                "cfg_scale": 7.5,
                "width": 512,
                "height": 768
            },
            "runners": ["stable_diffusion"]
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://stablehorde.net/api/v2/generate/async",
                headers={"apikey": HORDE_API_KEY},
                json=horde_data
            )
        if response.status_code == 200:
            result = response.json()
            # base64 image en “img”
            image_b64 = result.get("generations", [{}])[0].get("img")
            if image_b64:
                logger.info("✅ Stable Horde image generated")
                return JSONResponse({"status": "ok", "image": image_b64})
    except Exception as e:
        logger.warning(f"⚠️ Stable Horde failed: {e}")

    ### 2️⃣ Fallback → DeepAI (requiere API key gratuita)
    if DEEPAI_API_KEY:
        try:
            logger.info("➡️ Trying DeepAI fallback")
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    "https://api.deepai.org/api/text2img",
                    headers={"api-key": DEEPAI_API_KEY},
                    data={"text": prompt}
                )
            if response.status_code == 200:
                result = response.json()
                img_url = result.get("output_url")
                if img_url:
                    async with httpx.AsyncClient() as client:
                        img_resp = await client.get(img_url)
                    img_b64 = base64.b64encode(img_resp.content).decode("utf-8")
                    logger.info("✅ DeepAI fallback success")
                    return JSONResponse({"status": "ok", "image": img_b64})
        except Exception as e:
            logger.warning(f"⚠️ DeepAI fallback failed: {e}")

    # --- Si todo falla ---
    return JSONResponse(
        {"status": "error", "message": "No se pudo generar imagen con servicios gratuitos."},
        status_code=500
    )
