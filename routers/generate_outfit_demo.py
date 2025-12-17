import os
import base64
import logging
import requests

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

import replicate

router = APIRouter()

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_outfit_demo")

# =========================
# Env keys
# =========================
HF_API_KEY = os.getenv("HF_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
}

# =========================
# MODELS
# =========================
HF_MODEL_URL = (
    "https://api-inference.huggingface.co/models/"
    "stabilityai/stable-diffusion-xl-base-1.0"
)

REPLICATE_MODEL = "stability-ai/sdxl"

# =========================
# ENDPOINT
# =========================
@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    """
    payload:
    {
        "image_base64": "data:image/png;base64,..."
    }
    """

    image_base64 = payload.get("image_base64")

    if not image_base64:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "image_base64 missing"},
        )

    # Limpiar base64
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    prompt = (
        "Using the attached photo of a real person, generate a new full-body outfit.\n"
        "Do NOT change face, skin tone, body shape, or proportions.\n"
        "Do NOT deform the face or body.\n"
        "Keep the same person identity.\n"
        "Only change the clothes.\n"
        "Clothes must fit naturally to the body.\n"
        "Realistic lighting, natural pose, high quality fashion photography."
    )

    # =========================
    # 1️⃣ Hugging Face (MAIN)
    # =========================
    if HF_API_KEY:
        try:
            logger.info("➡️ Generating image with Hugging Face (SDXL)")

            response = requests.post(
                HF_MODEL_URL,
                headers=HF_HEADERS,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "num_inference_steps": 30,
                        "guidance_scale": 7.5,
                    },
                },
                timeout=120,
            )

            if response.status_code == 200:
                img_bytes = response.content
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                logger.info("✅ Hugging Face image generated")

                return JSONResponse(
                    {
                        "status": "ok",
                        "source": "huggingface",
                        "image": f"data:image/png;base64,{img_b64}",
                    }
                )

            logger.warning(
                f"⚠️ HF error {response.status_code}: {response.text}"
            )

        except Exception as e:
            logger.warning(f"⚠️ HF exception: {e}")

    # =========================
    # 2️⃣ Replicate (FALLBACK)
    # =========================
    if REPLICATE_API_KEY:
        try:
            logger.info("➡️ Generating image with Replicate (SDXL)")

            output = replicate.run(
                REPLICATE_MODEL,
                input={
                    "prompt": prompt,
                    "image": image_base64,
                    "width": 768,
                    "height": 1024,
                },
            )

            if isinstance(output, list) and len(output) > 0:
                img_url = output[0]
                img_resp = requests.get(img_url)

                if img_resp.status_code == 200:
                    img_b64 = base64.b64encode(img_resp.content).decode("utf-8")

                    logger.info("✅ Replicate image generated")

                    return JSONResponse(
                        {
                            "status": "ok",
                            "source": "replicate",
                            "image": f"data:image/png;base64,{img_b64}",
                        }
                    )

        except Exception as e:
            logger.warning(f"⚠️ Replicate exception: {e}")

    # =========================
    # ❌ FAIL
    # =========================
    logger.error("❌ No AI could generate image")

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Image generation failed (HF + Replicate)",
        },
    )
