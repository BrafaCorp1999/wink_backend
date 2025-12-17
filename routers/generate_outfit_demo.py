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
# Environment variables
# =========================
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

if not HF_API_TOKEN:
    logger.warning("⚠️ HF_API_TOKEN missing → Hugging Face disabled")

if REPLICATE_API_KEY:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY
else:
    logger.warning("⚠️ REPLICATE_API_KEY missing → Replicate disabled")

# =========================
# Hugging Face config
# =========================
HF_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "stabilityai/stable-diffusion-xl-base-1.0"
)

HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
}

# =========================
# Endpoint
# =========================
@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "person")

    prompt = (
        "Using the attached image of the person, generate a new realistic full-body outfit. "
        "Do NOT change the face, skin tone, body shape, or proportions. "
        "Preserve identity, facial features, and measurements exactly. "
        "Only change clothing. "
        "The outfit should fit naturally and look photorealistic. "
        f"The person is a {gender}."
    )

    images_base64: list[str] = []

    # ==========================================================
    # 1️⃣ TRY HUGGING FACE
    # ==========================================================
    if HF_API_TOKEN:
        try:
            logger.info("Trying Hugging Face generation...")

            response = requests.post(
                HF_MODEL_URL,
                headers=HF_HEADERS,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "guidance_scale": 7.5,
                        "num_inference_steps": 30,
                    },
                },
                timeout=90,
            )

            if response.status_code == 200:
                image_bytes = response.content
                images_base64.append(
                    "data:image/png;base64,"
                    + base64.b64encode(image_bytes).decode("utf-8")
                )
                logger.info("✅ Hugging Face generation success")
                return JSONResponse(
                    {"status": "ok", "demo_outfits": images_base64}
                )

            else:
                logger.warning(
                    f"HF failed {response.status_code}: {response.text}"
                )

        except Exception as e:
            logger.warning(f"HF exception: {e}")

    # ==========================================================
    # 2️⃣ TRY REPLICATE (FALLBACK)
    # ==========================================================
    if REPLICATE_API_KEY:
        try:
            logger.info("Trying Replicate fallback...")

            output = replicate.run(
                "stability-ai/stable-diffusion",
                input={
                    "prompt": prompt,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 768,
                },
            )

            if isinstance(output, list) and len(output) > 0:
                for url in output:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200:
                        images_base64.append(
                            "data:image/png;base64,"
                            + base64.b64encode(r.content).decode("utf-8")
                        )

                if images_base64:
                    logger.info("✅ Replicate generation success")
                    return JSONResponse(
                        {"status": "ok", "demo_outfits": images_base64}
                    )

            logger.warning("Replicate returned no images")

        except Exception as e:
            logger.error(f"Replicate failed: {e}")

    # ==========================================================
    # ❌ TOTAL FAILURE
    # ==========================================================
    logger.error("Image generation failed (HF + Replicate)")
    raise HTTPException(
        status_code=500,
        detail="Image generation failed (HF + Replicate)",
    )
