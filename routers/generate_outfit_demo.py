# routers/generate_outfit_demo.py
import os
import base64
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx
import replicate

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_outfit_demo")

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_KEY")

if not HF_API_TOKEN:
    logger.warning("⚠️ HF_API_TOKEN not set — Hugging Face disabled")
if not REPLICATE_API_TOKEN:
    logger.warning("⚠️ REPLICATE_API_KEY not set — Replicate disabled")
else:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

def build_prompt(gender: str) -> str:
    return (
        f"A realistic full-body photo of a {gender} in everyday modern clothes, "
        "keeping the face, skin tone, and body proportions natural. "
        "Photorealistic, no cartoon/animation, high resolution, realistic textures and lighting."
    )

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "person")
    image_base64 = payload.get("image_base64")

    # --- Hugging Face Router API ---
    if HF_API_TOKEN:
        logger.info("Trying Hugging Face generation...")
        hf_url = "https://router.huggingface.co/models/CompVis/stable-diffusion-v1-5"
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        }
        data = {
            "inputs": build_prompt(gender),
            "options": {"wait_for_model": True, "use_gpu": True},
        }
        if image_base64:
            data["parameters"] = {
                "init_image": image_base64,
                "strength": 0.7
            }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(hf_url, headers=headers, json=data)
            response.raise_for_status()
            result_json = response.json()
            image_b64 = result_json.get("images", [None])[0]
            if image_b64:
                return JSONResponse({"status": "ok", "image": image_b64})
            else:
                logger.warning("HF returned no image")
        except Exception as e:
            logger.warning(f"HF failed: {e}")

    # --- Replicate fallback ---
    if REPLICATE_API_TOKEN:
        logger.info("Trying Replicate fallback...")
        try:
            output = replicate.run(
                "stability-ai/sdxl",
                input={
                    "prompt": build_prompt(gender),
                    "num_outputs": 1,
                    "image_dimensions": "1024x1024",
                },
            )
            if output and len(output) > 0:
                image_b64 = output[0].split(",")[-1] if "," in output[0] else output[0]
                return JSONResponse({"status": "ok", "image": image_b64})
        except Exception as e:
            logger.error(f"Replicate failed: {e}")

    return JSONResponse(
        {"status": "error", "message": "Image generation failed (HF + Replicate)"},
        status_code=500
    )
