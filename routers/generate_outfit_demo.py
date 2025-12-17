# routers/generate_outfit_demo.py
import os
import base64
import logging
import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import replicate

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_outfit_demo")

# =========================
# Cargar llaves
# =========================
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_KEY")

if not HF_API_TOKEN:
    logger.warning("⚠️ HF_API_TOKEN not set — Hugging Face disabled")
if not REPLICATE_API_TOKEN:
    logger.warning("⚠️ REPLICATE_API_KEY not set — Replicate disabled")
else:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# =========================
# Prompt robusto
# =========================
def build_prompt(gender: str) -> str:
    return (
        f"A highly realistic, full-body photo of a {gender} in everyday modern clothes, "
        "preserving the face, skin tone, and body proportions accurately. "
        "The outfit is stylish and natural, no animation or cartoon style, "
        "photorealistic lighting, realistic textures, high-resolution."
    )

# =========================
# Endpoint para generar outfit
# =========================
@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "person")
    image_base64 = payload.get("image_base64")  # opcional, puede usarse para referencia

    # --- Intentar Hugging Face Router API ---
    if HF_API_TOKEN:
        logger.info("Trying Hugging Face generation...")
        hf_url = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-5"
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
            # HF Router devuelve image en base64
            image_b64 = result_json.get("images", [None])[0]
            if not image_b64:
                raise ValueError("No image returned from HF")
            return JSONResponse({"status": "ok", "image": image_b64})
        except Exception as e:
            logger.warning(f"HF failed: {e}")

    # --- Intentar Replicate como fallback ---
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
            if not output or len(output) == 0:
                raise ValueError("No image returned from Replicate")
            image_b64 = output[0].split(",")[-1] if "," in output[0] else output[0]
            return JSONResponse({"status": "ok", "image": image_b64})
        except Exception as e:
            logger.error(f"Replicate failed: {e}")

    # --- Error si ambos fallan ---
    return JSONResponse(
        {"status": "error", "message": "Image generation failed (HF + Replicate)"},
        status_code=500
    )
