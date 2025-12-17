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
    logger.warning("⚠️ REPLICATE_API_KEY missing → Replicate disabled")
else:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

def build_prompt(gender: str) -> str:
    return (
        f"A photorealistic full-body image of a {gender} person wearing a modern realistic outfit. "
        "Keep the face, skin tone, and body proportions as close as possible to the original reference. "
        "Natural lighting, realistic textures, no cartoon, no game style."
    )

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "person")
    image_base64 = payload.get("image_base64")

    # Validación
    if not image_base64:
        return JSONResponse({"status": "error", "message": "Missing image_base64"}, status_code=400)

    try:
        logger.info("➡️ Running Replicate image generation...")

        # Usa versión específica de SDXL que sí existe
        model_id = (
            "stability-ai/sdxl:"
            "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
        )

        output = replicate.run(
            model_id,
            input={
                "prompt": build_prompt(gender),
                # Puedes incluir init_image para refinar con la cara
                "image": image_base64,
                "num_outputs": 1,
                "image_dimensions": "1024x1024"
            },
        )

        logger.info(f"Replicate output: {output}")

        if not output or len(output) == 0:
            raise Exception("No image returned by Replicate")

        # El output contiene URLs
        img_url = output[0]
        async with httpx.AsyncClient() as client:
            img_resp = await client.get(img_url)
        if img_resp.status_code != 200:
            raise Exception("Failed to download image")

        img_bytes = img_resp.content
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        return JSONResponse({"status": "ok", "image": img_b64})

    except Exception as e:
        logger.error(f"✖ Replicate failed: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
