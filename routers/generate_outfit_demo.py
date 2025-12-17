# routers/generate_outfit_demo.py
import os
import base64
import logging
import json
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import httpx

import replicate

logger = logging.getLogger("generate_outfit_demo")
logging.basicConfig(level=logging.INFO)

router = APIRouter()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Hugging Face token
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
if REPLICATE_API_KEY:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(request: Request):
    """
    Genera outfit IA usando Hugging Face (principal) y Replicate (fallback).
    Recibe JSON: { "gender": "female/male", "image_base64": "<base64>" }
    Retorna JSON con imagen base64.
    """
    data = await request.json()
    gender = data.get("gender", "female")
    image_base64 = data.get("image_base64")

    if not image_base64:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Image missing"})

    prompt = (
        "Generate a realistic photo of the person wearing a new outfit. "
        "Preserve the original face and skin, adjust clothing naturally to body shape and size. "
        "Natural lighting, realistic textures, high-resolution. "
        "Avoid cartoon, anime, or game-style. "
        f"Gender: {gender}."
    )

    # =======================
    # 1️⃣ Intentar Hugging Face
    # =======================
    try:
        logger.info("Trying Hugging Face generation...")
        hf_headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
        hf_payload = {
            "inputs": prompt,
            "parameters": {
                "image": image_base64,
                "num_images": 1,
                "guidance_scale": 7.5,
                "size": "1024x1024"
            }
        }

        async with httpx.AsyncClient(timeout=120) as client:
            hf_response = await client.post("https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-5", 
                                            headers=hf_headers, json=hf_payload)
            if hf_response.status_code == 200:
                hf_json = hf_response.json()
                # hf_json puede variar según modelo, aquí asumimos que devuelve base64 en hf_json["images"][0]
                if "images" in hf_json and len(hf_json["images"]) > 0:
                    img_base64 = hf_json["images"][0]
                    logger.info("✅ Hugging Face generation success")
                    return JSONResponse(content={"status": "ok", "image": img_base64})
            logger.warning(f"HF failed: {hf_response.status_code}, {hf_response.text}")
    except Exception as e:
        logger.warning(f"HF failed: {e}")

    # =======================
    # 2️⃣ Intentar Replicate como fallback
    # =======================
    try:
        logger.info("Trying Replicate fallback...")
        model = replicate.models.get("stability-ai/sdxl")
        prediction = model.predict(
            prompt=prompt,
            init_image=image_base64,
            num_outputs=1,
            width=1024,
            height=1024
        )
        if prediction and len(prediction) > 0:
            logger.info("✅ Replicate generation success")
            return JSONResponse(content={"status": "ok", "image": prediction[0]})
    except Exception as e:
        logger.error(f"Replicate failed: {e}")

    return JSONResponse(status_code=500, content={"status": "error", "message": "Image generation failed (HF + Replicate)"})
