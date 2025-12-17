import os
import base64
import logging
import requests

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import replicate

router = APIRouter()
logger = logging.getLogger("generate_outfit_demo")
logging.basicConfig(level=logging.INFO)

HF_TOKEN = os.getenv("HF_API_TOKEN")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

if REPLICATE_API_KEY:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY


# =========================
# Request schema
# =========================
class OutfitDemoRequest(BaseModel):
    image_base64: str
    gender: str


# =========================
# HF image-to-image
# =========================
def generate_with_hf(image_base64: str) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF token missing")

    url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-refiner-1.0"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    prompt = (
        "Generate a realistic fashion outfit on the same person. "
        "Do not change face, skin tone, body proportions or pose. "
        "Keep the same person identity. "
        "Outfit should look natural and well fitted."
    )

    payload = {
        "inputs": {
            "image": image_base64,
            "prompt": prompt,
        }
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)

    if r.status_code != 200:
        raise RuntimeError(f"HF error {r.status_code}: {r.text}")

    return base64.b64encode(r.content).decode("utf-8")


# =========================
# Replicate fallback
# =========================
def generate_with_replicate(image_base64: str) -> str:
    prompt = (
        "Generate a new realistic outfit on the same person. "
        "Do not modify face, skin, body shape or proportions. "
        "Only change clothing. High realism, fashion photography."
    )

    output = replicate.run(
        "stability-ai/sdxl",
        input={
            "prompt": prompt,
            "image": image_base64,
            "strength": 0.35,
        },
    )

    if not output:
        raise RuntimeError("Replicate returned empty output")

    image_url = output[0]
    img_bytes = requests.get(image_url).content
    return base64.b64encode(img_bytes).decode("utf-8")


# =========================
# API endpoint
# =========================
@router.post("/generate_outfit_demo")
def generate_outfit_demo(data: OutfitDemoRequest):
    try:
        logger.info("Trying Hugging Face generation...")
        img = generate_with_hf(data.image_base64)
        return {"demo_outfits": [img]}

    except Exception as hf_error:
        logger.warning(f"HF failed: {hf_error}")

    try:
        logger.info("Trying Replicate fallback...")
        img = generate_with_replicate(data.image_base64)
        return {"demo_outfits": [img]}

    except Exception as rep_error:
        logger.error(f"Replicate failed: {rep_error}")

    raise HTTPException(
        status_code=500,
        detail="Image generation failed (HF + Replicate)",
    )
