# routers/generate_outfit_demo.py
import os
import logging
import base64
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

logger = logging.getLogger(__name__)
router = APIRouter()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Hugging Face
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")  # Replicate

if REPLICATE_API_KEY:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY

class OutfitRequest(BaseModel):
    gender: str
    image_base64: str

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: OutfitRequest):
    prompt = (
        "Generate a realistic full-body outfit on this person without changing their face, "
        "skin tone, or body proportions. "
        "Use the reference image provided. "
        "The outfit should look natural and realistic, not animated or cartoonish. "
        "One high-quality image only."
    )

    image_bytes = base64.b64decode(payload.image_base64.split(",")[-1])

    # --- 1️⃣ Try Hugging Face first ---
    if HF_API_TOKEN:
        logger.info("Trying Hugging Face generation...")
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        hf_url = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-5"

        data = {
            "inputs": prompt,
            "parameters": {"init_image": payload.image_base64, "num_images": 1, "guidance_scale": 7.5}
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(hf_url, headers=headers, json=data)
            if resp.status_code == 200:
                output = resp.json()
                if isinstance(output, list) and len(output) > 0 and "image_base64" in output[0]:
                    return {"status": "ok", "image": output[0]["image_base64"]}
                else:
                    raise Exception("HF did not return image")
            else:
                logger.warning(f"HF failed: {resp.status_code}, {resp.text}")
        except Exception as e:
            logger.warning(f"HF failed: {e}")

    # --- 2️⃣ Fallback Replicate ---
    if REPLICATE_API_KEY:
        logger.info("Trying Replicate fallback...")
        replicate_url = "https://api.replicate.com/v1/predictions"
        model_version = "stability-ai/sdxl:latest"  # Ajusta según modelo que tengas disponible

        data = {
            "version": model_version,
            "input": {
                "prompt": prompt,
                "image": payload.image_base64,
                "num_outputs": 1,
            },
        }

        headers = {"Authorization": f"Token {REPLICATE_API_KEY}"}
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(replicate_url, headers=headers, json=data)
            if resp.status_code in [200, 201]:
                output = resp.json()
                # Replicate devuelve url de imagen
                img_url = output.get("output", [None])[0]
                if img_url:
                    # Descargar imagen y convertir a base64
                    async with httpx.AsyncClient() as client:
                        img_resp = await client.get(img_url)
                    return {"status": "ok", "image": f"data:image/png;base64,{base64.b64encode(img_resp.content).decode()}"}
            else:
                logger.warning(f"Replicate failed: {resp.status_code}, {resp.text}")
        except Exception as e:
            logger.warning(f"Replicate failed: {e}")

    raise HTTPException(status_code=500, detail="Image generation failed (HF + Replicate)")
