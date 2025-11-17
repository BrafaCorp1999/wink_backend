from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import base64, traceback, io, os
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

router = APIRouter()

# Cargar modelo globalmente para no recargar en cada request
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
)
PIPELINE = PIPELINE.to(DEVICE)

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    """
    Generate 3 demo outfits using provided user data (face + body measurements).
    Returns: JSON with 3 base64 images.
    """
    try:
        measurements = payload.get("measurements")
        face_base64 = payload.get("face_base64")
        gender = payload.get("gender", "unknown")
        base_photo_url = payload.get("base_photo_url")

        if not measurements or not face_base64:
            raise HTTPException(status_code=400, detail="Missing measurements or face image.")

        # --- Convert face_base64 to PIL image ---
        header, encoded = face_base64.split(",", 1)
        face_img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

        # --- Prompts para 3 outfits aleatorios ---
        outfit_styles = ["casual", "elegant", "sporty"]
        demo_images = []

        for style in outfit_styles:
            prompt = f"Full body portrait of a {gender} person, wearing {style} outfit, maintaining exact body measurements and face from reference image."
            
            # Generar imagen
            with torch.autocast(DEVICE):
                image = PIPELINE(prompt=prompt, init_image=face_img, strength=0.7, guidance_scale=7.5).images[0]

            # Convertir a base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            b64_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
            demo_images.append(b64_str)

        return JSONResponse({
            "status": "ok",
            "demo_outfits": demo_images,
            "generation_mode": "stable_diffusion",
            "fallback_used": False
        })

    except Exception as e:
        print("❌ Error in /generate_outfit_demo:", traceback.format_exc())

        # Fallback: devolver 3 imágenes genéricas vacías si falla
        empty_b64 = "data:image/png;base64," + base64.b64encode(np.zeros((512,512,3), dtype=np.uint8).tobytes()).decode()
        return JSONResponse({
            "status": "ok",
            "demo_outfits": [empty_b64]*3,
            "generation_mode": "fallback",
            "fallback_used": True,
            "error": str(e)
        })
