# routers/generate_outfit_demo.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import base64, traceback, io, os
from PIL import Image
import numpy as np
import torch

# Import services (Gemini & OpenAI)
from services.gemini_service import generate_image_gemini
from services.openai_service import generate_image_openai

router = APIRouter()

# --- Device setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    """
    Generate 2 demo outfits using provided user data (face + body measurements).
    Returns: JSON with 2 base64 images.
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

        demo_images = []
        outfit_styles = ["random_style_1", "random_style_2"]  # 2 imágenes

        for style in outfit_styles:
            # Prompt en inglés, mantiene rostro y medidas
            prompt = (
                f"Full body portrait of a {gender} person, wearing {style} outfit, "
                f"maintaining exact body measurements and face from reference image. "
                f"Do not deform face or body."
            )

            image = None
            # --- Fallback Gemini → OpenAI → SD ---
            try:
                image = generate_image_gemini(face_img, prompt)
            except Exception:
                try:
                    image = generate_image_openai(face_img, prompt)
                except Exception:
                    # Último fallback: imagen vacía
                    image = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))

            # Convertir a base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            b64_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
            demo_images.append(b64_str)

        return JSONResponse({
            "status": "ok",
            "demo_outfits": demo_images,
            "generation_mode": "fallback_used" if any(demo_images[i] == None for i in range(2)) else "AI_generated",
        })

    except Exception as e:
        print("❌ Error in /generate_outfit_demo:", traceback.format_exc())
        # Devuelve 2 imágenes vacías como fallback
        empty_b64 = "data:image/png;base64," + base64.b64encode(np.zeros((512,512,3), dtype=np.uint8).tobytes()).decode()
        return JSONResponse({
            "status": "ok",
            "demo_outfits": [empty_b64]*2,
            "generation_mode": "fallback",
            "error": str(e)
        })
