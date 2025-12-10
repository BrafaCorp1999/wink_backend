# routers/generate_outfit_demo.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import base64, traceback, io
from PIL import Image
import numpy as np

# Servicios
from utils.gemini_service import gemini_generate_image
from utils.openai_service import openai_generate_image
from utils.sd_service import sd_generate_image

router = APIRouter()

# --- Endpoint para generar outfits ---
@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    """
    Generate 3 demo outfits using provided user data (face + body measurements).
    Returns JSON with 3 base64 images.
    """
    try:
        measurements = payload.get("measurements")
        face_base64 = payload.get("face_base64")
        gender = payload.get("gender", "unknown")
        base_photo_url = payload.get("base_photo_url")

        if not measurements or not face_base64:
            raise HTTPException(status_code=400, detail="Missing measurements or face image.")

        # Convertir face_base64 a PIL image
        header, encoded = face_base64.split(",", 1)
        face_img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

        # --- Prompts para 3 estilos de outfit ---
        outfit_styles = ["casual", "elegant", "sporty"]
        demo_images = []
        generation_mode = "none"
        fallback_used = False

        for style in outfit_styles:
            prompt = f"Full body portrait of a {gender} person, wearing {style} outfit, keeping exact body measurements and face from reference image, realistic, no distortion."

            # --- Intentar Gemini ---
            try:
                image = gemini_generate_image(prompt, face_img)
                generation_mode = "gemini"
            except Exception as e1:
                print("⚠️ Gemini failed:", e1)
                # --- Intentar OpenAI ---
                try:
                    image = openai_generate_image(prompt, face_img)
                    generation_mode = "openai"
                except Exception as e2:
                    print("⚠️ OpenAI failed:", e2)
                    # --- Intentar Stable Diffusion ---
                    try:
                        image = sd_generate_image(prompt, face_img)
                        generation_mode = "stable_diffusion"
                    except Exception as e3:
                        print("❌ All generation methods failed:", e3)
                        # --- Fallback: imagen vacía ---
                        image = Image.fromarray(np.zeros((512,512,3), dtype=np.uint8))
                        fallback_used = True

            # Convertir a base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            b64_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
            demo_images.append(b64_str)

        return JSONResponse({
            "status": "ok",
            "demo_outfits": demo_images,
            "generation_mode": generation_mode,
            "fallback_used": fallback_used
        })

    except Exception as e:
        print("❌ Error in /generate_outfit_demo:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
