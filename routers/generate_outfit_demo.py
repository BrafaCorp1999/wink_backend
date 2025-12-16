# routers/generate_outfit_demo.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import base64
import traceback
import numpy as np
import asyncio

# Servicios AI
from utils.openai_service import openai_generate_image
from utils.replicate_service import replicate_generate_image

router = APIRouter()

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    try:
        measurements = payload.get("measurements")
        face_base64 = payload.get("face_base64")
        gender = payload.get("gender", "unknown")

        if not measurements or not face_base64:
            raise HTTPException(status_code=400, detail="Missing measurements or face image.")

        demo_images = []

        # Generar solo 1 outfit para pruebas
        prompt = (
            f"Full body portrait of a {gender} person wearing a simple casual outfit. "
            "Maintain realistic body proportions and do not deform the face. "
            "Clothes must fit naturally."
        )

        image_b64 = None

        # --- 1️⃣ Intentar OpenAI DALL·E ---
        try:
            print(f"➡️ Generating outfit with OpenAI")
            image_b64 = await openai_generate_image(prompt)
            if not image_b64:
                raise ValueError("OpenAI returned empty image")
            print(f"✅ OpenAI generated image")

        # --- 2️⃣ Si falla OpenAI, intentar Replicate ---
        except Exception as e_openai:
            print(f"⚠️ OpenAI failed: {e_openai}")
            try:
                print(f"➡️ Trying Replicate")
                image_b64 = await replicate_generate_image(prompt)
                if not image_b64:
                    raise ValueError("Replicate returned empty image")
                print(f"✅ Replicate generated image")
            except Exception as e_replicate:
                print(f"⚠️ Replicate also failed: {e_replicate}")

        # --- 3️⃣ Si ambos fallan, fallback PNG negro ---
        if not image_b64 or not isinstance(image_b64, str):
            print(f"🔴 Both AI services failed, using fallback image")
            dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
            image_b64 = "data:image/png;base64," + base64.b64encode(dummy_image.tobytes()).decode()

        demo_images.append(image_b64)

        return JSONResponse({
            "status": "ok",
            "demo_outfits": demo_images,
            "generation_mode": "AI_generated" if image_b64 else "fallback",
        })

    except Exception as e:
        print("❌ Unexpected error in /generate_outfit_demo:", traceback.format_exc())
        fallback_b64 = "data:image/png;base64," + base64.b64encode(
            np.zeros((256, 256, 3), dtype=np.uint8).tobytes()
        ).decode()
        return JSONResponse({
            "status": "ok",
            "demo_outfits": [fallback_b64],
            "generation_mode": "fallback",
            "error": str(e)
        })
