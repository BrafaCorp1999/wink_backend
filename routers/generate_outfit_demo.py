from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import base64
import traceback
from PIL import Image
import io
import asyncio

# Servicios AI
from utils.openai_service import openai_generate_image
from utils.replicate_service import replicate_generate_image

router = APIRouter()

def generate_valid_fallback():
    # Fallback negro válido PNG 256x256
    img = Image.new("RGB", (256, 256), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    try:
        gender = payload.get("gender", "unknown")
        # Para prueba, ignoramos face/measurements
        demo_images = []

        prompt = f"Simple illustration of a {gender} person wearing casual outfit, random style, full body, do not deform face"

        image_b64 = None

        # --- 1️⃣ Intentar OpenAI DALL·E ---
        try:
            print(f"➡️ Generating outfit with OpenAI for prompt: {prompt}")
            image_b64 = await openai_generate_image(prompt)
            if not image_b64:
                raise ValueError("OpenAI returned empty image")
            print(f"✅ OpenAI generated image")

        # --- 2️⃣ Si falla OpenAI, intentar Replicate ---
        except Exception as e_openai:
            print(f"⚠️ OpenAI failed: {e_openai}")
            try:
                print(f"➡️ Trying Replicate for prompt: {prompt}")
                image_b64 = await replicate_generate_image(prompt)
                if not image_b64:
                    raise ValueError("Replicate returned empty image")
                print(f"✅ Replicate generated image")
            except Exception as e_replicate:
                print(f"⚠️ Replicate also failed: {e_replicate}")

        # --- 3️⃣ Fallback si ambos fallan ---
        if not image_b64 or not isinstance(image_b64, str):
            print(f"🔴 Both AI services failed, using fallback image")
            image_b64 = generate_valid_fallback()

        demo_images.append(image_b64)

        return JSONResponse({
            "status": "ok",
            "demo_outfits": demo_images,
            "generation_mode": "AI_generated"
        })

    except Exception as e:
        print("❌ Unexpected error:", traceback.format_exc())
        return JSONResponse({
            "status": "ok",
            "demo_outfits": [generate_valid_fallback()],
            "generation_mode": "fallback",
            "error": str(e)
        })
