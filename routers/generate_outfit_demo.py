# routers/generate_outfit_demo.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import base64, traceback, numpy as np

# --- Import services ---
from utils.gemini_service import gemini_generate_image
from utils.openai_service import openai_generate_image

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
        outfit_styles = ["random_style_1", "random_style_2"]

        for style in outfit_styles:
            prompt = (
                f"Full body portrait of a {gender} person wearing {style} outfit, "
                "maintaining exact body measurements and face from reference image. "
                "Do not deform face or body. Clothes must fit the body properly."
            )

            image_b64 = None

            try:
                # 1️⃣ Intentar Gemini
                gemini_result = await gemini_generate_image(prompt)
                if gemini_result and isinstance(gemini_result, dict):
                    enriched_prompt = gemini_result.get("content", prompt)
                else:
                    enriched_prompt = prompt

                if enriched_prompt:
                    image_b64 = await gemini_generate_image(enriched_prompt)
                    if image_b64 and isinstance(image_b64, str):
                        print(f"✅ Gemini generó imagen para style: {style}")
                    else:
                        image_b64 = None  # forzar fallback a OpenAI

                # 2️⃣ Si Gemini falla, usar OpenAI
                if not image_b64:
                    print(f"⚠️ Gemini falló, intentando OpenAI para style: {style}")
                    image_b64 = await openai_generate_image(enriched_prompt)
                    if image_b64 and isinstance(image_b64, str):
                        print(f"✅ OpenAI generó imagen para style: {style}")
                    else:
                        raise ValueError("OpenAI returned empty image")

            except Exception as e:
                print(f"⚠️ Ambos servicios fallaron para style {style}: {e}")
                # fallback real
                image_b64 = "data:image/png;base64," + base64.b64encode(
                    np.zeros((512, 512, 3), dtype=np.uint8).tobytes()
                ).decode()

            demo_images.append(image_b64)

        return JSONResponse({
            "status": "ok",
            "demo_outfits": demo_images,
            "generation_mode": "AI_generated"
        })

    except Exception as e:
        print("❌ Error in /generate_outfit_demo:", traceback.format_exc())
        empty_b64 = "data:image/png;base64," + base64.b64encode(
            np.zeros((512,512,3), dtype=np.uint8).tobytes()
        ).decode()
        return JSONResponse({
            "status": "ok",
            "demo_outfits": [empty_b64]*2,
            "generation_mode": "fallback",
            "error": str(e)
        })
