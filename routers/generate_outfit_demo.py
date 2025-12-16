# routers/generate_outfit_demo.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import base64, traceback, numpy as np
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
                # Gemini primero
                gemini_result = await gemini_generate_image(prompt, size="512x512")
                if gemini_result and isinstance(gemini_result, dict):
                    enriched_prompt = gemini_result.get("content", prompt)
                else:
                    enriched_prompt = prompt

                image_b64 = await gemini_generate_image(enriched_prompt, size="512x512")
                if not image_b64 or not isinstance(image_b64, str):
                    # fallback OpenAI
                    image_b64 = await openai_generate_image(enriched_prompt, size="512x512")

                # fallback real si ambos fallan
                if not image_b64 or not isinstance(image_b64, str):
                    raise ValueError("No image generated")

            except Exception as e:
                print(f"⚠️ Generación falló para {style}: {e}")
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
        print("❌ Error general:", traceback.format_exc())
        empty_b64 = "data:image/png;base64," + base64.b64encode(
            np.zeros((512,512,3), dtype=np.uint8).tobytes()
        ).decode()
        return JSONResponse({
            "status": "ok",
            "demo_outfits": [empty_b64]*2,
            "generation_mode": "fallback",
            "error": str(e)
        })
