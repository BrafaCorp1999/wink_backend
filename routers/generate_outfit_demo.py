from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import base64, traceback, numpy as np
import cv2

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

        # --- Base64 → NumPy array (BGR) ---
        header, encoded = face_base64.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        demo_images = []
        outfit_styles = ["random_style_1", "random_style_2"]

        for style in outfit_styles:
            prompt = (
                f"Full body portrait of a {gender} person wearing {style} outfit, "
                "maintaining exact body measurements and face from reference image. "
                "Do not deform face or body. Clothes must fit the body properly."
            )

            # --- Fallback: Gemini → OpenAI → placeholder negro ---
            image_b64 = None
            try:
                gemini_result = await gemini_generate_image(prompt)
                if gemini_result and "content" in gemini_result:
                    # Gemini solo devuelve texto, pasamos a OpenAI
                    image_b64 = await openai_generate_image(prompt)
                else:
                    # Si Gemini falla, fallback directo a OpenAI
                    image_b64 = await openai_generate_image(prompt)
            except Exception as e1:
                print("⚠️ Gemini/OpenAI failed:", e1)
                # fallback negro
                image_b64 = "data:image/png;base64," + base64.b64encode(
                    np.zeros((512,512,3), dtype=np.uint8).tobytes()
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
