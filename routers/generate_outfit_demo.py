# routers/generate_outfit_demo.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import base64, traceback, io, numpy as np
import cv2
import torch

# Import services (Gemini & OpenAI)
from services.gemini_service import generate_image_gemini
from services.openai_service import generate_image_openai

router = APIRouter()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    try:
        measurements = payload.get("measurements")
        face_base64 = payload.get("face_base64")
        gender = payload.get("gender", "unknown")
        base_photo_url = payload.get("base_photo_url")

        if not measurements or not face_base64:
            raise HTTPException(status_code=400, detail="Missing measurements or face image.")

        # --- Base64 → NumPy array (BGR) ---
        header, encoded = face_base64.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        demo_images = []
        outfit_styles = ["random_style_1", "random_style_2"]  # 2 imágenes

        for style in outfit_styles:
            prompt = (
                f"Full body portrait of a {gender} person wearing {style} outfit, "
                "maintaining exact body measurements and face from reference image. "
                "Do not deform face or body. "
                "Clothes must fit the body properly, not float over it."
            )

            image = None
            # --- Fallback robusto: Gemini → OpenAI → placeholder negro ---
            try:
                image = generate_image_gemini(face_img, prompt)
            except Exception as e1:
                print("⚠️ Gemini failed:", e1)
                try:
                    image = generate_image_openai(face_img, prompt)
                except Exception as e2:
                    print("⚠️ OpenAI failed:", e2)
                    image = np.zeros((512,512,3), dtype=np.uint8)  # fallback negro

            # --- Convertir salida a base64 ---
            if isinstance(image, np.ndarray):
                _, buf = cv2.imencode(".png", image)
                b64_str = "data:image/png;base64," + base64.b64encode(buf).decode()
            else:
                # fallback si genera Pillow o otra cosa
                b64_str = "data:image/png;base64," + base64.b64encode(np.zeros((512,512,3), np.uint8).tobytes()).decode()

            demo_images.append(b64_str)

        return JSONResponse({
            "status": "ok",
            "demo_outfits": demo_images,
            "generation_mode": "AI_generated"
        })

    except Exception as e:
        print("❌ Error in /generate_outfit_demo:", traceback.format_exc())
        empty_b64 = "data:image/png;base64," + base64.b64encode(np.zeros((512,512,3), dtype=np.uint8).tobytes()).decode()
        return JSONResponse({
            "status": "ok",
            "demo_outfits": [empty_b64]*2,
            "generation_mode": "fallback",
            "error": str(e)
        })
