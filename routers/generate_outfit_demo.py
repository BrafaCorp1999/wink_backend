# routers/generate_outfit_demo.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import base64, traceback, numpy as np

from utils.openai_service import openai_generate_image
from utils.replicate_service import replicate_generate_image  # nueva

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

            # 1️⃣ OpenAI DALL·E
            try:
                image_b64 = await openai_generate_image(prompt, size="512x512")
            except Exception as e:
                print(f"⚠️ OpenAI fail for style {style}: {e}")
                image_b64 = None

            # 2️⃣ Replicate fallback (Stable Diffusion)
            if not image_b64:
                try:
                    # replicate returns a URL -- descarga y conviértela a base64
                    image_url = await replicate_generate_image(prompt, width=512, height=512)
                    if image_url:
                        # descarga bytes
                        import requests
                        resp = requests.get(image_url)
                        if resp.status_code == 200:
                            image_bytes = resp.content
                            image_b64 = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
                except Exception as e:
                    print(f"⚠️ Replicate fail for style {style}: {e}")
                    image_b64 = None

            # 3️⃣ Fallback seguro (imagen negra)
            if not image_b64:
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
        print("❌ Error in generate_outfit_demo:", traceback.format_exc())
        empty_b64 = "data:image/png;base64," + base64.b64encode(
            np.zeros((512,512,3), dtype=np.uint8).tobytes()
        ).decode()
        return JSONResponse({
            "status": "ok",
            "demo_outfits": [empty_b64]*2,
            "generation_mode": "fallback",
            "error": str(e)
        })
