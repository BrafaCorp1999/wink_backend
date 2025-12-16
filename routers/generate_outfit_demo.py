from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import io
import base64
import os
import requests
import traceback

router = APIRouter()

# 🔹 Asegúrate de tener tus keys en Environment Variables
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
REPLICATE_KEY = os.environ.get("REPLICATE_API_KEY")

import openai
openai.api_key = OPENAI_KEY

try:
    import replicate
except ImportError:
    replicate = None  # opcional si no quieres Replicate ahora

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "unknown")
    images_b64 = []

    try:
        # -------------------------
        # 1️⃣ DALL·E
        # -------------------------
        dalle_prompt = f"Generate a full-body outfit for a {gender} person. Maintain face features and posture."
        try:
            dalle_resp = openai.Image.create(
                prompt=dalle_prompt,
                n=1,
                size="512x768"
            )
            dalle_b64 = dalle_resp['data'][0]['b64_json']
            dalle_bytes = base64.b64decode(dalle_b64)
            images_b64.append("data:image/png;base64," + base64.b64encode(dalle_bytes).decode("utf-8"))
            print(f"✅ DALL·E generado correctamente")
        except Exception as e:
            print(f"⚠️ Error DALL·E: {e}")
            print(traceback.format_exc())

        # -------------------------
        # 2️⃣ Replicate
        # -------------------------
        if replicate:
            replicate_prompt = f"Generate a realistic outfit for a {gender} person. Keep user's facial features."
            try:
                model = replicate.models.get("stability-ai/stable-diffusion")
                output_urls = model.predict(prompt=replicate_prompt)
                if output_urls:
                    replicate_bytes = requests.get(output_urls[0]).content
                    images_b64.append("data:image/png;base64," + base64.b64encode(replicate_bytes).decode("utf-8"))
                    print(f"✅ Replicate generado correctamente")
            except Exception as e:
                print(f"⚠️ Error Replicate: {e}")
                print(traceback.format_exc())

        # -------------------------
        # Si ninguna IA funcionó
        # -------------------------
        if not images_b64:
            print("❌ Ninguna IA generó imágenes")
            return JSONResponse({
                "status": "error",
                "message": "No se pudo generar ninguna imagen. Revisa los logs."
            })

        # -------------------------
        # Retornar resultados a Flutter
        # -------------------------
        return JSONResponse({
            "status": "ok",
            "demo_outfits": images_b64
        })

    except Exception as e:
        print(f"❌ ERROR GENERAL: {e}")
        print(traceback.format_exc())
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })
