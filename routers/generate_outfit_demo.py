from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import io
import os
import requests
import base64
from pathlib import Path

router = APIRouter()

# 🔹 Leer API keys desde variables de entorno
DALLE_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_KEY = os.getenv("REPLICATE_API_KEY")

# 🔹 Ruta del fallback (opcional)
FALLBACK_IMAGE_PATH = Path(__file__).parent / "demo_outfit_fallback.png"

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "unknown")
    try:
        # -------------------------
        # 1️⃣ DALL·E
        # -------------------------
        dalle_prompt = f"Generate a full-body outfit for a {gender} person. Maintain face features and posture."
        dalle_bytes = None
        if DALLE_KEY:
            try:
                dalle_resp = requests.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {DALLE_KEY}"},
                    json={"prompt": dalle_prompt, "size": "512x768"}
                )
                dalle_resp.raise_for_status()
                dalle_data = dalle_resp.json()
                # Base64 de la imagen generada
                dalle_bytes = base64.b64decode(dalle_data['data'][0]['b64_json'])
            except Exception as e:
                print(f"⚠️ Error DALL·E: {e}")

        # -------------------------
        # 2️⃣ Replicate
        # -------------------------
        replicate_prompt = f"Generate a realistic outfit for a {gender} person. Keep user's facial features."
        replicate_bytes = None
        if REPLICATE_KEY:
            try:
                replicate_resp = requests.post(
                    "https://api.replicate.com/v1/predictions",
                    headers={"Authorization": f"Token {REPLICATE_KEY}"},
                    json={
                        "version": "YOUR_MODEL_VERSION_ID",  # reemplaza con tu versión de modelo
                        "input": {"prompt": replicate_prompt}
                    }
                )
                replicate_resp.raise_for_status()
                replicate_data = replicate_resp.json()
                # Suponiendo que replicate devuelve URL directa de imagen
                image_url = replicate_data['output'][0]
                img_resp = requests.get(image_url)
                img_resp.raise_for_status()
                replicate_bytes = img_resp.content
            except Exception as e:
                print(f"⚠️ Error Replicate: {e}")

        # -------------------------
        # Fallback si alguna IA falla
        # -------------------------
        if dalle_bytes is None:
            print("⚠️ Usando fallback para DALL·E")
            with open(FALLBACK_IMAGE_PATH, "rb") as f:
                dalle_bytes = f.read()
        if replicate_bytes is None:
            print("⚠️ Usando fallback para Replicate")
            with open(FALLBACK_IMAGE_PATH, "rb") as f:
                replicate_bytes = f.read()

        # -------------------------
        # Codificar a base64 para enviar a Flutter
        # -------------------------
        images_b64 = [
            "data:image/png;base64," + base64.b64encode(dalle_bytes).decode("utf-8"),
            "data:image/png;base64," + base64.b64encode(replicate_bytes).decode("utf-8"),
        ]

        return JSONResponse({"status": "ok", "demo_outfits": images_b64})

    except Exception as e:
        print(f"❌ Error general backend: {e}")
        return JSONResponse({"status": "error", "message": str(e)})
