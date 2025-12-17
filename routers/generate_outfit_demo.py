# routers/generate_outfit_demo.py
import os
import time
import base64
import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

NANOBANANA_API_KEY = os.getenv("NANOBANANA_API_KEY")
NANOBANANA_URL = "https://api.nanobananaapi.ai/api/v1/nanobanana/generate"

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    if not NANOBANANA_API_KEY:
        raise HTTPException(status_code=500, detail="Nano Banana API key no configurada")

    gender = payload.get("gender", "female")
    image_base64 = payload.get("image_base64")

    if not image_base64:
        raise HTTPException(status_code=400, detail="Se requiere image_base64")

    # Prompt para generar outfit realista preservando rostro
    prompt = f"""
    Ultra-realistic full-body photo of a {gender} person wearing a modern, stylish outfit.
    Preserve original facial identity, facial proportions and skin tone.
    Natural human face, realistic anatomy, real fabric textures.
    Studio lighting, DSLR photo, sharp focus.
    No cartoon, no animation, no CGI, no game style, no distortion.
    """

    headers = {
        "Authorization": f"Bearer {NANOBANANA_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "prompt": prompt.strip(),
        "numImages": 1,
        "type": "TEXTTOIAMGE",
        "image_size": "3:4",
        # Para demo rápida: no callback
    }

    try:
        # 1️⃣ Crear tarea
        response = requests.post(NANOBANANA_URL, json=body, headers=headers, timeout=60)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Nano Banana no respondió")

        data = response.json().get("data")
        if not data or "taskId" not in data:
            raise HTTPException(status_code=500, detail="No se recibió taskId de NanoBanana")

        task_id = data["taskId"]
        result_url = f"https://api.nanobananaapi.ai/api/v1/nanobanana/result/{task_id}"

        # 2️⃣ Polling interno hasta que la imagen esté lista (max 60s)
        for _ in range(30):
            time.sleep(2)
            result_resp = requests.get(result_url, headers=headers, timeout=30).json()
            result_data = result_resp.get("data", {})

            status = result_data.get("status")
            if status == "SUCCESS":
                image_url = result_data["images"][0]
                image_bytes = requests.get(image_url, timeout=30).content
                image_base64_result = base64.b64encode(image_bytes).decode("utf-8")
                return JSONResponse({"status": "ok", "image": image_base64_result})

            elif status == "FAILED":
                raise HTTPException(status_code=500, detail="NanoBanana falló generando la imagen")

        # Timeout
        raise HTTPException(status_code=500, detail="Timeout: la imagen no se generó a tiempo")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error llamando a NanoBanana: {str(e)}")
