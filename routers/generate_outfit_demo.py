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
    """
    Genera un outfit completo usando Nano Banana Text-to-Image.
    Espera hasta ~120s por la imagen si es necesario.
    """

    if not NANOBANANA_API_KEY:
        raise HTTPException(status_code=500, detail="Nano Banana API key no configurada")

    gender = payload.get("gender", "female")

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
        "image_size": "3:4",  # puedes cambiar 3:4, 4:5, 9:16 según preferencia
    }

    try:
        # 1️⃣ Crear tarea
        response = requests.post(NANOBANANA_URL, json=body, headers=headers, timeout=120)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al crear tarea Nano Banana: {str(e)}")

    try:
        data = response.json()
    except Exception:
        raise HTTPException(status_code=500, detail="Nano Banana no devolvió JSON válido")

    # Caso 1: tarea lista inmediatamente
    if data.get("status") == "success":
        image_url = data.get("data", {}).get("images", [None])[0]
        if not image_url:
            raise HTTPException(status_code=500, detail="No se recibió URL de imagen")
        
        image_bytes = requests.get(image_url).content
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return JSONResponse({"status": "ok", "image": image_base64})

    # Caso 2: tarea pendiente → polling
    task_id = data.get("data", {}).get("taskId")
    if not task_id:
        raise HTTPException(status_code=500, detail="No se recibió taskId ni imagen")

    result_url = f"https://api.nanobananaapi.ai/api/v1/nanobanana/result/{task_id}"

    for _ in range(60):  # hasta ~120s (2s sleep * 60)
        time.sleep(2)
        try:
            result = requests.get(result_url, headers=headers, timeout=60).json()
        except Exception:
            continue

        if result.get("data", {}).get("status") == "SUCCESS":
            image_url = result["data"]["images"][0]
            image_bytes = requests.get(image_url).content
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            return JSONResponse({"status": "ok", "image": image_base64})

        if result.get("data", {}).get("status") == "FAILED":
            raise HTTPException(status_code=500, detail="Nano Banana falló generando la imagen")

    raise HTTPException(status_code=500, detail="Timeout generando imagen")
