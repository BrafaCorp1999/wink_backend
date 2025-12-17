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
    }

    # Crear tarea
    try:
        response = requests.post(NANOBANANA_URL, json=body, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nano Banana no respondió: {str(e)}")

    data = response.json()
    task_id = data.get("data", {}).get("taskId")
    if not task_id:
        raise HTTPException(status_code=500, detail="No se recibió taskId")

    # Polling rápido (solo 5-10s)
    result_url = f"https://api.nanobananaapi.ai/api/v1/nanobanana/result/{task_id}"
    for _ in range(5):  # 5 iteraciones x 2s = 10s máximo
        time.sleep(2)
        try:
            result = requests.get(result_url, headers=headers, timeout=10).json()
        except:
            continue

        status = result.get("data", {}).get("status")
        if status == "SUCCESS":
            image_url = result["data"]["images"][0]
            image_bytes = requests.get(image_url).content
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            return JSONResponse({"status": "ok", "image": image_base64})

        if status == "FAILED":
            break

    # Si no está lista, devolvemos pending
    return JSONResponse({"status": "pending", "message": "Imagen en proceso, intente nuevamente en unos segundos"})
