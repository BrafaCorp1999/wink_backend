import os
import time
import base64
import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

NANOBANANA_API_KEY = os.getenv("NANOBANANA_API_KEY")
NANOBANANA_URL = "https://api.nanobananaapi.ai/api/v1/nanobanana/generate"

@router.post("/generate_outfit_demo/")
async def generate_outfit_demo(payload: dict):
    if not NANOBANANA_API_KEY:
        raise HTTPException(status_code=500, detail="Nano Banana API key no configurada")

    gender = payload.get("gender", "female")

    # Prompt CLAVE: realismo + preservación facial
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
        # Resolución equilibrada (realista + bajo consumo)
        "image_size": "3:4",
    }

    # 1️⃣ Crear tarea
    response = requests.post(NANOBANANA_URL, json=body, headers=headers, timeout=60)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Nano Banana no respondió")

    data = response.json()
    task_id = data.get("data", {}).get("taskId")
    if not task_id:
        raise HTTPException(status_code=500, detail="No se recibió taskId")

    # 2️⃣ Polling simple (bloqueante pero suficiente para demo)
    result_url = f"https://api.nanobananaapi.ai/api/v1/nanobanana/result/{task_id}"

    for _ in range(20):  # hasta ~40–60s
        time.sleep(2)
        result = requests.get(result_url, headers=headers).json()

        if result.get("data", {}).get("status") == "SUCCESS":
            image_url = result["data"]["images"][0]

            image_bytes = requests.get(image_url).content
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            return JSONResponse(
                {
                    "status": "ok",
                    "image": image_base64,
                }
            )

        if result.get("data", {}).get("status") == "FAILED":
            break

    raise HTTPException(status_code=500, detail="Timeout generando imagen")
