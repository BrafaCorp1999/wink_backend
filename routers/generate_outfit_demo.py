import os
import time
import base64
import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    from fastapi.responses import JSONResponse
    import requests, base64, time, os

    NANOBANANA_API_KEY = os.getenv("NANOBANANA_API_KEY")
    if not NANOBANANA_API_KEY:
        return JSONResponse({"status": "error", "message": "API key no configurada"}, status_code=500)

    gender = payload.get("gender", "female")
    prompt = f"Ultra-realistic full-body photo of a {gender} person wearing a modern, stylish outfit. Preserve original facial identity, proportions and skin tone."

    # 1️⃣ Crear tarea
    response = requests.post(
        "https://api.nanobananaapi.ai/api/v1/nanobanana/generate",
        json={"prompt": prompt, "numImages": 1, "type": "TEXTTOIAMGE", "image_size": "3:4"},
        headers={"Authorization": f"Bearer {NANOBANANA_API_KEY}"}
    )
    if response.status_code != 200 or not response.json().get("data"):
        return JSONResponse({"status": "error", "message": "NanoBanana no respondió"}, status_code=500)

    task_id = response.json()["data"].get("taskId")
    if not task_id:
        return JSONResponse({"status": "error", "message": "No se recibió taskId"}, status_code=500)

    # 2️⃣ Polling
    for _ in range(40):  # hasta ~80s
        time.sleep(2)
        result = requests.get(f"https://api.nanobananaapi.ai/api/v1/nanobanana/result/{task_id}",
                              headers={"Authorization": f"Bearer {NANOBANANA_API_KEY}"}).json()
        status = result.get("data", {}).get("status")
        if status == "SUCCESS":
            image_url = result["data"]["images"][0]
            image_bytes = requests.get(image_url).content
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            return JSONResponse({"status": "ok", "image": image_base64})
        if status == "FAILED":
            break

    return JSONResponse({"status": "error", "message": "Timeout generando imagen"}, status_code=500)

