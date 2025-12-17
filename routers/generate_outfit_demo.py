import os
import time
import base64
import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os, requests

router = APIRouter()
NANOBANANA_API_KEY = os.getenv("NANOBANANA_API_KEY")
NANOBANANA_URL = "https://api.nanobananaapi.ai/api/v1/nanobanana/generate"

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    if not NANOBANANA_API_KEY:
        raise HTTPException(status_code=500, detail="Nano Banana API key no configurada")

    gender = payload.get("gender", "female")
    prompt = f"Ultra-realistic full-body photo of a {gender} person wearing a modern, stylish outfit. Preserve face."

    headers = {"Authorization": f"Bearer {NANOBANANA_API_KEY}", "Content-Type": "application/json"}
    body = {"prompt": prompt, "numImages": 1, "type": "TEXTTOIAMGE", "image_size": "3:4"}

    response = requests.post(NANOBANANA_URL, json=body, headers=headers, timeout=60)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Nano Banana no respondió")

    data = response.json()
    task_id = data.get("data", {}).get("taskId")
    if not task_id:
        raise HTTPException(status_code=500, detail="No se recibió taskId")

    # ✅ Devuelve taskId inmediatamente
    return {"status": "pending", "taskId": task_id}

@router.get("/check_outfit_task/{task_id}")
async def check_outfit_task(task_id: str):
    headers = {"Authorization": f"Bearer {NANOBANANA_API_KEY}"}
    result_url = f"https://api.nanobananaapi.ai/api/v1/nanobanana/result/{task_id}"
    result = requests.get(result_url, headers=headers, timeout=10).json()
    if result.get("data", {}).get("status") == "SUCCESS":
        image_url = result["data"]["images"][0]
        image_bytes = requests.get(image_url).content
        return {"status": "ok", "image": base64.b64encode(image_bytes).decode("utf-8")}
    elif result.get("data", {}).get("status") == "FAILED":
        return {"status": "failed"}
    else:
        return {"status": "pending"}

