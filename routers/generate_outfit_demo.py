import os
import base64
import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

DEAPI_API_KEY = os.getenv("DEAPI_API_KEY")
CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")

# =========================
# 🟢 Endpoint principal
# =========================
@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "female")
    print(f"[LOG] 🔹 Solicitud de outfit para gender={gender}")

    prompt = f"Ultra-realistic full body photo of a {gender} person wearing a modern stylish outfit. No cartoon."

    # 1️⃣ Intentar deAPI
    if DEAPI_API_KEY:
        try:
            print("[LOG] 🔹 Intentando deAPI...")
            img_bytes = try_deapi(prompt)
            print("[LOG] ✅ deAPI generó imagen")
            return JSONResponse({"status":"ok", "image": base64.b64encode(img_bytes).decode("utf-8")})
        except Exception as e:
            print(f"[WARN] ⚠️ deAPI falló: {e}")

    # 2️⃣ Intentar Cloudflare Workers AI
    if CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID:
        try:
            print("[LOG] 🔹 Intentando Cloudflare Workers AI...")
            img_bytes = try_cloudflare(prompt)
            print("[LOG] ✅ Cloudflare Workers generó imagen")
            return JSONResponse({"status":"ok", "image": base64.b64encode(img_bytes).decode("utf-8")})
        except Exception as e:
            print(f"[WARN] ⚠️ Cloudflare Workers falló: {e}")

    # 3️⃣ Intentar SubNP Free API
    try:
        print("[LOG] 🔹 Intentando SubNP Free API...")
        img_bytes = try_subnp(prompt)
        print("[LOG] ✅ SubNP generó imagen")
        return JSONResponse({"status":"ok", "image": base64.b64encode(img_bytes).decode("utf-8")})
    except Exception as e:
        print(f"[WARN] ⚠️ SubNP Free API falló: {e}")

    # Ninguno funcionó
    raise HTTPException(status_code=500, detail="Ningún servicio pudo generar la imagen")

# =========================
# 🟢 Función deAPI
# =========================
def try_deapi(prompt: str) -> bytes:
    url = "https://api.deapi.ai/v1/image/text2image"
    headers = {"Authorization": f"Bearer {DEAPI_API_KEY}", "Content-Type": "application/json"}
    body = {"prompt": prompt, "width": 512, "height": 768, "num_images": 1}
    resp = requests.post(url, json=body, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise Exception(f"deAPI status={resp.status_code}, body={resp.text}")
    data = resp.json()
    img_url = data.get("data", {}).get("image_url")
    img_b64 = data.get("data", {}).get("image_base64")
    if img_url:
        return requests.get(img_url).content
    elif img_b64:
        return base64.b64decode(img_b64.split(",")[-1])
    else:
        raise Exception("deAPI no devolvió imagen")

# =========================
# 🟢 Función Cloudflare Workers AI
# =========================
def try_cloudflare(prompt: str) -> bytes:
    url = f"https://{CLOUDFLARE_ACCOUNT_ID}.cloudflareworkers.ai/ai/run/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {CLOUDFLARE_API_KEY}", "Content-Type": "application/json"}
    body = {"prompt": prompt}
    resp = requests.post(url, json=body, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise Exception(f"Cloudflare status={resp.status_code}, body={resp.text}")
    data = resp.json()
    img_b64 = data.get("result", {}).get("image_base64")
    if not img_b64:
        raise Exception("Cloudflare no devolvió imagen")
    return base64.b64decode(img_b64.split(",")[-1])

# =========================
# 🟢 Función SubNP Free API
# =========================
def try_subnp(prompt: str) -> bytes:
    url = "https://subnp.com/api/free/generate"
    resp = requests.post(url, json={"prompt": prompt}, headers={"Content-Type": "application/json"}, timeout=120)
    if resp.status_code != 200:
        raise Exception(f"SubNP status={resp.status_code}, body={resp.text}")
    data = resp.json()
    img_url = data.get("image_url")
    if not img_url:
        raise Exception("SubNP no devolvió imagen")
    return requests.get(img_url).content
