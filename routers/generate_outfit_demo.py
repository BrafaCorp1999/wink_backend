# routers/generate_outfit_demo.py
import os
import base64
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

# =========================
# Leer llaves de entorno
# =========================
CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")

# =========================
# ENDPOINT
# =========================
@router.post("/generate_outfit_demo")
async def generate_outfit_demo(request: Request):
    try:
        data = await request.json()
        gender = data.get("gender")
        image_base64 = data.get("image_base64")

        if not gender or not image_base64:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Faltan datos"})

        # =========================
        # 1️⃣ Intentar Cloudflare Workers AI
        # =========================
        if CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID:
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    cf_payload = {
                        "prompt": f"Outfit for a {gender} based on the user's body image",
                        "type": "IMAGETOIAMGE",
                        "imageUrls": [f"data:image/png;base64,{image_base64}"],
                        "numImages": 1,
                        "image_size": "1:1"
                    }
                    cf_headers = {
                        "Authorization": f"Bearer {CLOUDFLARE_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    cf_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/workers/ai/generate"
                    cf_resp = await client.post(cf_url, json=cf_payload, headers=cf_headers)
                    cf_data = cf_resp.json()
                    print("🔹 Cloudflare response:", cf_data)

                    if cf_resp.status_code == 200 and "data" in cf_data and len(cf_data["data"]) > 0:
                        img_base64 = cf_data["data"][0]["b64_json"]
                        return {"status": "ok", "image": img_base64}
                    else:
                        print("⚠️ Cloudflare no devolvió imagen válida, intentando fallback...")
            except Exception as e:
                print("⚠️ Error Cloudflare:", e)

        # =========================
        # 2️⃣ Fallback: API gratuita sin key
        # =========================
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                free_payload = {
                    "prompt": f"Outfit for a {gender} based on the user's body image",
                    "image_base64": image_base64
                }
                free_resp = await client.post(
                    "https://subnp-free-ai.vercel.app/api/generate_outfit_demo",
                    json=free_payload
                )
                free_data = free_resp.json()
                print("🔹 Free API response:", free_data)

                if free_resp.status_code == 200 and free_data.get("status") == "ok":
                    return {"status": "ok", "image": free_data["image"]}
                else:
                    return JSONResponse(status_code=500, content={"status": "error", "message": "No se pudo generar la imagen con ninguna API"})
        except Exception as e:
            print("⚠️ Error Free API:", e)
            return JSONResponse(status_code=500, content={"status": "error", "message": "Error interno al generar imagen"})

    except Exception as e:
        print("❌ ERROR GENERAL:", e)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
