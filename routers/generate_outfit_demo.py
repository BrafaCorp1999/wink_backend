import os
import time
import base64
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx

router = APIRouter()
logger = logging.getLogger("generate_outfit_demo")
logging.basicConfig(level=logging.INFO)

# Variables de entorno
CLOUDFLARE_AI_TOKEN = os.getenv("CLOUDFLARE_AI_TOKEN")  # Token de Cloudflare Workers AI
STABLE_HORDE_KEY = os.getenv("STABLE_HORDE_KEY", "0000000000")  # Key de Horde

def build_prompt(gender: str) -> str:
    return (
        f"High‑quality realistic full body outfit for a {gender} person, "
        "keeping the face and proportions natural without distortion."
    )

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "person")
    prompt = build_prompt(gender)

    #########################################
    # 1️⃣ Intentar Cloudflare Workers AI
    #########################################
    if CLOUDFLARE_AI_TOKEN:
        try:
            logger.info("➡️ Trying Cloudflare Workers AI generation")

            cf_url = f"https://api.cloudflare.com/client/v4/accounts/{{YOUR_ACCOUNT_ID}}/ai/run/@cf/stabilityai/stable-diffusion-xl-base-1.0"
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    cf_url,
                    headers={
                        "Authorization": f"Bearer {CLOUDFLARE_AI_TOKEN}",
                        "Content-Type": "application/json"
                    },
                    json={"prompt": prompt}
                )

            # Si éxito y image base64 viene en el JSON
            if resp.status_code == 200:
                result = resp.json()
                image_b64 = result.get("image_base64")
                if image_b64:
                    logger.info("✅ Cloudflare AI success")
                    return JSONResponse({"status": "ok", "image": image_b64})
        except Exception as e:
            logger.warning(f"⚠️ Cloudflare AI failed: {e}")

    #########################################
    # 2️⃣ Intentar Stable Horde (polling)
    #########################################
    try:
        logger.info("➡️ Trying Stable Horde generation (async)")

        # 1) Enviar job
        async with httpx.AsyncClient(timeout=120) as client:
            submit_resp = await client.post(
                "https://stablehorde.net/api/v2/generate/async",
                headers={"apikey": STABLE_HORDE_KEY},
                json={"prompt": prompt, "params": {"steps": 25, "width": 512, "height": 768}}
            )

        if submit_resp.status_code == 202:
            job = submit_resp.json().get("id")
            logger.info(f"  Job ID: {job}")

            # 2) Hacer polling hasta que esté listo
            async with httpx.AsyncClient(timeout=120) as client:
                for _ in range(30):  # hasta ~30 polls (≈30‑40s)
                    status_resp = await client.get(
                        f"https://stablehorde.net/api/v2/generate/check/{job}"
                    )
                    status_json = status_resp.json()

                    if status_json.get("done") == 1:
                        gens = status_json.get("generations", [])
                        if gens:
                            img_b64 = gens[0].get("img")
                            if img_b64:
                                logger.info("✅ Stable Horde image ready")
                                return JSONResponse({"status": "ok", "image": img_b64})
                    # esperar antes de intentar de nuevo
                    time.sleep(2)

    except Exception as e:
        logger.warning(f"⚠️ Stable Horde polling failed: {e}")

    #########################################
    # 3️⃣ Si todo falla
    #########################################
    return JSONResponse(
        {"status": "error", "message": "No se pudo generar imagen con servicios gratuitos."},
        status_code=500
    )
