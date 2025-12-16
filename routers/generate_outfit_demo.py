from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
import requests

router = APIRouter()

# 🔹 Simulación de integración de dos IA's
#   DALL·E + Replicate
#   Aquí reemplazarías los "requests.get" con tus llamadas a la API real

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "unknown")

    try:
        # -------------------------
        # 1️⃣ DALL·E: generar imagen
        # -------------------------
        dalle_prompt = f"Generate a full-body outfit for a {gender} person. Maintain face features and posture."
        # 🔹 Ejemplo de placeholder: reemplazar por llamada real a DALL·E
        dalle_url = f"https://via.placeholder.com/512x768.png?text=DALL-E+{gender}"
        dalle_resp = requests.get(dalle_url)
        if dalle_resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Error generando imagen DALL·E")
        dalle_bytes = dalle_resp.content

        # -------------------------
        # 2️⃣ Replicate: generar otra imagen
        # -------------------------
        replicate_prompt = f"Generate a realistic outfit for a {gender} person. Keep user's facial features."
        # 🔹 Ejemplo de placeholder: reemplazar por llamada real a Replicate
        replicate_url = f"https://via.placeholder.com/512x768.png?text=Replicate+{gender}"
        replicate_resp = requests.get(replicate_url)
        if replicate_resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Error generando imagen Replicate")
        replicate_bytes = replicate_resp.content

        # -------------------------
        # Enviar como JSON base64 (opcional) o enviar bytes individuales
        # Para Flutter: preferible enviar lista de bytes codificados como base64 corto
        # Alternativa: crear endpoint /image/1 y /image/2 para GET directo
        # Aquí usamos base64 para simplificar la integración Flutter
        import base64
        images_b64 = [
            "data:image/png;base64," + base64.b64encode(dalle_bytes).decode("utf-8"),
            "data:image/png;base64," + base64.b64encode(replicate_bytes).decode("utf-8"),
        ]

        return JSONResponse({
            "status": "ok",
            "demo_outfits": images_b64
        })

    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })
