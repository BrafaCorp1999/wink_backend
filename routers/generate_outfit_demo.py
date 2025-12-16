from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import httpx
import os
from io import BytesIO
from PIL import Image

router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "unknown")
    measurements = payload.get("measurements")
    face_base64 = payload.get("face_base64")

    if not measurements or not face_base64:
        raise HTTPException(status_code=400, detail="Missing measurements or face image.")

    # 🔹 Prompt para la IA
    prompt = (
        f"Genera un outfit moderno para una persona {gender}, "
        "manteniendo sus rasgos faciales y proporciones exactas del cuerpo según la imagen de referencia. "
        "No deformar la cara ni cambiar postura. Fondo neutro, estilo de fotografía realista."
    )

    # Convertir base64 de la cara a bytes
    try:
        face_bytes = BytesIO()
        face_bytes.write(base64.b64decode(face_base64.split(",")[1]))
        face_bytes.seek(0)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid face image base64.")

    # 🔹 Intentar con OpenAI
    try:
        import openai
        openai.api_key = OPENAI_API_KEY

        result = openai.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            image=face_bytes  # referencia de la cara
        )

        img_data = result.data[0].b64_json
        img_bytes = BytesIO(base64.b64decode(img_data)).read()
        return Response(content=img_bytes, media_type="image/png")

    except Exception as e_openai:
        # 🔹 Fallback Replicate
        try:
            headers = {"Authorization": f"Token {REPLICATE_API_KEY}"}
            json_payload = {"prompt": prompt, "image": face_bytes.read(), "size": "1024x1024"}
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post("https://api.replicate.com/v1/predictions", headers=headers, json=json_payload)
                resp.raise_for_status()
                data = resp.json()
                # Aquí se asume que la respuesta devuelve URL o base64
                replicate_img_url = data["output"][0]
                # Descargar la imagen
                img_resp = await client.get(replicate_img_url)
                img_resp.raise_for_status()
                return Response(content=img_resp.content, media_type="image/png")

        except Exception as e_repl:
            raise HTTPException(
                status_code=500,
                detail=f"AI generation failed: OpenAI error: {str(e_openai)}, Replicate error: {str(e_repl)}"
            )
