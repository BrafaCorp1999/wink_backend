from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from google import genai
from google.genai import types
from googletrans import Translator
import base64, traceback, os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=GEMINI_API_KEY)
translator = Translator()


def traducir_y_resumir(texto_en: str) -> str:
    """Traduce al español y resume el texto a 1–2 líneas."""
    try:
        if not texto_en:
            return "✨ Look moderno y cómodo, ideal para destacar con estilo."

        # Traducir al español
        texto_es = translator.translate(texto_en, src="en", dest="es").text.strip()

        # Resumir (máx. 2 líneas o ~180 caracteres)
        if len(texto_es) > 180:
            texto_es = texto_es[:180].rsplit('.', 1)[0] + "..."
        return texto_es
    except Exception as e:
        print("⚠️ Error traduciendo texto:", e)
        return "✨ Look moderno y cómodo, ideal para destacar con estilo."


@router.post("/generate-image")
async def generate_image(
    person_image: UploadFile = File(...),
    prompt: str = Form(...),
    model_type: str = Form("realistic"),
    gender: str = Form("female"),
    style: str = Form("modern"),
):
    """
    Genera un outfit completo sobre la imagen del usuario usando un prompt personalizado.
    Retorna imagen + descripción corta en español.
    """
    try:
        MAX_IMAGE_SIZE_MB = 10
        ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

        # --- Validar imagen ---
        user_bytes = await person_image.read()
        if len(user_bytes) / (1024 * 1024) > MAX_IMAGE_SIZE_MB:
            raise HTTPException(status_code=400, detail="person_image exceeds 10MB")
        if person_image.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {person_image.content_type}"
            )

        # --- Prompt completo para Gemini ---
        full_prompt = f"""
        Generate a photorealistic full-body outfit for the user in the image.
        STRICTLY PRESERVE:
        - User's face, hair, and natural body shape.
        Instructions: {prompt}
        Context:
        - Model Type: {model_type}
        - Gender: {gender}
        - Style: {style}
        Include a short 2–3 sentence English description of the outfit.
        """

        contents = [
            full_prompt,
            types.Part.from_bytes(data=user_bytes, mime_type=person_image.content_type),
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )

        # --- Procesar respuesta ---
        image_data = None
        text_response = "No description available."
        if response.candidates:
            parts = response.candidates[0].content.parts
            for part in parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    image_data = part.inline_data.data
                    image_mime_type = getattr(part.inline_data, "mime_type", "image/png")
                elif hasattr(part, "text") and part.text:
                    text_response = part.text.strip()

        # --- Codificar imagen ---
        if image_data:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            image_url = f"data:{image_mime_type};base64,{image_base64}"
        else:
            image_url = None

        # --- Traducir y resumir descripción ---
        text_es = traducir_y_resumir(text_response)

        return JSONResponse(content={"image": image_url, "text": text_es})

    except Exception as e:
        print("❌ Error en /generate-image:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
