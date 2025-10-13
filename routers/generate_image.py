from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from google import genai
from google.genai import types
import base64, traceback, os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=GEMINI_API_KEY)


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
    Retorna imagen + texto resumido (2-3 líneas) describiendo el outfit sugerido.
    """
    try:
        MAX_IMAGE_SIZE_MB = 10
        ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

        # ---- Validar imagen del usuario ----
        user_bytes = await person_image.read()
        if len(user_bytes) / (1024*1024) > MAX_IMAGE_SIZE_MB:
            raise HTTPException(status_code=400, detail="person_image exceeds 10MB")
        if person_image.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {person_image.content_type}")

        # ---- Construir prompt completo estrictamente ----
        full_prompt = f"""
        Generate a photorealistic full-body outfit for the user in the image.
        STRICTLY PRESERVE:
        - User's face, hair, neck and facial identity
        - Full body proportions
        - Natural appearance
        Instructions: {prompt}
        Context:
        - Model Type: {model_type}
        - Gender: {gender}
        - Style: {style}
        Apply modern clothing items (tops, bottoms, shoes, accessories) according to the instructions.
        Include a short description of the outfit in 2-3 lines.
        """

        # ---- Preparar contenido para Gemini ----
        contents = [
            full_prompt,
            types.Part.from_bytes(data=user_bytes, mime_type=person_image.content_type)
        ]

        # ---- Generar imagen + texto con Gemini ----
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=['TEXT','IMAGE'])
        )

        # ---- Procesar respuesta ----
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

        if image_data:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            image_url = f"data:{image_mime_type};base64,{image_base64}"
        else:
            image_url = None

        return JSONResponse(content={"image": image_url, "text": text_response})

    except Exception as e:
        print("❌ Error in /api/generate-image endpoint:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
