from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
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

@router.post("/generate-body-model")
async def generate_body_model(
    Rostro: UploadFile = File(...),
    Torso: UploadFile = File(...),
    Piernas: UploadFile = File(...),
):
    try:
        # Validar tamaño y tipo
        MAX_IMAGE_SIZE_MB = 10
        ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
        for img in [Rostro, Torso, Piernas]:
            bytes_img = await img.read()
            if len(bytes_img)/(1024*1024) > MAX_IMAGE_SIZE_MB:
                raise HTTPException(status_code=400, detail=f"{img.filename} exceeds {MAX_IMAGE_SIZE_MB}MB")
            if img.content_type not in ALLOWED_MIME_TYPES:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {img.content_type}")
            img.file.seek(0)  # reset para leer después

        # Construir prompt
        prompt = f"""
        Generate a photorealistic full-body model combining 3 images of a user:
        - Face: preserve identity and expression
        - Torso: preserve body shape
        - Legs: preserve proportions
        - Combine seamlessly, natural posture, realistic lighting
        - Avoid cropping or deforming any part
        """

        # Preparar inputs
        contents = [
            prompt,
            types.Part.from_bytes(data=await Rostro.read(), mime_type=Rostro.content_type),
            types.Part.from_bytes(data=await Torso.read(), mime_type=Torso.content_type),
            types.Part.from_bytes(data=await Piernas.read(), mime_type=Piernas.content_type),
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=['TEXT','IMAGE'])
        )

        # Procesar respuesta
        image_data = None
        if response.candidates:
            parts = response.candidates[0].content.parts
            for part in parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    image_data = part.inline_data.data
                    image_mime_type = getattr(part.inline_data, "mime_type", "image/png")

        if image_data:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            image_url = f"data:{image_mime_type};base64,{image_base64}"
        else:
            image_url = None

        return JSONResponse(content={"image": image_url, "text": "Modelo corporal generado correctamente"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
