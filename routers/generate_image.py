from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import base64, traceback, os, json
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

# --- Cargar API key y cliente ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

@router.post("/generate-image")
async def generate_image(
    person_image: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Genera dos imágenes realistas de cuerpo completo y una descripción corta en español.
    """
    try:
        # -------------------- VALIDACIONES --------------------
        MAX_MB = 10
        ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

        img_bytes = await person_image.read()
        if len(img_bytes) / (1024 * 1024) > MAX_MB:
            raise HTTPException(status_code=400, detail="Image exceeds 10MB limit.")
        if person_image.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=400, detail="Unsupported image type.")

        # -------------------- DEBUG --------------------
        print(f"📥 Prompt recibido: {prompt[:200]}...")  # mostrar primeros 200 caracteres
        print(f"📥 Tamaño de imagen: {len(img_bytes)/1024:.2f} KB, tipo: {person_image.content_type}")

        # -------------------- GEMINI GENERATION --------------------
        contents = [
            types.Part.from_bytes(data=img_bytes, mime_type=person_image.content_type),
            types.Part.from_text(prompt)  # ✅ Solo un argumento posicional
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",  # tu modelo anterior
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.7,
            ),
        )

        # -------------------- PARSE RESPONSE --------------------
        images_base64, descriptions = [], []

        if response.candidates:
            cand = response.candidates[0]
            for part in cand.content.parts:
                # Si viene imagen
                if getattr(part, "inline_data", None):
                    data = part.inline_data.data
                    mime = getattr(part.inline_data, "mime_type", "image/png")
                    images_base64.append(
                        f"data:{mime};base64,{base64.b64encode(data).decode()}"
                    )
                # Si viene texto
                elif getattr(part, "text", None):
                    text_str = part.text.strip()
                    # Separar por línea y traducir al español si quieres (simple ejemplo)
                    if text_str:
                        descriptions.append(text_str)

        # Validar cantidad de imágenes y descripciones
        if len(images_base64) < 2:
            print("⚠️ Se generó menos de 2 imágenes.")
        images_base64 = images_base64[:2]
        descriptions = descriptions[:2]
        if not descriptions:
            descriptions = ["Outfit generado.", "Outfit generado."]

        # -------------------- RETURN RESPONSE --------------------
        return JSONResponse(content={
            "success": True,
            "images": images_base64,
            "descriptions": descriptions
        })

    except Exception as e:
        print("❌ Error en /generate-image:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Error en /generate-image: {e}")
