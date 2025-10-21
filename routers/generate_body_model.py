from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import base64, traceback, os, json
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
    user_data: str = Form(...),  # JSON string con medidas, género, estilo, etc.
    prendas: str = Form(...),    # JSON string con array de objetos {categoria, imagen_base64}
):
    """
    Genera una imagen de cuerpo completo del usuario combinando rostro, torso, piernas
    y superpone prendas seleccionadas según las categorías enviadas.
    """

    try:
        # Validaciones básicas
        MAX_IMAGE_SIZE_MB = 10
        ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

        for img in [Rostro, Torso, Piernas]:
            bytes_img = await img.read()
            if len(bytes_img)/(1024*1024) > MAX_IMAGE_SIZE_MB:
                raise HTTPException(status_code=400, detail=f"{img.filename} excede {MAX_IMAGE_SIZE_MB}MB")
            if img.content_type not in ALLOWED_MIME_TYPES:
                raise HTTPException(status_code=400, detail=f"Tipo no permitido: {img.content_type}")
            img.file.seek(0)

        # Parsear datos JSON
        try:
            user_info = json.loads(user_data)
            prendas_info = json.loads(prendas)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error en el formato de JSON: {str(e)}")

        # Construir descripción a partir del usuario y las prendas
        desc_usuario = (
            f"Genero: {user_info.get('genero', 'no especificado')}, "
            f"Estilo preferido: {user_info.get('estilo', 'casual')}, "
            f"Contextura: {user_info.get('contextura', 'media')}, "
            f"Tono de piel: {user_info.get('tono_piel', 'claro')}, "
            f"Altura: {user_info.get('altura', 'promedio')}."
        )

        prendas_descripcion = ", ".join([p.get("categoria", "") for p in prendas_info])
        prompt = f"""
        Genera un modelo de cuerpo completo del usuario combinando rostro, torso y piernas.
        Usa la siguiente información de referencia del usuario:
        {desc_usuario}

        Viste al modelo con las prendas proporcionadas: {prendas_descripcion}.
        Mantén proporciones naturales, buena iluminación y estilo fotográfico realista.
        Evita deformaciones, transparencias o errores anatómicos.
        """

        # Crear contenido para Gemini
        contents = [
            prompt,
            types.Part.from_bytes(data=await Rostro.read(), mime_type=Rostro.content_type),
            types.Part.from_bytes(data=await Torso.read(), mime_type=Torso.content_type),
            types.Part.from_bytes(data=await Piernas.read(), mime_type=Piernas.content_type),
        ]

        # Adjuntar las imágenes de las prendas
        for prenda in prendas_info:
            try:
                img_bytes = base64.b64decode(prenda["imagen_base64"].split(",")[-1])
                contents.append(
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                )
            except Exception as e:
                print(f"[⚠️] Error procesando prenda: {e}")

        # Petición a Gemini (2 imágenes)
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT','IMAGE'],
                candidate_count=2
            )
        )

        # Procesar respuesta
        images = []
        if response.candidates:
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        image_data = part.inline_data.data
                        mime_type = getattr(part.inline_data, "mime_type", "image/png")
                        base64_img = base64.b64encode(image_data).decode("utf-8")
                        images.append(f"data:{mime_type};base64,{base64_img}")

        if not images:
            raise HTTPException(status_code=500, detail="No se recibieron imágenes generadas del modelo.")

        return JSONResponse(
            content={
                "status": "ok",
                "images": images,
                "message": "Modelo corporal generado correctamente con las prendas seleccionadas."
            }
        )

    except Exception as e:
        print("❌ Error en generate_body_model:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
