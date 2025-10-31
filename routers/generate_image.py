from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import base64, os, traceback, json

router = APIRouter()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@router.post("/generate-image")
async def generate_image(
    person_image: UploadFile = File(...),
    prompt: str = Form(...),
):
    try:
        # Leer imagen
        img_bytes = await person_image.read()

        # 🧠 Prompt final (claro y enfocado)
        final_prompt = f"""
        {prompt}

        Usa la imagen proporcionada como referencia del rostro y cuerpo.
        No deformes el rostro, ni cambies el tono de piel o las proporciones del cuerpo.
        Genera exactamente dos imágenes de cuerpo completo en diferentes combinaciones de atuendos,
        y una descripción breve en español (máx 20 palabras) sobre el estilo general.
        """

        # 🚀 Enviar a tu modelo (ej. "gemini-1.5-flash" o tu versión previa)
        response = client.models.generate_content(
            model="gemini-1.5-flash",  # ✅ Cambia aquí por tu modelo exacto (ej: "gemini-1.0-pro-vision" o similar)
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type=person_image.content_type),
                types.Part.from_text(final_prompt)
            ],
            generation_config=types.GenerationConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "images": {
                            "type": "array",
                            "items": {"type": "string", "description": "Base64 image data URIs"}
                        },
                        "description": {"type": "string"}
                    },
                    "required": ["images", "description"]
                }
            )
        )

        raw_output = response.text
        print("🟩 RAW OUTPUT:", raw_output)

        # Intentar parsear JSON seguro
        try:
            data = json.loads(raw_output)
        except Exception as e:
            print("⚠️ Error parseando JSON:", e)
            data = {}

        if not data or "images" not in data:
            raise HTTPException(status_code=500, detail="El modelo no devolvió imágenes válidas.")

        # Validar formato
        if len(data["images"]) == 0:
            raise HTTPException(status_code=500, detail="No se generaron imágenes.")

        return JSONResponse(content=data)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Error en /generate-image: {str(e)}")
