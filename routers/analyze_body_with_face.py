from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from google.genai import types
from utils.gemini_client import client
import cv2
import numpy as np
import base64
import traceback
import json

router = APIRouter(prefix="/analyze-body-with-face", tags=["AI Analyze Body+Face"])

@router.post("/")
async def analyze_body_with_face(person_image: UploadFile = File(...)):
    """
    Analiza el cuerpo y extrae rostro/base del usuario.
    Devuelve medidas estimadas + rostro recortado (base64).
    """
    try:
        # --- Validaciones ---
        ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
        user_bytes = await person_image.read()
        if person_image.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=400, detail="Unsupported image type")

        # --- IA analiza cuerpo para obtener medidas ---
        prompt = """
Analiza esta imagen de una persona de cuerpo entero y devuelve solo un JSON estructurado con:
{
  "gender": "male|female",
  "body_shape": "slim|average|curvy|muscular",
  "height_estimated": <cm>,
  "weight_estimated": <kg>,
  "waist_estimated": <cm>,
  "hips_estimated": <cm>,
  "shoulders_estimated": <cm>,
  "description": "breve descripción en español"
}
Sin texto adicional ni formato fuera del JSON.
"""

        contents = [
            prompt.strip(),
            types.Part.from_bytes(data=user_bytes, mime_type=person_image.content_type),
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=contents,
            config={"temperature": 0.3, "response_modalities": ["TEXT"]},
        )

        text_output = response.candidates[0].content.parts[0].text.strip()
        try:
            body_data = json.loads(text_output)
        except Exception:
            body_data = {"raw_text": text_output}

        # --- Detección y recorte de rostro ---
        np_img = np.frombuffer(user_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            raise HTTPException(status_code=404, detail="No se detectó rostro en la imagen")

        x, y, w, h = faces[0]
        margin_y = int(h * 0.4)
        y0 = max(0, y - margin_y)
        y1 = min(img.shape[0], y + h + margin_y)

        face_crop = img[y0:y1, x:x+w]
        _, buffer = cv2.imencode(".png", face_crop)
        face_base64 = base64.b64encode(buffer).decode("utf-8")
        face_data_uri = f"data:image/png;base64,{face_base64}"

        # --- Respuesta final ---
        return JSONResponse(content={
            "status": "ok",
            "message": "Cuerpo analizado y rostro extraído correctamente.",
            "body_data": body_data,
            "face_image_base64": face_data_uri
        })

    except Exception as e:
        print("❌ Error in /analyze-body-with-face:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
