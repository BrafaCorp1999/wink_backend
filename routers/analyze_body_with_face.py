# routers/analyze_body_with_face.py
import os
import base64
import logging
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np

router = APIRouter()
logger = logging.getLogger("analyze_body_with_face")
logging.basicConfig(level=logging.INFO)

@router.post("/analyze-body-with-face")
async def analyze_body_with_face(
    gender_hint: str = Form(...),
    person_image: UploadFile = File(...)
):
    """
    Analiza cuerpo + rostro para obtener medidas y bounding box de cara.
    Retorna imagen base64 y datos de referencia.
    """
    try:
        # Leer imagen
        contents = await person_image.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        height, width = img.shape[:2]

        # Bounding box simple para rostro (ejemplo con Haar Cascade, OpenCV)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        face_box = None
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_box = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

        # Convertir imagen a base64 para referencia (opcional)
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        response = {
            "status": "ok",
            "gender_hint": gender_hint,
            "image_base64": img_base64,
            "height": height,
            "width": width,
            "face_box": face_box
        }
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error analyzing body: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
