from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from google.genai import types
import base64
import traceback
import json
import numpy as np
import os

# Intentar importar cv2 (modo seguro)
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False
    print("⚠️ OpenCV no disponible — el recorte de rostro será omitido.")

# === Intentar importar el cliente Gemini central ===
try:
    from utils.gemini_client import client
except Exception:
    from google import genai
    from dotenv import load_dotenv

    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY in environment or .env file")

    client = genai.Client(api_key=GEMINI_API_KEY)
    print("⚠️ Usando cliente Gemini fallback desde analyze_body_with_face.py")

router = APIRouter(
    prefix="/analyze-body-with-face",
    tags=["AI Analyze Body+Face"]
)

@router.post("/")
async def analyze_body_with_face(person_image: UploadFile = File(...)):
    """Analiza una foto de cuerpo completo para estimar medidas y extraer rostro sin alterar colores."""
    try:
        ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
        user_bytes = await person_image.read()

        if person_image.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=400, detail="Unsupported image type")

        # === Prompt de análisis corporal ===
        prompt = """
Analyze the provided full-body image of a single person.
Return a valid JSON object ONLY with the following structure, no explanations or extra text:

{
  "gender": "male" | "female",
  "body_shape": "slim" | "average" | "curvy" | "muscular",
  "height_estimated_cm": <integer>,
  "weight_estimated_kg": <integer>,
  "shoulders_cm": <integer>,
  "chest_cm": <integer>,
  "waist_cm": <integer>,
  "hips_cm": <integer>,
  "body_description": "1 short English sentence describing posture and build."
}

The estimation should be realistic based on the visible body proportions, not idealized.
Do NOT include any text or symbols outside the JSON.
        """

        contents = [
            prompt.strip(),
            types.Part.from_bytes(data=user_bytes, mime_type=person_image.content_type),
        ]

        # === Llamada a Gemini ===
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=contents,
            config={"temperature": 0.2, "response_modalities": ["TEXT"]},
        )

        text_output = response.candidates[0].content.parts[0].text.strip()

        try:
            body_data = json.loads(text_output)
        except Exception:
            body_data = {"raw_output": text_output, "note": "Parsing issue — not strict JSON."}

        # === Face detection ===
        face_data_uri = None

        if OPENCV_AVAILABLE:
            np_img = np.frombuffer(user_bytes, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]

                margin_y_top = int(h * 0.4)
                margin_y_bottom = int(h * 0.3)
                margin_x = int(w * 0.15)

                x0 = max(0, x - margin_x)
                x1 = min(img.shape[1], x + w + margin_x)
                y0 = max(0, y - margin_y_top)
                y1 = min(img.shape[0], y + h + margin_y_bottom)

                face_crop = img[y0:y1, x0:x1]

                _, buffer = cv2.imencode(".png", face_crop)
                face_base64 = base64.b64encode(buffer).decode("utf-8")
                face_data_uri = f"data:image/png;base64,{face_base64}"

        return JSONResponse(content={
            "status": "ok",
            "message": "Body successfully analyzed.",
            "body_data": body_data,
            "face_image_base64": face_data_uri or "No face crop (OpenCV unavailable)"
        })

    except Exception as e:
        print("❌ Error in /analyze-body-with-face:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
