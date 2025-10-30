from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from google.genai import types
import base64, traceback, json, numpy as np, os

try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False
    print("⚠️ OpenCV no disponible — el recorte de rostro será omitido.")

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

router = APIRouter(prefix="/analyze-body-with-face", tags=["AI Analyze Body+Face"])


@router.post("/")
async def analyze_body_with_face(person_image: UploadFile = File(...)):
    """Analiza una foto completa: rostro + cuerpo, devolviendo rostro separado y medidas aproximadas."""
    try:
        ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
        user_bytes = await person_image.read()

        if person_image.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=400, detail="Unsupported image type")

        # 🧠 Prompt robusto para análisis corporal + postura
        prompt = """
Analyze the provided full-body image of a person. Return ONLY valid JSON, with no explanations.
Structure:
{
  "gender": "male" | "female",
  "body_shape": "slim" | "average" | "curvy" | "muscular",
  "height_estimated_cm": int,
  "weight_estimated_kg": int,
  "shoulders_cm": int,
  "chest_cm": int,
  "waist_cm": int,
  "hips_cm": int,
  "arms_cm": int,
  "body_description": "1 short English sentence about posture and build"
}
Ensure realism, not idealized measurements.
"""

        contents = [
            prompt.strip(),
            types.Part.from_bytes(data=user_bytes, mime_type=person_image.content_type),
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=contents,
            config={"temperature": 0.2, "response_modalities": ["TEXT"]},
        )

        raw_text = response.candidates[0].content.parts[0].text.strip()

        try:
            body_data = json.loads(raw_text)
        except Exception:
            body_data = {"raw_output": raw_text, "note": "Could not parse clean JSON"}

        # 🧍‍♀️ Recorte de rostro con OpenCV
        face_data_uri = None
        if OPENCV_AVAILABLE:
            np_img = np.frombuffer(user_bytes, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]

                # Margen para capturar cabello y parte del cuello
                x0 = max(0, x - int(w * 0.2))
                y0 = max(0, y - int(h * 0.4))
                x1 = min(img.shape[1], x + w + int(w * 0.2))
                y1 = min(img.shape[0], y + h + int(h * 0.3))

                face_crop = img[y0:y1, x0:x1]
                _, buf = cv2.imencode(".png", face_crop)
                face_data_uri = f"data:image/png;base64,{base64.b64encode(buf).decode()}"

        return JSONResponse(content={
            "status": "ok",
            "message": "Body and face analyzed successfully.",
            "body_data": body_data,
            "face_image_base64": face_data_uri or None
        })

    except Exception as e:
        print("❌ Error in /analyze-body-with-face:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
