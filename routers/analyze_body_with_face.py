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
    Analyze a full-body photo to estimate measurements and extract a clean, unaltered face crop.
    Returns:
        - JSON with body proportions (height, shoulders, waist, etc.)
        - Base64 PNG face crop (unaltered, natural colors)
    """
    try:
        ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
        user_bytes = await person_image.read()
        if person_image.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=400, detail="Unsupported image type")

        # === Gemini prompt (in English for better understanding) ===
        prompt = """
Analyze the provided full-body image of a single person.
Return a valid JSON object ONLY with the following structure, no explanations or extra text:

{
  "gender": "male" | "female",
  "body_shape": "slim" | "average" | "curvy" | "muscular",
  "height_estimated_cm": <integer>,        // approximate height in centimeters
  "weight_estimated_kg": <integer>,        // approximate weight in kilograms
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

        # === Face detection and cropping ===
        np_img = np.frombuffer(user_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            raise HTTPException(status_code=404, detail="No face detected in the image")

        # Pick the largest face (in case multiple are found)
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces[0]

        # Add margin for neck and upper hair area
        margin_y_top = int(h * 0.4)
        margin_y_bottom = int(h * 0.3)
        margin_x = int(w * 0.15)

        x0 = max(0, x - margin_x)
        x1 = min(img.shape[1], x + w + margin_x)
        y0 = max(0, y - margin_y_top)
        y1 = min(img.shape[0], y + h + margin_y_bottom)

        face_crop = img[y0:y1, x0:x1]

        # Ensure no color/style modifications
        _, buffer = cv2.imencode(".png", face_crop)
        face_base64 = base64.b64encode(buffer).decode("utf-8")
        face_data_uri = f"data:image/png;base64,{face_base64}"

        return JSONResponse(content={
            "status": "ok",
            "message": "Body successfully analyzed and unaltered face extracted.",
            "body_data": body_data,
            "face_image_base64": face_data_uri
        })

    except Exception as e:
        print("❌ Error in /analyze-body-with-face:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
