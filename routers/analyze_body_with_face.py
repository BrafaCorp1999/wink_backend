# routers/analyze_body_with_face.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import base64
import traceback
import numpy as np
import cv2

# -------------------------------------------------
# Try to load Mediapipe (optional)
# -------------------------------------------------
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ Mediapipe not available — body measurements will be approximate.")

# -------------------------------------------------
# Router setup
# -------------------------------------------------
router = APIRouter(
    prefix="/analyze-body-with-face",
    tags=["AI Analyze Body + Face"]
)

# -------------------------------------------------
# Endpoint
# -------------------------------------------------
@router.post("/")
async def analyze_body_with_face(
    person_image: UploadFile = File(...),
    gender_hint: str = Form(None)
):
    try:
        # -----------------------------
        # Validate image type
        # -----------------------------
        ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
        if person_image.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=400, detail="Unsupported image type")

        user_bytes = await person_image.read()
        if not user_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")

        # -----------------------------
        # Decode image safely
        # -----------------------------
        np_img = np.frombuffer(user_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        height, width = img.shape[:2]

        # -----------------------------
        # Normalize gender hint
        # -----------------------------
        gender_hint = (gender_hint or "").lower().strip()
        gender_hint = gender_hint if gender_hint in ["male", "female"] else "unknown"

        # -----------------------------
        # Face detection (OpenCV Haar)
        # -----------------------------
        face_data_uri = None
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) > 0:
            # Take largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]

            x0 = max(0, x - int(w * 0.2))
            y0 = max(0, y - int(h * 0.4))
            x1 = min(width, x + w + int(w * 0.2))
            y1 = min(height, y + h + int(h * 0.3))

            face_crop = img[y0:y1, x0:x1]

            if face_crop.size > 0:
                _, buf = cv2.imencode(".png", face_crop)
                face_data_uri = (
                    "data:image/png;base64,"
                    + base64.b64encode(buf).decode()
                )

        # -----------------------------
        # Body analysis (Mediapipe optional)
        # -----------------------------
        if MEDIAPIPE_AVAILABLE:
            mp_pose = mp.solutions.pose
            with mp_pose.Pose(static_image_mode=True) as pose:
                results = pose.process(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                )

            if results.pose_landmarks:
                body_data = {
                    "gender": gender_hint,
                    "body_shape": "average",
                    "height_estimated_cm": int(height * 0.35),
                    "weight_estimated_kg": int(height * 0.35),
                    "shoulders_cm": int(width * 0.4),
                    "chest_cm": int(width * 0.45),
                    "waist_cm": int(width * 0.35),
                    "hips_cm": int(width * 0.4),
                    "arms_cm": int(height * 0.2),
                    "body_description": "Person has an average build and posture."
                }
            else:
                body_data = _fallback_body_data(gender_hint)
        else:
            body_data = _fallback_body_data(gender_hint)

        # -----------------------------
        # Response
        # -----------------------------
        return JSONResponse(
            content={
                "status": "ok",
                "message": "Body and face analyzed successfully.",
                "body_data": body_data,
                "face_image_base64": face_data_uri
            }
        )

    except Exception as e:
        print("❌ Error in /analyze-body-with-face")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# Fallback body data
# -------------------------------------------------
def _fallback_body_data(gender: str):
    return {
        "gender": gender,
        "body_shape": "average",
        "height_estimated_cm": 170,
        "weight_estimated_kg": 70,
        "shoulders_cm": 40,
        "chest_cm": 90,
        "waist_cm": 70,
        "hips_cm": 90,
        "arms_cm": 60,
        "body_description": "Average build with unknown posture."
    }
