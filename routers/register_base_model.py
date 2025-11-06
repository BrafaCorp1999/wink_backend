# routes/register_base_model.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import base64, json, os, traceback
from dotenv import load_dotenv
from firebase_admin import firestore, storage

load_dotenv()
router = APIRouter()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

db = firestore.client()
bucket = storage.bucket()

@router.post("/register_base_model")
async def register_base_model(
    user_id: str = Form(...),
    full_body: UploadFile = File(...),
):
    """
    Scans a full-body image to obtain the user's face and detailed body measurements.
    Returns: cropped face image in base64, body measurements, and body type.
    """
    try:
        image_bytes = await full_body.read()

        # ---------------------- PROMPT ----------------------
        prompt = """
You are an expert AI in body analysis and facial recognition.
Analyze the following full-body image and return the following:

1. A cropped image of the face (do not distort, no filters, neutral background).
2. A set of body measurements in centimeters in the following JSON format:
{
 "measurements": {
    "height_cm": ...,
    "shoulders_cm": ...,
    "chest_cm": ...,
    "waist_cm": ...,
    "hips_cm": ...,
    "thigh_cm": ...,
    "calf_cm": ...,
    "leg_length_cm": ...,
    "neck_cm": ...,
    "torso_cm": ...
 },
 "body_type": "slim | normal | athletic | plus",
 "notes": "optional details or estimates"
}
Measurements must be realistic and proportional for a human body.

3. Also return the cropped face image in base64 (field name: "face_base64").
Do not generate clothes or change posture.
"""

        result = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[
                types.Part.from_text(prompt),
                types.Part.from_bytes(image_bytes, mime_type=full_body.content_type),
            ],
        )

        text_response = result.text.strip()
        print("🧠 AI Raw Response:", text_response[:300])

        # Extract the JSON from the response
        json_start = text_response.find("{")
        json_end = text_response.rfind("}") + 1
        json_part = text_response[json_start:json_end]
        data_json = json.loads(json_part)

        # Validate fields
        measurements = data_json.get("measurements", {})
        body_type = data_json.get("body_type", "normal")
        face_b64 = data_json.get("face_base64")

        if not face_b64:
            raise HTTPException(status_code=400, detail="No face_base64 returned.")

        # Save face and full-body image to Cloud Storage
        face_bytes = base64.b64decode(face_b64)
        face_blob = bucket.blob(f"users/{user_id}/face.png")
        face_blob.upload_from_string(face_bytes, content_type="image/png")
        face_url = face_blob.public_url

        full_blob = bucket.blob(f"users/{user_id}/full_body.png")
        full_blob.upload_from_string(image_bytes, content_type=full_body.content_type)
        full_url = full_blob.public_url

        # Save to Firestore
        model_data = {
            "user_id": user_id,
            "base_model": True,
            "face_url": face_url,
            "full_body_url": full_url,
            "measurements": measurements,
            "body_type": body_type,
            "created_at": firestore.SERVER_TIMESTAMP,
        }

        db.collection("user_models").document(user_id).set(model_data)

        return JSONResponse({
            "status": "ok",
            "message": "Base model registered successfully",
            "face_url": face_url,
            "measurements": measurements,
            "body_type": body_type
        })

    except Exception as e:
        print("❌ Error in /register_base_model:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
