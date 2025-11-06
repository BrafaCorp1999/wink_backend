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
    try:
        # Leer bytes de la imagen enviada
        image_bytes = await full_body.read()

        # --- PROMPT para escaneo facial + corporal ---
        prompt = """
You are an AI specialized in facial-body mapping for fashion.
Analyze the given full-body image and:
1. Detect and crop the face region (keep natural proportions, no edits).
2. Estimate approximate body measurements (height, shoulders, chest, waist, hips).
3. Return a JSON structure like:
{
 "measurements": {"height_cm": ..., "chest_cm": ..., "waist_cm": ..., "hips_cm": ..., "shoulders_cm": ...},
 "contexture": "slim/average/athletic/plus",
 "notes": "optional details"
}
Also return the cropped face as a base64 image.
Do NOT generate new clothes or modify appearance.
"""

        # Llamada a Gemini
        result = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[
                types.Part.from_text(prompt),
                types.Part.from_bytes(image_bytes, mime_type=full_body.content_type),
            ],
        )

        text_response = result.text.strip()
        print("🧠 AI Raw Response:", text_response)

        # Extraer JSON del resultado
        json_start = text_response.find("{")
        json_end = text_response.rfind("}") + 1
        json_part = text_response[json_start:json_end]
        data_json = json.loads(json_part)

        # Obtener imagen del rostro
        face_base64 = data_json.get("face_image")
        if not face_base64:
            raise HTTPException(status_code=400, detail="Face image not returned.")

        face_bytes = base64.b64decode(face_base64)

        # Guardar en Storage
        face_blob = bucket.blob(f"users/{user_id}/face.png")
        face_blob.upload_from_string(face_bytes, content_type="image/png")
        face_url = face_blob.public_url

        full_blob = bucket.blob(f"users/{user_id}/full_body.png")
        full_blob.upload_from_string(image_bytes, content_type=full_body.content_type)
        full_url = full_blob.public_url

        # Guardar en Firestore
        model_data = {
            "user_id": user_id,
            "model_type": "scan_body",
            "face_ref": face_url,
            "full_body_ref": full_url,
            "measurements": data_json.get("measurements", {}),
            "contexture": data_json.get("contexture", "average"),
            "created_at": firestore.SERVER_TIMESTAMP,
        }

        db.collection("user_models").document(user_id).set(model_data)

        return JSONResponse(
            {
                "status": "ok",
                "message": "Base model registered successfully",
                "face_url": face_url,
                "measurements": data_json.get("measurements"),
                "contexture": data_json.get("contexture"),
            }
        )

    except Exception as e:
        print("❌ Error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
