from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import json
import os
import firebase_admin
from firebase_admin import credentials, firestore

# =========================
# Firestore setup
# =========================
if not firebase_admin._apps:
    cred = credentials.Certificate("path/to/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

router = APIRouter()

# =========================
# Prompt template
# =========================
IMAGE_TO_IMAGE_PROMPT = """
Use the provided base image strictly as reference for the SAME person.

IDENTITY LOCK:
- Preserve facial features, hair, skin tone, body proportions.

CLOTHING:
- Generate a modern, realistic outfit including top, bottoms, and shoes.
- Add subtle accessories (belt, bag, jewelry).
- Match the indicated style: {style}.
"""

# =========================
# Helper: asegurar PNG
# =========================
def ensure_png_upload(upload: UploadFile) -> BytesIO:
    try:
        image_bytes = upload.file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = "input.png"
        return buffer
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# =========================
# Endpoint
# =========================
@router.post("/generate-outfit/image-to-image")
async def generate_outfit_image_to_image(
    uid: str = Form(...),
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: str = Form("casual"),
    base_image_file: UploadFile = File(...),
):
    # -------------------------
    # Control de límite
    # -------------------------
    user_doc = db.collection("users").document(uid)
    data = user_doc.get().to_dict() or {}
    used_count = data.get("image_to_image_count", 0)
    if used_count >= 2:
        return {"status": "error", "message": "Límite alcanzado. Suscríbete a Wink Pro"}

    # -------------------------
    # Validar traits
    # -------------------------
    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    # -------------------------
    # Preparar imagen base
    # -------------------------
    base_image = ensure_png_upload(base_image_file)

    # -------------------------
    # OpenAI Client
    # -------------------------
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = IMAGE_TO_IMAGE_PROMPT.format(style=style)

    try:
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=base_image,
            prompt=prompt,
            n=1,
            size="512x512"
        )

        if not response.data:
            raise Exception("Empty image response")

        images_b64 = [response.data[0].b64_json]

        # -------------------------
        # Actualizar contador
        # -------------------------
        user_doc.set({"image_to_image_count": used_count + 1}, merge=True)

        return {
            "status": "ok",
            "images": images_b64,
            "traits_used": traits
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image-to-Image failed: {str(e)}")
