from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import json
import os

router = APIRouter()

# =========================
# PROMPT – BODY PHOTO
# =========================

BODY_PHOTO_PROMPT = """
Use the uploaded full-body image strictly as a visual reference for the SAME real person.

IDENTITY LOCK:
- Preserve facial features (face shape, eyes, nose, lips, skin tone).
- Preserve hairstyle and hair color.
- Maintain body proportions, height, and body type.
- Do NOT alter identity or pose.

CLOTHING REPLACEMENT:
- Replace the original outfit entirely.
- Create a new {style} outfit.
- Include top, bottoms, shoes.
- Add subtle accessories if appropriate.
- Fabrics must look realistic with natural folds.
- Colors harmonious and modern.

POSE & COMPOSITION:
- Full-body, head to toe.
- Natural standing pose.
- Eye-level camera.
- Do NOT crop or distort anatomy.

LIGHTING & QUALITY:
- Clean studio or neutral background.
- Soft natural lighting.
- DSLR-quality, photorealistic.
- No illustration, no CGI.

OUTPUT:
- Generate 2 distinct outfit variations.
"""

# =========================
# UTIL: asegurar PNG válido
# =========================

def ensure_png_upload(upload: UploadFile) -> BytesIO:
    """
    Convierte cualquier imagen recibida a PNG válido
    con filename y mimetype aceptados por OpenAI.
    """
    try:
        image_bytes = upload.file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        # 🔥 CRÍTICO: nombre del archivo
        buffer.name = "input.png"

        return buffer

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )

# =========================
# ENDPOINT
# =========================

@router.post("/generate-outfits/body-photo")
async def generate_outfits_from_body_photo(
    gender: str = Form(...),
    body_traits: str = Form(...),   # JSON string desde analyze-body-with-face
    style: str = Form("casual"),
    image_file: UploadFile = File(...)
):
    # -------------------------
    # Validar traits
    # -------------------------
    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    # -------------------------
    # Preparar imagen válida
    # -------------------------
    image_stream = ensure_png_upload(image_file)

    # -------------------------
    # OpenAI Client
    # -------------------------
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.images.edit(
            model="gpt-image-1.5",
            image=image_stream,
            prompt=BODY_PHOTO_PROMPT.format(style=style),
            n=2,
            size="1024x1024"
        )

        images_b64 = [img.b64_json for img in response.data]

        if not images_b64:
            raise HTTPException(
                status_code=500,
                detail="No images generated"
            )

        return {
            "status": "ok",
            "mode": "body_photo",
            "images": images_b64,
            "traits_used": traits
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )
