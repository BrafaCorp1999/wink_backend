from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
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
    # Leer imagen
    # -------------------------
    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    image_stream = BytesIO(image_bytes)

    # -------------------------
    # OpenAI
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

        return {
            "status": "ok",
            "images": images_b64,
            "traits_used": traits,
            "mode": "body_photo"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )
