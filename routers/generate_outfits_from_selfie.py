from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
import base64
import json
import os
from PIL import Image

router = APIRouter()

# =========================
# PROMPT – SELFIE + MEDIDAS
# =========================
SELFIE_PROMPT = """
Use the provided full-body selfie as reference for the SAME real person.

IDENTITY LOCK:
- Preserve facial features (face shape, eyes, nose, lips, skin tone).
- Maintain proportions according to body traits.
- Do NOT change pose or identity.

BODY TRAITS:
- Height: {height_cm} cm
- Weight: {weight_kg} kg
- Waist: {waist_cm} cm
- Hips: {hips_cm} cm
- Shoulders: {shoulders_cm} cm
- Neck: {neck_cm} cm
- Body type: {body_type}

CLOTHING:
- Generate a realistic full-body outfit including top, bottoms, and shoes.
- Add subtle accessories (belt, bag, jewelry) to make the outfit distinct.
- Use colors, patterns, and fabrics clearly different from previous outfits or the person's original outfit.
- Fabrics must look realistic with natural folds.
- Colors harmonious and modern.
- Outfit should match the indicated style: {style}.

POSE & COMPOSITION:
- Full-body, head to toe including feet.
- Natural standing pose.
- Eye-level camera.
- Do NOT crop or distort anatomy.

LIGHTING & QUALITY:
- Soft, natural lighting.
- DSLR-quality, photorealistic.
- No illustration or CGI.

OUTPUT:
- Generate a single, realistic outfit.
- Ensure the outfit looks visually distinct from any previous outfit for this person.
"""


# =========================
# UTIL: asegurar PNG válido
# =========================
def ensure_png_upload(upload: UploadFile) -> BytesIO:
    try:
        image_bytes = upload.file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = "selfie.png"
        return buffer
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# =========================
# ENDPOINT
# =========================
@router.post("/generate-outfits/selfie")
async def generate_outfits_from_selfie(
    gender: str = Form(...),
    body_traits: str = Form(...),   # JSON string con medidas manuales
    style: str = Form("casual"),
    selfie_file: UploadFile = File(...)
):
    # -------------------------
    # Validar traits
    # -------------------------
    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    # -------------------------
    # Preparar imagen selfie
    # -------------------------
    base_image = ensure_png_upload(selfie_file)

    # -------------------------
    # OpenAI Client
    # -------------------------
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        # 🔁 Solo 1 imagen para demo
        variation_prompt = SELFIE_PROMPT.format(
            height_cm=traits.get("height_cm", "unknown"),
            weight_kg=traits.get("weight_kg", "unknown"),
            waist_cm=traits.get("waist_cm", "unknown"),
            hips_cm=traits.get("hips_cm", "unknown"),
            shoulders_cm=traits.get("shoulders_cm", "unknown"),
            neck_cm=traits.get("neck_cm", "unknown"),
            body_type=traits.get("body_type", "average"),
            style=style
        )

        response = client.images.generate(
            model="gpt-image-1-mini",  # modelo barato para demo
            prompt=variation_prompt,
            n=1,               # solo 1 imagen
            size="512x512"     # tamaño mínimo suficiente para demo
        )

        if not response.data:
            raise Exception("Empty image response")

        images_b64 = [response.data[0].b64_json]

        return {
            "status": "ok",
            "mode": "selfie_manual",
            "images": images_b64,
            "traits_used": traits
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )
