from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
import base64
import json
import os

router = APIRouter()

# =========================
# PROMPT – SELFIE + MEDIDAS
# =========================
SELFIE_PROMPT = """
Use the provided information to generate a photorealistic full-body outfit
for the person described below.

BODY TRAITS:
- Height: {height_cm} cm
- Weight: {weight_kg} kg
- Waist: {waist_cm} cm
- Hips: {hips_cm} cm
- Shoulders: {shoulders_cm} cm
- Neck: {neck_cm} cm
- Body type: {body_type}

IDENTITY LOCK:
- Preserve facial features from the uploaded selfie.
- Maintain proportions according to body traits.
- Do NOT change pose or identity.

CLOTHING:
- Generate a complete outfit including top, bottoms, and shoes.
- Add subtle accessories if appropriate.
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
- Generate 2 distinct outfit variations.
"""

# =========================
# UTIL: asegurar PNG válido
# =========================
def ensure_png_upload(upload: UploadFile) -> BytesIO:
    try:
        image_bytes = upload.file.read()
        from PIL import Image
        from io import BytesIO
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

    images_b64: list[str] = []

    try:
        # 🔁 LOOP INTERNO PARA 2 VARIACIONES
        for i in range(2):
            variation_prompt = SELFIE_PROMPT.format(
                height_cm=traits.get("height_cm", "unknown"),
                weight_kg=traits.get("weight_kg", "unknown"),
                waist_cm=traits.get("waist_cm", "unknown"),
                hips_cm=traits.get("hips_cm", "unknown"),
                shoulders_cm=traits.get("shoulders_cm", "unknown"),
                neck_cm=traits.get("neck_cm", "unknown"),
                body_type=traits.get("body_type", "average"),
                style=style
            ) + f"\n\nOUTFIT VARIATION #{i+1}: Make this outfit clearly different from the previous one."

            response = client.images.generate(
                model="gpt-image-1.5",
                prompt=variation_prompt,
                n=1,               # siempre 1 imagen por iteración
                size="1024x1024"     # menos créditos y separadas
            )

            if not response.data:
                raise Exception("Empty image response")

            images_b64.append(response.data[0].b64_json)

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
