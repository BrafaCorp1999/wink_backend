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
Create a photorealistic full-body image of a real person based on the following physical traits.

IDENTITY CONSISTENCY:
- The person must look realistic and human.
- Maintain consistent facial structure and skin tone.
- Hair style and hair color must remain natural and coherent.
- Body proportions must match the described measurements.
- Do NOT exaggerate or stylize the body.

BODY & PROPORTIONS:
- Respect height, weight, waist, hips, shoulders and body type.
- Natural anatomy with correct limb length.
- No deformation or unrealistic shapes.

OUTFIT:
- Generate a modern, realistic outfit including top, bottoms and shoes.
- Outfit must match the style: {style}.
- Use harmonious colors and realistic fabrics.
- Clothing must fit naturally to the body.
- Optional subtle accessories (belt, bag, watch).

POSE & COMPOSITION:
- Full-body view, head to toe.
- Natural standing pose.
- Eye-level camera.
- Centered composition.

LIGHTING & QUALITY:
- Neutral or studio background.
- Soft natural lighting.
- DSLR-quality, photorealistic.
- No illustration, no CGI, no anime.

OUTPUT:
- One single realistic full-body image.
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
        buffer.name = "input.png"
        return buffer
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

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
    # Preparar imagen base
    # -------------------------
    base_image = ensure_png_upload(image_file)

    # -------------------------
    # OpenAI Client
    # -------------------------
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        variation_prompt = (
            BODY_PHOTO_PROMPT.format(style=style)
            + "\n\nUSER BODY TRAITS:\n"
            f"- Gender: {gender}\n"
            f"- Height: {traits.get('height_cm', 'unknown')} cm\n"
            f"- Weight: {traits.get('weight_kg', 'unknown')} kg\n"
            f"- Waist: {traits.get('waist_cm', 'unknown')} cm\n"
            f"- Hips: {traits.get('hips_cm', 'unknown')} cm\n"
            f"- Shoulders: {traits.get('shoulders_cm', 'unknown')} cm\n"
            f"- Neck: {traits.get('neck_cm', 'unknown')} cm\n"
            f"- Body type: {traits.get('body_type', 'average')}\n"
        )

        # Opcional: edad y cabello si vienen en traits
        if 'age' in traits:
            variation_prompt += f"- Age: {traits['age']}\n"
        if 'hair_length' in traits:
            variation_prompt += f"- Hair length: {traits['hair_length']}\n"
        if 'hair_type' in traits:
            variation_prompt += f"- Hair type: {traits['hair_type']}\n"

        response = client.images.generate(
        model="gpt-image-1-mini",
        prompt=variation_prompt,
        n=1,
        size="512x512"
        )

        if not response.data:
            raise Exception("Empty image response")

        images_b64 = [response.data[0].b64_json]

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
