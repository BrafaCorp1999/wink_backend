from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import json
import os
import random

router = APIRouter()

# =========================
# PROMPT – BODY PHOTO (NO TOCADO)
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
- Outfit must match a random casual/modern or elegant style.
- Use harmonious colors and realistic fabrics.
- Clothing must fit naturally to the body.
- Optional subtle accessories (belt, bag, watch).

POSE & COMPOSITION:
- Full-body view, head to toe.
- Natural standing pose.
- Eye-level camera.
- Centered composition.

LIGHTING & QUALITY:
- Neutral or studio background that matches with the style taken.
- Soft natural lighting.
- DSLR-quality, photorealistic.
- No illustration, no CGI, no anime.

OUTPUT:
- One single realistic full-body image.
"""

# =========================
# NORMALIZAR GENDER
# =========================
def normalize_gender(value: str) -> str:
    value = value.lower().strip()
    if value in ("male", "man", "hombre"):
        return "male"
    if value in ("female", "woman", "mujer"):
        return "female"
    return "female"

# =========================
# UTIL: asegurar PNG válido (ASYNC + LIMIT)
# =========================
async def ensure_png_upload(upload: UploadFile) -> BytesIO:
    try:
        image_bytes = await upload.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Limitar tamaño (clave para iPhone)
        MAX_SIZE = 1024
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE))

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
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
    user_id: str = Form(...),
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: str = Form("casual"),
    image_file: UploadFile = File(...)
):
    import uuid
    request_id = str(uuid.uuid4())
    print(f"[IMAGE_GEN_START][BODY] {request_id}")

    # check_limit(user_id)

    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    gender = normalize_gender(gender)
    base_image = await ensure_png_upload(image_file)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    variation_prompt = BODY_PHOTO_PROMPT.format(style=style) + f"""
    Gender: {gender}
    Height: {traits.get('height_cm')}
    Weight: {traits.get('weight_kg')}
    Body type: {traits.get('body_type')}
    """

    try:
        response = client.images.generate(
            model="gpt-image-1-mini",
            prompt=variation_prompt,
            n=1,
            size="512x512"
        )

        if not response.data or not response.data[0].b64_json:
            raise Exception("Empty image response")

        print(f"[IMAGE_GEN_END][BODY] {request_id}")

        return {
            "status": "ok",
            "mode": "body_photo",
            "image": response.data[0].b64_json,
            "traits_used": traits
        }

    except Exception as e:
        raise HTTPException(500, detail=str(e))

