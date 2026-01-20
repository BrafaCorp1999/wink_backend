from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import json
import os
import uuid

router = APIRouter()

# =========================
# PROMPT – BODY PHOTO (CLONADO CORPORAL)
# =========================
BODY_PHOTO_PROMPT = """
You are editing a real photograph of a real person for a fashion application.

CRITICAL – DO NOT CHANGE THE PERSON:
- The body must remain EXACTLY the same as in the original image.
- Do NOT change body proportions, size, height, weight, or measurements.
- Do NOT slim, widen, lengthen, shorten, or stylize the body.
- Do NOT modify posture significantly.
- Do NOT modify facial features, face shape, expression, or identity.
- Do NOT change skin tone or physical traits.

POSE:
- Keep a subtle fashion modeling posture.
- Natural and relaxed stance.
- Slightly confident posture, minimal arm movement.
- No dramatic or exaggerated pose.

OUTFIT REPLACEMENT (VERY IMPORTANT):
- Replace the outfit with a SIMILAR outfit type to the original.
- Maintain the same clothing category:
  - If the person wears jeans → generate jeans.
  - If the person wears a dress → generate a dress.
  - If the person wears sneakers → generate sneakers.
- Only change colors, textures, or minor design details.
- Clothing must fit the body naturally and realistically.
- No extreme fashion, no costume, no runway looks.

BACKGROUND:
- Softly blurred, relaxed environment (park, garden, calm urban area).
- Background must NOT be empty.
- Background must NOT distract from the person.

LIGHTING & QUALITY:
- Natural soft lighting.
- Photorealistic.
- DSLR-quality photo.
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
# UTIL: asegurar PNG válido
# =========================
async def ensure_png_upload(upload: UploadFile) -> BytesIO:
    try:
        image_bytes = await upload.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

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
# ENDPOINT – BODY PHOTO REGISTRATION
# =========================
@router.post("/generate-outfits/body-photo")
async def generate_outfits_from_body_photo(
    user_id: str = Form(...),
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: str = Form("casual"),
    image_file: UploadFile = File(...)
):
    request_id = str(uuid.uuid4())
    print(f"[IMAGE_GEN_START][BODY] {request_id}")

    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    gender = normalize_gender(gender)
    base_image = await ensure_png_upload(image_file)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    final_prompt = BODY_PHOTO_PROMPT + f"""
Additional context:
- Gender: {gender}
- Style preference: {style}
"""

    try:
        response = client.images.edits(
            model="gpt-image-1",
            image=base_image,
            prompt=final_prompt,
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
        raise HTTPException(status_code=500, detail=str(e))
