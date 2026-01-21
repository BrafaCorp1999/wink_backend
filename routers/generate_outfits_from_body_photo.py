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
BODY_PHOTO_PROMPT_MODERADO = """
You are editing a real photograph of a person for a fashion application.

IMPORTANT:
- Preserve the person's body exactly as in the original image.
- Do not alter height, weight, body proportions, or measurements.
- Keep facial features, expression, and identity unchanged.
- Maintain natural skin tone and physical traits.

POSE:
- Natural and relaxed stance, minimal movement.
- Subtle fashion pose only.

OUTFIT:
- Replace the outfit with a similar one of the same type.
- Adjust colors or small design details only.
- Clothes must fit naturally and realistically.
- Avoid extreme fashion or costumes.

BACKGROUND & LIGHTING:
- Soft, relaxed environment (park, garden, calm urban area).
- Natural lighting, photorealistic.
- No illustrations, anime, CGI, or fantasy elements.

OUTPUT:
- One realistic full-body image with the new outfit, maintaining the person's identity and body.
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
        response = client.images.edit(
            model="gpt-image-1",
            image=base_image,
            prompt=final_prompt,
            size="auto"
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
