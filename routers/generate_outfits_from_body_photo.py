from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import json
import os
import uuid

router = APIRouter()

# =========================
# PROMPT – BODY PHOTO (MOBILE, ACTUALIZADO)
# =========================
BODY_PHOTO_PROMPT_MOBILE = """
You are editing a fashion reference photo.

IDENTITY & BODY LOCK (STRICT):
- Preserve face, hairstyle, skin tone, natural imperfections, freckles, and any unique marks.
- Do NOT change body size, posture, or volume.
- Keep all proportions exactly as in the original image.
- Do NOT smooth, whiten, or alter facial or skin features.

CLOTHING REPLACEMENT:
- Replace the outfit with a similar clothing category if needed.
- Ensure realistic fit and natural appearance.
- Do NOT alter any other clothing, skin, or body parts not specified.

STYLE & OUTPUT:
- Maintain a neutral, realistic photographic style.
- Full-body photo (head to feet visible)
- Soft natural lighting, no strong shadows
- Neutral background, no fantasy or CGI effects
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
# ENDPOINT – BODY PHOTO REGISTRATION (MOBILE)
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

    # Parse body traits (si lo usas después para registro o logs)
    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    gender = normalize_gender(gender)
    base_image = await ensure_png_upload(image_file)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    final_prompt = BODY_PHOTO_PROMPT_MOBILE + f"""
Additional context:
- Gender: {gender}
- Style: {style}
- Place the person in a similar type of environment as the original photo.
- Keep lighting, perspective, and camera angle natural and realistic.
"""

    try:
        response = client.images.edit(
            model="gpt-image-1",
            image=base_image,
            prompt=final_prompt,
            size="1024x1024"
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
