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
Edit this photo to change the person's outfit for a fashion application.

IMPORTANT – PRESERVE THE PERSON:
- Keep the face unchanged.
- Do not modify facial features, expression, or identity.
- Keep overall body shape natural; minor posture adjustments are acceptable.
- Avoid extreme edits to skin tone or physical traits.

OUTFIT REPLACEMENT:
- Replace the outfit with a similar type to the original.
- Maintain the same clothing category (e.g., jeans, dress, sneakers).
- Adjust colors, textures, or minor design details only.
- Clothes should fit naturally and realistically.
- Avoid fantasy, costume, or exaggerated fashion.

BACKGROUND:
- Keep a simple, relaxed environment (park, garden, calm urban area).
- Background should not distract from the person.

LIGHTING & QUALITY:
- Natural, soft lighting.
- Photorealistic full-body image.
- High-quality, DSLR-like photo.
- No illustrations, CGI, anime, or unrealistic edits.

OUTPUT:
- One realistic full-body image with ready for fashion visualization.
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
