from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
import json
import os
from PIL import Image

router = APIRouter()

# =========================
# PROMPT – SELFIE + MANUAL DATA
# =========================
SELFIE_PROMPT = """
Use the provided selfie image as the PRIMARY visual reference of the SAME real person.

IDENTITY LOCK (CRITICAL):
- Preserve the exact facial features, face shape, skin tone and expression.
- Do NOT beautify, stylize or modify the face.
- Keep hairstyle and hair color consistent.
- Body proportions must remain realistic and natural.

BODY CONSISTENCY:
- Respect the person's natural body structure.
- Do NOT exaggerate curves, muscles or height.
- Measurements are only a soft guideline, NOT a replacement of the image.

OUTFIT:
- Generate a realistic, modern outfit suitable for everyday fashion.
- Style should feel trendy but natural (not extreme, not runway).
- Outfit must be different from the original clothing.
- Use realistic fabrics, natural folds and correct fitting.
- Colors should be harmonious and fashionable.
- Optional minimal accessories (small bag, watch, belt).

POSE & EXPRESSION:
- Full-body view, head to toe.
- Light fashion-model pose:
  - Relaxed posture
  - Subtle weight shift
  - Arms naturally positioned (not hanging stiffly)
- Natural, calm expression.

ENVIRONMENT:
- Soft, realistic outdoor or lifestyle background.
- Examples: park, garden, urban green area, relaxed public space.
- Background must be lightly blurred and subtle.
- Do NOT dominate the image.

LIGHTING & QUALITY:
- Soft natural daylight.
- Photorealistic DSLR quality.
- No CGI, no illustration, no anime.

OUTPUT:
- Generate exactly ONE photorealistic full-body image.
"""

# =========================
# NORMALIZE TRAITS (SOFT ONLY)
# =========================
def normalize_traits(traits: dict, gender: str) -> dict:
    return {
        "height_cm": traits.get("height_cm"),
        "weight_kg": traits.get("weight_kg"),
        "body_type": traits.get("body_type", "average"),
    }

# =========================
# UTIL: ensure PNG (ASYNC + SAFE)
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
        buffer.name = "selfie.png"
        return buffer

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )

# =========================
# ENDPOINT
# =========================
@router.post("/generate-outfits/selfie")
async def generate_outfits_from_selfie(
    user_id: str = Form(...),
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: str = Form("modern"),
    selfie_file: UploadFile = File(...)
):
    import uuid
    request_id = str(uuid.uuid4())
    print(f"[IMAGE_GEN_START][SELFIE] {request_id}")

    try:
        raw_traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    traits = normalize_traits(raw_traits, gender)
    base_image = await ensure_png_upload(selfie_file)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.images.generate(
            model="gpt-image-1-mini",
            prompt=SELFIE_PROMPT,
            image=base_image,
            n=1,
            size="auto"
        )

        if not response.data or not response.data[0].b64_json:
            raise Exception("Empty image response")

        print(f"[IMAGE_GEN_END][SELFIE] {request_id}")

        return {
            "status": "ok",
            "mode": "selfie_manual",
            "image": response.data[0].b64_json,
            "traits_used": traits
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
