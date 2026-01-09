from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
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
- Preserve facial features, skin tone, hairstyle and proportions.
- Do NOT change identity, face shape or body structure.

BODY TRAITS:
- Height: {height_cm} cm
- Weight: {weight_kg} kg
- Waist: {waist_cm}
- Hips: {hips_cm}
- Shoulders: {shoulders_cm}
- Neck: {neck_cm}
- Body type: {body_type}

CLOTHING:
- Generate a realistic full-body outfit (top, bottoms, shoes).
- Outfit must be visually different from original clothing.
- Use modern colors, realistic fabrics and natural folds.
- Add subtle accessories if appropriate.
- Chose a random style casual/modern/elegant.

POSE & COMPOSITION:
- Full-body, head to feet.
- Natural standing pose.
- Eye-level camera.
- No cropping or distortion.

LIGHTING & QUALITY:
- Soft natural lighting.
- Photorealistic.
- Try to match the place with the style seleted not more realistic but with some details that seems to be in a place.
- No illustration, no CGI.

OUTPUT:
- Generate exactly ONE realistic outfit image.
"""

# =========================
# UTIL: normalizar traits
# =========================
def normalize_traits(traits: dict, gender: str) -> dict:
    return {
        "height_cm": traits.get("height_cm") or (175 if gender == "male" else 165),
        "weight_kg": traits.get("weight_kg") or (70 if gender == "male" else 60),
        "waist_cm": traits.get("waist_cm", "unknown"),
        "hips_cm": traits.get("hips_cm", "unknown"),
        "shoulders_cm": traits.get("shoulders_cm", "unknown"),
        "neck_cm": traits.get("neck_cm", "unknown"),
        "body_type": traits.get("body_type", "average"),
    }

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
    user_id: str = Form(...),
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: str = Form("casual"),
    selfie_file: UploadFile = File(...)
):
    import uuid
    request_id = str(uuid.uuid4())
    print(f"[IMAGE_GEN_START][SELFIE] {request_id}")

    # check_limit(user_id)

    try:
        raw_traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    traits = normalize_traits(raw_traits, gender)
    base_image = ensure_png_upload(selfie_file)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = SELFIE_PROMPT.format(
        height_cm=traits["height_cm"],
        weight_kg=traits["weight_kg"],
        waist_cm=traits["waist_cm"],
        hips_cm=traits["hips_cm"],
        shoulders_cm=traits["shoulders_cm"],
        neck_cm=traits["neck_cm"],
        body_type=traits["body_type"],
        style=style
    )

    try:
        response = client.images.generate(
            model="gpt-image-1-mini",
            prompt=prompt,
            image=base_image,
            n=1,
            size="512x512"
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
        raise HTTPException(500, detail=str(e))
