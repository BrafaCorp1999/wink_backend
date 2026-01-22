from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import os
import uuid

router = APIRouter()

# =========================
# PROMPT – BODY PHOTO (WEB)
# MISMO CONCEPTO QUE MÓVIL
# =========================
BODY_PHOTO_PROMPT_WEB = """
You are editing a fashion reference photo.

Preserve the person’s overall body shape, proportions, and facial structure.
Do not exaggerate or stylize the face or body.

Replace the outfit with a similar clothing category.
Ensure realistic fit and natural appearance.

Maintain a neutral, realistic photographic style.
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
# REQUEST MODEL (WEB)
# =========================
class BodyPhotoWebRequest(BaseModel):
    gender: str
    image_base64: str

# =========================
# ENDPOINT – BODY PHOTO (WEB)
# =========================
@router.post("/generate-outfits/body-photo-web")
async def generate_outfits_from_body_photo_web(data: BodyPhotoWebRequest):
    request_id = str(uuid.uuid4())
    print(f"[IMAGE_GEN_START][BODY_WEB] {request_id}")

    gender = normalize_gender(data.gender)

    # 🔹 Decodificar imagen base64
    try:
        image_bytes = base64.b64decode(data.image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        MAX_SIZE = 1024
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE))

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = "input.png"

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image base64: {str(e)}"
        )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    final_prompt = BODY_PHOTO_PROMPT_WEB + f"""

Additional context:
- Gender: {gender}
- Place the person in a similar type of environment, 
  with slight variation in background and surroundings.
- Keep lighting, perspective, and camera angle natural and realistic.
"""

    try:
        response = client.images.edit(
            model="gpt-image-1",
            image=buffer,
            prompt=final_prompt,
            size="auto"
        )

        if not response.data or not response.data[0].b64_json:
            raise Exception("Empty image response")

        print(f"[IMAGE_GEN_END][BODY_WEB] {request_id}")

        return {
            "status": "ok",
            "mode": "body_photo_web",
            "image": response.data[0].b64_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
