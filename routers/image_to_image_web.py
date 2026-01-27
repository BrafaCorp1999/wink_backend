from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import os

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Request schema
# =========================
class GenerateOutfitWebRequest(BaseModel):
    image_base64: str
    description: str

# =========================
# Helper
# =========================
def resize_image(image_b64: str, size=1024) -> str:
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((size, size))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")

# =========================
# Endpoint WEB
# =========================
@router.post("/ai/generate-outfit-from-form-web")
async def generate_outfit_from_form_web(data: GenerateOutfitWebRequest):
    try:
        base_image = resize_image(data.image_base64)

        prompt = f"""
You are a virtual fashion stylist.
Preserve the person exactly as they are.
Do NOT change body, face, pose or proportions.

Apply a realistic outfit based on this description:
{data.description}

High quality fashion editorial style.
"""

        # 1️⃣ Generate image
        image_result = client.images.edit(
            model="gpt-image-1-mini",
            image=base_image,
            prompt=prompt,
            size="1024x1024"
        )

        generated_image = image_result.data[0].b64_json

        # 2️⃣ Generate text
        text_prompt = """
Describe the applied outfit briefly.
Focus on colors, style and occasion.
"""

        text_result = client.responses.create(
            model="gpt-4.1-mini",
            input=text_prompt
        )

        recommendation = text_result.output_text.strip()

        return {
            "status": "ok",
            "image": generated_image,
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
