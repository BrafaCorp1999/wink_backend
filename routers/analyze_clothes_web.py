from fastapi import APIRouter, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import os
import uuid
import logging
import json

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO)

# =========================
# Helpers
# =========================
def prepare_image_from_b64(image_b64: str, size=1024) -> BytesIO:
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((size, size))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.name = "input.png"
    buffer.seek(0)
    return buffer

def combine_clothes_prompt(descriptions):
    return f"""
Combine the following clothing items into a single full-body outfit, realistic photo:

{descriptions}

Rules:
- Same person, same face and body.
- Keep pose, lighting, and background unchanged.
- Full body visible from head to toes.
- Realistic fashion photo.
- Apply garments exactly as described.
"""

# =========================
# Endpoint WEB
# =========================
@router.post("/ai/combine-clothes-web")
async def combine_clothes_web(
    base_image_b64: str = Form(...),
    clothes_images_b64: str = Form(...),
    gender: str = Form(...),
    style: str = Form(...),
    clothes_categories: str = Form(...)
):
    request_id = str(uuid.uuid4())
    logging.info(f"[COMBINE-CLOTHES-WEB] {request_id}")

    try:
        # Analizar cada prenda
        clothes_list = json.loads(clothes_images_b64)
        descriptions = []

        for img_b64 in clothes_list:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Analyze this clothing item for virtual try-on.\n"
                                "Describe ONLY visual characteristics: type, colors, fit, length, sleeve/neckline, texture/pattern.\n"
                                "Do not mention brand or model."
                            )
                        },
                        {"type": "input_image", "image_base64": img_b64}
                    ]
                }]
            )
            descriptions.append(response.output_text.strip())

        combined_prompt = combine_clothes_prompt("\n".join(descriptions))

        # Preparar imagen base
        base_img = prepare_image_from_b64(base_image_b64)

        # Generar imagen final
        result = client.images.edit(
            model="gpt-image-1-mini",
            image=("base.png", base_img.read(), "image/png"),
            prompt=combined_prompt,
            size="1024x1024"
        )

        return {
            "status": "ok",
            "request_id": request_id,
            "description": "\n".join(descriptions),
            "image": result.data[0].b64_json
        }

    except Exception as e:
        logging.error(f"[COMBINE-CLOTHES-WEB][ERROR] {e}")
        raise HTTPException(status_code=500, detail="Combine clothes generation failed")
