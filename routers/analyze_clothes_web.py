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
- Same person, same face and body, do not clarify the face, mantain face details and tone skin.
- Keep pose, lighting, and background change a little.
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
        clothes_list = json.loads(clothes_images_b64)
        categories_list = json.loads(clothes_categories)

        if len(clothes_list) != len(categories_list):
            raise HTTPException(status_code=400, detail="Mismatch images vs categories")

        descriptions = []

        for cat, img_b64 in zip(categories_list, clothes_list):
            try:
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    f"Analyze this clothing item for virtual try-on.\n"
                                    f"Category: {cat}\n"
                                    "Describe ONLY visual characteristics: type, color, fit, length, sleeve/neckline, texture/pattern.\n"
                                    "STRICTLY DO NOT mention brand, model, or change face/body/height/proportions.\n"
                                    "Focus on color and style details, e.g., skinny, oversize, pattern, texture, length."
                                )
                            },
                            {"type": "input_image", "image_base64": img_b64}
                        ]
                    }]
                )
                descriptions.append(f"{cat}: {response.output_text.strip()}")
            except Exception as e:
                logging.warning(f"[WEB] Responses API failed for one item: {e}")
                descriptions.append(f"{cat}: descripción simulada")  # fallback demo

        # Validación: si es vestido + zapatos no pasar de 2
        if "vestidos" in categories_list and len(categories_list) > 2:
            raise HTTPException(status_code=400, detail="Solo puedes combinar vestido + zapatos como máximo 2 prendas")

        # Prompt final
        combined_prompt = f"""
STRICT INSTRUCTIONS:
- DO NOT CHANGE FACE, BODY, HEIGHT, or PROPORTIONS.
- Keep pose, lighting, and try to change a little the background.
- Full body visible from head to toes.

Apply the following clothing changes exactly over the base image:

{chr(10).join(descriptions)}

Combine garments realistically into a full-body outfit, keeping person unchanged.
"""

        base_img = prepare_image_from_b64(base_image_b64)

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

