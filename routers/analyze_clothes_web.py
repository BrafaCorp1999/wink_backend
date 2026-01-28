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
    buffer.seek(0)
    return buffer


def combine_clothes_prompt(descriptions: list[str]) -> str:
    return f"""
BASE IMAGE:
- STRICTLY DO NOT change face, body shape, proportions, height, weight or skin tone.
- Keep the same person identity.

GARMENTS TO APPLY:
{chr(10).join(f"- {d}" for d in descriptions)}

STRICT RULES:
- Replace ONLY the garments described above.
- Preserve exact colors, tones and garment types.
- Do NOT invent, recolor or ignore any garment.
- If two garments are provided, BOTH must be visible.
- Full body visible from head to toes.
- Realistic fashion photo.
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
        descriptions: list[str] = []

        for idx, img_b64 in enumerate(clothes_list):
            try:
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "Analyze this clothing item for virtual try-on.\n"
                                        "Describe ONLY visual characteristics.\n"
                                        "MANDATORY: include garment type and main color.\n"
                                        "Include fit/model (e.g. skinny, oversized), length, sleeves, texture.\n"
                                        "Do NOT mention brand, price or person."
                                    )
                                },
                                {
                                    "type": "input_image",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                )

                desc = response.output_text.strip()
                descriptions.append(desc)
                logging.info(f"[WEB][OK] Prenda {idx+1} descripción: {desc}")

            except Exception as e:
                logging.error(f"[WEB][ANALYSIS FAILED] Prenda {idx+1}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to analyze one clothing item"
                )

        # 🔴 SOLO PARA DEBUG (opcional)
        # Comenta generación si quieres solo validar descripciones
        base_img = prepare_image_from_b64(base_image_b64)
        combined_prompt = combine_clothes_prompt(descriptions)

        result = client.images.edit(
            model="gpt-image-1-mini",
            image=("base.png", base_img.read(), "image/png"),
            prompt=combined_prompt,
            size="1024x1024"
        )

        return {
            "status": "ok",
            "request_id": request_id,
            "description": descriptions,  # 👈 ahora ES ARRAY (mejor debug)
            "image": result.data[0].b64_json
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[COMBINE-CLOTHES-WEB][ERROR] {e}")
        raise HTTPException(status_code=500, detail="Combine clothes failed")
