from fastapi import APIRouter, UploadFile, File, Form, HTTPException
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
def image_to_png(upload: UploadFile) -> BytesIO:
    img = Image.open(upload.file).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def combine_clothes_prompt(descriptions):
    # Combina 1-2 descripciones de prendas en un solo outfit coherente
    return f"""
Combine the following clothing items into a single full-body outfit, realistic photo:

{descriptions}

STRICT RULES:
- Same person, same face and body.
- Keep pose, lighting, and background unchanged.
- Full body visible from head to toes.
- Realistic fashion photo.
- Apply garments exactly as described.
"""

# =========================
# Endpoint MOBILE
# =========================
@router.post("/ai/combine-clothes")
async def combine_clothes(
    base_image_file: UploadFile = File(...),
    clothes_files: list[UploadFile] = File(...),
    gender: str = Form(...),
    style: str = Form(...),
    clothes_categories: str = Form(...)
):
    request_id = str(uuid.uuid4())
    logging.info(f"[COMBINE-CLOTHES-MOBILE] {request_id}")

    try:
        # Analizar cada prenda
        descriptions = []
        for cloth in clothes_files:
            img_b64 = base64.b64encode(image_to_png(cloth).read()).decode("utf-8")
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
        base_img = image_to_png(base_image_file)

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
        logging.error(f"[COMBINE-CLOTHES-MOBILE][ERROR] {e}")
        raise HTTPException(status_code=500, detail="Combine clothes generation failed")
