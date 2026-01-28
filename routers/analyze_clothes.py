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
    return f"""
Combine the following clothing items into a single full-body outfit, realistic photo:

{descriptions}

STRICT RULES:
- Same person, same face and body, do not clarify the face, mantain face details and tone skin.
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
        categories_list = json.loads(clothes_categories)

        if len(clothes_files) != len(categories_list):
            raise HTTPException(status_code=400, detail="Mismatch images vs categories")

        descriptions = []

        for cat, cloth in zip(categories_list, clothes_files):
            img_b64 = base64.b64encode(image_to_png(cloth).read()).decode("utf-8")
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
                logging.warning(f"[MOBILE] Responses API failed for one item: {e}")
                descriptions.append(f"{cat}: descripción simulada")  # fallback demo

        # Validación vestido + zapatos
        if "vestidos" in categories_list and len(categories_list) > 2:
            raise HTTPException(status_code=400, detail="Solo puedes combinar vestido + zapatos como máximo 2 prendas")

        combined_prompt = f"""
STRICT INSTRUCTIONS:
- DO NOT CHANGE FACE, BODY, HEIGHT, or PROPORTIONS.
- Keep pose, lighting, and try to change a little the background of the original image.
- Full body visible from head to toes.

Apply the following clothing changes exactly over the base image:

{chr(10).join(descriptions)}

Combine garments realistically into a full-body outfit, keeping person unchanged.
"""

        base_img = image_to_png(base_image_file)

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

