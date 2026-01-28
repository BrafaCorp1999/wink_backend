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
def upload_to_png(upload: UploadFile, size: int = 1024) -> BytesIO:
    image = Image.open(upload.file).convert("RGB")
    image = image.resize((size, size))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def combine_clothes_prompt(descriptions: list[str]) -> str:
    garments_text = "\n".join(f"- {d}" for d in descriptions)

    return f"""
VIRTUAL TRY-ON TASK.

BASE IMAGE (STRICT — DO NOT VIOLATE):
- Do NOT change face, facial features, expression, or identity.
- Do NOT change body shape, proportions, height, weight, or measurements.
- Do NOT change skin tone.
- Same person, same anatomy.
- Background/place: Change a little the background/enivronment of the original image attached, due to make the photography more reaslistic and fashionable.

GARMENTS TO APPLY:
{garments_text}

STRICT RULES:
- Replace ONLY the garments listed above.
- Preserve EXACT colors, tones, patterns, and garment types.
- Do NOT invent, recolor, or remove garments.
- If ONE garment is provided, apply ONLY that garment.
- If TWO garments are provided, BOTH must be visible.
- Natural fit according to body proportions.
- Full body visible from head to toes.
- Photorealistic fashion photo.
"""

# =========================
# Endpoint MOBILE FINAL
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
        categories = json.loads(clothes_categories)

        if not clothes_files or len(clothes_files) == 0:
            raise HTTPException(status_code=400, detail="No clothing images provided")

        if len(clothes_files) > 2:
            raise HTTPException(status_code=400, detail="Maximum 2 garments allowed")

        if len(clothes_files) != len(categories):
            raise HTTPException(status_code=400, detail="Mismatch clothes vs categories")

        descriptions: list[str] = []

        # =========================
        # 1️⃣ ANALYZE GARMENTS
        # =========================
        for idx, (cat, cloth) in enumerate(zip(categories, clothes_files)):
            try:
                png_buf = upload_to_png(cloth)
                img_b64 = base64.b64encode(png_buf.read()).decode("utf-8")

                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Analyze this clothing item for virtual try-on.\n"
                                    f"Category: {cat}\n"
                                    "Describe ONLY visual characteristics.\n"
                                    "MANDATORY: include garment type and EXACT main color.\n"
                                    "Include fit/model, length, sleeves, neckline, texture or pattern.\n"
                                    "Do NOT invent colors.\n"
                                    "Do NOT mention brand, price, person or background."
                                )
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{img_b64}"
                            }
                        ]
                    }]
                )

                desc = response.output_text.strip()
                if not desc:
                    raise ValueError("Empty description returned")

                descriptions.append(desc)
                logging.info(f"[MOBILE][OK] Garment {idx + 1}: {desc}")

            except Exception as e:
                logging.error(f"[MOBILE][ANALYSIS FAILED] Garment {idx + 1}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to analyze one clothing item"
                )

        # =========================
        # 2️⃣ GENERATE FINAL IMAGE
        # =========================
        base_img = upload_to_png(base_image_file)
        final_prompt = combine_clothes_prompt(descriptions)

        result = client.images.edit(
            model="gpt-image-1-mini",
            image=("base.png", base_img.read(), "image/png"),
            prompt=final_prompt,
            size="1024x1024"
        )

        return {
            "status": "ok",
            "request_id": request_id,
            "descriptions": descriptions,  # 👈 útil para debug si lo necesitas
            "image": result.data[0].b64_json
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[COMBINE-CLOTHES-MOBILE][ERROR] {e}")
        raise HTTPException(status_code=500, detail="Combine clothes failed")
