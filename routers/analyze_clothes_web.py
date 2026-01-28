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
def prepare_image_from_b64(image_b64: str, size: int = 1024) -> BytesIO:
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
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
- Do NOT change the face, facial features, expression, or identity.
- Do NOT change body shape, proportions, height, weight, or measurements.
- Do NOT change skin tone.
- Keep the same person, same pose, same anatomy.
- Bckground/place: change a little the ambient/background of the original image to make the photography more realistic. 

GARMENTS TO APPLY:
{garments_text}

STRICT RULES:
- Replace ONLY the garments listed above.
- Preserve EXACT colors, tones, patterns, and garment types.
- Do NOT recolor, invent, remove, or ignore any garment.
- If only ONE garment is provided, apply ONLY that garment.
- If TWO garments are provided, BOTH must be clearly visible.
- Ensure natural fit according to body proportions.
- Full body visible from head to toes.
- Photorealistic fashion photography.
- Natural lighting and realistic shadows.
"""

# =========================
# Endpoint WEB + MOBILE
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
    logging.info(f"[COMBINE-CLOTHES] {request_id}")

    try:
        clothes_list = json.loads(clothes_images_b64)

        if not isinstance(clothes_list, list) or len(clothes_list) == 0:
            raise HTTPException(status_code=400, detail="No clothing images provided")

        if len(clothes_list) > 2:
            raise HTTPException(status_code=400, detail="Maximum 2 garments allowed")

        descriptions: list[str] = []

        # =========================
        # 1️⃣ ANALYZE EACH GARMENT
        # =========================
        for idx, img_b64 in enumerate(clothes_list):
            try:
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Analyze this clothing item for virtual try-on.\n"
                                    "Describe ONLY visual characteristics.\n"
                                    "MANDATORY: include garment type and EXACT main color.\n"
                                    "Include fit/model, length, sleeves, neckline, texture or pattern.\n"
                                    "Do NOT invent colors.\n"
                                    "Do NOT mention brand, price, person, or background."
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
                logging.info(f"[ANALYSIS][OK] Garment {idx + 1}: {desc}")

            except Exception as e:
                logging.error(f"[ANALYSIS][FAILED] Garment {idx + 1}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to analyze one clothing item"
                )

        # =========================
        # 2️⃣ GENERATE FINAL IMAGE
        # =========================
        base_img = prepare_image_from_b64(base_image_b64)
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
            "descriptions": descriptions,  # 👈 útil para debug (web + mobile)
            "image": result.data[0].b64_json
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[COMBINE-CLOTHES][ERROR] {e}")
        raise HTTPException(status_code=500, detail="Combine clothes failed")
