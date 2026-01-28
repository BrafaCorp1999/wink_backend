from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import os, uuid, logging

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO)

def image_to_png(upload: UploadFile) -> BytesIO:
    img = Image.open(upload.file).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

@router.post("/ai/generate-tryon")
async def generate_tryon(
    base_image: UploadFile = File(...),
    clothes_description: str = Form(...)
):
    request_id = str(uuid.uuid4())
    logging.info(f"[GENERATE-TRYON] {request_id}")

    try:
        base_img = image_to_png(base_image)

        prompt = f"""
Replace the person's clothing with the following outfit:

{clothes_description}

Rules:
- Full body visible from head to toes.
- Same person, same face, same body.
- Do not alter facial features or pose.
- Preserve background and lighting.
- Realistic fashion photo.
- Add natural folds, shadows, and fabric texture.
- If the outfit includes a dress and shoes, ensure the dress length and shoes are fully visible.
- Maintain proper proportions for all clothing items.
"""

        result = client.images.edit(
            model="gpt-image-1-mini",
            image=("base.png", base_img.read(), "image/png"),
            prompt=prompt,
            size="1024x1024"
        )

        return {
            "status": "ok",
            "request_id": request_id,
            "image": result.data[0].b64_json
        }

    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail="Try-on generation failed")
