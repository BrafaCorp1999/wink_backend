from fastapi import APIRouter, Form, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64, os, uuid, logging

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO)

def prepare_image_from_b64(image_b64: str, size=1024) -> BytesIO:
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((size, size))
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.name = "input.png"
    buf.seek(0)
    return buf

@router.post("/ai/generate-tryon-web")
async def generate_tryon_web(
    base_image_b64: str = Form(...),
    clothes_description: str = Form(...)
):
    request_id = str(uuid.uuid4())
    logging.info(f"[GENERATE-TRYON-WEB] {request_id}")

    try:
        base_img = prepare_image_from_b64(base_image_b64)

        prompt = f"""
Replace the person's clothing with the following outfit description:

{clothes_description}

Rules:
- Keep same person, same face, same body.
- Do not alter facial features or pose.
- Preserve background and lighting.
- Full body visible.
- Realistic fashion photography.
- Add natural folds, shadows, and fabric texture.
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
