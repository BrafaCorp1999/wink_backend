from fastapi import APIRouter, Form, HTTPException
from openai import OpenAI
import os
import uuid
import logging

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO)

@router.post("/ai/generate-tryon-web")
async def generate_tryon_web(
    base_image_b64: str = Form(...),
    clothes_description: str = Form(...)
):
    request_id = str(uuid.uuid4())
    logging.info(f"[GENERATE-TRYON-WEB] {request_id}")

    try:
        prompt = f"""
Replace the person's clothing with the following outfit description:

{clothes_description}

Rules:
- Same person.
- Same face and body.
- Realistic photo.
- Try to match exactly the description's clothes above.
"""

        result = client.images.generate(
            model="gpt-image-1-mini",
            prompt=prompt,
            size="1024x1024"
        )

        return {
            "status": "ok",
            "request_id": request_id,
            "image": result.data[0].b64_json
        }

    except Exception:
        raise HTTPException(status_code=500, detail="Try-on generation failed")
