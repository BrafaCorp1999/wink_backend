from fastapi import APIRouter, UploadFile, File, HTTPException
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import os
import uuid
import logging

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO)

def image_to_base64(upload: UploadFile) -> str:
    img = Image.open(upload.file).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@router.post("/ai/analyze-clothes")
async def analyze_clothes(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    logging.info(f"[ANALYZE-CLOTHES] {request_id}")

    try:
        img_b64 = image_to_base64(file)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Describe this clothing item for virtual try-on. "
                            "Be objective and concise. Include: "
                            "type, color, material, fit, length, patterns. "
                            "Do not mention brand or person."
                        )
                    },
                    {
                        "type": "input_image",
                        "image_base64": img_b64
                    }
                ]
            }]
        )

        description = response.output_text.strip()

        return {
            "status": "ok",
            "request_id": request_id,
            "description": description
        }

    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail="Clothing analysis failed")
