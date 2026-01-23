from fastapi import APIRouter, Form, HTTPException
from openai import OpenAI
import base64
import os
import uuid
import logging

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO)

@router.post("/ai/analyze-clothes-web")
async def analyze_clothes_web(
    image_b64: str = Form(...)
):
    request_id = str(uuid.uuid4())
    logging.info(f"[ANALYZE-CLOTHES-WEB] {request_id}")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Describe this clothing item for virtual try-on. "
                            "Include type, color, material, fit, length, patterns."
                        )
                    },
                    {
                        "type": "input_image",
                        "image_base64": image_b64
                    }
                ]
            }]
        )

        return {
            "status": "ok",
            "request_id": request_id,
            "description": response.output_text.strip()
        }

    except Exception:
        raise HTTPException(status_code=500, detail="Clothing analysis failed")
