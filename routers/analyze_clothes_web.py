from fastapi import APIRouter, Form, HTTPException
from openai import OpenAI
import os
import uuid
import logging

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.basicConfig(level=logging.INFO)

# =========================
# Endpoint WEB
# =========================
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
                            "Analyze this clothing item for realistic virtual try-on.\n\n"
                            "Describe ONLY visual characteristics that affect how it looks when worn.\n"
                            "Use neutral fashion terminology.\n\n"
                            "Include:\n"
                            "- garment type\n"
                            "- dominant and secondary colors\n"
                            "- fabric appearance\n"
                            "- fit\n"
                            "- length and cut\n"
                            "- sleeve type or neckline if applicable\n"
                            "- pattern or texture if present\n\n"
                            "Do NOT mention brand names or people.\n"
                            "Keep it concise but visually precise."
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

    except Exception as e:
        logging.error(f"[ANALYZE-CLOTHES-WEB][ERROR] {e}")
        raise HTTPException(status_code=500, detail="Clothing analysis failed")
