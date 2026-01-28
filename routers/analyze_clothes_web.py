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

# =========================
# Helper
# =========================
def image_to_base64(upload: UploadFile) -> str:
    image = Image.open(upload.file).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# =========================
# Endpoint MOBILE
# =========================
@router.post("/ai/combine-clothes")
async def analyze_clothes(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    logging.info(f"[ANALYZE-CLOTHES-MOBILE] {request_id}")

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
                            "Analyze this clothing item for realistic virtual try-on.\n\n"
                            "Describe ONLY visual characteristics that affect how it looks when worn.\n"
                            "Use neutral fashion terminology.\n\n"
                            "Include:\n"
                            "- garment type\n"
                            "- dominant and secondary colors\n"
                            "- fabric appearance (cotton, denim, knit, satin, leather-like, etc.)\n"
                            "- fit (tight, fitted, relaxed, oversized)\n"
                            "- length and cut\n"
                            "- sleeve type or neckline if applicable\n"
                            "- pattern or texture if present\n\n"
                            "Do NOT mention brand names.\n"
                            "Do NOT mention any person or mannequin.\n"
                            "Keep it concise but visually precise."
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
        logging.error(f"[ANALYZE-CLOTHES-MOBILE][ERROR] {e}")
        raise HTTPException(status_code=500, detail="Clothing analysis failed")
