from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import base64, traceback, os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

@router.post("/generate-image")
async def generate_image(
    person_image: UploadFile = File(...),
    prompt: str = Form(...),
):
    """
    Generate 2 ultra-realistic outfit variations using a pre-built prompt from frontend.
    """
    try:
        # -------------------- VALIDATIONS --------------------
        MAX_MB = 10
        ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

        img_bytes = await person_image.read()
        if len(img_bytes) / (1024 * 1024) > MAX_MB:
            raise HTTPException(status_code=400, detail="Image exceeds 10MB limit.")
        if person_image.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=400, detail="Unsupported image type.")

        # -------------------- GEMINI GENERATION --------------------
        contents = [
            types.Part.from_text(prompt),  # ✅ solo un argumento
            types.Part.from_bytes(data=img_bytes, mime_type=person_image.content_type),
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.7,
            ),
        )

        # -------------------- PARSE RESPONSE --------------------
        images_base64, text_response = [], ""
        if response.candidates:
            cand = response.candidates[0]
            for part in cand.content.parts:
                if getattr(part, "inline_data", None):
                    data = part.inline_data.data
                    mime = getattr(part.inline_data, "mime_type", "image/png")
                    images_base64.append(
                        f"data:{mime};base64,{base64.b64encode(data).decode()}"
                    )
                elif getattr(part, "text", None):
                    text_response += part.text.strip() + " "

        if not images_base64:
            raise HTTPException(status_code=500, detail="No images generated from model.")

        # -------------------- RETURN RESPONSE --------------------
        return JSONResponse(content={
            "success": True,
            "images": images_base64[:2],
            "description": text_response.strip()
        })

    except Exception as e:
        print("❌ Error in /generate-image:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
