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
    gender: str = Form("female"),
    model_type: str = Form("realistic"),
    style: str = Form("modern"),
    body_shape: str = Form(None),
    waist: float = Form(None),
    hips: float = Form(None),
    height: float = Form(None),
):
    """
    Generate 2 full-body photorealistic outfit images based on a user's photo,
    preserving the face, hair, skin tone, and body proportions exactly.
    """
    try:
        MAX_MB = 10
        ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
        user_bytes = await person_image.read()

        if len(user_bytes) / (1024 * 1024) > MAX_MB:
            raise HTTPException(status_code=400, detail="Image exceeds 10MB")
        if person_image.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=400, detail="Unsupported image type")

        gender_label = "female" if str(gender).lower().startswith("f") else "male"
        body_shape = body_shape or "average"
        style = style or "casual"
        model_type = model_type or "realistic"

        measurements = []
        if waist: measurements.append(f"waist {waist} cm")
        if hips: measurements.append(f"hips {hips} cm")
        if height: measurements.append(f"height {height} cm")
        measure_text = ", ".join(measurements) if measurements else "average body proportions"

        base_prompt = f"""
Generate 2 distinct full-body photorealistic fashion images of the person in the uploaded photo.

STRICT REQUIREMENTS:
- Preserve 100% the face, hairstyle, hair texture, skin tone.
- Do NOT modify facial features, expression, ethnicity, or body proportions.
- Show full body, head to feet, natural standing pose, centered in frame.
- Only change clothing and outfit, respecting provided measurements.
- Gender: {gender_label}, Body type: {body_shape}, Measurements: {measure_text}, Style: {style}.
- Additional instructions: {prompt}
- Return a short Spanish 1–2 sentence description for each image.
"""

        contents = [
            base_prompt.strip(),
            types.Part.from_bytes(data=user_bytes, mime_type=person_image.content_type),
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.7,
            ),
        )

        images_base64 = []
        text_response = ""

        if response.candidates:
            cand = response.candidates[0]
            for part in cand.content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    img_data = part.inline_data.data
                    mime = getattr(part.inline_data, "mime_type", "image/png")
                    images_base64.append(f"data:{mime};base64,{base64.b64encode(img_data).decode()}")
                elif hasattr(part, "text") and part.text:
                    text_response = part.text.strip()

        if len(images_base64) > 2:
            images_base64 = images_base64[:2]  # solo 2 imágenes

        if not images_base64:
            raise HTTPException(status_code=500, detail="Image generation failed")

        return JSONResponse(content={
            "images": images_base64,
            "text": text_response,
            "context_used": {
                "gender": gender_label,
                "body_shape": body_shape,
                "measurements": measure_text,
                "style": style,
                "model_type": model_type
            }
        })

    except Exception as e:
        print("❌ Error in /generate-image:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
