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
    """Generate two realistic outfit variations while preserving identity and realism."""
    try:
        MAX_MB = 10
        ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
        img_bytes = await person_image.read()

        if len(img_bytes) / (1024 * 1024) > MAX_MB:
            raise HTTPException(status_code=400, detail="Image exceeds 10MB")
        if person_image.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=400, detail="Unsupported image type")

        gender_label = "female" if str(gender).lower().startswith("f") else "male"
        body_shape = body_shape or "average"
        style = style or "casual"

        measure_text = ", ".join(
            f"{k} {v} cm"
            for k, v in {
                "waist": waist,
                "hips": hips,
                "height": height
            }.items() if v
        ) or "average body proportions"

        # 🧠 Prompt mejorado para realismo del outfit
        base_prompt = f"""
Generate 2 full-body ultra-realistic fashion outfit images of the person in the uploaded photo.

STRICT RULES:
- Maintain exact face, hairstyle, skin tone, and expression.
- Preserve body shape and posture 100%.
- Ensure clothing looks naturally worn on the body — no overlay or sticker effect.
- Include natural lighting, shadows, and realistic cloth textures.
- Avoid AI distortion or exaggerated proportions.
- Gender: {gender_label}, Body shape: {body_shape}, Measurements: {measure_text}.
- Outfit style: {style}.
- {prompt}

Output: 2 distinct high-quality outfit variations and a short Spanish text (1–2 sentences) describing them.
"""

        contents = [
            base_prompt.strip(),
            types.Part.from_bytes(data=img_bytes, mime_type=person_image.content_type),
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.6,
            ),
        )

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
            raise HTTPException(status_code=500, detail="Image generation failed")

        return JSONResponse(content={
            "images": images_base64[:2],
            "text": text_response.strip(),
            "context_used": {
                "gender": gender_label,
                "body_shape": body_shape,
                "measurements": measure_text,
                "style": style
            }
        })

    except Exception as e:
        print("❌ Error in /generate-image:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
