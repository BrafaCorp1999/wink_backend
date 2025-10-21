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
    body_shape: str = Form(None),  # slim / average / curvy / muscular
    waist: float = Form(None),
    hips: float = Form(None),
    height: float = Form(None),
):
    """
    Generate 2 full-body realistic outfit images based on a user's photo,
    preserving the natural face, hair, skin tone, and body proportions.
    """

    try:
        # --- Validations ---
        MAX_MB = 10
        ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
        user_bytes = await person_image.read()

        if len(user_bytes) / (1024 * 1024) > MAX_MB:
            raise HTTPException(status_code=400, detail="Image exceeds 10MB")
        if person_image.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=400, detail="Unsupported image type")

        # --- Default attributes ---
        gender_label = "female" if str(gender).lower().startswith("f") else "male"
        body_shape = body_shape or "average"
        style = style or "casual"
        model_type = model_type or "realistic"

        # --- Build body measurements ---
        measurements = []
        if waist:
            measurements.append(f"waist {waist} cm")
        if hips:
            measurements.append(f"hips {hips} cm")
        if height:
            measurements.append(f"height {height} cm")
        measure_text = ", ".join(measurements) if measurements else "average body proportions"

        # --- Strong prompt that forces preserving the face/hair/skin and respecting gender ---
        english_prompt = f"""
Generate a full-body photorealistic fashion image of the person in the uploaded photo.

STRICT REQUIREMENTS:
- Preserve 100% the person's face, hairstyle, hair texture, skin tone, and body proportions exactly as in the uploaded photo.
- Do NOT modify the person's ethnicity, facial structure, facial expression, or skin tone.
- Use the face/head region of the uploaded photo as the primary reference for identity — keep it identical.
- Change ONLY clothing and outfit; clothing must be realistically integrated on the body.
- Respect the provided gender: {gender_label}. If the provided gender conflicts with the face, follow the provided gender but keep the face features unchanged.

FASHION TASK:
Create two similar but distinct full-body outfits for a {gender_label} with a {body_shape} body type and {measure_text}.
Style: {style}.
Additional instructions: {prompt}

REQUIREMENTS:
- Both images must be high-quality, photorealistic, full-body, consistent lighting and background (studio/minimalist).
- Do not alter the face or hair. Do not lighten/darken skin or change hair texture.
- Adjust clothing size and fit according to the measurements and body type provided.
- Provide a short Spanish 1–2 sentence description for the outfit(s) (preferably for each image).

Return image data (inline) and a text description.
"""

        # --- First generation (base) ---
        contents1 = [
            english_prompt.strip(),
            types.Part.from_bytes(data=user_bytes, mime_type=person_image.content_type),
        ]
        response1 = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents1,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.7,
            ),
        )

        # --- Parse first response ---
        images_base64 = []
        text_response = ""
        if response1.candidates:
            cand = response1.candidates[0]
            for part in cand.content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    img_data = part.inline_data.data
                    mime = getattr(part.inline_data, "mime_type", "image/png")
                    images_base64.append(f"data:{mime};base64,{base64.b64encode(img_data).decode()}")
                elif hasattr(part, "text") and part.text:
                    # prefer Spanish short description if returned
                    text_response = part.text.strip()

        # --- Second generation: small variation instruction appended ---
        variation_prompt = english_prompt.strip() + "\nVariation: produce a second similar outfit with different clothing choices (same constraints)."
        contents2 = [
            variation_prompt,
            types.Part.from_bytes(data=user_bytes, mime_type=person_image.content_type),
        ]
        response2 = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents2,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.7,
            ),
        )

        if response2.candidates:
            cand2 = response2.candidates[0]
            for part in cand2.content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    img_data = part.inline_data.data
                    mime = getattr(part.inline_data, "mime_type", "image/png")
                    images_base64.append(f"data:{mime};base64,{base64.b64encode(img_data).decode()}")
                elif hasattr(part, "text") and part.text:
                    # if there is extra text, append or keep
                    if not text_response:
                        text_response = part.text.strip()

        if not images_base64:
            raise HTTPException(status_code=500, detail="Image generation failed")

        # If the model returned text in English, you could translate here (optional).
        # We prefer the model to return Spanish description; if not, we keep returned text.

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
