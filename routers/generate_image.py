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
    preserving the natural face, hair, skin tone, body proportions,
    adjusting clothing according to measurements and body_shape.
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
        gender_label = "female" if gender.lower().startswith("f") else "male"
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

        # --- Prompt reforzado ---
        english_prompt = f"""
Generate a full-body **photorealistic fashion image** of the person in the uploaded photo.

STRICT REQUIREMENTS:
- Preserve 100% the person’s **face**, **hairstyle**, **hair texture**, **skin tone**, and **body proportions** exactly as in the uploaded photo.
- Do NOT modify the person's ethnicity, facial structure, expression, or lighting tone of the face.
- The face and hair must remain **natural, detailed, and identical** to the original image.
- Change ONLY clothing and outfit style.
- Keep the background neutral and elegant (studio or minimalist style).

Fashion Task:
Create 2 similar outfits for a {gender_label} with a {body_shape} body type and {measure_text}.
The outfits should respect the user's measurements and body proportions, adjusting clothing size accordingly.
The outfit style should be {style}.
Follow these additional user instructions: {prompt}.
- Provide a short fashion description in Spanish (1–2 sentences) for the outfits.

Make the results suitable for a fashion app — consistent, realistic, elegant, and human-looking.
"""

        # --- First image ---
        contents1 = [
            english_prompt.strip(),
            types.Part.from_bytes(data=user_bytes, mime_type=person_image.content_type),
        ]
        response1 = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents1,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.7
            ),
        )

        # --- Parse first image ---
        images_base64 = []
        text_response = "No hay descripción disponible."

        if response1.candidates:
            candidate = response1.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    img_data = part.inline_data.data
                    mime = getattr(part.inline_data, "mime_type", "image/png")
                    images_base64.append(f"data:{mime};base64,{base64.b64encode(img_data).decode()}")
                elif hasattr(part, "text") and part.text:
                    text_response = part.text.strip()

        # --- Second image (small variation) ---
        contents2 = [
            english_prompt.strip() + "\nVariation: slightly different outfit style",
            types.Part.from_bytes(data=user_bytes, mime_type=person_image.content_type),
        ]
        response2 = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents2,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.7
            ),
        )

        # --- Parse second image ---
        if response2.candidates:
            candidate = response2.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    img_data = part.inline_data.data
                    mime = getattr(part.inline_data, "mime_type", "image/png")
                    images_base64.append(f"data:{mime};base64,{base64.b64encode(img_data).decode()}")

        if not images_base64:
            raise HTTPException(status_code=500, detail="Image generation failed")

        return JSONResponse(content={
            "images": images_base64,
            "text": text_response,  # descripción de la primera imagen
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
