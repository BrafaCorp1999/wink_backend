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
    gender: str = Form(...),
    body_shape: str = Form("average"),
    height: float = Form(None),
    weight: float = Form(None),
    style: str = Form("casual"),
    occasion: str = Form("daily"),
    climate: str = Form("temperate"),
    preferred_colors: str = Form("neutral tones"),
    model_type: str = Form("realistic"),
):
    """
    Generate 2 ultra-realistic outfit variations for the given person image.
    The system preserves identity and realism based on physical attributes.
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

        gender_label = "female" if str(gender).lower().startswith("f") else "male"
        body_shape = body_shape or "average"
        style = style or "casual"
        occasion = occasion or "daily"
        climate = climate or "temperate"
        preferred_colors = preferred_colors or "neutral tones"

        measurements = []
        if height: measurements.append(f"height {height} cm")
        if weight: measurements.append(f"weight {weight} kg")
        measure_text = ", ".join(measurements) if measurements else "average body proportions"

        # -------------------- PROMPT --------------------
        prompt = f"""
Generate 2 ultra-realistic, full-body fashion outfit variations for the person in the uploaded image.

CONTEXT:
- Gender: {gender_label}
- Body type: {body_shape}
- Measurements: {measure_text}
- Preferred style: {style}
- Occasion: {occasion}
- Climate: {climate}
- Preferred colors: {preferred_colors}
- Model type: {model_type}

STRICT IMAGE RULES:
- Preserve the user's face, hairstyle, and skin tone exactly.
- Do not change facial features, ethnicity, or proportions.
- Maintain the same pose and lighting conditions.
- Replace only clothing and accessories with realistic fashion outfits.
- Output must look natural, photorealistic, and consistent with the original person.

OUTPUT REQUIREMENTS:
- Provide 2 high-quality full-body images (base64).
- Include one short English sentence describing each outfit.
        """

        contents = [
            types.Part.from_text(prompt.strip()),
            types.Part.from_bytes(data=img_bytes, mime_type=person_image.content_type),
        ]

        # -------------------- GEMINI GENERATION --------------------
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

        # -------------------- RETURN STRUCTURED RESPONSE --------------------
        return JSONResponse(content={
            "success": True,
            "images": images_base64[:2],
            "description": text_response.strip(),
            "context_used": {
                "gender": gender_label,
                "body_shape": body_shape,
                "measurements": measure_text,
                "style": style,
                "occasion": occasion,
                "climate": climate,
                "preferred_colors": preferred_colors,
                "model_type": model_type
            }
        })

    except Exception as e:
        print("❌ Error in /generate-image:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
