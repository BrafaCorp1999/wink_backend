# routers/register_generate_base_images.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import base64
import json
import os
from openai import OpenAI
from io import BytesIO

router = APIRouter()

# =========================
# PROMPTS OPTIMIZADOS
# =========================

BODY_PHOTO_PROMPT = """
Use the uploaded full-body image strictly as a visual reference for the same real person.

IDENTITY LOCK:
- Preserve facial features: face shape, eyes, nose, lips, skin tone.
- Maintain hairstyle and hair color.
- Keep body proportions, height, and body type identical.
- Do NOT alter identity.

CLOTHING REPLACEMENT:
- Replace the original outfit entirely.
- Create a new outfit with {style} style.
- Include realistic and natural-looking clothing: top, bottoms, shoes.
- Add subtle accessories if appropriate (e.g., watch, belt, scarf).
- Ensure fabrics have natural folds and textures.
- Colors can be varied but harmonious with the style.

POSE & COMPOSITION:
- Full-body from head to toe, natural standing pose.
- Eye-level camera angle, proportional anatomy.
- Do NOT crop or distort body parts.

ENVIRONMENT & LIGHTING:
- Clean studio or neutral background.
- Natural soft lighting, realistic shadows.
- DSLR-quality, photorealistic, no CGI or illustration.

OUTPUT:
- Generate 2 distinct variations of the outfit.
- Maintain realism and the user's original proportions.
"""

SELFIE_PROMPT = """
Generate a photorealistic full-body image of a real person based on the provided selfie and body measurements.

IDENTITY & BODY:
- Match facial features: face shape, eyes, nose, lips, skin tone.
- Preserve hairstyle and color.
- Height: {height_cm} cm, Weight: {weight_kg} kg, Body type: {body_type}.

CLOTHING & STYLE:
- Complete outfit in {style} style, realistic and modern.
- Well-fitted top and bottoms with natural textures.
- Shoes visible and appropriate for the outfit.
- Optional subtle accessories (watch, belt, scarf).
- Colors harmonious and suitable for the style.

POSE & COMPOSITION:
- Full-body, natural standing pose.
- Eye-level camera, correct anatomy.
- Background clean and unobtrusive.

LIGHTING & QUALITY:
- Soft, natural lighting with realistic shadows.
- DSLR-quality, ultra-realistic, no illustrations or CGI.

OUTPUT:
- Generate 2 distinct outfit variations maintaining realism and proportions.
"""

# =========================
# ENDPOINT
# =========================

@router.post("/register_generate_base_images")
async def register_generate_base_images(
    mode: str = Form(...),  # "photo_body" | "selfie_manual"
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: Optional[str] = Form("casual"),
    image_file: UploadFile = File(...)
):
    # Validar traits
    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    # Selección de prompt
    if mode == "photo_body":
        final_prompt = BODY_PHOTO_PROMPT.format(style=style)
    elif mode == "selfie_manual":
        final_prompt = SELFIE_PROMPT.format(
            height_cm=traits.get("height_cm", "unknown"),
            weight_kg=traits.get("weight_kg", "unknown"),
            body_type=traits.get("body_type", "average"),
            style=style
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    # Leer imagen
    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty")

    # Inicializar cliente OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        if mode == "photo_body":
            response = client.images.edit(
                model="gpt-image-1.5",
                prompt=final_prompt,
                image=BytesIO(image_bytes),
                n=2,
                size="1024x1024"
            )
        else:
            response = client.images.generate(
                model="gpt-image-1.5",
                prompt=final_prompt,
                n=2,
                size="1024x1024"
            )

        generated_images_base64 = [img.b64_json for img in response.data]
        if not generated_images_base64:
            raise HTTPException(status_code=500, detail="No images generated")

        return {
            "status": "ok",
            "images": generated_images_base64,
            "prompt_used": final_prompt,
            "mode": mode
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
