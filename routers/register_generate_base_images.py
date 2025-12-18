# routers/register_generate_base_images.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import base64
import json
import os
from openai import OpenAI
import tempfile

router = APIRouter()

# =========================
# PROMPTS BASE
# =========================

BODY_PHOTO_PROMPT = """
Use the uploaded image strictly as a visual reference for the same real person.

IDENTITY LOCK:
- Use the image ONLY to preserve the same real person.
- Preserve identical facial features: face shape, eyes, nose, lips, skin tone.
- Preserve hairstyle and hair color.
- Preserve body proportions, height, body type.
- Do NOT alter facial structure or identity.

CLOTHING REPLACEMENT:
- Replace the entire outfit.
- Do NOT replicate original clothing.
- New outfit, different colors and garments.

OUTFIT REQUIREMENTS:
- Modern, realistic outfit for daily wear.
- Well-fitted top with realistic fabric.
- Matching bottoms with natural folds.
- Appropriate shoes clearly visible.
- Optional subtle accessories.

POSE & COMPOSITION:
- Full-body shot from head to toe.
- Natural standing pose.
- Eye-level camera, proportional anatomy.

ENVIRONMENT:
- Clean indoor studio or neutral outdoor space.
- Background must not distract from subject.

LIGHTING & REALISM:
- Natural soft lighting, realistic shadows.
- DSLR-style ultra-photorealistic photography.
- No illustration or CGI.
"""

SELFIE_PROMPT = """
Generate a photorealistic full-body image of a real person based on the provided selfie.

FACE REFERENCE:
- Match facial structure, skin tone, eyes, nose, lips, hairstyle.
- Do NOT change identity.

BODY CHARACTERISTICS:
- Height: {height_cm} cm
- Weight: {weight_kg} kg
- Body type: {body_type}

OUTFIT:
- Complete modern outfit suitable for daily lifestyle.
- Well-fitted top, complementary bottoms.
- Appropriate footwear.
- Subtle realistic accessories.

COMPOSITION:
- Full-body from head to toe, natural pose.
- Eye-level camera, proportional anatomy.

ENVIRONMENT:
- Clean indoor or neutral outdoor background.

LIGHTING & QUALITY:
- Natural soft lighting, realistic textures, DSLR-quality.
- Ultra-realistic, no illustration or CGI.
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
    """
    mode:
    - photo_body  -> imagen de cuerpo completo (transformación)
    - selfie_manual -> selfie + medidas (generación)
    """

    # =========================
    # PARSE BODY TRAITS
    # =========================
    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    # =========================
    # SELECCIÓN DE PROMPT
    # =========================
    if mode == "photo_body":
        final_prompt = BODY_PHOTO_PROMPT
    elif mode == "selfie_manual":
        final_prompt = SELFIE_PROMPT.format(
            height_cm=traits.get("height_cm", "unknown"),
            weight_kg=traits.get("weight_kg", "unknown"),
            body_type=traits.get("body_type", "average"),
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    # =========================
    # LEER IMAGEN
    # =========================
    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file is empty")

    # =========================
    # GENERACIÓN DE IMÁGENES CON GPT-IMAGE-1.5
    # =========================
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        if mode == "photo_body":
            # =========================
            # Image-to-image generation
            # =========================
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
                tmp_file.write(image_bytes)
                tmp_file.flush()

                response = client.images.edit(
                    model="gpt-image-1.5",
                    image=open(tmp_file.name, "rb"),
                    prompt=final_prompt,
                    size="1024x1024",
                    n=2
                )
        else:
            # =========================
            # Text + body measurements -> generate from zero
            # =========================
            response = client.images.generate(
                model="gpt-image-1.5",
                prompt=final_prompt,
                size="1024x1024",
                n=2
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
