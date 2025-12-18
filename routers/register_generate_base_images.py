from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import base64
import json

router = APIRouter()

# =========================
# PROMPTS BASE
# =========================

BODY_PHOTO_PROMPT = """
Use the uploaded image strictly as a visual reference for the same real person.

The generated image must depict the exact same individual, preserving:
- identical facial features
- same face shape, eyes, nose, lips and skin tone
- same body proportions, height and body type
- same hairstyle and hair color

Do NOT modify the person's identity or facial structure in any way.

IMPORTANT:
Do NOT replicate the original clothing from the reference image.
The outfit must be completely different from the original photo.

Generate a new, photorealistic full-body image of this person, standing naturally,
with the entire body visible from head to toe, including shoes and hair.

The person is wearing a modern, well-fitted outfit composed of:
- a clean, stylish top appropriate for daily wear
- matching bottoms with realistic fabric texture and natural folds
- appropriate footwear clearly visible
- subtle accessories if applicable

The outfit should look natural, realistic, and suitable for the person's body type.

The person is positioned in a clean, well-lit environment such as:
a modern indoor space or an open, minimal outdoor area.
The background must not distract from the subject.

Lighting should be natural and soft, similar to a professional lifestyle photo.
Camera angle should be at eye level, realistic, and proportional.

The final image must look like a real photograph taken with a high-quality camera,
not a rendering or illustration.
"""

SELFIE_PROMPT = """
Generate a photorealistic full-body image of a real person based on the following information.

The face must closely resemble the person shown in the provided selfie image.
Preserve:
- facial structure
- skin tone
- eyes, nose, lips
- hair color and hairstyle

Do NOT change the person's identity.

Body characteristics:
- Height: {height_cm} cm
- Weight: {weight_kg} kg
- Body type: {body_type}

Based on these measurements, generate realistic body proportions.
The body must look natural and anatomically correct.

Generate a full-body image with the entire body visible from head to toe,
including shoes and hair.

The person is wearing a complete, realistic outfit suitable for everyday use:
- well-fitted top
- complementary bottoms
- appropriate footwear
- subtle, realistic accessories

The clothing should naturally match the person's body type and proportions.

The scene should be realistic, such as:
a clean indoor environment or a neutral outdoor location.
Lighting must be soft and natural, similar to a real lifestyle photograph.

The final image must look like a real photograph,
with realistic textures, lighting, shadows, and proportions.
"""

# =========================
# ENDPOINT
# =========================

@router.post("/api/register_generate_base_images")
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
    # GENERACIÓN DE IMÁGENES
    # =========================
    # AQUÍ conectas tu motor real (OpenAI / Gemini / SD / etc)
    # Esto es conceptual y encaja con lo que ya tienes funcionando

    generated_images_base64 = generate_images_with_ai(
        prompt=final_prompt,
        reference_image=image_bytes,
        gender=gender,
        style=style,
        mode=mode
    )

    if not generated_images_base64:
        raise HTTPException(status_code=500, detail="No images generated")

    return {
        "status": "ok",
        "images": generated_images_base64,
        "prompt_used": final_prompt,
        "mode": mode
    }


# =========================
# MOTOR IA (EJEMPLO)
# =========================

def generate_images_with_ai(
    prompt: str,
    reference_image: bytes,
    gender: str,
    style: str,
    mode: str
):
    """
    Esta función representa TU integración actual.
    Aquí NO devuelvas la imagen original.
    """

    # ⚠️ CLAVE:
    # - usar reference_image SOLO como conditioning
    # - no como output

    # EJEMPLO (mock):
    generated_image_bytes = reference_image  # <-- reemplaza por tu IA real

    return [
        base64.b64encode(generated_image_bytes).decode("utf-8"),
        base64.b64encode(generated_image_bytes).decode("utf-8"),
    ]
