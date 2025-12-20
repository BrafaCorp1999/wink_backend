# routers/register_generate_base_images.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import json
import os
from io import BytesIO
from openai import OpenAI

router = APIRouter()

BODY_PHOTO_PROMPT = """
Use the uploaded full-body image strictly as a visual reference for the same real person.
IDENTITY LOCK: Preserve facial features, body proportions, hairstyle.
CLOTHING REPLACEMENT: Replace outfit with {style} style, realistic clothing.
POSE & COMPOSITION: Full-body, natural pose.
ENVIRONMENT & LIGHTING: Clean background, realistic shadows.
OUTPUT: Generate 2 distinct outfit variations.
"""

SELFIE_PROMPT = """
Generate a photorealistic full-body image based on the provided body traits.
Height: {height_cm} cm, Weight: {weight_kg} kg, Body type: {body_type}.
Style: {style}, realistic clothing, shoes visible.
OUTPUT: Generate 2 distinct outfit variations maintaining realism.
"""

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.post("/register_generate_base_images")
async def register_generate_base_images(
    mode: str = Form(...),  # "photo_body" | "selfie_manual"
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: Optional[str] = Form("casual"),
    image_file: Optional[UploadFile] = File(None)
):
    """
    photo_body: recibe archivo real, usa images.edit
    selfie_manual: recibe solo traits en JSON, usa images.generate
    """
    try:
        traits = json.loads(body_traits)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body_traits JSON")

    if mode == "photo_body":
        if image_file is None:
            raise HTTPException(status_code=400, detail="image_file is required for photo_body")
        final_prompt = BODY_PHOTO_PROMPT.format(style=style)
        try:
            image_bytes = await image_file.read()
            response = client.images.edit(
                model="gpt-image-1.5",
                image=BytesIO(image_bytes),
                prompt=final_prompt,
                n=2,
                size="1024x1024"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image edit failed: {str(e)}")

    elif mode == "selfie_manual":
        final_prompt = SELFIE_PROMPT.format(
            height_cm=traits.get("height_cm", "unknown"),
            weight_kg=traits.get("weight_kg", "unknown"),
            body_type=traits.get("body_type", "average"),
            style=style
        )
        try:
            response = client.images.generate(
                model="gpt-image-1.5",
                prompt=final_prompt,
                n=2,
                size="1024x1024"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image generate failed: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    images_base64 = [img.b64_json for img in response.data]

    return {
        "status": "ok",
        "images": images_base64,
        "prompt_used": final_prompt,
        "mode": mode
    }
