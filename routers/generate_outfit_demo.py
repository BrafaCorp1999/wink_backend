from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import base64, traceback, io, os
from PIL import Image
import requests
from dotenv import load_dotenv

load_dotenv()
router = APIRouter(prefix="/generate-outfit-demo", tags=["AI Outfit Demo"])

# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# -------------------------------
# Gemini Outfit Generation
# -------------------------------
def generate_with_gemini(prompt: str, image_bytes: bytes):
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)

        b64_img = base64.b64encode(image_bytes).decode()

        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        result = model.generate_images(
            prompt=prompt,
            image={"mime_type": "image/jpeg", "data": b64_img},
            size="1024x1024",
        )

        if not result or not result.images:
            return None

        return result.images[0].image_bytes

    except Exception:
        return None


# -------------------------------
# OpenAI Outfit Generation (fallback)
# -------------------------------
def generate_with_openai(prompt: str, image_bytes: bytes):
    try:
        url = "https://api.openai.com/v1/images/edits"

        files = {"image": ("input.jpg", image_bytes, "image/jpeg")}
        data = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "size": "1024x1024",
        }
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

        r = requests.post(url, files=files, data=data, headers=headers)

        if r.status_code != 200:
            return None

        img_base64 = r.json()["data"][0]["b64_json"]
        return base64.b64decode(img_base64)

    except Exception:
        return None


# -------------------------------
# Main Demo Endpoint
# -------------------------------
@router.post("/")
async def generate_outfit_demo(payload: dict):
    """
    Creates 3 outfit images using Gemini and OpenAI as fallback.
    """
    try:
        face_base64 = payload.get("face_base64")
        gender = payload.get("gender", "person")

        if not face_base64:
            raise HTTPException(status_code=400, detail="Missing face_base64")

        # Convert base64 → bytes
        header, data = face_base64.split(",", 1)
        img_bytes = base64.b64decode(data)

        outfit_styles = ["casual outfit", "elegant outfit", "sporty outfit"]
        final_images = []

        for style in outfit_styles:

            # English prompt (strict face preservation)
            prompt = f"""
            Generate a realistic full-body image of the same person in the reference image.
            KEEP the exact same face, facial structure, proportions, and identity.
            DO NOT modify the face, skin tone, eyes, nose, lips, hairline, or expression.
            ONLY change the clothing.

            Style requested: {style}.
            """

            # Try Gemini
            gemini_img = generate_with_gemini(prompt, img_bytes)

            if gemini_img:
                final_images.append(
                    "data:image/png;base64," + base64.b64encode(gemini_img).decode()
                )
                continue

            # Fallback: OpenAI
            openai_img = generate_with_openai(prompt, img_bytes)

            if openai_img:
                final_images.append(
                    "data:image/png;base64," + base64.b64encode(openai_img).decode()
                )
                continue

            # If both fail: empty image placeholder
            final_images.append(
                "data:image/png;base64," +
                base64.b64encode(b"\x00" * (512 * 512 * 3)).decode()
            )

        return JSONResponse({
            "status": "ok",
            "generated_by": "gemini_or_openai",
            "images": final_images
        })

    except Exception:
        print("❌ Error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")
