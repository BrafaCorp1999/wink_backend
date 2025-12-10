# services/gemini_service.py
import base64
import google.generativeai as genai

gemini_client = None


def init_gemini(api_key: str):
    """Initialize Gemini client globally."""
    global gemini_client
    genai.configure(api_key=api_key)
    gemini_client = genai.GenerativeModel("gemini-1.0-pro-vision")
    print("✅ Gemini initialized successfully")


async def gemini_generate_image(prompt: str):
    """Generate image using Gemini Vision."""
    global gemini_client
    if gemini_client is None:
        return None  # Gemini not initialized

    try:
        result = gemini_client.generate_images(
            prompt=prompt
        )

        if result.generated_images:
            img_bytes = result.generated_images[0]
            b64 = base64.b64encode(img_bytes).decode()
            return f"data:image/png;base64,{b64}"

        return None

    except Exception as e:
        print("⚠️ Gemini generation error:", e)
        return None
