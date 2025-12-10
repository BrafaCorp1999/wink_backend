# services/openai_service.py
import base64
from openai import OpenAI

openai_client = None


def init_openai(api_key: str):
    """Initialize OpenAI client globally."""
    global openai_client
    openai_client = OpenAI(api_key=api_key)
    print("✅ OpenAI initialized successfully")


async def openai_generate_image(prompt: str):
    """Generate image with OpenAI DALL·E 3.1 Mid."""
    global openai_client
    if openai_client is None:
        return None

    try:
        response = openai_client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024"
        )

        image_base64 = response.data[0].b64_json
        return f"data:image/png;base64,{image_base64}"

    except Exception as e:
        print("⚠️ OpenAI generation error:", e)
        return None
