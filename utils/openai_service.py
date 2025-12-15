# utils/openai_service.py
from openai import OpenAI

openai_client = None


def init_openai(api_key: str):
    global openai_client
    openai_client = OpenAI(api_key=api_key)
    print("✅ OpenAI initialized (FALLBACK)")


async def openai_generate_image(prompt: str):
    global openai_client
    if openai_client is None:
        return None

    try:
        result = openai_client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024"
        )

        return f"data:image/png;base64,{result.data[0].b64_json}"

    except Exception as e:
        print("⚠️ OpenAI image error:", e)
        return None
