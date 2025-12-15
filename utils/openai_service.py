# utils/openai_service.py
import os
import openai

openai_client = None

def init_openai(api_key: str):
    global openai_client
    os.environ["OPENAI_API_KEY"] = api_key  # setea la variable de entorno
    openai_client = openai
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
