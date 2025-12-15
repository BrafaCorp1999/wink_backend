# utils/gemini_service.py
import base64
import google.generativeai as genai

gemini_client = None


def init_gemini(api_key: str):
    global gemini_client
    genai.configure(api_key=api_key)

    # Modelo de texto → imagen
    gemini_client = genai.GenerativeModel("models/gemini-1.5-flash")
    print("✅ Gemini initialized (PRIMARY)")


async def gemini_generate_image(prompt: str):
    global gemini_client
    if gemini_client is None:
        return None

    try:
        response = gemini_client.generate_content(prompt)

        # Gemini NO devuelve imagen directa como SD
        # Se usa para enriquecer prompt o fallback lógico
        return {
            "type": "prompt",
            "content": response.text
        }

    except Exception as e:
        print("⚠️ Gemini error:", e)
        return None
