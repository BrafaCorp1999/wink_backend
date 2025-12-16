# utils/replicate_service.py
import os
import replicate
import requests
import base64
import traceback

# Inicializar cliente con token de entorno
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

async def replicate_generate_image(prompt: str, width=512, height=512):
    try:
        # Ejecuta el modelo Stable Diffusion
        output_urls = replicate_client.run(
            "stability-ai/stable-diffusion-3.5-medium",
            input={"prompt": prompt, "width": width, "height": height}
        )

        if not output_urls or len(output_urls) == 0:
            return None

        # Descargar la primera imagen resultante
        image_url = output_urls[0]
        resp = requests.get(image_url)
        if resp.status_code != 200:
            return None

        image_bytes = resp.content
        image_b64 = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
        return image_b64

    except Exception as e:
        print("⚠️ Replicate image error:", e)
        print(traceback.format_exc())
        return None
