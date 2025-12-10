# services/sd_service.py
import base64
import io
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load once
try:
    sd = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    sd = sd.to(DEVICE)
    print("✅ Stable Diffusion initialized")
except Exception as e:
    print("⚠️ SD load failed:", e)
    sd = None


async def sd_generate_image(prompt: str):
    """Generate image using Stable Diffusion."""
    global sd
    if sd is None:
        return None

    try:
        with torch.autocast(DEVICE):
            img = sd(prompt).images[0]

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        print("⚠️ SD generation error:", e)
        return None
