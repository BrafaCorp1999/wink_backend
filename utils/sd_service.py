import io
import base64
import torch
import numpy as np
import cv2
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
    global sd
    if sd is None:
        return None

    try:
        with torch.autocast(DEVICE):
            pil_img = sd(prompt).images[0]

        # Convert PIL Image to NumPy array (BGR for OpenCV)
        img = np.array(pil_img)[:, :, ::-1]  # RGB -> BGR

        # Encode image as PNG with OpenCV
        _, buf = cv2.imencode(".png", img)
        b64 = base64.b64encode(buf.tobytes()).decode()
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        print("⚠️ SD generation error:", e)
        return None
