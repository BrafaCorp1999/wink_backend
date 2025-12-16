# routers/generate_outfit_demo.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import traceback

# 🔹 Aquí se define el router
router = APIRouter()

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    try:
        gender = payload.get("gender", "unknown")

        # Prompt simplificado para demo
        prompt = f"A simple outfit for a {gender} person. Neutral pose, plain background."

        # 🔹 Aquí iría tu llamada a OpenAI/Replicate
        # Para demo, devolvemos un PNG negro como placeholder
        import base64
        import numpy as np
        image_b64 = "data:image/png;base64," + base64.b64encode(
            np.zeros((256, 256, 3), dtype=np.uint8).tobytes()
        ).decode()

        return JSONResponse({
            "status": "ok",
            "demo_outfits": [image_b64],
            "generation_mode": "AI_generated"
        })

    except Exception as e:
        print("❌ Unexpected error:", traceback.format_exc())
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })
