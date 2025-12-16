from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import traceback
from utils.openai_service import openai_generate_image
from utils.replicate_service import replicate_generate_image

router = APIRouter()

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    try:
        gender = payload.get("gender", "unknown")

        # Prompt simple para demo, sin foto ni medidas
        prompt = f"A full-body portrait of a {gender} person wearing a casual outfit, photorealistic."

        image_b64 = None

        # --- 1️⃣ Intentar OpenAI ---
        try:
            image_b64 = await openai_generate_image(prompt)
            if not image_b64:
                raise ValueError("OpenAI returned empty image")
        except Exception as e_openai:
            # --- 2️⃣ Si falla OpenAI, intentar Replicate ---
            try:
                image_b64 = await replicate_generate_image(prompt)
                if not image_b64:
                    raise ValueError("Replicate returned empty image")
            except Exception as e_replicate:
                raise HTTPException(status_code=500, detail=f"AI generation failed: {e_openai}, {e_replicate}")

        return JSONResponse({
            "status": "ok",
            "demo_outfits": [image_b64],  # Solo 1 imagen
            "generation_mode": "AI_generated"
        })

    except Exception as e:
        print("❌ Unexpected error in /generate_outfit_demo:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
