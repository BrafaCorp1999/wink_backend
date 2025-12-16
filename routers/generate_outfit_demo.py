from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    gender = payload.get("gender", "unknown")

    # Aquí iría OpenAI/Replicate
    # Por ahora, prompt simple para test
    try:
        # 🔹 Simulación de generación AI (PNG válido 1x1 negro)
        demo_image = "data:image/png;base64," + \
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMA" \
            "AQAABQABDQottAAAAABJRU5ErkJggg=="

        return JSONResponse({
            "status": "ok",
            "demo_outfits": [demo_image]
        })

    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })
