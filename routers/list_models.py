from fastapi import APIRouter
from google import genai
import os

router = APIRouter()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@router.get("/api/list_models")
async def list_models():
    models_info = []
    try:
        models = client.models.list_models()
        for m in models:
            models_info.append({
                "name": m.name,
                "display_name": getattr(m, "display_name", None),
                "supported_modalities": getattr(m, "supported_modalities", None),
                "description": getattr(m, "description", None),
                "release_date": getattr(m, "release_date", None),
                "capabilities": getattr(m, "capabilities", None)
            })
        return {"status": "ok", "models": models_info}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
