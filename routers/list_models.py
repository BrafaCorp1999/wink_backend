from fastapi import APIRouter
from fastapi.responses import JSONResponse
from google import genai
import os

router = APIRouter()

# Usa tu API key actual
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

@router.get("/list_models")
async def list_models():
    try:
        models = client.models.list()  # ✅ método correcto
        model_info = [
            {
                "name": m.name,
                "description": getattr(m, "description", ""),
                "supported_modalities": getattr(m, "supported_modalities", []),
                "max_output_tokens": getattr(m, "max_output_tokens", None),
            }
            for m in models
        ]
        return JSONResponse({"status": "ok", "models": model_info})
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)})
