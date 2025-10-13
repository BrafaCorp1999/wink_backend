from fastapi import APIRouter, UploadFile, File
import base64
from typing import Optional

router = APIRouter()

@router.post("/generate-body-model")
async def generate_body_model(
    Rostro: Optional[UploadFile] = File(None),
    Torso: Optional[UploadFile] = File(None),
    Piernas: Optional[UploadFile] = File(None),
):
    """
    Combina 3 imágenes escaneadas (rostro, torso, piernas)
    y genera un modelo corporal completo.
    """

    print("🧠 Recibiendo imágenes:")
    for img in [Rostro, Torso, Piernas]:
        if img:
            print(f" - {img.filename}")

    # 🔹 (Próximamente: aquí se usaría tu pipeline IA real)
    # Por ahora devolvemos una imagen de ejemplo
    with open("static/generated/body_model_example.png", "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")

    return {
        "status": "ok",
        "message": "Modelo corporal generado correctamente",
        "image": f"data:image/png;base64,{encoded}"
    }
