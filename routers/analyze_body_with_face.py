# analyze_body_with_face.py
from fastapi import APIRouter, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO

router = APIRouter()

# --- Función auxiliar para simular extracción de medidas ---
def extract_body_features(image: Image.Image, gender: str) -> dict:
    """
    Lógica de análisis simulada.
    Retorna medidas y rasgos basados en la imagen (demo).
    """
    width, height = image.size
    aspect_ratio = height / width

    # Contextura simulada según proporción
    if aspect_ratio > 2.2:
        body_type = "slim"
    elif aspect_ratio < 1.6:
        body_type = "plus"
    else:
        body_type = "average"

    # Altura y peso estimados (demo)
    if gender.lower() == "male":
        height_cm = 175
        weight_kg = 70
    else:
        height_cm = 165
        weight_kg = 60

    hair_type = "medium length, straight"

    return {
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "body_type": body_type,
        "hair_type": hair_type
    }

# =========================
# Endpoint corregido para Flutter Multipart
# =========================
@router.post("/analyze-body-with-face")
async def analyze_body_with_face(
    gender_hint: str = Form(...),
    person_image: UploadFile = None
):
    try:
        if person_image is None:
            return JSONResponse(status_code=400, content={"detail": "Imagen no proporcionada"})

        # Leer bytes de la imagen
        image_bytes = await person_image.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Extraemos rasgos del cuerpo
        features = extract_body_features(image, gender_hint)

        # DEBUG: log para ver medidas y rasgos
        print(f"✅ Rasgos extraídos: {features}")

        return {"status": "ok", **features}

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Error analyzing body: {str(e)}"})
