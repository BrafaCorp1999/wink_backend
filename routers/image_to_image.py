from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import os
import json

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Helpers
# =========================
def prepare_image(file: UploadFile, size=1024) -> BytesIO:
    image = Image.open(file.file).convert("RGB")
    image.thumbnail((size, size))  # Mantener proporciones

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.name = "input.png"
    buffer.seek(0)
    return buffer

# =========================
# Endpoint Mobile
# =========================
@router.post("/ai/generate-outfit-from-form")
async def generate_outfit_from_form(
    gender: str = Form(...),
    body_traits: str = Form(...),
    style: str = Form(...),
    occasion: str = Form(...),
    climate: str = Form(...),
    colors: str = Form(...),
    base_image_file: UploadFile = File(...)
):
    try:
        # Imagen del usuario
        image_file = prepare_image(base_image_file)

        # 1️⃣ Texto - Outfit en español + lista de prendas
        text_prompt = f"""
Eres un estilista profesional.

Perfil del usuario:
- Género: {gender}
- Estilo: {style}
- Ocasión: {occasion}
- Clima: {climate}
- Colores preferidos: {colors}

Describe UN outfit completo en español de manera concisa (3-4 líneas).
Devuelve además un JSON con las prendas mencionadas, ejemplo:
{{"prendas": ["vestido", "zapatos", "bolso"]}}
"""

        text_result = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Eres un estilista profesional. Responde en español."},
                {"role": "user", "content": text_prompt}
            ],
            max_tokens=120
        )

        raw_text = text_result.choices[0].message.content.strip()

        # Extraer recommendation y prendas_detectadas
        try:
            parts = raw_text.rsplit("```json", 1)
            recommendation = parts[0].strip()
            prendas_detectadas = json.loads(parts[1].strip().strip("```"))
        except:
            recommendation = raw_text
            prendas_detectadas = {"prendas": []}

        # 2️⃣ Imagen - aplicar outfit
        image_prompt = f"""
Aplica el siguiente outfit a la persona en la imagen:

{recommendation}

Preserva estrictamente el rostro, las proporciones del cuerpo, la altura, peso y la pose original del usuario.
Solo cambia el outfit y el entorno según la ocasión, estilo y clima especificados.
Alta calidad, estilo editorial de moda.
"""

        image_result = client.images.edit(
            model="gpt-image-1-mini",
            image=image_file,
            prompt=image_prompt,
            size="1024x1024"
        )

        generated_image = image_result.data[0].b64_json

        return {
            "status": "ok",
            "image": generated_image,
            "recommendation": recommendation,
            "prendas_detectadas": prendas_detectadas.get("prendas", [])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
