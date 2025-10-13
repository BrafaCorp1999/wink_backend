from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import base64, traceback
from openai import types, Client

router = APIRouter()
client = Client()

@router.post("/try-on")
async def try_on(
    person_image: UploadFile = File(...),
    cloth_images: list[UploadFile] = File(...),  # 🔹 ahora soporta múltiples prendas
    instructions: str = Form(""),
    model_type: str = Form("realistic"),
    gender: str = Form("female"),
    style: str = Form("modern"),
):
    try:
        MAX_IMAGE_SIZE_MB = 10
        ALLOWED_MIME_TYPES = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/heic",
            "image/heif",
        }

        # ---- Validar person_image ----
        if person_image.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {person_image.content_type}")
        user_bytes = await person_image.read()
        if len(user_bytes) / (1024*1024) > MAX_IMAGE_SIZE_MB:
            raise HTTPException(status_code=400, detail="person_image exceeds 10MB")

        # ---- Validar cloth_images ----
        cloth_bytes_list = []
        for cloth in cloth_images:
            if cloth.content_type not in ALLOWED_MIME_TYPES:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {cloth.content_type}")
            b = await cloth.read()
            if len(b) / (1024*1024) > MAX_IMAGE_SIZE_MB:
                raise HTTPException(status_code=400, detail=f"{cloth.filename} exceeds 10MB")
            cloth_bytes_list.append({"name": cloth.filename, "data": b, "mime": cloth.content_type})

        # ---- Prompt extenso ----
        cloth_descriptions = ", ".join([c["name"] for c in cloth_bytes_list])
        prompt = f"""
        {{
            "objective": "Generate a photorealistic virtual try-on image, integrating the selected clothing items ({cloth_descriptions}) onto a person, rigidly preserving the facial identity, proportions, and natural posture.",
            "task": "High-Fidelity Virtual Try-On with Identity/Garment Preservation and Full-Body Output",
            "inputs": {{
                "person_image": {{"description": "Source image containing the target person.", "id": "input_1"}},
                "garment_images": [
                    {', '.join([f'{{"description": "Clothing item: {c["name"]}", "id": "input_{i+2}"}}' for i,c in enumerate(cloth_bytes_list)])}
                ]
            }},
            "focus_instructions": "Apply smart zoom for each garment type (blouse, shoes, pants, etc.) to highlight it, but do not deform the face or body. Keep realistic proportions.",
            "style_instructions": "{style}",
            "special_instructions": "{instructions}"
        }}
        """

        # ---- Preparar inputs para Gemini ----
        contents = [prompt, types.Part.from_bytes(data=user_bytes, mime_type=person_image.content_type)]
        for c in cloth_bytes_list:
            contents.append(types.Part.from_bytes(data=c["data"], mime_type=c["mime"]))

        # ---- Generar imagen ----
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
        )

        # ---- Procesar respuesta ----
        image_data = None
        text_response = "No description available."
        if response.candidates:
            parts = response.candidates[0].content.parts
            for part in parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    image_data = part.inline_data.data
                    image_mime_type = getattr(part.inline_data, "mime_type", "image/png")
                elif hasattr(part, "text") and part.text:
                    text_response = part.text

        if image_data:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            image_url = f"data:{image_mime_type};base64,{image_base64}"
        else:
            image_url = None

        return JSONResponse(content={"image": image_url, "text": text_response})

    except Exception as e:
        print("❌ Error in /api/try-on endpoint:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
