from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import base64, os, traceback, json
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

@router.post("/generate-outfits-from-manual")
async def generate_outfits_from_manual(
    selfie: UploadFile = File(...),
    gender: str = Form(...),
    height: float = Form(...),
    weight: float = Form(...),
    chest: float = Form(...),
    waist: float = Form(...),
    hips: float = Form(...),
    arms: float = Form(...),
    body_shape: str = Form(...),
):
    """
    Generate two outfit base suggestions (casual + mixed) from selfie + manual measures.
    """
    try:
        selfie_bytes = await selfie.read()
        base64_selfie = base64.b64encode(selfie_bytes).decode("utf-8")

        # ✅ Prompt en inglés para IA
        prompt = f"""
You are a professional fashion AI stylist.
Analyze the provided selfie and the body measurements below.
Based on this, generate TWO photorealistic OUTFIT BASE looks for the user:
1️⃣ One *casual everyday look* (relaxed, modern, comfortable)
2️⃣ One *alternative or elegant look* (optional: semi-formal, stylish, trend-aware)

Do not distort the body or face. Keep realistic proportions.
Ensure the outfits fit naturally based on the measurements.

USER PROFILE:
- Gender: {gender}
- Body shape: {body_shape}
- Height: {height} cm
- Weight: {weight} kg
- Chest: {chest} cm
- Waist: {waist} cm
- Hips: {hips} cm
- Arms: {arms} cm

Return a JSON with the following structure:
{{
  "outfits": [
    {{
      "type": "casual",
      "description": "short summary of the outfit (colors, style, vibe)",
      "image_base64": "<base64 image>"
    }},
    {{
      "type": "alternative",
      "description": "short summary of the outfit",
      "image_base64": "<base64 image>"
    }}
  ]
}}
        """

        # ✅ Generar respuesta IA con imagen
        result = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_text(prompt),
                        types.Part.from_data(mime_type=selfie.content_type, data=selfie_bytes)
                    ]
                )
            ],
            generation_config=types.GenerationConfig(
                response_mime_type="application/json"
            )
        )

        ai_output = result.text
        try:
            data = json.loads(ai_output)
        except Exception:
            data = {"raw_output": ai_output}

        data["input_selfie_base64"] = base64_selfie
        data["profile"] = {
            "gender": gender,
            "height": height,
            "weight": weight,
            "chest": chest,
            "waist": waist,
            "hips": hips,
            "arms": arms,
            "body_shape": body_shape
        }

        return JSONResponse(content=data)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating outfits: {str(e)}")
