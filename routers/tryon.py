@router.post("/try-on")
async def try_on(
    person_image: UploadFile = File(...),
    cloth_images: List[UploadFile] = File(...),
    categories: List[str] = Form(...),  # etiquetas de cada prenda
    instructions: str = Form(""),
    model_type: str = Form(""),
    gender: str = Form(""),
    garment_type: str = Form(""),
    style: str = Form(""),
):
    try:
        # Validación y lectura de imágenes igual que antes
        MAX_IMAGE_SIZE_MB = 10
        ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

        user_bytes = await person_image.read()
        if len(user_bytes) / (1024*1024) > MAX_IMAGE_SIZE_MB:
            raise HTTPException(status_code=400, detail="person_image exceeds 10MB")

        cloth_bytes_list = []
        for img in cloth_images:
            if img.content_type not in ALLOWED_MIME_TYPES:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {img.content_type}")
            bytes_img = await img.read()
            if len(bytes_img) / (1024*1024) > MAX_IMAGE_SIZE_MB:
                raise HTTPException(status_code=400, detail="cloth_image exceeds 10MB")
            cloth_bytes_list.append(bytes_img)

        # Construir prompt dinámico según las prendas
        garments_info = "\n".join([f"- {cat}: apply realistically" for cat in categories])
        prompt = f"""
        Generate a photorealistic full-body try-on image for the user.
        Strictly preserve the user's face and proportions.
        Apply all selected garments realistically:
        {garments_info}
        Maintain correct proportions, preserve facial identity, do not crop the person.
        Instructions: {instructions}
        Model Type: {model_type}
        Gender: {gender}
        Style: {style}
        """

        # Preparar inputs para Gemini (puedes usar tu código actual)
        contents = [prompt, types.Part.from_bytes(data=user_bytes, mime_type=person_image.content_type)]
        for b in cloth_bytes_list:
            contents.append(types.Part.from_bytes(data=b, mime_type="image/png"))

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
        )

        # Procesar respuesta igual que antes
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
