@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    try:
        gender = payload.get("gender", "unknown")

        # Prompt simplificado
        prompt = f"A simple outfit for a {gender} person. Neutral pose, plain background."

        image_b64 = None

        # OpenAI
        try:
            image_b64 = await openai_generate_image(prompt)
            print("✅ OpenAI generated image")
        except Exception as e_openai:
            print(f"⚠️ OpenAI failed: {e_openai}")
            try:
                image_b64 = await replicate_generate_image(prompt)
                print("✅ Replicate generated image")
            except Exception as e_replicate:
                print(f"⚠️ Replicate also failed: {e_replicate}")

        if not image_b64:
            raise HTTPException(status_code=500, detail="AI generation failed")

        return JSONResponse({
            "status": "ok",
            "demo_outfits": [image_b64],
            "generation_mode": "AI_generated"
        })

    except Exception as e:
        print("❌ Unexpected error in /generate_outfit_demo:", traceback.format_exc())
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })
