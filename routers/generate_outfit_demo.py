@router.post("/generate_outfit_demo")
async def generate_outfit_demo(payload: dict):
    try:
        gender = payload.get("gender", "unknown")

        prompt = f"Full body portrait of a {gender} person wearing a simple outfit. Neutral pose, no face distortion."

        image_b64 = None

        # 1️⃣ Intentar OpenAI
        try:
            print(f"➡️ Generating outfit with OpenAI")
            image_b64 = await openai_generate_image(prompt)
            print(f"✅ OpenAI generated image")

        # 2️⃣ Si falla OpenAI, intentar Replicate
        except Exception as e_openai:
            print(f"⚠️ OpenAI failed: {e_openai}")
            try:
                print(f"➡️ Trying Replicate")
                image_b64 = await replicate_generate_image(prompt)
                print(f"✅ Replicate generated image")
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
