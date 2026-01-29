from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Outfit Backend",
    version="2.1",
    description="AI Stylist + Clothing Analysis + Virtual Try-On (Demo)"
)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5000",
        "http://localhost:8000",
        "http://127.0.0.1",
        "https://wink-e51d9.web.app",
    ],
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Routers – BODY / AI STYLIST
# =========================
from routers.analyze_body_with_face import router as analyze_router
from routers.generate_outfits_from_body_photo import router as body_photo_router
from routers.generate_outfits_from_selfie import router as selfie_router
from routers.image_to_image import router as image_to_image_router

from routers.analyze_body_web import router as analyze_web_router
from routers.generate_outfits_from_body_photo_web import router as body_photo_web_router
from routers.image_to_image_web import router as image_to_image_web_router
from routers.keep_alive import router as keep_alive_router

# =========================
# Routers – CLOTHING ANALYSIS & TRY-ON (DEMO)
# =========================
from routers.analyze_clothes import router as analyze_clothes_router
from routers.analyze_clothes_web import router as analyze_clothes_web_router
from routers.generate_try_on import router as generate_tryon_router
from routers.generate_try_on_web import router as generate_tryon_web_router

# =========================
# Register routers
# =========================
app.include_router(analyze_router, prefix="/api")
app.include_router(body_photo_router, prefix="/api")
app.include_router(selfie_router, prefix="/api")
app.include_router(image_to_image_router, prefix="/api")

app.include_router(analyze_web_router, prefix="/api")
app.include_router(body_photo_web_router, prefix="/api")
app.include_router(image_to_image_web_router, prefix="/api")

app.include_router(analyze_clothes_router, prefix="/api")
app.include_router(analyze_clothes_web_router, prefix="/api")
app.include_router(generate_tryon_router, prefix="/api")
app.include_router(generate_tryon_web_router, prefix="/api")
app.include_router(keep_alive_router, prefix="/api")

# =========================
# Root / Health
# =========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "AI Outfit Backend running"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ai-outfit-backend",
        "version": "2.1",
        "features": [
            "AI Stylist",
            "Clothing analysis (1–2 items)",
            "Virtual Try-On (demo)",
        ]
    }
