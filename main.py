from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Outfit Backend",
    version="2.0",
    description="Body analysis + AI outfit generation"
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
# Routers / Endpoints
# =========================
from routers.analyze_body_with_face import router as analyze_router
from routers.generate_outfits_from_body_photo import router as body_photo_router
from routers.generate_outfits_from_selfie import router as selfie_router
from routers.image_to_image import router as image_to_image_router
from routers.combine_clothes import router as combine_clothes_router
from routers.analyze_body_web import router as analyze_web_router

app.include_router(analyze_router, prefix="/api")
app.include_router(body_photo_router, prefix="/api")
app.include_router(selfie_router, prefix="/api")
app.include_router(image_to_image_router, prefix="/api")
app.include_router(combine_clothes_router, prefix="/api")
app.include_router(analyze_web_router, prefix="/api")

# =========================
# Root / Health
# =========================
@app.get("/")
def root():
    return {"status": "ok", "message": "Backend running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ai-outfit-backend",
        "version": "2.0"
    }
