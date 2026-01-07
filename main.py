from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Outfit Backend",
    version="2.0",
    description="Body analysis + AI outfit generation"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "https://*.web.app",
        "https://*.firebaseapp.com",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routers.analyze_body_with_face import router as analyze_router
from routers.generate_outfits_from_body_photo import router as body_photo_router
from routers.generate_outfits_from_selfie import router as selfie_router
from routers.generate_outfit_image_to_image import router as image_to_image_router

app.include_router(analyze_router, prefix="/api")
app.include_router(body_photo_router, prefix="/api")
app.include_router(selfie_router, prefix="/api")
app.include_router(image_to_image_router, prefix="/api")

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
