# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Crear app
# =========================
app = FastAPI(
    title="AI Outfit Backend",
    version="2.0",
    description="Body analysis + AI outfit generation"
)

# =========================
# Middleware CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Routers
# =========================
from routers.analyze_body_with_face import router as analyze_router
from routers.generate_outfits_from_body_photo import router as body_photo_router
from routers.generate_outfits_from_selfie import router as selfie_router

app.include_router(analyze_router, prefix="/api")
app.include_router(body_photo_router, prefix="/api")
app.include_router(selfie_router, prefix="/api")

# =========================
# Health check
# =========================
@app.get("/")
def root():
    return {"status": "ok", "message": "Backend running"}
