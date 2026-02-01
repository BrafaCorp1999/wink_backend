from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.api_route("/api/ping", methods=["GET", "HEAD"])
async def ping():
    return JSONResponse(content={"status": "ok"})
