from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

@router.api_route("/ping", methods=["GET", "HEAD"])
async def ping(request: Request):
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Detectar origen del ping
    if "GitHub-Hookshot" in user_agent:
        source = "GitHub Actions"
    elif "UptimeRobot" in user_agent:
        source = "UptimeRobot"
    else:
        source = "Otro / navegador"

    # Imprime en logs de Render
    print(f"[PING] Recibido desde: {source}, User-Agent: {user_agent}")

    return JSONResponse(content={"status": "ok", "source": source})
