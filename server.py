"""
BioSim — Serveur FastAPI
- GET  /          → interface web
- GET  /api/world → état initial (JSON)
- WS   /ws        → stream des ticks en temps réel
- POST /api/speed → change la vitesse de simulation
"""
import asyncio
import json
import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
# uvicorn n'est importé que pour l'exécution directe (`python server.py`), pas pour
# l'import du module en tant qu'app ASGI (`uvicorn server:app`, tests…) → import paresseux.

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _get_sysinfo() -> dict:
    cpu = ram = temp = None
    if _HAS_PSUTIL:
        cpu = psutil.cpu_percent(interval=None)
        vm  = psutil.virtual_memory()
        ram = round(vm.used / vm.total * 100, 1)
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = round(int(f.read().strip()) / 1000, 1)
    except Exception:
        pass
    return {"cpu": cpu, "ram": ram, "temp": temp}

from engine.world import World
from engine.simulation import Simulation
from engine.entities import EntityType


# ── Init simulation ──────────────────────────────────────────────────────────
world = World(width=220, height=160)
sim   = Simulation(world)
sim.populate()

app = FastAPI(title="BioSim")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# ── Clients WebSocket connectés ──────────────────────────────────────────────
clients: set[WebSocket] = set()

# ── Vitesse : ticks par seconde ──────────────────────────────────────────────
tick_interval = 0.5   # secondes entre chaque tick (modifiable via API)
sim_running   = True

MAX_WS_CLIENTS         = 64   # plafond de connexions WebSocket simultanées (anti-DoS)
MAX_CONSECUTIVE_ERRORS = 20   # au-delà, on laisse la boucle crasher (systemd redémarre)
_SEND_TIMEOUT          = 2.0  # timeout d'envoi par client (un client lent ne gèle plus tout)


# ── Loop de simulation (tourne en arrière-plan) ─────────────────────────────
async def _send_one(ws: WebSocket, msg: str):
    """Envoie à un client avec timeout ; retourne le ws s'il faut le retirer."""
    try:
        await asyncio.wait_for(ws.send_text(msg), timeout=_SEND_TIMEOUT)
        return None
    except Exception:
        return ws


async def _broadcast(msg: str):
    """Diffuse en parallèle sur un snapshot du set (qui peut muter pendant les
    await → itérer `clients` directement lèverait RuntimeError). Purge morts/lents."""
    targets = list(clients)
    if not targets:
        return
    results = await asyncio.gather(*[_send_one(ws, msg) for ws in targets])
    dead = {ws for ws in results if ws is not None}
    if dead:
        clients.difference_update(dead)


async def simulation_loop():
    global tick_interval, sim_running
    consecutive_errors = 0
    while True:
        if sim_running and clients:
            try:
                data = sim.step()
            except Exception:
                consecutive_errors += 1
                traceback.print_exc()
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    # bug déterministe : on ne masque plus indéfiniment (spam SD +
                    # sim gelée). On laisse la task crasher → systemd redémarre propre.
                    raise
                await asyncio.sleep(tick_interval)
                continue
            consecutive_errors = 0
            msg = json.dumps({"type": "tick", "data": data})
            await _broadcast(msg)
        await asyncio.sleep(tick_interval)


@app.on_event("startup")
async def startup():
    asyncio.create_task(simulation_loop())


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/world")
async def get_world():
    return sim.full_state()


@app.post("/api/speed")
async def set_speed(tps: float = Query(1.0, gt=0, le=20)):
    global tick_interval
    # tps ∈ (0, 20] garanti par Query → pas de ZeroDivisionError, borne haute maîtrisée
    tick_interval = max(0.05, min(5.0, 1.0 / tps))
    return {"tps": tps, "interval": tick_interval}


@app.get("/api/sysinfo")
async def sysinfo():
    return _get_sysinfo()


@app.post("/api/pause")
async def pause():
    global sim_running
    sim_running = not sim_running
    return {"running": sim_running}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if len(clients) >= MAX_WS_CLIENTS:
        # plafond atteint → on refuse proprement (anti-DoS par accumulation de sockets)
        await websocket.close(code=1013)  # 1013 = Try Again Later
        return
    clients.add(websocket)
    # Envoie l'état complet au nouveau client
    try:
        await websocket.send_text(json.dumps({
            "type": "init",
            "data": sim.full_state()
        }))
        while True:
            # Garde la connexion ouverte, attend des messages (ping/pong)
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception:
        clients.discard(websocket)


if __name__ == "__main__":
    # host/port configurables via env. Défaut 0.0.0.0 pour conserver l'accès LAN
    # (ex. http://megatron:8080). Poser BIOSIM_HOST=127.0.0.1 pour restreindre au
    # loopback (derrière un reverse-proxy / tunnel) si on veut durcir l'exposition.
    import uvicorn
    host = os.environ.get("BIOSIM_HOST", "0.0.0.0")
    port = int(os.environ.get("BIOSIM_PORT", "8080"))
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
