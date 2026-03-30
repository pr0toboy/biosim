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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

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


# ── Loop de simulation (tourne en arrière-plan) ─────────────────────────────
async def simulation_loop():
    global tick_interval, sim_running
    while True:
        if sim_running and clients:
            try:
                data = sim.step()
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(tick_interval)
                continue
            msg  = json.dumps({"type": "tick", "data": data})
            dead = set()
            for ws in clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.add(ws)
            clients.difference_update(dead)
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
async def set_speed(tps: float = 1.0):
    global tick_interval
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
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
