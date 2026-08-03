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
import threading
import time
import traceback
from contextlib import asynccontextmanager
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

from engine.world import World, _grid_to_b64
from engine.simulation import Simulation, site_name, SITE_OCCUPIED_R, _TOPO_ON
from engine.entities import EntityType


# ── Init simulation ──────────────────────────────────────────────────────────
# Gabarit du monde DÉPLOYÉ, configurable sans toucher au code (le gabarit de TEST du
# determinism_guard reste 220x160 et ne suit PAS cette variable : le monde servi peut
# différer du monde des goldens, comme depuis toujours).
_W = int(os.environ.get("BIOSIM_WORLD_W", "220"))
_H = int(os.environ.get("BIOSIM_WORLD_H", "160"))
world = World(width=_W, height=_H)
sim   = Simulation(world)
sim.populate()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie ASGI (remplace les @app.on_event dépréciés, retirés en Starlette 1.0).
    Le corps de démarrage/arrêt reste dans _on_startup/_on_shutdown, définis plus bas près
    de leurs helpers (_save_is_valid/_write_save) ; ils ne sont résolus qu'à l'appel (au
    démarrage/arrêt du serveur), donc la référence anticipée est sûre."""
    await _on_startup()
    yield
    await _on_shutdown()


app = FastAPI(title="BioSim", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# ── Clients WebSocket connectés ──────────────────────────────────────────────
clients: set[WebSocket] = set()

# Verrou d'état : sérialise sim.step() (offloadé dans un thread executor) et les
# lectures sim.full_state() faites sur l'event-loop (/api/world, init WS). Sans lui,
# un full_state pourrait itérer sim.entities pendant que step() la remplace → lecture
# incohérente ou "list changed size during iteration".
state_lock = asyncio.Lock()

# ── Vitesse : ticks par seconde ──────────────────────────────────────────────
tick_interval = 0.5   # secondes entre chaque tick (modifiable via API)
sim_running   = True

MAX_WS_CLIENTS         = 64   # plafond de connexions WebSocket simultanées (anti-DoS)
MAX_CONSECUTIVE_ERRORS = 20   # au-delà, on laisse la boucle crasher (systemd redémarre)
_SEND_TIMEOUT          = 2.0  # timeout d'envoi par client (un client lent ne gèle plus tout)

# ── Persistance ───────────────────────────────────────────────────────────────
# Défauts inactifs → comportement inchangé (monde neuf à chaque démarrage). Les
# endpoints /api/save et /api/load restent disponibles à la demande.
SAVE_PATH       = os.environ.get("BIOSIM_SAVE_PATH",
                                 os.path.join(os.path.dirname(__file__), "save.json"))
AUTOSAVE_TICKS  = int(os.environ.get("BIOSIM_AUTOSAVE_TICKS", "0"))  # 0 = autosave off
LOAD_ON_START   = bool(os.environ.get("BIOSIM_LOAD_ON_START"))       # charge SAVE_PATH au boot


_save_lock = threading.Lock()


def _write_save(snap: dict):
    """Écrit le snapshot JSON de façon atomique, crash-safe et sérialisée (durcissement I3,
    audit #13/#21/#50).
    - Verrou : autosave, POST /api/save et le save d'arrêt visent tous le disque via
      l'executor → sans lock, deux écrivains entrelacent leurs écritures et se disputent la
      rotation .prev, corrompant save.json.
    - tmp UNIQUE par process (le .tmp fixe partagé était le vecteur de corruption).
    - fsync du fichier PUIS du répertoire → survit à une coupure d'alim (sinon os.replace peut
      précéder le flush des données sur disque = save.json vide au prochain boot).
    - rotation .prev : le save précédent reste récupérable."""
    with _save_lock:
        tmp = f"{SAVE_PATH}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(snap, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(SAVE_PATH):
            os.replace(SAVE_PATH, SAVE_PATH + ".prev")
        os.replace(tmp, SAVE_PATH)
        dfd = os.open(os.path.dirname(SAVE_PATH) or ".", os.O_RDONLY)
        try:
            os.fsync(dfd)          # durabilité du rename lui-même
        finally:
            os.close(dfd)


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
        # ... et on FERME vraiment la socket (finding d'audit) : la retirer du set ne la ferme
        # pas. Un client lent restait connecté sans plus rien recevoir, et surtout HORS QUOTA —
        # il ne comptait plus dans MAX_WS_CLIENTS alors que sa socket vivait toujours, donc en
        # accumulant on contournait le plafond anti-DoS. Best-effort : une socket déjà morte
        # lève à la fermeture, ce n'est pas une erreur.
        await asyncio.gather(*[_close_quietly(ws) for ws in dead])


async def _close_quietly(ws: WebSocket):
    try:
        await ws.close(code=1011)      # 1011 = Internal Error : le serveur abandonne ce client
    except Exception:
        pass


async def simulation_loop():
    global tick_interval, sim_running
    consecutive_errors = 0
    loop = asyncio.get_running_loop()
    while True:
        _t_cycle = time.perf_counter()   # cadence COMPENSÉE, cf. le sleep en fin de boucle
        # La sim avance qu'il y ait des spectateurs ou non : le monde vit et s'écrit
        # tout seul (chronique, autosave). Seul le broadcast dépend des clients.
        if sim_running:
            try:
                # step() est offloadé dans un thread : un tick long (forte population)
                # ne gèle plus l'event-loop → WS, API et nouvelles connexions restent
                # réactifs pendant le calcul. Le state_lock garantit qu'aucun full_state
                # ne lit l'état pendant que step() le mute.
                async with state_lock:
                    data = await loop.run_in_executor(None, sim.step)
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
            # dumps + broadcast HORS du verrou : `data` est un snapshot de primitifs,
            # inutile de bloquer les ticks/full_state pendant la sérialisation + I/O réseau.
            # Sans client, on saute aussi la sérialisation (pur gaspillage CPU).
            if clients:
                msg = json.dumps({"type": "tick", "data": data})
                await _broadcast(msg)
            # Autosave optionnel : snapshot sous verrou (pas de step concurrent),
            # écriture disque hors verrou (I/O) → ne fige pas la boucle.
            if AUTOSAVE_TICKS and data["tick"] % AUTOSAVE_TICKS == 0:
                async with state_lock:
                    snap = sim.save_state()
                await loop.run_in_executor(None, _write_save, snap)
        # Cadence COMPENSÉE : on dort ce qui RESTE de l'intervalle, pas l'intervalle entier.
        # Avant, la période valait `travail + tick_interval` — le monde servi tournait donc à
        # 1,32 tick/s pour une cible de 2 (mesuré : step 236 ms + 500 ms = 736 ms de période),
        # et la dérive s'aggravait avec la population sans que rien ne le signale. Le serveur
        # n'était pourtant pas saturé (21 % d'un cœur) : ce n'était pas un manque de puissance,
        # c'était de la comptabilité.
        # Plancher de 10 ms quand le travail déborde l'intervalle : sans lui, un tick plus long
        # que `tick_interval` ferait tourner la boucle à plein régime et affamerait l'event-loop
        # (WS, API, nouvelles connexions) au lieu de dégrader proprement la cadence.
        await asyncio.sleep(max(0.01, tick_interval - (time.perf_counter() - _t_cycle)))


def _sim_task_done(task: "asyncio.Task"):
    """B1 : si la boucle de simulation meurt (bug déterministe → `raise` après
    MAX_CONSECUTIVE_ERRORS), on termine le PROCESS pour de vrai. Une exception
    dans une task asyncio ne tue pas uvicorn → sans ça la sim reste gelée, le
    serveur vivant, et systemd aveugle (l'intention « systemd redémarre propre »
    ne se réalisait jamais). os._exit(1) → Restart=always relance réellement.
    Un arrêt propre du serveur annule la task → on ne tue rien dans ce cas."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        traceback.print_exc()
    print("[biosim] boucle de simulation terminée → arrêt du process (redémarrage systemd)",
          flush=True)
    os._exit(1)


def _save_is_valid(path: str, probe_ticks: int = 3) -> bool:
    """Charge `path` dans une sim JETABLE et fait quelques steps de sonde. False si le
    chargement OU un step lève (durcissement I4, audit #33/#35). Sans ça, un save
    empoisonné/corrompu était rejoué tel quel à chaque restart → même crash → StartLimit
    fige le service en mort DÉFINITIVE. La sonde attrape le crash au rechargement AVANT
    d'engager la vraie boucle, ce qui autorise un repli sur .prev (ou un monde neuf)."""
    try:
        from engine.world import World as _W
        probe = Simulation(_W(width=2, height=2))
        probe.load(path)
        for _ in range(probe_ticks):
            probe.step()
        return True
    except Exception:
        traceback.print_exc()
        return False


async def _on_startup():
    # Reprise optionnelle depuis un save au démarrage, AVEC repli anti-crash-loop (I4).
    # On valide chaque candidat sur une sonde jetable ; sim.load ne s'applique qu'à un
    # save prouvé sain → un save empoisonné ne peut plus tuer le service en boucle.
    if LOAD_ON_START:
        loaded_from = None
        for candidate in (SAVE_PATH, SAVE_PATH + ".prev"):
            if not os.path.exists(candidate):
                continue
            if _save_is_valid(candidate):
                sim.load(candidate)   # déjà validé sur une sonde → chargement sûr
                loaded_from = candidate
                break
            print(f"[biosim] save '{candidate}' INVALIDE (échec chargement/sonde) → repli",
                  flush=True)
        if loaded_from:
            # flush : sous systemd stdout est bufferisé → sinon le print traîne dans le journal.
            print(f"[biosim] état rechargé depuis {loaded_from} (tick {sim.tick_count})",
                  flush=True)
        else:
            print("[biosim] aucun save valide → démarrage d'un monde neuf", flush=True)
    # Garder la ref (sinon le GC peut ramasser la task en plein vol) + callback de
    # fin qui fait redémarrer le process si la boucle meurt (B1).
    task = asyncio.create_task(simulation_loop())
    task.add_done_callback(_sim_task_done)
    app.state.sim_task = task


async def _on_shutdown():
    """Arrêt propre (SIGTERM systemd : restart, reboot) : sauvegarde l'état pour ne
    pas perdre jusqu'à AUTOSAVE_TICKS ticks de monde. On prend le verrou AVANT
    d'annuler la task : un step en cours se termine d'abord, et la task annulée ne
    peut plus muter l'état pendant le snapshot (elle bloquerait sur ce même verrou).
    `_sim_task_done` ignore une task annulée → pas d'os._exit(1) sur ce chemin."""
    task = getattr(app.state, "sim_task", None)
    snap = None
    async with state_lock:
        if task is not None:
            task.cancel()
        if AUTOSAVE_TICKS:  # persistance active seulement (défaut inchangé : rien)
            snap = sim.save_state()
    if snap is not None:
        await asyncio.get_running_loop().run_in_executor(None, _write_save, snap)
        print(f"[biosim] état sauvegardé à l'arrêt (tick {snap['tick_count']})",
              flush=True)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/world")
async def get_world():
    async with state_lock:
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


@app.get("/api/metrics")
async def metrics():
    async with state_lock:
        return sim.metrics()


@app.get("/api/chronicle")
async def chronicle():
    """Annales du monde (bloc K) : jalons persistants (âges, forges, extinctions…)."""
    async with state_lock:
        return {"chronicle": sim.chronicle}


@app.get("/api/territory")
async def territory():
    """Owner-grid du territoire (T1) : int16 H×W (clan_id possédant chaque tuile, -1 =
    sauvage/eau) encodé b64 avec son dtype (le front décode selon dtype). int16 car les
    clan_id sont un compteur monotone non borné qui déborderait l'int8 (audit #2).
    Découplé du flux de ticks (poll léger côté front, comme la
    chronique) → n'entre pas dans le hash déterministe. Le front mappe les valeurs vers
    les couleurs via la liste `clans` qu'il reçoit déjà à chaque tick."""
    async with state_lock:
        g = sim.territory_grid
        if g is None:
            return {"available": False}
        return {"available": True, "grid": _grid_to_b64(g)}


@app.get("/api/trails")
async def trails():
    """Grille des SENTIERS (P6 F3) : uint16 H×W (passages cumulés des humains sur chaque tuile,
    saturés à 65535), encodée b64 avec son dtype (le front décode selon dtype — leçon de la
    régression int16 du territoire). Purement COSMÉTIQUE : aucune décision moteur ne la lit.
    Découplée du flux de ticks (poll léger côté front, comme le territoire et la chronique) →
    n'entre pas dans le hash déterministe. Le front applique 2 seuils d'usure."""
    async with state_lock:
        g = getattr(sim.world, "trail_grid", None)
        if g is None:
            return {"available": False}
        return {"available": True, "grid": _grid_to_b64(g)}


@app.get("/api/explored")
async def explored():
    """Carte EXPLORÉE (P7 G1) : uint8 H×W, 1 = tuile déjà vue par un humain du monde. Même
    plomberie que les sentiers (b64 + dtype, poll léger hors flux de ticks) → hors hash.
    Cosmétique stricte : aucun brouillard de guerre, aucune décision moteur ne la lit."""
    async with state_lock:
        g = getattr(sim.world, "explored_grid", None)
        if g is None:
            return {"available": False}
        return {"available": True, "grid": _grid_to_b64(g)}


@app.get("/api/sites")
async def sites():
    """Sites DÉCOUVERTS (P7 G4) : union des `known_sites` des clans VIVANTS — donc dérivé, jamais
    stocké, et il rétrécit si les clans qui savaient s'éteignent. Un site jamais atteint n'est pas
    listé : le joueur voit ce que son monde a appris, pas ce que le catalogue contient.
    Hors hash, poll léger comme les sentiers et la carte explorée. `occupied` est dérivé des feux
    vivants (même rayon que le moteur) et sert au front à distinguer une terre prise d'une terre
    offerte — il ne remplace PAS le prédicat de réservation, qui reste interne."""
    async with state_lock:
        if not _TOPO_ON:
            return {"available": False}
        cat = sim.world.site_catalogue()
        connus = set()
        for c in sim.clans:
            connus.update(getattr(c, "known_sites", []) or [])
        feux = [(b.x, b.y) for b in sim.buildings if b.btype == "campfire"]
        r2 = SITE_OCCUPIED_R ** 2
        out = []
        for sid, x, y, score in cat:
            if sid not in connus:
                continue
            out.append({"id": sid, "x": x, "y": y,
                        "name": site_name(sim.world.seed, sid), "score": int(score),
                        "occupied": any((fx - x) ** 2 + (fy - y) ** 2 <= r2 for fx, fy in feux)})
        return {"available": True, "sites": out}


@app.post("/api/pause")
async def pause():
    global sim_running
    sim_running = not sim_running
    return {"running": sim_running}


@app.post("/api/save")
async def api_save():
    # Snapshot sous verrou (cohérent vs step), écriture disque hors verrou.
    async with state_lock:
        snap = sim.save_state()
    await asyncio.get_running_loop().run_in_executor(None, _write_save, snap)
    return {"ok": True, "path": SAVE_PATH, "tick": snap["tick_count"],
            "entities": len(snap["entities"])}


@app.post("/api/load")
async def api_load():
    if not os.path.exists(SAVE_PATH):
        return {"ok": False, "error": "aucun fichier de sauvegarde"}
    # Chargement sous verrou du début à la fin (mutation intégrale de l'état).
    async with state_lock:
        try:
            sim.load(SAVE_PATH)
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        tick, n = sim.tick_count, len(sim.entities)
    return {"ok": True, "tick": tick, "entities": n}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if len(clients) >= MAX_WS_CLIENTS:
        # plafond atteint → on refuse proprement (anti-DoS par accumulation de sockets)
        await websocket.close(code=1013)  # 1013 = Try Again Later
        return
    # Envoie l'état complet AVANT d'entrer dans `clients` (finding d'audit) : inscrit d'abord,
    # le nouveau venu pouvait recevoir un delta de tick AVANT son init — le front travaillait
    # alors sur un `worldData` inexistant. On accepte en échange de manquer au plus UN payload
    # de tick (le suivant arrive 500 ms après, et chaque payload se suffit) : rater un tick est
    # bénin, en recevoir un avant l'init ne l'est pas.
    # Le snapshot est pris sous state_lock (cohérent vs step()), sérialisé et envoyé hors verrou.
    try:
        async with state_lock:
            init_state = sim.full_state()
        await websocket.send_text(json.dumps({
            "type": "init",
            "data": init_state
        }))
        clients.add(websocket)
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
