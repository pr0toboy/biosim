"""
BioSim — Boucle de simulation
Appelée à chaque tick. Gère IA, déplacements, reproduction, mort.
"""
import random
import math
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING
from .entities import Entity, EntityType, Sex, State, SPECS, spawn, BUILDING_SPECS
from .world import Biome, TREE_STUMP_THRESHOLD, STONE_STUMP_THRESHOLD, FERTILITY_TRAMPLE

# ── Clans ─────────────────────────────────────────────────────────────────────
N_CLANS = 4
CLAN_COLORS = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6"]

@dataclass
class Clan:
    id: int
    cx: float        # centre du territoire (feu de camp)
    cy: float
    color: str
    chief_id: int

    def to_dict(self):
        return {"id": self.id, "cx": self.cx, "cy": self.cy,
                "color": self.color, "chief_id": self.chief_id}


_next_building_id = 0

@dataclass
class Building:
    id: int
    clan_id: int
    x: int
    y: int
    btype: str = "house"
    wood: int  = 0          # bois stocké dans ce bâtiment
    stone: int = 0          # pierre stockée dans ce bâtiment
    level: int = 1          # niveau du bâtiment (1 ou 2)
    stage: int = 1          # wheatfield : stade de croissance 1-4 (4 = mûr)
    grow_ticks: int = 0     # ticks accumulés dans le stade courant
    watered_ticks: int = 0  # ticks restants de croissance accélérée (arrosoir)
    wheat: int = 0          # mill : blé stocké (livré par les fermiers)
    bread: int = 0          # mill : pains prêts à consommer
    mill_ticks: int = 0     # mill : ticks de cuisson en cours
    work_done: int = 0      # ticks de travail déjà effectués sur un chantier
    work_needed: int = 0    # ticks de travail total nécessaires pour terminer

    def to_dict(self):
        d = {"id": self.id, "clan_id": self.clan_id,
             "x": self.x, "y": self.y, "btype": self.btype,
             "wood": self.wood, "stone": self.stone, "level": self.level}
        if self.work_needed > 0:
            d["work_done"] = self.work_done
            d["work_needed"] = self.work_needed
        if self.btype == "wheatfield":
            d["stage"]   = self.stage
            d["watered"] = self.watered_ticks > 0
        elif self.btype == "mill":
            d["wheat"] = self.wheat
            d["bread"] = self.bread
        return d


# ── Bâtiments & outils ────────────────────────────────────────────────────────
AXE_CRAFT_COST      = 5    # bois stocké pour fabriquer une hache en bois
AXE_BONUS           = 3    # bois bonus par coup avec hache bois (total 2+3=5)
STONE_AXE_COST      = 5    # pierre stockée pour upgrader → hache en pierre
STONE_AXE_BONUS     = 5    # bois bonus par coup avec hache pierre (total 2+5=7)
PICK_WOOD_COST      = 8    # bois stocké pour fabriquer une pioche en bois
STONE_PICK_COST     = 5    # pierre stockée pour upgrader → pioche en pierre
STONE_PICK_BONUS    = 1    # pierre bonus par minage avec pioche pierre (total 1+1=2)
MAX_STONE_CARRY     = 3    # pierre max transportable par un humain

# ── Champ de blé ──────────────────────────────────────────────────────────────
WHEAT_TICKS_PER_STAGE  = 180   # ticks pour passer d'un stade au suivant (×3 = 540 total)
WHEAT_HARVEST_FOOD     = 45.0  # réduction de faim lors de la récolte
WHEAT_HUNGER_THRESH    = 35    # faim min pour décider de récolter (chercher de la nourriture)
WHEAT_WORK_THRESH      = 65   # faim max pour travailler les champs (planter/arroser quand reposé)
SICKLE_STONE_COST      = 5    # pierre pour fabriquer une faucille
SICKLE_HARVEST_BONUS   = 25.0 # nourriture bonus récupérée avec la faucille
WATERING_CAN_WOOD_COST = 6    # bois pour fabriquer un arrosoir
WATERED_TICKS          = 25   # ticks d'accélération de croissance après arrosage (×2)
FISHING_ROD_WOOD_COST  = 4    # bois pour fabriquer une canne à pêche
FISHING_CATCH_PROB     = 0.10 # probabilité de prise par tick (~1 poisson / 10 ticks)
FISHING_FOOD           = 35.0 # réduction de faim par prise
FISHING_HUNGER_THRESH  = 40   # faim min pour décider de pêcher

# ── Météo / Pluie ─────────────────────────────────────────────────────────────
RAIN_PROB = {
    "spring": 0.004,   # ~1 épisode / 250 ticks
    "summer": 0.001,
    "autumn": 0.006,
    "winter": 0.002,
}
STORM_PROB = {        # probabilité qu'un épisode pluvieux soit un orage
    "spring": 0.25,
    "summer": 0.45,   # orages d'été fréquents
    "autumn": 0.20,
    "winter": 0.05,
}
RAIN_DURATION_MIN  = 40    # ticks min d'un épisode pluvieux
RAIN_DURATION_MAX  = 120   # ticks max
RAIN_THIRST_REDUCE = 0.06  # soif retirée / tick sous la pluie
RAIN_SPEED_MULT    = 0.65  # vitesse humains sous la pluie
RAIN_WHEAT_BONUS   = 1     # grow_ticks bonus / tick (cumul avec arrosoir)
LIGHTNING_PROB     = 0.015 # probabilité de foudre par tick d'orage (frappe un arbre aléatoire)

# ── Canicule ──────────────────────────────────────────────────────────────────
HEATWAVE_PROB         = 0.0015  # probabilité de déclenchement par tick (été uniquement)
HEATWAVE_DURATION_MIN = 60      # ticks
HEATWAVE_DURATION_MAX = 180
HEATWAVE_THIRST_MULT  = 2.2     # multiplicateur de soif pendant la canicule
HEATWAVE_FIRE_PROB    = 0.008   # probabilité d'incendie spontané par tick de canicule

# ── Moulin ────────────────────────────────────────────────────────────────────
MILL_BREAD_COST_WHEAT  = 1    # 1 récolte de blé (stockée) par pain
MILL_BREAD_TICKS       = 80   # ticks de production d'un pain
MILL_BREAD_FOOD        = 65.0 # réduction de faim en mangeant un pain
MILL_MAX_BREAD         = 5    # stock max de pains dans un moulin

# ── Saisons ──────────────────────────────────────────────────────────────────
TICKS_PER_SEASON = 300   # 1200 ticks = 1 année complète

SEASON_NAMES = ["spring", "summer", "autumn", "winter"]

# Multiplicateur de régénération de nourriture par saison
SEASON_REGEN_MULT = {
    "spring": 1.6,
    "summer": 1.1,
    "autumn": 0.7,
    "winter": 0.32,
}

def get_season(tick: int) -> str:
    return SEASON_NAMES[(tick // TICKS_PER_SEASON) % 4]

# Températures moyennes par saison (°C) avec amplitude ±
SEASON_TEMP = {
    "spring": (12.0, 6.0),   # moy, amplitude
    "summer": (28.0, 6.0),
    "autumn": (10.0, 5.0),
    "winter": (-3.0, 5.0),
}

def get_temperature(tick: int) -> float:
    """Retourne la température en °C (sinusoïde lisse dans la saison)."""
    season = get_season(tick)
    tick_in_season = tick % TICKS_PER_SEASON
    phase = (tick_in_season / TICKS_PER_SEASON) * math.pi  # 0 → π
    mean, amp = SEASON_TEMP[season]
    return round(mean + amp * math.sin(phase), 1)

def _hunger_mult(temp_c: float) -> float:
    """Froid → faim accélérée. < 5°C : ×1.5 max, > 20°C : ×0.85 (chaleur coupe l'appétit)."""
    if temp_c <= 5.0:
        return 1.0 + (5.0 - temp_c) * 0.033   # ≈ ×1.5 à -10°C
    elif temp_c >= 20.0:
        return max(0.85, 1.0 - (temp_c - 20.0) * 0.01)
    return 1.0

def _thirst_mult(temp_c: float) -> float:
    """Chaud → soif accélérée. > 20°C : ×2.0 max, < 5°C : ×0.5."""
    if temp_c >= 20.0:
        return 1.0 + (temp_c - 20.0) * 0.067   # ≈ ×2.0 à 35°C
    elif temp_c <= 5.0:
        return max(0.5, 1.0 - (5.0 - temp_c) * 0.033)
    return 1.0

# La reproduction n'est possible qu'au printemps, été et automne (début)
SEASON_REPRO_ALLOWED = {
    "spring": True,
    "summer": True,
    "autumn": True,   # une dernière portée avant l'hiver
    "winter": False,
}


# ── Puit ──────────────────────────────────────────────────────────────────────
WELL_DRINK_AMOUNT = 85.0   # soif retirée en buvant au puit (même valeur qu'une tuile eau)


# ── Soif ─────────────────────────────────────────────────────────────────────
THIRST_RATE = 0.04    # soif gagnée par tick
MAX_THIRST  = 100.0   # mort au-delà
DRINK_AMOUNT = 85.0   # soif retirée en buvant une fois

# Biomes considérés comme sources d'eau potable
WATER_BIOMES = {int(Biome.WATER), int(Biome.RIVER)}

if TYPE_CHECKING:
    from .world import World


def _dist(ax, ay, bx, by) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _dist_hitbox(entity: Entity, target: Entity) -> float:
    """Distance minimale entre les tuiles du hitbox de entity et target.
    Utilise numpy pour les entités multi-tuiles (hitbox_width > 1 ou hitbox_height > 1).
    Exemple requin : hitbox_width=1, hitbox_height=2 → grille 1×2 centrée sur entity."""
    hw = entity.spec.hitbox_width
    hh = entity.spec.hitbox_height
    if hw <= 1 and hh <= 1:
        return _dist(entity.x, entity.y, target.x, target.y)
    x_off = np.linspace(-(hw - 1) / 2.0, (hw - 1) / 2.0, hw)  # shape (hw,)
    y_off = np.linspace(-(hh - 1) / 2.0, (hh - 1) / 2.0, hh)  # shape (hh,)
    # Grille cartésienne hw × hh de points du hitbox
    gx = entity.x + x_off[:, None]          # (hw, 1)  → broadcast → (hw, hh)
    gy = entity.y + y_off[None, :]          # (1,  hh) → broadcast → (hw, hh)
    dists = np.hypot(gx - target.x, gy - target.y)
    return float(dists.min())


STUCK_TICKS_RESET = 25   # ticks sans déplacement avant reset de la cible

def _move_toward(entity: Entity, tx: float, ty: float, speed: float, world: "World"):
    """Déplace l'entité d'un pas vers (tx, ty), sans dépasser la vitesse max.
    Si le chemin direct est bloqué (eau), glisse le long de l'obstacle.
    Si vraiment bloqué (STUCK_TICKS_RESET ticks), reset la cible et essaie une échappée."""
    dx = tx - entity.x
    dy = ty - entity.y
    d  = math.sqrt(dx * dx + dy * dy)
    if d < 0.01:
        entity._stuck_ticks = 0
        return
    step = min(speed, d)
    nx = entity.x + (dx / d) * step
    ny = entity.y + (dy / d) * step
    aq = entity.spec.aquatic
    ox, oy = entity.x, entity.y   # position avant mouvement
    moved = False
    if world.is_walkable(int(nx), int(ny), aq):
        entity.x = max(0, min(world.width  - 0.01, nx))
        entity.y = max(0, min(world.height - 0.01, ny))
        moved = True
    elif world.is_walkable(int(nx), int(entity.y), aq):
        entity.x = max(0, min(world.width  - 0.01, nx))
        moved = True
    elif world.is_walkable(int(entity.x), int(ny), aq):
        entity.y = max(0, min(world.height - 0.01, ny))
        moved = True

    # Micro-mouvement (< 0.01 tile/tick) : traité comme bloqué pour éviter les boucles
    # de virgule flottante (ex : entité à x=42.999 glissant vers x=43.0 au rythme de 0.0001/tick)
    really_moved = moved and ((entity.x - ox)**2 + (entity.y - oy)**2 >= 0.0001)
    if really_moved:
        entity._stuck_ticks = 0
    else:
        entity._stuck_ticks += 1
    if entity._stuck_ticks >= STUCK_TICKS_RESET:
        # Cible inaccessible : reset + annule chantier planifié + échappée aléatoire
        entity.target_x = None
        entity.target_y = None
        entity._build_target_type = None
        entity._build_target_x    = None
        entity._build_target_y    = None
        entity._stuck_ticks = 0
        dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        random.shuffle(dirs)
        for ddx, ddy in dirs:
            ex = max(0.0, min(world.width  - 0.01, entity.x + ddx))
            ey = max(0.0, min(world.height - 0.01, entity.y + ddy))
            if world.is_walkable(int(ex), int(ey), aq):
                entity.x, entity.y = ex, ey
                break


def _random_walk(entity: Entity, world: "World"):
    """Déambule aléatoirement."""
    tx, ty = entity.target_x, entity.target_y
    if tx is None or (
        (entity.x - tx)**2 + (entity.y - ty)**2 < 0.25
    ):
        entity.target_x = None
        entity.target_y = None
        for radius in (5, 12, 25):
            for _ in range(12):
                tx = entity.x + random.uniform(-radius, radius)
                ty = entity.y + random.uniform(-radius, radius)
                tx = max(0, min(world.width  - 1, tx))
                ty = max(0, min(world.height - 1, ty))
                if world.is_walkable(int(tx), int(ty), entity.spec.aquatic):
                    entity.target_x = tx
                    entity.target_y = ty
                    break
            if entity.target_x is not None:
                break
    if entity.target_x is not None:
        _move_toward(entity, entity.target_x, entity.target_y,
                     entity.traits["speed"] * 0.6, world)


MAX_CARRY = 5          # bois max transportable par un humain
MAX_WOOD_PER_HOUSE = 50  # bois max stocké par maison individuelle
CLAN_WOOD_CAP = 100      # total bois clan → arrête de couper (laisse de la place aux autres tâches)
MAX_PER_SPECIES = 200  # cap d'entités par espèce

HERD_RADIUS = 8    # tuiles de détection des congénères
CHOP_COOLDOWN = 12  # ticks minimum entre deux coupes d'arbre
HERD_MIN    = 2    # membres min pour déclencher le comportement de troupeau


def _herd_move(entity: Entity, all_entities: list[Entity], world: "World") -> bool:
    """Déplace vers le centroïde du troupeau proche. Retourne True si appliqué."""
    sx, sy, count = 0.0, 0.0, 0
    herd_r_sq = HERD_RADIUS * HERD_RADIUS
    ex, ey = entity.x, entity.y
    etype = entity.etype
    for e in all_entities:
        if e is entity or not e.alive or e.etype != etype:
            continue
        dx = ex - e.x; dy = ey - e.y
        if dx*dx + dy*dy <= herd_r_sq:
            sx += e.x
            sy += e.y
            count += 1
            if count >= 15:   # cap pour ne pas scanner tout le monde
                break
    if count < HERD_MIN:
        return False
    cx, cy = sx / count, sy / count
    dx = ex - cx; dy = ey - cy
    if dx*dx + dy*dy < 6.25:
        # Déjà au cœur du troupeau → disperser activement
        angle = random.uniform(0, 2 * math.pi)
        spread_r = random.uniform(3, 7)
        tx = max(0, min(world.width  - 1, entity.x + math.cos(angle) * spread_r))
        ty = max(0, min(world.height - 1, entity.y + math.sin(angle) * spread_r))
        if world.is_walkable(int(tx), int(ty), entity.spec.aquatic):
            entity.target_x = tx
            entity.target_y = ty
            _move_toward(entity, tx, ty, entity.traits["speed"] * 0.7, world)
        return True
    # Léger bruit pour éviter les trajectoires rigides
    tx = max(0, min(world.width  - 1, cx + random.uniform(-2, 2)))
    ty = max(0, min(world.height - 1, cy + random.uniform(-2, 2)))
    if not world.is_walkable(int(tx), int(ty), entity.spec.aquatic):
        return False
    entity.target_x = tx
    entity.target_y = ty
    _move_toward(entity, tx, ty, entity.traits["speed"] * 0.5, world)
    return True


def _find_nearest(entity: Entity, entities: list[Entity],
                  etype: EntityType, max_dist: float) -> "Entity|None":
    best, best_d_sq = None, max_dist * max_dist
    ex, ey = entity.x, entity.y
    for e in entities:
        if e is entity or not e.alive or e.etype != etype:
            continue
        dx = ex - e.x; dy = ey - e.y
        d_sq = dx*dx + dy*dy
        if d_sq < best_d_sq:
            best_d_sq = d_sq
            best = e
    return best


def _find_predator_nearby(entity: Entity, entities: list[Entity],
                          predators: list = None,
                          predator_grid: dict = None) -> "Entity|None":
    fd = entity.spec.flee_distance
    fd_sq = fd * fd
    ex, ey = entity.x, entity.y
    etype = entity.etype
    # Grille spatiale : ne cherche que dans les cellules proches (8×8 tuiles)
    if predator_grid is not None:
        cx, cy = entity.ix >> 3, entity.iy >> 3
        for gcx in range(cx - 1, cx + 2):
            for gcy in range(cy - 1, cy + 2):
                for e in predator_grid.get((gcx, gcy), ()):
                    if not e.alive: continue
                    if etype in e.spec.prey_types:
                        dx = ex - e.x; dy = ey - e.y
                        if dx*dx + dy*dy <= fd_sq:
                            return e
        return None
    scan = predators if predators is not None else entities
    for e in scan:
        if not e.alive:
            continue
        if etype in e.spec.prey_types:
            dx = ex - e.x; dy = ey - e.y
            if dx*dx + dy*dy <= fd_sq:
                return e
    return None


def _find_water_spot(entity: Entity, world: "World", max_r: int = 40) -> "tuple[float,float]|None":
    """Trouve la tuile marchable adjacente à de l'eau la plus proche.
    Utilise la carte BFS pré-calculée world._nearest_water_tile pour répondre en O(1)."""
    ex, ey = entity.ix, entity.iy
    if not world.is_valid(ex, ey):
        return None
    pos = world._nearest_water_tile[ey, ex]
    if int(pos[0]) >= 0:
        return (float(pos[0]), float(pos[1]))
    return None


def _drink_or_seek_water(entity: Entity, world: "World", events: list[dict],
                         buildings: list = None, cb: dict = None) -> bool:
    """Essaie de boire si adjacent à de l'eau ou à un puit, sinon cherche et va vers la source.
    Retourne True si l'action soif a été traitée (l'entité ne doit pas faire autre chose)."""
    # Vérifie si une tuile adjacente est de l'eau
    for adx in range(-1, 2):
        for ady in range(-1, 2):
            if adx == 0 and ady == 0:
                continue
            wx, wy = entity.ix + adx, entity.iy + ady
            if world.is_valid(wx, wy) and int(world.biome_grid[wy, wx]) in WATER_BIOMES:
                entity.state = State.DRINKING
                entity.thirst = max(0.0, entity.thirst - DRINK_AMOUNT)
                entity.target_x = None
                entity.target_y = None
                return True

    # Vérifie si un puit est adjacent (utilise l'index cb si disponible)
    _wells = (cb.get("well", []) if cb is not None
              else ([b for b in buildings if b.btype == "well"] if buildings else []))
    ex, ey = entity.x, entity.y
    for b in _wells:
        dx = ex - b.x; dy = ey - b.y
        if dx*dx + dy*dy < 2.25:
            entity.state = State.DRINKING
            entity.thirst = max(0.0, entity.thirst - WELL_DRINK_AMOUNT)
            entity.target_x = None
            entity.target_y = None
            return True

    # Pas d'eau adjacente : cherche un puit proche d'abord, sinon une tuile eau
    entity.state = State.SEEKING_WATER
    # Invalider la cible si elle pointe vers une tuile non-marchable (ex: eau)
    # Note : on laisse _move_toward gérer les blocages (STUCK_TICKS_RESET=25) — ne pas
    # réinitialiser prématurément la cible sur _stuck_ticks, ce qui boucle à l'infini.
    if entity.target_x is not None:
        _tx, _ty = int(entity.target_x), int(entity.target_y)
        if (not world.is_valid(_tx, _ty)
                or not world.is_walkable(_tx, _ty, entity.spec.aquatic)):
            entity.target_x = None
            entity.target_y = None
            entity._stuck_ticks = 0
        else:
            _move_toward(entity, entity.target_x, entity.target_y,
                         entity.traits["speed"] * 1.2, world)
            return True

    # Puit du clan en priorité (évite les longs trajets jusqu'à la rivière)
    if _wells:
        nearest = min(_wells, key=lambda b: (ex - b.x)**2 + (ey - b.y)**2)
        entity.target_x = float(nearest.x)
        entity.target_y = float(nearest.y)
        _move_toward(entity, entity.target_x, entity.target_y,
                     entity.traits["speed"] * 1.2, world)
        return True

    spot = _find_water_spot(entity, world)
    if spot:
        entity.target_x, entity.target_y = spot
        _move_toward(entity, entity.target_x, entity.target_y,
                     entity.traits["speed"] * 1.2, world)
        return True
    return False


def tick_entity(entity: Entity, world: "World", all_entities: list[Entity],
                births: list[Entity], events: list[dict], tick: int,
                season: str = "spring", clans: dict = None,
                buildings: list = None,
                temp_c: float = 12.0, species_counts: dict = None,
                raining: bool = False, heatwave: bool = False,
                clan_bldg: dict = None, predators: list = None,
                predator_grid: dict = None):
    """clan_bldg : {clan_id: {btype: [Building]}} pré-calculé dans step() pour éviter O(n²).
    predators : liste pré-filtrée des prédateurs actifs pour _find_predator_nearby."""
    if not entity.alive:
        return

    # Reset du verrou intra-tick de construction
    entity.building_type = None
    # Décrémente cooldown de coupe
    if entity.chop_cooldown_left > 0:
        entity.chop_cooldown_left -= 1

    spec = entity.spec

    # Index de bâtiments du clan (évite les list comprehensions répétées)
    _cb = clan_bldg.get(entity.clan_id, {}) if clan_bldg and entity.clan_id is not None else {}

    # Sanity clamp : corrige la vitesse si elle a dévié hors de ±60% de la spec de base
    _spec_spd = entity.spec.speed
    if not (_spec_spd * 0.40 <= entity.traits["speed"] <= _spec_spd * 1.80):
        entity.traits["speed"] = round(_spec_spd * random.uniform(0.95, 1.05), 4)

    # Vitesse effective ce tick (pluie ralentit les humains sans modifier le trait permanent)
    _eff_speed = (round(entity.traits["speed"] * RAIN_SPEED_MULT, 4)
                  if raining and entity.etype == EntityType.HUMAN
                  else entity.traits["speed"])

    # ── Récupération si sur une tuile non-franchissable (eau) ─────────────
    if not world.is_walkable(entity.ix, entity.iy, spec.aquatic):
        for r in range(1, 10):
            for ddx, ddy in [(-r,0),(r,0),(0,-r),(0,r),
                              (-r,-r),(r,-r),(-r,r),(r,r)]:
                nx, ny = entity.ix + ddx, entity.iy + ddy
                if world.is_valid(nx, ny) and world.is_walkable(nx, ny, spec.aquatic):
                    entity.x = float(nx) + 0.5
                    entity.y = float(ny) + 0.5
                    entity.target_x = None
                    entity.target_y = None
                    break
            else:
                continue
            break

    # ── Vieillissement, faim & soif ───────────────────────────────────────
    entity.age    += 1
    entity.hunger += entity.traits["hunger_rate"] * _hunger_mult(temp_c)
    if not spec.aquatic:
        entity.thirst += THIRST_RATE * _thirst_mult(temp_c)
        if heatwave:
            entity.thirst += THIRST_RATE * (HEATWAVE_THIRST_MULT - 1.0)
        if raining:
            entity.thirst = max(0.0, entity.thirst - RAIN_THIRST_REDUCE)
        # Piétinement : présence sur herbe → fertilité réduite chaque tick
        world.consume_fertility(entity.ix, entity.iy, FERTILITY_TRAMPLE)

    if entity.repro_cooldown_left > 0:
        entity.repro_cooldown_left -= 1

    # Mort de vieillesse
    if entity.age > spec.max_age:
        entity.alive = False
        entity.state = State.DEAD
        events.append({"type": "death", "cause": "age",
                       "etype": entity.etype.value, "x": entity.ix, "y": entity.iy})
        return

    # Mort de faim
    if entity.hunger >= spec.max_hunger:
        entity.alive = False
        entity.state = State.DEAD
        events.append({"type": "death", "cause": "hunger",
                       "etype": entity.etype.value, "x": entity.ix, "y": entity.iy})
        return

    # Mort de soif
    if not spec.aquatic and entity.thirst >= MAX_THIRST:
        entity.alive = False
        entity.state = State.DEAD
        events.append({"type": "death", "cause": "thirst",
                       "etype": entity.etype.value, "x": entity.ix, "y": entity.iy})
        return

    # ── Gestation ─────────────────────────────────────────────────────────
    if entity.gestation_left > 0:
        entity.gestation_left -= 1
        if entity.gestation_left == 0:
            # Cap par espèce : inclut les naissances déjà prévues ce tick
            current = (species_counts or {}).get(entity.etype.value, 0)
            births_same = sum(1 for b in births if b.etype == entity.etype)
            if current + births_same >= MAX_PER_SPECIES:
                entity.state = State.RESTING
                return
            n = random.randint(*spec.litter_size)
            for _ in range(n):
                if current + births_same >= MAX_PER_SPECIES:
                    break
                jx = entity.x + random.uniform(-1, 1)
                jy = entity.y + random.uniform(-1, 1)
                jx = max(0, min(world.width  - 1, jx))
                jy = max(0, min(world.height - 1, jy))
                if world.is_walkable(int(jx), int(jy), spec.aquatic):
                    baby = spawn(entity.etype, jx, jy)
                    baby.age = 0
                    baby.hunger = 20
                    baby.clan_id = entity.clan_id
                    # Hérite des traits de la mère + mutation ±5%
                    baby.traits = {
                        "speed":       max(0.1,  round(entity.traits["speed"]       * random.uniform(0.95, 1.05), 4)),
                        "vision":      max(1,     round(entity.traits["vision"]      * random.uniform(0.95, 1.05))),
                        "hunger_rate": max(0.005, round(entity.traits["hunger_rate"] * random.uniform(0.95, 1.05), 5)),
                    }
                    births.append(baby)
                    births_same += 1
            events.append({"type": "birth", "etype": entity.etype.value,
                           "count": n, "x": entity.ix, "y": entity.iy,
                           "clan": entity.clan_id})
        entity.state = State.RESTING
        return

    # ── Machine à états ─────────────────────────────────────────────────────

    # 1. Fuite (priorité max pour les proies)
    if spec.flee_distance > 0:
        predator = _find_predator_nearby(entity, all_entities, predators, predator_grid)
        if predator:
            entity.state = State.FLEEING
            dx = entity.x - predator.x
            dy = entity.y - predator.y
            d  = math.sqrt(dx*dx + dy*dy) or 1
            # Essaie plusieurs angles de fuite (direct puis ±45° puis ±90°)
            fled = False
            for angle_offset in (0, 0.785, -0.785, 1.571, -1.571):
                ca, sa = math.cos(angle_offset), math.sin(angle_offset)
                fdx = (dx/d) * ca - (dy/d) * sa
                fdy = (dx/d) * sa + (dy/d) * ca
                tx = max(0, min(world.width  - 1, entity.x + fdx * 6))
                ty = max(0, min(world.height - 1, entity.y + fdy * 6))
                if world.is_walkable(int(tx), int(ty), spec.aquatic):
                    entity.target_x = tx
                    entity.target_y = ty
                    fled = True
                    break
            if fled:
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed * 1.4, world)
            return

    # 2.5 Soif urgente (prioritaire sur la chasse)
    if not spec.aquatic and entity.thirst > 70:
        if _drink_or_seek_water(entity, world, events, buildings, cb=_cb):
            return

    # 2. Chasse (prédateurs)
    if spec.is_predator and entity.hunger > 55:
        for prey_type in spec.prey_types:
            # Préserve les proies si leur population est trop basse
            if (species_counts or {}).get(prey_type.value, 0) < 30:
                continue
            prey = _find_nearest(entity, all_entities, prey_type, entity.traits["vision"])
            if prey:
                entity.state  = State.HUNTING
                entity.target_id = prey.id
                _move_toward(entity, prey.x, prey.y, _eff_speed, world)
                # Capture si assez proche (hitbox étendu pour entités larges)
                catch_r = 0.8 + (entity.spec.hitbox_width - 1) * 0.5
                if _dist_hitbox(entity, prey) < catch_r:
                    prey.alive = False
                    prey.state = State.DEAD
                    entity.hunger = max(0, entity.hunger - spec.eat_meat)
                    events.append({"type": "kill",
                                   "predator": entity.etype.value,
                                   "prey": prey.etype.value,
                                   "x": entity.ix, "y": entity.iy})
                return

    # 2b. Conflit inter-clan (humains très affamés attaquent clan ennemi)
    if (entity.etype == EntityType.HUMAN and entity.clan_id is not None
            and entity.hunger > 65 and clans):
        for e in all_entities:
            if (e is not entity and e.alive
                    and e.etype == EntityType.HUMAN
                    and e.clan_id is not None
                    and e.clan_id != entity.clan_id):
                d = _dist(entity.x, entity.y, e.x, e.y)
                if d < entity.traits["vision"]:
                    entity.state = State.HUNTING
                    _move_toward(entity, e.x, e.y, _eff_speed * 1.1, world)
                    if d < 0.8:
                        e.alive = False
                        e.state = State.DEAD
                        entity.hunger = max(0, entity.hunger - 25)
                        events.append({"type": "clan_fight",
                                       "attacker_clan": entity.clan_id,
                                       "victim_clan":   e.clan_id,
                                       "x": entity.ix, "y": entity.iy})
                    return

    # 3.0 Mange le pain au moulin (humains affamés, moulin adjacent avec du pain)
    if entity.spec.can_build and entity.clan_id is not None and entity.hunger > 35:
        clan_mills = _cb.get("mill", [])
        mill_adj = next((m for m in clan_mills
                         if m.bread > 0 and _dist(entity.x, entity.y, m.x, m.y) < 1.5), None)
        if mill_adj is not None:
            mill_adj.bread -= 1
            entity.hunger   = max(0.0, entity.hunger - MILL_BREAD_FOOD)
            entity.state    = State.EATING
            return
        # Moulin avec pain mais pas adjacent → se déplace
        mill_ripe = next((m for m in clan_mills if m.bread > 0), None)
        if mill_ripe is not None and entity.hunger > 50:
            entity.state    = State.SEEKING_FOOD
            entity.target_x = float(mill_ripe.x)
            entity.target_y = float(mill_ripe.y)
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed * 1.1, world)
            return

    # 3. Mange ou cherche de la nourriture
    if spec.eat_amount > 0 and entity.hunger > 30:
        food = world.get_food(entity.ix, entity.iy)
        if food >= 5:
            entity.state = State.EATING
            consumed = world.consume_food(entity.ix, entity.iy, spec.eat_amount)
            entity.hunger = max(0, entity.hunger - consumed * 1.5)
            if not spec.aquatic:
                world.consume_fertility(entity.ix, entity.iy)
            # Efface la cible après manger → force à réévaluer au lieu de revenir
            entity.target_x = None
            entity.target_y = None
            return
        # Soif modérée interrompt la recherche de nourriture (mais pas le repas en cours)
        if not spec.aquatic and entity.thirst > 50:
            if _drink_or_seek_water(entity, world, events, buildings, cb=_cb):
                return
        # Cherche une tuile avec de la nourriture dans son champ de vision (numpy)
        entity.state = State.SEEKING_FOOD
        r = int(entity.traits["vision"])
        ex_i, ey_i = entity.ix, entity.iy
        x0 = max(0, ex_i - r); x1 = min(world.width,  ex_i + r + 1)
        y0 = max(0, ey_i - r); y1 = min(world.height, ey_i + r + 1)
        food_sub = world.food_grid[y0:y1, x0:x1].copy()
        walk_mask = (world._aquatic_walkable if spec.aquatic else world._walkable)[y0:y1, x0:x1]
        food_sub[~walk_mask] = 0.0
        if not spec.aquatic:
            food_sub[world._near_water[y0:y1, x0:x1]] *= 0.35
        best_idx = int(np.argmax(food_sub))
        best_f = float(food_sub.flat[best_idx])
        if best_f > 5:
            ly, lx = np.unravel_index(best_idx, food_sub.shape)
            entity.target_x = float(x0 + int(lx))
            entity.target_y = float(y0 + int(ly))
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed, world)
            return
        # Aucune nourriture dans le champ de vision → scan global (comme pour les arbres)
        if not spec.aquatic:
            _food_tiles = np.argwhere(world.food_grid > 8)
            if len(_food_tiles):
                _wk = world._walkable
                _food_tiles = _food_tiles[_wk[_food_tiles[:, 0], _food_tiles[:, 1]]]
            if len(_food_tiles):
                _dists = (_food_tiles[:, 1] - entity.x)**2 + (_food_tiles[:, 0] - entity.y)**2
                _best = _food_tiles[int(np.argmin(_dists))]
                entity.target_x = float(_best[1])
                entity.target_y = float(_best[0])
                entity.state    = State.EXPLORING
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed, world)
                return

    # 3.7 Soif légère (après manger, avant bois/construction)
    if not spec.aquatic and entity.thirst > 35:
        if _drink_or_seek_water(entity, world, events, buildings, cb=_cb):
            return

    # 3.8 Pêche (canne à pêche + affamé, après échec de la recherche de nourriture)
    if (entity.spec.can_build and entity.fishing_rod is not None
            and entity.hunger > FISHING_HUNGER_THRESH):
        # Adjacent à une tuile eau → pêche
        fishing_spot = None
        for adx in range(-1, 2):
            for ady in range(-1, 2):
                wx, wy = entity.ix + adx, entity.iy + ady
                if world.is_valid(wx, wy) and int(world.biome_grid[wy, wx]) in WATER_BIOMES:
                    fishing_spot = (wx, wy)
                    break
            if fishing_spot:
                break
        if fishing_spot:
            entity.state = State.FISHING
            if random.random() < FISHING_CATCH_PROB:
                entity.hunger = max(0.0, entity.hunger - FISHING_FOOD)
                pass  # fish_catch — événement interne silencieux
            return
        # Pas adjacent : chercher une berge accessible
        spot = _find_water_spot(entity, world)
        if spot:
            entity.state = State.FISHING
            entity.target_x, entity.target_y = spot
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed, world)
            return

    # 3.5 Dépôt des ressources à la maison du clan
    if ((entity.spec.can_chop or entity.spec.can_mine)
            and (entity.wood > 0 or entity.stone > 0)
            and entity.clan_id is not None):
        clan_houses = _cb.get("house", [])
        if clan_houses:
            # Préfère une maison avec de la place libre; si tout est plein, jette le bois
            _houses_with_space = [h for h in clan_houses if h.wood < MAX_WOOD_PER_HOUSE]
            if not _houses_with_space:
                entity.wood = 0   # tout plein → abandonne le bois, fait autre chose
            else:
                nearest = min(_houses_with_space, key=lambda b: _dist(entity.x, entity.y, b.x, b.y))
            dist_to_house = _dist(entity.x, entity.y, nearest.x, nearest.y) if _houses_with_space else 999
            if dist_to_house < 1.5:
                # Adjacent : dépose (bois capé, pierre non capée)
                _wood_space = max(0, MAX_WOOD_PER_HOUSE - nearest.wood)
                nearest.wood  += min(entity.wood, _wood_space)
                entity.wood   = max(0, entity.wood - _wood_space)
                nearest.stone += entity.stone; entity.stone = 0
                # Fabrication d'outils (priorité : pioche bois > hache bois > pioche pierre > hache pierre)
                if entity.pick is None and nearest.wood >= PICK_WOOD_COST:
                    nearest.wood -= PICK_WOOD_COST
                    entity.pick = "wood_pick"
                    events.append({"type": "craft_pick", "pick": "wood_pick",
                                   "clan_id": entity.clan_id})
                elif entity.tool is None and nearest.wood >= AXE_CRAFT_COST:
                    nearest.wood -= AXE_CRAFT_COST
                    entity.tool = "axe"
                    events.append({"type": "craft_axe", "tool": "axe",
                                   "clan_id": entity.clan_id})
                elif entity.pick == "wood_pick" and nearest.stone >= STONE_PICK_COST:
                    nearest.stone -= STONE_PICK_COST
                    entity.pick = "stone_pick"
                    events.append({"type": "craft_pick", "pick": "stone_pick",
                                   "clan_id": entity.clan_id})
                elif entity.tool == "axe" and nearest.stone >= STONE_AXE_COST:
                    nearest.stone -= STONE_AXE_COST
                    entity.tool = "stone_axe"
                    events.append({"type": "craft_axe", "tool": "stone_axe",
                                   "clan_id": entity.clan_id})
                elif entity.watering_can is None and nearest.wood >= WATERING_CAN_WOOD_COST:
                    nearest.wood -= WATERING_CAN_WOOD_COST
                    entity.watering_can = "watering_can"
                    events.append({"type": "craft_watering_can",
                                   "clan_id": entity.clan_id})
                elif entity.sickle is None and nearest.stone >= SICKLE_STONE_COST:
                    nearest.stone -= SICKLE_STONE_COST
                    entity.sickle = "sickle"
                    events.append({"type": "craft_sickle",
                                   "clan_id": entity.clan_id})
                elif entity.fishing_rod is None and nearest.wood >= FISHING_ROD_WOOD_COST:
                    nearest.wood -= FISHING_ROD_WOOD_COST
                    entity.fishing_rod = "fishing_rod"
                    events.append({"type": "craft_fishing_rod",
                                   "clan_id": entity.clan_id})
            elif entity.wood >= MAX_CARRY or entity.stone >= MAX_STONE_CARRY:
                # Portée pleine → aller déposer
                entity.state = State.WANDERING
                entity.target_x = float(nearest.x)
                entity.target_y = float(nearest.y)
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed, world)
                return

    # 4.1 Travailler sur un chantier en cours du clan (n'importe quel humain peut contribuer)
    # Ne pas interrompre un humain qui se rend déjà sur son propre chantier planifié
    if (entity.spec.can_build and entity.clan_id is not None and entity.hunger < 80
            and entity._build_target_type is None):
        clan_sites = [b for btl in [v for k, v in _cb.items() if k.startswith("site_")] for b in btl if b.work_done < b.work_needed]
        if clan_sites:
            near = min(clan_sites,
                       key=lambda b: _dist(entity.x, entity.y, b.x, b.y))
            entity.state = State.BUILDING
            if _dist(entity.x, entity.y, near.x, near.y) < 1.5:
                near.work_done += 1
                entity.target_x = None
                entity.target_y = None
            else:
                entity.target_x = float(near.x)
                entity.target_y = float(near.y)
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed, world)
            return

    # 4. Reproduction (bloquée en automne et en hiver)
    if (SEASON_REPRO_ALLOWED[season]
            and entity.hunger < spec.repro_hunger_min
            and entity.repro_cooldown_left == 0
            and entity.sex == Sex.FEMALE
            and entity.age > spec.max_age * 0.20):
        # Cap population par maisons pour les clans humains
        repro_allowed = True
        if entity.spec.can_build and entity.clan_id is not None:
            base = BUILDING_SPECS["house"].pop_bonus
            cap = sum(base * b.level for b in _cb.get("house", []))
            clan_pop = sum(1 for e in all_entities
                           if e.alive and e.etype == EntityType.HUMAN
                           and e.clan_id == entity.clan_id)
            if clan_pop >= cap:
                repro_allowed = False
        if repro_allowed:
            partner = None
            for e in all_entities:
                same_clan = (entity.clan_id is None or e.clan_id == entity.clan_id)
                if (e is not entity and e.alive
                        and e.etype == entity.etype
                        and e.sex == Sex.MALE
                        and e.hunger < spec.repro_hunger_min
                        and same_clan
                        and _dist(entity.x, entity.y, e.x, e.y) < 3):
                    partner = e
                    break
            if partner:
                entity.state = State.SEEKING_MATE
                entity.gestation_left = spec.gestation
                entity.repro_cooldown_left = spec.repro_cooldown
                partner.repro_cooldown_left = spec.repro_cooldown
                return

    # 4.24 Se déplacer vers le chantier planifié et le démarrer une fois sur place
    if (entity.spec.can_build
            and entity._build_target_type is not None
            and entity.clan_id is not None):
        btype_t = entity._build_target_type
        btx = entity._build_target_x
        bty = entity._build_target_y
        bx_t = int(btx)
        by_t = int(bty)
        # Annuler si un chantier du même type existe déjà pour ce clan
        existing_site = _cb.get(f"site_{btype_t}", [])
        # Annuler aussi si l'emplacement est devenu invalide
        if (existing_site
                or not world.is_valid(bx_t, by_t)
                or not world.is_walkable(bx_t, by_t)):
            entity._build_target_type = None
            entity._build_target_x    = None
            entity._build_target_y    = None
        elif _dist(entity.x, entity.y, btx, bty) < 1.5:
            # Arrivé à l'emplacement : vérifier ressources et créer le chantier
            bspec_t    = BUILDING_SPECS[btype_t]
            clan_houses_t = _cb.get("house", [])
            can_build_t   = False
            # Éviter double déduction si un autre humain a émis le même start_site ce tick
            already_this_tick = any(
                ev.get("type") == "start_site"
                and ev.get("btype") == btype_t
                and ev.get("clan_id") == entity.clan_id
                for ev in events
            )
            if not already_this_tick:
                if btype_t == "house":
                    if not clan_houses_t:
                        if entity.wood >= bspec_t.first_cost:
                            entity.wood -= bspec_t.first_cost
                            can_build_t = True
                    else:
                        donor_t = max(clan_houses_t, key=lambda b: b.wood)
                        if donor_t.wood >= bspec_t.wood_cost:
                            donor_t.wood -= bspec_t.wood_cost
                            can_build_t = True
                elif btype_t in ("well", "mill"):
                    dw = max(clan_houses_t, key=lambda b: b.wood)  if clan_houses_t else None
                    ds = max(clan_houses_t, key=lambda b: b.stone) if clan_houses_t else None
                    if (dw and ds
                            and dw.wood  >= bspec_t.wood_cost
                            and ds.stone >= bspec_t.stone_cost):
                        dw.wood  -= bspec_t.wood_cost
                        ds.stone -= bspec_t.stone_cost
                        can_build_t = True
            if can_build_t:
                entity.building_type = btype_t
                events.append({"type": "start_site", "btype": btype_t,
                               "x": bx_t, "y": by_t,
                               "clan_id": entity.clan_id,
                               "work_needed": bspec_t.build_time})
            entity._build_target_type = None
            entity._build_target_x    = None
            entity._build_target_y    = None
            return
        else:
            # En chemin vers le chantier prévu
            entity.state   = State.BUILDING
            entity.target_x = btx
            entity.target_y = bty
            _move_toward(entity, btx, bty, _eff_speed, world)
            return

    # 4.25 Démarre une construction de maison (l'humain doit rejoindre l'emplacement)
    if (entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_type is None
            and entity._build_target_type is None
            and entity.hunger < 65
            and clans):
        clan = clans.get(entity.clan_id)
        if clan:
            bspec = BUILDING_SPECS["house"]
            clan_houses = _cb.get("house", [])
            clan_sites_h = _cb.get("site_house", [])
            # Cap atteint ou chantier déjà en cours → pas de nouvelle construction
            _total_h = len(clan_houses) + len(clan_sites_h)
            if bspec.max_per_clan == 0 or _total_h < bspec.max_per_clan:
                # Ne construire que si la population dépasse ~70% de la capacité actuelle
                _pop_cap_h = sum(bspec.pop_bonus * b.level for b in clan_houses)
                _clan_pop_h = sum(1 for e in all_entities
                                  if e.alive and e.etype == EntityType.HUMAN
                                  and e.clan_id == entity.clan_id)
                _need_house = (not clan_houses) or (_clan_pop_h >= _pop_cap_h * 0.70)
                # Vérifier si un autre humain a déjà planifié une maison ce tick
                already_planned = (
                    _need_house and (
                        any(ev.get("type") == "start_site" and ev.get("btype") == "house"
                            and ev.get("clan_id") == entity.clan_id for ev in events)
                        or any(e._build_target_type == "house" and e.clan_id == entity.clan_id
                               for e in all_entities if e.alive and e is not entity)
                    )
                )
                # Source de bois : porté (1re maison) ou stockage (maisons suivantes)
                if not clan_houses:
                    can_build_house = (_need_house and not already_planned
                                       and entity.wood >= bspec.first_cost)
                else:
                    donor = max(clan_houses, key=lambda b: b.wood)
                    can_build_house = (_need_house and not already_planned
                                       and donor.wood >= bspec.wood_cost)
                if can_build_house:
                    for _ in range(30):
                        angle = random.uniform(0, 2 * math.pi)
                        dist  = random.uniform(bspec.min_from_fire, bspec.max_from_fire)
                        bx = int(clan.cx + math.cos(angle) * dist)
                        by = int(clan.cy + math.sin(angle) * dist)
                        if not world.is_valid(bx, by) or not world.is_walkable(bx, by):
                            continue
                        if any(_dist(bx, by, b.x, b.y) < bspec.min_dist
                               for b in (buildings or [])
                               if b.btype in ("house", "site_house")):
                            continue
                        # Emplacement valide → planifier le déplacement (pas de déduction bois encore)
                        entity._build_target_type = "house"
                        entity._build_target_x    = float(bx)
                        entity._build_target_y    = float(by)
                        entity.state   = State.BUILDING
                        entity.target_x = float(bx)
                        entity.target_y = float(by)
                        _move_toward(entity, entity.target_x, entity.target_y,
                                     _eff_speed, world)
                        return

    # 4.26 Construction d'un moulin (si le clan a ≥2 champs et pas encore de moulin)
    if (entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_type is None
            and entity._build_target_type is None
            and entity.hunger < 65
            and clans):
        bspec_m = BUILDING_SPECS["mill"]
        clan_mills   = _cb.get("mill", [])
        clan_sites_m = _cb.get("site_mill", [])
        clan_fields  = _cb.get("wheatfield", [])
        if (len(clan_fields) >= 2 and not clan_sites_m
                and (bspec_m.max_per_clan == 0 or len(clan_mills) < bspec_m.max_per_clan)):
            clan_houses = _cb.get("house", [])
            donor = max(clan_houses, key=lambda b: b.wood) if clan_houses else None
            donor_s = max(clan_houses, key=lambda b: b.stone) if clan_houses else None
            has_res = (donor and donor.wood >= bspec_m.wood_cost
                       and donor_s and donor_s.stone >= bspec_m.stone_cost)
            if has_res:
                # Placement adjacent à un champ de blé existant
                for field in clan_fields:
                    for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
                        mx, my = field.x + ddx, field.y + ddy
                        if not world.is_valid(mx, my) or not world.is_walkable(mx, my):
                            continue
                        if any(_dist(mx, my, b.x, b.y) < bspec_m.min_dist
                               for b in (buildings or []) if b.btype == "mill"):
                            continue
                        donor.wood   -= bspec_m.wood_cost
                        donor_s.stone -= bspec_m.stone_cost
                        entity.building_type = "mill"
                        events.append({"type": "start_site", "btype": "mill",
                                       "x": mx, "y": my,
                                       "clan_id": entity.clan_id,
                                       "work_needed": bspec_m.build_time})
                        return

    # 4.27 Construction d'un puit (si le clan a ≥1 maison et pas encore de puit)
    if (entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_type is None
            and entity._build_target_type is None
            and entity.hunger < 65
            and clans):
        bspec_well = BUILDING_SPECS["well"]
        clan_wells  = _cb.get("well", [])
        clan_sites_w = _cb.get("site_well", [])
        clan_houses = _cb.get("house", [])
        if (len(clan_houses) >= 2 and not clan_sites_w
                and (bspec_well.max_per_clan == 0 or len(clan_wells) < bspec_well.max_per_clan)):
            donor   = max(clan_houses, key=lambda b: b.wood)   if clan_houses else None
            donor_s = max(clan_houses, key=lambda b: b.stone)  if clan_houses else None
            has_res = (donor   and donor.wood    >= bspec_well.wood_cost
                       and donor_s and donor_s.stone >= bspec_well.stone_cost)
            if has_res:
                clan = clans.get(entity.clan_id)
                already_planned_w = (
                    any(ev.get("type") == "start_site" and ev.get("btype") == "well"
                        and ev.get("clan_id") == entity.clan_id for ev in events)
                    or any(e._build_target_type == "well" and e.clan_id == entity.clan_id
                           for e in all_entities if e.alive and e is not entity)
                )
                if clan and not already_planned_w:
                    for _ in range(30):
                        angle = random.uniform(0, 2 * math.pi)
                        dist  = random.uniform(bspec_well.min_from_fire, bspec_well.max_from_fire)
                        wx = int(clan.cx + math.cos(angle) * dist)
                        wy = int(clan.cy + math.sin(angle) * dist)
                        if not world.is_valid(wx, wy) or not world.is_walkable(wx, wy):
                            continue
                        if any(_dist(wx, wy, b.x, b.y) < bspec_well.min_dist
                               for b in (buildings or []) if b.btype == "well"):
                            continue
                        # Planifier le déplacement (la déduction des ressources se fait à l'arrivée)
                        entity._build_target_type = "well"
                        entity._build_target_x    = float(wx)
                        entity._build_target_y    = float(wy)
                        entity.state   = State.BUILDING
                        entity.target_x = float(wx)
                        entity.target_y = float(wy)
                        _move_toward(entity, entity.target_x, entity.target_y,
                                     _eff_speed, world)
                        return

    # 4.3a Récolte : champ mûr (priorité élevée : affamé OU adjacent à un champ mûr)
    if entity.spec.can_build and entity.clan_id is not None:
        clan_fields = _cb.get("wheatfield", [])
        clan_mills  = _cb.get("mill", [])
        ripe_adj = next((b for b in clan_fields
                         if b.stage >= 4 and _dist(entity.x, entity.y, b.x, b.y) < 1.5), None)
        if ripe_adj is not None:
            entity.state = State.FARMING
            ripe_adj.stage = 1
            ripe_adj.grow_ticks = 0
            ripe_adj.watered_ticks = 0
            mill_dst = next((m for m in clan_mills if m.wheat < MILL_MAX_BREAD * 2), None)
            if mill_dst is not None:
                mill_dst.wheat += 1
            else:
                food = WHEAT_HARVEST_FOOD + (SICKLE_HARVEST_BONUS if entity.sickle else 0.0)
                entity.hunger = max(0.0, entity.hunger - food)
            return
        # Chercher un champ mûr si affamé
        if entity.hunger > WHEAT_HUNGER_THRESH:
            ripe_far = next((b for b in clan_fields if b.stage >= 4), None)
            if ripe_far is not None:
                entity.state    = State.FARMING
                entity.target_x = float(ripe_far.x)
                entity.target_y = float(ripe_far.y)
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed, world)
                return

    # 4.3b/c Entretien des champs : arrosage + plantation (quand reposé et non trop affamé)
    if (entity.spec.can_build and entity.clan_id is not None
            and entity.hunger < WHEAT_WORK_THRESH):
        clan_fields = _cb.get("wheatfield", [])
        clan_mills  = _cb.get("mill", [])

        # 4.3b Arrosage (si possède l'arrosoir)
        if entity.watering_can is not None:
            if entity.can_filled:
                # Champ à arroser adjacent (pas encore arrosé, pas encore mûr)
                dry = next((b for b in clan_fields
                            if b.stage < 4 and b.watered_ticks == 0
                            and _dist(entity.x, entity.y, b.x, b.y) < 1.5), None)
                if dry is not None:
                    entity.state    = State.FARMING
                    entity.can_filled = False
                    dry.watered_ticks = WATERED_TICKS
                    return
                # Se déplacer vers le champ le plus sec
                dry_far = next((b for b in clan_fields
                                if b.stage < 4 and b.watered_ticks == 0), None)
                if dry_far is not None:
                    entity.state    = State.FARMING
                    entity.target_x = float(dry_far.x)
                    entity.target_y = float(dry_far.y)
                    _move_toward(entity, entity.target_x, entity.target_y,
                                 _eff_speed, world)
                    return
            else:
                # Arrosoir vide → chercher de l'eau : puit du clan d'abord, sinon tuile eau
                clan_wells = _cb.get("well", [])
                near_well = next((b for b in clan_wells
                                  if _dist(entity.x, entity.y, b.x, b.y) < 1.5), None)
                if near_well is not None:
                    entity.state      = State.FARMING
                    entity.can_filled = True
                    return
                for adx in range(-2, 3):
                    for ady in range(-2, 3):
                        wx, wy = entity.ix + adx, entity.iy + ady
                        if (world.is_valid(wx, wy)
                                and world.biome_grid[wy, wx] in WATER_BIOMES):
                            entity.state    = State.FARMING
                            entity.can_filled = True
                            return
                # Se déplacer vers le puit le plus proche du clan
                if clan_wells:
                    nearest_well = min(clan_wells,
                                       key=lambda b: _dist(entity.x, entity.y, b.x, b.y))
                    entity.state    = State.FARMING
                    entity.target_x = float(nearest_well.x)
                    entity.target_y = float(nearest_well.y)
                    _move_toward(entity, entity.target_x, entity.target_y,
                                 _eff_speed, world)
                    return
                # Cherche une source d'eau dans le champ de vision
                r = int(entity.traits["vision"])
                best_w, best_wd = None, float("inf")
                for vdx in range(-r, r + 1):
                    for vdy in range(-r, r + 1):
                        wx, wy = entity.ix + vdx, entity.iy + vdy
                        if (world.is_valid(wx, wy)
                                and world.biome_grid[wy, wx] in WATER_BIOMES):
                            d = _dist(entity.x, entity.y, wx, wy)
                            if d < best_wd:
                                best_wd = d
                                best_w  = (wx, wy)
                if best_w:
                    entity.state    = State.FARMING
                    entity.target_x = float(best_w[0])
                    entity.target_y = float(best_w[1])
                    _move_toward(entity, entity.target_x, entity.target_y,
                                 _eff_speed, world)
                    return

        # 4.3c Plantation : si le clan peut encore planter
        bspec_w = BUILDING_SPECS["wheatfield"]
        if (entity.building_type is None
                and (bspec_w.max_per_clan == 0 or len(clan_fields) < bspec_w.max_per_clan)
                and clans):
            clan = clans.get(entity.clan_id)
            if clan:
                for _ in range(30):
                    angle = random.uniform(0, 2 * math.pi)
                    dist  = random.uniform(bspec_w.min_from_fire, bspec_w.max_from_fire)
                    fx = int(clan.cx + math.cos(angle) * dist)
                    fy = int(clan.cy + math.sin(angle) * dist)
                    if (not world.is_valid(fx, fy)
                            or not world.is_walkable(fx, fy)
                            or world.biome_grid[fy, fx] not in (int(Biome.GRASS), int(Biome.DIRT))):
                        continue
                    if any(_dist(fx, fy, b.x, b.y) < bspec_w.min_dist for b in clan_fields):
                        continue
                    entity.building_type = "wheatfield"
                    events.append({"type": "start_site", "btype": "wheatfield",
                                   "x": fx, "y": fy,
                                   "clan_id": entity.clan_id,
                                   "work_needed": bspec_w.build_time})
                    return

    # 4.5 Coupe du bois (l'humain doit être SUR la tuile forêt pour couper)
    # Arrête de couper si le total de bois du clan dépasse le seuil global
    _clan_houses_wood = _cb.get("house", [])
    _total_clan_wood = sum(b.wood for b in _clan_houses_wood)
    _wood_full = bool(_clan_houses_wood) and _total_clan_wood >= CLAN_WOOD_CAP
    if (entity.spec.can_chop and entity.wood < MAX_CARRY
            and entity.hunger < 70 and entity.chop_cooldown_left == 0
            and not _wood_full):
        # Coupe si debout sur une tuile d'arbre debout
        if world.is_choppable(entity.ix, entity.iy):
            entity.state = State.CHOPPING
            wood = world.chop_tree(entity.ix, entity.iy)
            if entity.tool == "stone_axe":
                wood += STONE_AXE_BONUS
            elif entity.tool == "axe":
                wood += AXE_BONUS
            entity.wood = min(MAX_CARRY, entity.wood + wood)
            entity.chop_cooldown_left = CHOP_COOLDOWN
            return
        # Arbre dans le champ de vision → se déplace vers le plus proche (numpy)
        r = int(entity.traits["vision"])
        ex_i, ey_i = entity.ix, entity.iy
        x0 = max(0, ex_i - r); x1 = min(world.width,  ex_i + r + 1)
        y0 = max(0, ey_i - r); y1 = min(world.height, ey_i + r + 1)
        chop_sub = (world._forest_mask[y0:y1, x0:x1]
                    & (world.tree_grid[y0:y1, x0:x1] >= TREE_STUMP_THRESHOLD))
        _ys, _xs = np.where(chop_sub)
        best_tree = None
        if len(_xs):
            abs_xs = _xs + x0; abs_ys = _ys + y0
            dists_sq = (abs_xs - entity.x)**2 + (abs_ys - entity.y)**2
            idx = int(np.argmin(dists_sq))
            best_tree = (int(abs_xs[idx]), int(abs_ys[idx]))
        if best_tree:
            entity.state = State.CHOPPING
            entity.target_x = float(best_tree[0])
            entity.target_y = float(best_tree[1])
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed, world)
            return
        # Aucun arbre visible → chercher la forêt la plus proche sur toute la map
        # (important pour les clans éloignés de la forêt)
        _need_wood = (
            entity.wood < MAX_CARRY
            and (not _clan_houses_wood or entity.wood < 2)
        )
        if _need_wood:
            forest_tiles = np.argwhere(
                world._forest_mask & (world.tree_grid >= TREE_STUMP_THRESHOLD)
            )
            if len(forest_tiles):
                dists = (forest_tiles[:, 1] - entity.x)**2 + (forest_tiles[:, 0] - entity.y)**2
                best = forest_tiles[int(np.argmin(dists))]
                entity.state = State.CHOPPING
                entity.target_x = float(best[1])
                entity.target_y = float(best[0])
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed, world)
                return

    # 4.6 Mine la pierre (entités avec can_mine + pioche, si portée non pleine et pas trop affamées)
    if (entity.spec.can_mine
            and entity.pick is not None
            and entity.stone < MAX_STONE_CARRY
            and entity.hunger < 65):
        # Roche adjacente → mine directement
        for adx in range(-1, 2):
            for ady in range(-1, 2):
                if adx == 0 and ady == 0:
                    continue
                tx, ty = entity.ix + adx, entity.iy + ady
                if world.is_mineable(tx, ty):
                    entity.state = State.MINING
                    stone = world.mine_stone(tx, ty)
                    if entity.pick == "stone_pick":
                        stone += STONE_PICK_BONUS
                    entity.stone = min(MAX_STONE_CARRY, entity.stone + stone)
                    return
        # Roche dans le champ de vision → se déplace vers la plus proche (numpy)
        r = int(entity.traits["vision"])
        ex_i, ey_i = entity.ix, entity.iy
        x0 = max(0, ex_i - r); x1 = min(world.width,  ex_i + r + 1)
        y0 = max(0, ey_i - r); y1 = min(world.height, ey_i + r + 1)
        mine_sub = (world._mountain_mask[y0:y1, x0:x1]
                    & (world.stone_grid[y0:y1, x0:x1] >= STONE_STUMP_THRESHOLD))
        _ys, _xs = np.where(mine_sub)
        best_rock = None
        if len(_xs):
            abs_xs = _xs + x0; abs_ys = _ys + y0
            dists_sq = (abs_xs - entity.x)**2 + (abs_ys - entity.y)**2
            idx = int(np.argmin(dists_sq))
            best_rock = (int(abs_xs[idx]), int(abs_ys[idx]))
        if best_rock:
            entity.state = State.MINING
            entity.target_x = float(best_rock[0])
            entity.target_y = float(best_rock[1])
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed * 0.8, world)
            return

    # 4.65 Continuation d'une exploration longue distance déjà engagée
    if (entity.etype == EntityType.HUMAN
            and entity.clan_id is not None
            and clans
            and entity.hunger < 68
            and entity.thirst < 58
            and entity.target_x is not None):
        clan = clans.get(entity.clan_id)
        if clan and _dist(clan.cx, clan.cy, entity.target_x, entity.target_y) > 75:
            entity.state = State.EXPLORING
            _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
            return

    # 4.7 Exploration (humain sans tâche urgente : explore loin du clan)
    if (entity.etype == EntityType.HUMAN
            and entity.clan_id is not None
            and entity.hunger < 65
            and entity.thirst < 55
            and clans
            and random.random() < 0.45):
        clan = clans.get(entity.clan_id)
        if clan:
            # Vérifie si la zone locale du clan est vide de ressources
            _cx_i, _cy_i = int(clan.cx), int(clan.cy)
            _r_loc = 30
            _x0l = max(0, _cx_i - _r_loc); _x1l = min(world.width,  _cx_i + _r_loc)
            _y0l = max(0, _cy_i - _r_loc); _y1l = min(world.height, _cy_i + _r_loc)
            _local_food  = float(world.food_grid[_y0l:_y1l, _x0l:_x1l].max())
            _local_trees = bool(world._forest_mask[_y0l:_y1l, _x0l:_x1l].any())
            _zone_vide   = _local_food < 5 and not _local_trees
            # Zone vide → exploration forcée; sinon 8 % de chance
            far_explore = _zone_vide or random.random() < 0.08
            dist_range  = (80, 200) if far_explore else (20, 70)
            for _ in range(12):
                angle = random.uniform(0, 2 * math.pi)
                dist  = random.uniform(*dist_range)
                tx = int(clan.cx + math.cos(angle) * dist)
                ty = int(clan.cy + math.sin(angle) * dist)
                tx = max(0, min(world.width  - 1, tx))
                ty = max(0, min(world.height - 1, ty))
                if world.is_walkable(tx, ty):
                    entity.target_x = float(tx)
                    entity.target_y = float(ty)
                    entity.state = State.EXPLORING if far_explore else State.WANDERING
                    _move_toward(entity, entity.target_x, entity.target_y,
                                 _eff_speed, world)
                    return

    # 5. Déambule
    entity.state = State.WANDERING
    # Troupeau : espèces avec can_herd suivent le groupe s'il y en a un proche
    if entity.spec.can_herd:
        if _herd_move(entity, all_entities, world):
            return
    # Humains : biais vers le territoire du clan (rayon proportionnel à la satiété)
    if (entity.etype == EntityType.HUMAN and entity.clan_id is not None
            and clans and random.random() < 0.6
            and (entity.target_x is None
                 or _dist(entity.x, entity.y, entity.target_x, entity.target_y) < 0.5)):
        clan = clans.get(entity.clan_id)
        if clan:
            # Affamé/assoiffé → reste proche; reposé → explore loin
            wander_r = 18 if (entity.hunger > 55 or entity.thirst > 55) else 50
            for _ in range(8):
                tx = clan.cx + random.uniform(-wander_r, wander_r)
                ty = clan.cy + random.uniform(-wander_r, wander_r)
                tx = max(0, min(world.width  - 1, tx))
                ty = max(0, min(world.height - 1, ty))
                if world.is_walkable(int(tx), int(ty)):
                    entity.target_x = tx
                    entity.target_y = ty
                    break
    # Répulsion légère entre humains trop proches (évite l'entassement)
    if entity.etype == EntityType.HUMAN:
        for other in all_entities:
            if other is entity or not other.alive or other.etype != EntityType.HUMAN:
                continue
            d = _dist(entity.x, entity.y, other.x, other.y)
            if 0.01 < d < 0.8:
                repulse_x = entity.x - other.x
                repulse_y = entity.y - other.y
                norm = math.sqrt(repulse_x**2 + repulse_y**2) or 1
                tx = max(0.0, min(world.width  - 1.0, entity.x + (repulse_x / norm) * 1.5))
                ty = max(0.0, min(world.height - 1.0, entity.y + (repulse_y / norm) * 1.5))
                if world.is_walkable(int(tx), int(ty)):
                    entity.target_x = tx
                    entity.target_y = ty
                break
    _random_walk(entity, world)


class Simulation:
    def __init__(self, world: "World"):
        from .world import World
        self.world    = world
        self.entities: list[Entity] = []
        self.tick_count = 0
        self.events_log: list[dict] = []
        self.stats_history: list[dict] = []
        self.clans: list[Clan] = []
        self.buildings: list[Building] = []
        self._next_building_id = 0
        self.raining: bool = False
        self.storming: bool = False   # orage (sous-type de pluie)
        self.rain_ticks_left: int = 0
        self.heatwave: bool = False
        self.heatwave_ticks_left: int = 0

    def populate(self, counts: dict[EntityType, int] = None):
        """Spawn initial des entités sur des cases marchables."""
        # Animaux : placement aléatoire (les humains sont gérés séparément par clan)
        defaults = {
            EntityType.BOAR:         10,
            EntityType.CHICKEN:      30,
            EntityType.HORNED_SHEEP: 20,
            EntityType.HORSE:        16,
            EntityType.PIG:          30,
            EntityType.SHEEP:        30,
            EntityType.FISH:         55,
            EntityType.SHARK:        3,
        }
        counts = counts or defaults
        for etype, n in counts.items():
            aquatic = SPECS[etype].aquatic
            for _ in range(n):
                for attempt in range(100):
                    x = random.randint(1, self.world.width  - 2)
                    y = random.randint(1, self.world.height - 2)
                    if self.world.is_walkable(x, y, aquatic):
                        self.entities.append(spawn(etype, x + random.random(), y + random.random()))
                        break

        # ── Placement des clans : 4 emplacements bien espacés sur terrain marchable ──
        MIN_CAMPFIRE_DIST = min(self.world.width, self.world.height) // (N_CLANS + 1)
        FOREST_SEARCH_R   = 25   # rayon max pour trouver une forêt proche du campfire

        def _has_forest_nearby(x, y, r=FOREST_SEARCH_R):
            x0 = max(0, x - r); x1 = min(self.world.width,  x + r + 1)
            y0 = max(0, y - r); y1 = min(self.world.height, y + r + 1)
            return bool((self.world.biome_grid[y0:y1, x0:x1] == int(Biome.FOREST)).any())

        clan_positions: list[tuple[int, int]] = []
        for _ in range(12000):
            x = random.randint(10, self.world.width  - 10)
            y = random.randint(10, self.world.height - 10)
            if not self.world.is_walkable(x, y):
                continue
            if not _has_forest_nearby(x, y):
                continue
            if all(_dist(x, y, cx, cy) >= MIN_CAMPFIRE_DIST for cx, cy in clan_positions):
                clan_positions.append((x, y))
            if len(clan_positions) == N_CLANS:
                break
        # Fallback : compléter sans contrainte de distance mais avec forêt
        if len(clan_positions) < N_CLANS:
            for _ in range(8000):
                x = random.randint(10, self.world.width  - 10)
                y = random.randint(10, self.world.height - 10)
                if (self.world.is_walkable(x, y)
                        and _has_forest_nearby(x, y)
                        and (x, y) not in clan_positions):
                    clan_positions.append((x, y))
                if len(clan_positions) == N_CLANS:
                    break
        # Dernier recours : sans contrainte de forêt (évite le crash)
        if len(clan_positions) < N_CLANS:
            for _ in range(5000):
                x = random.randint(10, self.world.width  - 10)
                y = random.randint(10, self.world.height - 10)
                if self.world.is_walkable(x, y) and (x, y) not in clan_positions:
                    clan_positions.append((x, y))
                if len(clan_positions) == N_CLANS:
                    break

        self.clans = []
        for cid, (cx, cy) in enumerate(clan_positions):
            # ── Chef : mâle, placé sur le campfire ──────────────────────────
            chief = spawn(EntityType.HUMAN,
                          cx + random.uniform(-0.3, 0.3),
                          cy + random.uniform(-0.3, 0.3),
                          sex=Sex.MALE)
            chief.clan_id = cid
            chief.hunger  = 10.0
            chief.age     = SPECS[EntityType.HUMAN].max_age * 0.2   # adulte
            self.entities.append(chief)

            # ── 4 membres : 2♂ + 2♀, dans un rayon de 4 tuiles du campfire ─
            for sex in (Sex.MALE, Sex.FEMALE, Sex.MALE, Sex.FEMALE):
                for _ in range(80):
                    mx = cx + random.uniform(-4, 4)
                    my = cy + random.uniform(-4, 4)
                    mx = max(1.0, min(self.world.width  - 2.0, mx))
                    my = max(1.0, min(self.world.height - 2.0, my))
                    if self.world.is_walkable(int(mx), int(my)):
                        member = spawn(EntityType.HUMAN, mx, my, sex=sex)
                        member.clan_id = cid
                        member.hunger  = random.uniform(5, 75)
                        member.age     = SPECS[EntityType.HUMAN].max_age * random.uniform(0.1, 0.35)
                        self.entities.append(member)
                        break

            self.clans.append(Clan(id=cid, cx=float(cx), cy=float(cy),
                                   color=CLAN_COLORS[cid], chief_id=chief.id))
            fire = Building(id=self._next_building_id,
                            clan_id=cid,
                            x=cx, y=cy,
                            btype="campfire")
            self._next_building_id += 1
            self.buildings.append(fire)


    def step(self) -> dict:
        """Avance d'un tick. Retourne les données à broadcaster."""
        self.tick_count += 1
        season = get_season(self.tick_count)
        temp_c = get_temperature(self.tick_count)

        births: list[Entity] = []
        tick_events: list[dict] = []

        # ── Météo : déclenchement / arrêt pluie / orage / canicule ──────────
        if self.raining:
            self.rain_ticks_left -= 1
            if self.rain_ticks_left <= 0:
                self.raining  = False
                self.storming = False
            # La pluie coupe la canicule
            if self.heatwave:
                self.heatwave = False
                self.heatwave_ticks_left = 0
                tick_events.append({"type": "heatwave_end"})
        elif self.heatwave:
            self.heatwave_ticks_left -= 1
            if self.heatwave_ticks_left <= 0:
                self.heatwave = False
                tick_events.append({"type": "heatwave_end"})
        elif season == "summer" and random.random() < HEATWAVE_PROB:
            self.heatwave = True
            self.heatwave_ticks_left = random.randint(HEATWAVE_DURATION_MIN, HEATWAVE_DURATION_MAX)
            tick_events.append({"type": "heatwave_start"})
        elif random.random() < RAIN_PROB[season]:
            self.raining  = True
            self.storming = random.random() < STORM_PROB[season]
            self.rain_ticks_left = random.randint(RAIN_DURATION_MIN, RAIN_DURATION_MAX)

        self.world.regen_food(SEASON_REGEN_MULT[season])
        # Arbres et roches qui franchissent le seuil de repousse ce tick
        tree_changes = [{"x": x, "y": y, "stump": False}
                        for x, y in self.world.regen_trees()]
        rock_changes = [{"x": x, "y": y, "depleted": False}
                        for x, y in self.world.regen_stones()]
        # Fertilité : DIRT → GRASS (regen lente), GRASS → DIRT collecté depuis consume_fertility
        biome_changes = self.world.regen_fertility(self.raining, season)
        biome_changes += self.world.drain_biome_changes()

        # Mélange pour éviter les biais d'ordre
        random.shuffle(self.entities)

        clans_dict = {c.id: c for c in self.clans}

        # Recruter les humains libres proches d'un membre de clan
        for e in self.entities:
            if e.alive and e.etype == EntityType.HUMAN and e.clan_id is None:
                for other in self.entities:
                    if (other is not e and other.alive
                            and other.etype == EntityType.HUMAN
                            and other.clan_id is not None
                            and _dist(e.x, e.y, other.x, other.y) < 4):
                        e.clan_id = other.clan_id
                        break

        # Comptage par espèce (pour cap MAX_PER_SPECIES)
        species_counts: dict[str, int] = {}
        for e in self.entities:
            if e.alive:
                species_counts[e.etype.value] = species_counts.get(e.etype.value, 0) + 1

        # Index bâtiments par clan pour éviter O(n_humains × n_bâtiments) dans tick_entity
        clan_bldg: dict = {}
        for _b in self.buildings:
            _cid = _b.clan_id
            if _cid not in clan_bldg:
                clan_bldg[_cid] = {}
            if _b.btype not in clan_bldg[_cid]:
                clan_bldg[_cid][_b.btype] = []
            clan_bldg[_cid][_b.btype].append(_b)

        # Liste des prédateurs actifs (pour _find_predator_nearby, évite O(n²))
        active_predators = [e for e in self.entities if e.alive and e.spec.is_predator]

        # Grille spatiale des prédateurs (cellule 8×8 tuiles) pour accélérer la détection
        _GCELL = 8
        predator_grid: dict = {}
        for _p in active_predators:
            _key = (_p.ix >> 3, _p.iy >> 3)
            if _key not in predator_grid:
                predator_grid[_key] = []
            predator_grid[_key].append(_p)

        for e in self.entities:
            tick_entity(e, self.world, self.entities, births, tick_events,
                        self.tick_count, season, clans_dict, self.buildings,
                        temp_c, species_counts, self.raining, self.heatwave,
                        clan_bldg, active_predators, predator_grid)

        # Traiter les constructions (dédoublonnage intra-tick inclus)
        new_this_tick: list[Building] = []
        for ev in tick_events:
            if ev["type"] == "build_house":
                all_houses = self.buildings + new_this_tick
                too_close = any(
                    _dist(ev["x"], ev["y"], b.x, b.y) < BUILDING_SPECS["house"].min_dist
                    for b in all_houses
                )
                if not too_close:
                    b = Building(id=self._next_building_id,
                                 clan_id=ev["clan_id"], x=ev["x"], y=ev["y"],
                                 btype="house")
                    self._next_building_id += 1
                    new_this_tick.append(b)
            elif ev["type"] == "build_wheatfield":
                all_fields = [b for b in self.buildings + new_this_tick
                              if b.btype == "wheatfield" and b.clan_id == ev["clan_id"]]
                too_close = any(
                    _dist(ev["x"], ev["y"], b.x, b.y) < BUILDING_SPECS["wheatfield"].min_dist
                    for b in all_fields
                )
                if not too_close:
                    b = Building(id=self._next_building_id,
                                 clan_id=ev["clan_id"], x=ev["x"], y=ev["y"],
                                 btype="wheatfield", stage=1, grow_ticks=0)
                    self._next_building_id += 1
                    new_this_tick.append(b)
            elif ev["type"] == "build_mill":
                all_mills = [b for b in self.buildings + new_this_tick
                             if b.btype == "mill" and b.clan_id == ev["clan_id"]]
                too_close = any(
                    _dist(ev["x"], ev["y"], b.x, b.y) < BUILDING_SPECS["mill"].min_dist
                    for b in all_mills
                )
                if not too_close:
                    b = Building(id=self._next_building_id,
                                 clan_id=ev["clan_id"], x=ev["x"], y=ev["y"],
                                 btype="mill")
                    self._next_building_id += 1
                    new_this_tick.append(b)
            elif ev["type"] == "build_well":
                all_wells = [b for b in self.buildings + new_this_tick
                             if b.btype == "well"]
                too_close = any(
                    _dist(ev["x"], ev["y"], b.x, b.y) < BUILDING_SPECS["well"].min_dist
                    for b in all_wells
                )
                if not too_close:
                    b = Building(id=self._next_building_id,
                                 clan_id=ev["clan_id"], x=ev["x"], y=ev["y"],
                                 btype="well")
                    self._next_building_id += 1
                    new_this_tick.append(b)
            elif ev["type"] == "start_site":
                btype = ev["btype"]
                # Ne créer le site que s'il n'en existe pas déjà un du même type pour ce clan
                already = any(
                    b.btype == f"site_{btype}" and b.clan_id == ev["clan_id"]
                    for b in self.buildings + new_this_tick
                )
                if not already:
                    site = Building(
                        id=self._next_building_id,
                        clan_id=ev["clan_id"],
                        x=ev["x"], y=ev["y"],
                        btype=f"site_{btype}",
                        work_needed=ev["work_needed"]
                    )
                    self._next_building_id += 1
                    new_this_tick.append(site)
        self.buildings.extend(new_this_tick)

        # Promouvoir les chantiers terminés en bâtiments réels
        for b in self.buildings:
            if b.btype.startswith("site_") and b.work_done >= b.work_needed > 0:
                real_btype = b.btype[5:]  # "site_house" → "house"
                b.btype = real_btype
                b.work_needed = 0
                tick_events.append({"type": f"build_{real_btype}",
                                    "clan_id": b.clan_id,
                                    "x": b.x, "y": b.y})

        # Production de pain dans les moulins
        for b in self.buildings:
            if b.btype != "mill":
                continue
            if b.bread >= MILL_MAX_BREAD or b.wheat <= 0:
                b.mill_ticks = 0
                continue
            # Vérifie qu'une tuile d'eau est accessible dans un rayon de 6 tuiles
            has_water = any(
                int(self.world.biome_grid[b.y + dy, b.x + dx]) in WATER_BIOMES
                for dy in range(-6, 7) for dx in range(-6, 7)
                if self.world.is_valid(b.x + dx, b.y + dy)
            )
            if not has_water:
                continue
            b.mill_ticks += 1
            if b.mill_ticks >= MILL_BREAD_TICKS:
                b.wheat      -= MILL_BREAD_COST_WHEAT
                b.bread      += 1
                b.mill_ticks  = 0

        # Croissance des champs de blé (×2 si arrosé, +1 bonus si pluie)
        for b in self.buildings:
            if b.btype == "wheatfield" and b.stage < 4:
                if b.watered_ticks > 0:
                    b.grow_ticks   += 2
                    b.watered_ticks -= 1
                else:
                    b.grow_ticks += 1
                if self.raining:
                    b.grow_ticks += RAIN_WHEAT_BONUS
                if b.grow_ticks >= WHEAT_TICKS_PER_STAGE:
                    b.stage      += 1
                    b.grow_ticks  = 0

        # ── Incendie spontané (canicule) ─────────────────────────────────────
        if self.heatwave and random.random() < HEATWAVE_FIRE_PROB:
            ys, xs = np.where(
                (self.world.tree_grid >= TREE_STUMP_THRESHOLD) & self.world._forest_mask
            )
            if len(xs):
                idx = random.randrange(len(xs))
                lx, ly = int(xs[idx]), int(ys[idx])
                if self.world.ignite(lx, ly):
                    tick_events.append({"type": "heatwave_fire", "x": lx, "y": ly})

        # ── Foudre (orage) : frappe un arbre aléatoire debout ─────────────────
        if self.storming and random.random() < LIGHTNING_PROB:
            # Tire une tuile forêt avec arbre debout au hasard
            ys, xs = np.where(
                (self.world.tree_grid >= TREE_STUMP_THRESHOLD) & self.world._forest_mask
            )
            if len(xs):
                idx = random.randrange(len(xs))
                lx, ly = int(xs[idx]), int(ys[idx])
                if self.world.ignite(lx, ly):
                    tick_events.append({"type": "lightning",
                                        "x": lx, "y": ly})

        # ── Propagation et extinction des feux de forêt ───────────────────────
        _was_standing = (self.world.tree_grid >= TREE_STUMP_THRESHOLD) & self.world._forest_mask
        fire_changes = self.world.step_fire(season, self.raining)
        _now_stump = _was_standing & (self.world.tree_grid < TREE_STUMP_THRESHOLD)
        if _now_stump.any():
            _ys, _xs = np.where(_now_stump)
            for _x, _y in zip(_xs.tolist(), _ys.tolist()):
                tree_changes.append({"x": int(_x), "y": int(_y), "stump": True})

        # Auto-upgrade L1 → L2 si le clan a assez de ressources en stock
        for b in self.buildings:
            bspec = BUILDING_SPECS.get(b.btype)
            if not bspec or bspec.max_level <= 1 or b.level >= bspec.max_level:
                continue
            clan_houses = [x for x in self.buildings
                           if x.clan_id == b.clan_id and x.btype == "house"]
            # Pour upgrader une maison L2, le feu de camp doit être L2 d'abord
            if b.btype == "house":
                clan_fire = next((x for x in self.buildings
                                  if x.clan_id == b.clan_id and x.btype == "campfire"), None)
                if clan_fire is None or clan_fire.level < 2:
                    continue
            total_wood  = sum(x.wood  for x in clan_houses)
            total_stone = sum(x.stone for x in clan_houses)
            if total_wood < bspec.upgrade_wood or total_stone < bspec.upgrade_stone:
                continue
            # Prélève les ressources (d'abord dans les maisons les plus pleines)
            rem_w = bspec.upgrade_wood
            for h in sorted(clan_houses, key=lambda x: -x.wood):
                take = min(int(h.wood), rem_w); h.wood -= take; rem_w -= take
                if rem_w == 0: break
            rem_s = bspec.upgrade_stone
            for h in sorted(clan_houses, key=lambda x: -x.stone):
                take = min(h.stone, rem_s); h.stone -= take; rem_s -= take
                if rem_s == 0: break
            b.level = 2
            tick_events.append({"type": "upgrade_building", "btype": b.btype,
                                 "clan_id": b.clan_id, "x": b.x, "y": b.y})

        # Ajoute les naissances
        self.entities.extend(births)

        # Purge les morts
        self.entities = [e for e in self.entities if e.alive]

        # Supprimer les bâtiments des clans entièrement éteints
        alive_clan_ids = {e.clan_id for e in self.entities
                          if e.alive and e.etype == EntityType.HUMAN
                          and e.clan_id is not None}
        dead_clan_ids = {c.id for c in self.clans} - alive_clan_ids
        if dead_clan_ids:
            self.buildings = [b for b in self.buildings
                              if b.clan_id not in dead_clan_ids]
            self.clans = [c for c in self.clans if c.id not in dead_clan_ids]

        # Log événements
        self.events_log.extend(tick_events)
        if len(self.events_log) > 200:
            self.events_log = self.events_log[-200:]

        # Stats
        stats = self._compute_stats()
        self.stats_history.append({"tick": self.tick_count, **stats})
        if len(self.stats_history) > 300:
            self.stats_history = self.stats_history[-300:]

        # Arbres abattus et roches minées ce tick (collectés depuis world)
        for _cx, _cy in self.world._chop_changes:
            tree_changes.append({"x": _cx, "y": _cy, "stump": True})
        for _cx, _cy in self.world._mine_changes:
            rock_changes.append({"x": _cx, "y": _cy, "depleted": True})
        self.world._chop_changes.clear()
        self.world._mine_changes.clear()

        return {
            "tick":         self.tick_count,
            "season":       season,
            "temp_c":       temp_c,
            "raining":      self.raining,
            "storming":     self.storming,
            "heatwave":     self.heatwave,
            "fire_changes": fire_changes,
            "entities":     [e.to_dict() for e in self.entities],
            "events":       tick_events,
            "stats":        stats,
            "clans":        [c.to_dict() for c in self.clans],
            "tree_changes":  tree_changes,
            "rock_changes":  rock_changes,
            "biome_changes": biome_changes,
            "buildings":     [b.to_dict() for b in self.buildings],
        }

    def _compute_stats(self) -> dict:
        counts = {t.value: 0 for t in EntityType}
        for e in self.entities:
            counts[e.etype.value] += 1
        return {"populations": counts, "total": len(self.entities)}

    def full_state(self) -> dict:
        """État complet pour un nouveau client qui se connecte."""
        return {
            "world":     self.world.to_dict(),
            "tick":      self.tick_count,
            "season":    get_season(self.tick_count),
            "temp_c":    get_temperature(self.tick_count),
            "raining":   self.raining,
            "storming":  self.storming,
            "heatwave":  self.heatwave,
            "entities":  [e.to_dict() for e in self.entities],
            "stats":     self._compute_stats(),
            "history":   self.stats_history[-100:],
            "events":    self.events_log[-50:],
            "clans":     [c.to_dict() for c in self.clans],
            "buildings": [b.to_dict() for b in self.buildings],
        }
