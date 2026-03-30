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

    def to_dict(self):
        d = {"id": self.id, "clan_id": self.clan_id,
             "x": self.x, "y": self.y, "btype": self.btype,
             "wood": self.wood, "stone": self.stone, "level": self.level}
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
    "autumn": 0.6,
    "winter": 0.10,
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

# La reproduction n'est possible qu'au printemps et en été
SEASON_REPRO_ALLOWED = {
    "spring": True,
    "summer": True,
    "autumn": False,
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

    if moved:
        entity._stuck_ticks = 0
    else:
        entity._stuck_ticks += 1
        if entity._stuck_ticks >= STUCK_TICKS_RESET:
            # Cible inaccessible : reset + échappée aléatoire
            entity.target_x = None
            entity.target_y = None
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
    if entity.target_x is None or (
        _dist(entity.x, entity.y, entity.target_x, entity.target_y) < 0.5
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
MAX_PER_SPECIES = 200  # cap d'entités par espèce

HERD_RADIUS = 12   # tuiles de détection des congénères
HERD_MIN    = 2    # membres min pour déclencher le comportement de troupeau


def _herd_move(entity: Entity, all_entities: list[Entity], world: "World") -> bool:
    """Déplace vers le centroïde du troupeau proche. Retourne True si appliqué."""
    sx, sy, count = 0.0, 0.0, 0
    for e in all_entities:
        if e is entity or not e.alive or e.etype != entity.etype:
            continue
        if _dist(entity.x, entity.y, e.x, e.y) <= HERD_RADIUS:
            sx += e.x
            sy += e.y
            count += 1
            if count >= 15:   # cap pour ne pas scanner tout le monde
                break
    if count < HERD_MIN:
        return False
    cx, cy = sx / count, sy / count
    if _dist(entity.x, entity.y, cx, cy) < 1.5:
        return False          # déjà au centre → dispersion naturelle via _random_walk
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
    best, best_d = None, max_dist + 1
    for e in entities:
        if e is entity or not e.alive or e.etype != etype:
            continue
        d = _dist(entity.x, entity.y, e.x, e.y)
        if d < best_d:
            best_d = d
            best = e
    return best


def _find_predator_nearby(entity: Entity, entities: list[Entity]) -> "Entity|None":
    fd = entity.spec.flee_distance
    for e in entities:
        if not e.alive:
            continue
        if entity.etype in e.spec.prey_types:
            if _dist(entity.x, entity.y, e.x, e.y) <= fd:
                return e
    return None


def _find_water_spot(entity: Entity, world: "World", max_r: int = 20) -> "tuple[float,float]|None":
    """Trouve la tuile marchable adjacente à de l'eau la plus proche dans un rayon max_r."""
    best_pos, best_d = None, float("inf")
    ex, ey = entity.ix, entity.iy
    for r in range(1, max_r + 1):
        found_this_ring = False
        for dx in range(-r, r + 1):
            for dy in (-r, r) if abs(dx) < r else range(-r, r + 1):
                wx, wy = ex + dx, ey + dy
                if not world.is_valid(wx, wy):
                    continue
                if int(world.biome_grid[wy, wx]) not in WATER_BIOMES:
                    continue
                # Cherche une tuile marchable adjacente à cette tuile d'eau
                for adx, ady in ((-1,0),(1,0),(0,-1),(0,1)):
                    lx, ly = wx + adx, wy + ady
                    if world.is_valid(lx, ly) and world.is_walkable(lx, ly, entity.spec.aquatic):
                        d = _dist(entity.x, entity.y, lx, ly)
                        if d < best_d:
                            best_d = d
                            best_pos = (float(lx), float(ly))
                            found_this_ring = True
        if found_this_ring and best_d < r + 2:
            break   # on a une bonne candidate à distance r, inutile de chercher plus loin
    return best_pos


def _drink_or_seek_water(entity: Entity, world: "World", events: list[dict],
                         buildings: list = None) -> bool:
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

    # Vérifie si un puit est adjacent
    if buildings:
        for b in buildings:
            if b.btype == "well" and _dist(entity.x, entity.y, b.x, b.y) < 1.5:
                entity.state = State.DRINKING
                entity.thirst = max(0.0, entity.thirst - WELL_DRINK_AMOUNT)
                entity.target_x = None
                entity.target_y = None
                return True

    # Pas d'eau adjacente : cherche un puit proche d'abord, sinon une tuile eau
    entity.state = State.SEEKING_WATER
    if entity.target_x is not None:
        _move_toward(entity, entity.target_x, entity.target_y,
                     entity.traits["speed"] * 1.2, world)
        return True

    # Puit du clan en priorité (évite les longs trajets jusqu'à la rivière)
    if buildings and entity.clan_id is not None:
        clan_wells = [b for b in buildings
                      if b.btype == "well" and b.clan_id == entity.clan_id]
        if clan_wells:
            nearest = min(clan_wells, key=lambda b: _dist(entity.x, entity.y, b.x, b.y))
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
                raining: bool = False, heatwave: bool = False):
    if not entity.alive:
        return

    spec = entity.spec

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
                           "count": n, "x": entity.ix, "y": entity.iy})
        entity.state = State.RESTING
        return

    # ── Machine à états ─────────────────────────────────────────────────────

    # 1. Fuite (priorité max pour les proies)
    if spec.flee_distance > 0:
        predator = _find_predator_nearby(entity, all_entities)
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
        if _drink_or_seek_water(entity, world, events, buildings):
            return

    # 2. Chasse (prédateurs)
    if spec.is_predator and entity.hunger > 35:
        for prey_type in spec.prey_types:
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
        clan_mills = [b for b in (buildings or [])
                      if b.clan_id == entity.clan_id and b.btype == "mill"]
        mill_adj = next((m for m in clan_mills
                         if m.bread > 0 and _dist(entity.x, entity.y, m.x, m.y) < 1.5), None)
        if mill_adj is not None:
            mill_adj.bread -= 1
            entity.hunger   = max(0.0, entity.hunger - MILL_BREAD_FOOD)
            entity.state    = State.EATING
            events.append({"type": "eat_bread", "clan_id": entity.clan_id,
                           "x": entity.ix, "y": entity.iy})
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
            return
        # Soif modérée interrompt la recherche de nourriture (mais pas le repas en cours)
        if not spec.aquatic and entity.thirst > 50:
            if _drink_or_seek_water(entity, world, events, buildings):
                return
        # Cherche une tuile avec de la nourriture dans son champ de vision
        entity.state = State.SEEKING_FOOD
        best_f, best_pos = -1, None
        r = int(entity.traits["vision"])
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                fx, fy = entity.ix + dx, entity.iy + dy
                if world.is_valid(fx, fy) and world.is_walkable(fx, fy, spec.aquatic):
                    f = world.get_food(fx, fy)
                    if f > best_f:
                        best_f = f
                        best_pos = (fx, fy)
        if best_pos and best_f > 5:
            entity.target_x = float(best_pos[0])
            entity.target_y = float(best_pos[1])
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed, world)
            return

    # 3.7 Soif légère (après manger, avant bois/construction)
    if not spec.aquatic and entity.thirst > 35:
        if _drink_or_seek_water(entity, world, events, buildings):
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
                events.append({"type": "fish_catch",
                               "clan_id": entity.clan_id,
                               "x": entity.ix, "y": entity.iy})
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
        clan_houses = [b for b in (buildings or [])
                       if b.clan_id == entity.clan_id and b.btype == "house"]
        if clan_houses:
            nearest = min(clan_houses, key=lambda b: _dist(entity.x, entity.y, b.x, b.y))
            dist_to_house = _dist(entity.x, entity.y, nearest.x, nearest.y)
            if dist_to_house < 1.5:
                # Adjacent : dépose tout
                nearest.wood  += entity.wood;  entity.wood  = 0
                nearest.stone += entity.stone; entity.stone = 0
            elif entity.wood >= MAX_CARRY or entity.stone >= MAX_STONE_CARRY:
                # Portée pleine → aller déposer
                entity.state = State.WANDERING
                entity.target_x = float(nearest.x)
                entity.target_y = float(nearest.y)
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed, world)
                return
                # Fabrication d'outils (priorité : pioche bois > hache bois > pioche pierre > hache pierre)
                if entity.pick is None and nearest.wood >= PICK_WOOD_COST:
                    nearest.wood -= PICK_WOOD_COST
                    entity.pick = "wood_pick"
                    events.append({"type": "craft_pick", "pick": "wood_pick",
                                   "clan_id": entity.clan_id,
                                   "x": entity.ix, "y": entity.iy})
                elif entity.tool is None and nearest.wood >= AXE_CRAFT_COST:
                    nearest.wood -= AXE_CRAFT_COST
                    entity.tool = "axe"
                    events.append({"type": "craft_axe", "tool": "axe",
                                   "clan_id": entity.clan_id,
                                   "x": entity.ix, "y": entity.iy})
                elif entity.pick == "wood_pick" and nearest.stone >= STONE_PICK_COST:
                    nearest.stone -= STONE_PICK_COST
                    entity.pick = "stone_pick"
                    events.append({"type": "craft_pick", "pick": "stone_pick",
                                   "clan_id": entity.clan_id,
                                   "x": entity.ix, "y": entity.iy})
                elif entity.tool == "axe" and nearest.stone >= STONE_AXE_COST:
                    nearest.stone -= STONE_AXE_COST
                    entity.tool = "stone_axe"
                    events.append({"type": "craft_axe", "tool": "stone_axe",
                                   "clan_id": entity.clan_id,
                                   "x": entity.ix, "y": entity.iy})
                elif entity.watering_can is None and nearest.wood >= WATERING_CAN_WOOD_COST:
                    nearest.wood -= WATERING_CAN_WOOD_COST
                    entity.watering_can = "watering_can"
                    events.append({"type": "craft_watering_can",
                                   "clan_id": entity.clan_id,
                                   "x": entity.ix, "y": entity.iy})
                elif entity.sickle is None and nearest.stone >= SICKLE_STONE_COST:
                    nearest.stone -= SICKLE_STONE_COST
                    entity.sickle = "sickle"
                    events.append({"type": "craft_sickle",
                                   "clan_id": entity.clan_id,
                                   "x": entity.ix, "y": entity.iy})
                elif entity.fishing_rod is None and nearest.wood >= FISHING_ROD_WOOD_COST:
                    nearest.wood -= FISHING_ROD_WOOD_COST
                    entity.fishing_rod = "fishing_rod"
                    events.append({"type": "craft_fishing_rod",
                                   "clan_id": entity.clan_id,
                                   "x": entity.ix, "y": entity.iy})
                return
            elif entity.wood >= MAX_CARRY or entity.stone >= MAX_STONE_CARRY:
                # Porté plein → aller déposer
                entity.state = State.WANDERING
                entity.target_x = float(nearest.x)
                entity.target_y = float(nearest.y)
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed, world)
                return

    # 4.1 Continue la construction en cours (prioritaire sur reproduction/bois/errance)
    if entity.building_ticks_left > 0:
        entity.building_ticks_left -= 1
        entity.state = State.BUILDING
        if entity.target_x is not None:
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed, world)
        if entity.building_ticks_left == 0:
            bx = int(round(entity.target_x)) if entity.target_x is not None else entity.ix
            by = int(round(entity.target_y)) if entity.target_y is not None else entity.iy
            btype = entity.building_type or "house"
            events.append({"type": f"build_{btype}", "clan_id": entity.clan_id,
                           "x": bx, "y": by})
            entity.building_type = None
        return

    # 4. Reproduction (bloquée en automne et en hiver)
    if (SEASON_REPRO_ALLOWED[season]
            and entity.hunger < spec.repro_hunger_min
            and entity.repro_cooldown_left == 0
            and entity.sex == Sex.FEMALE
            and entity.age > spec.max_age * 0.1):
        # Cap population par maisons pour les clans humains
        repro_allowed = True
        if entity.spec.can_build and entity.clan_id is not None:
            base = BUILDING_SPECS["house"].pop_bonus
            cap = max(3, sum(base * b.level
                             for b in (buildings or [])
                             if b.clan_id == entity.clan_id and b.btype == "house"))
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

    # 4.25 Démarre une construction (pas déjà en chantier, source de bois disponible)
    if (entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_ticks_left == 0
            and entity.hunger < 65
            and clans):
        clan = clans.get(entity.clan_id)
        if clan:
            bspec = BUILDING_SPECS["house"]
            clan_houses = [b for b in (buildings or [])
                           if b.clan_id == entity.clan_id and b.btype == "house"]
            # Cap village atteint → pas de nouvelle construction
            if bspec.max_per_clan == 0 or len(clan_houses) < bspec.max_per_clan:
                # Source de bois : porté (1re maison) ou stockage (maisons suivantes)
                if not clan_houses:
                    can_build_house = entity.wood >= bspec.first_cost
                    wood_source = None      # None = prélève sur le porteur
                else:
                    donor = max(clan_houses, key=lambda b: b.wood)
                    can_build_house = donor.wood >= bspec.wood_cost
                    wood_source = donor     # prélève dans le stockage de cette maison
                if can_build_house:
                    for _ in range(30):
                        angle = random.uniform(0, 2 * math.pi)
                        dist  = random.uniform(bspec.min_from_fire, bspec.max_from_fire)
                        bx = int(clan.cx + math.cos(angle) * dist)
                        by = int(clan.cy + math.sin(angle) * dist)
                        if not world.is_valid(bx, by) or not world.is_walkable(bx, by):
                            continue
                        if any(_dist(bx, by, b.x, b.y) < bspec.min_dist for b in clan_houses):
                            continue
                        # Emplacement valide → prélève le bois et démarre le chantier
                        if wood_source is None:
                            entity.wood -= bspec.first_cost
                        else:
                            wood_source.wood -= bspec.wood_cost
                        entity.building_ticks_left = bspec.build_time
                        entity.building_type = "house"
                        entity.target_x = float(bx)
                        entity.target_y = float(by)
                        return

    # 4.26 Construction d'un moulin (si le clan a ≥2 champs et pas encore de moulin)
    if (entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_ticks_left == 0
            and entity.hunger < 65
            and clans):
        bspec_m = BUILDING_SPECS["mill"]
        clan_mills   = [b for b in (buildings or [])
                        if b.clan_id == entity.clan_id and b.btype == "mill"]
        clan_fields  = [b for b in (buildings or [])
                        if b.clan_id == entity.clan_id and b.btype == "wheatfield"]
        if (len(clan_fields) >= 2
                and (bspec_m.max_per_clan == 0 or len(clan_mills) < bspec_m.max_per_clan)):
            clan_houses = [b for b in (buildings or [])
                           if b.clan_id == entity.clan_id and b.btype == "house"]
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
                        entity.building_ticks_left = bspec_m.build_time
                        entity.building_type = "mill"
                        entity.target_x = float(mx)
                        entity.target_y = float(my)
                        return

    # 4.27 Construction d'un puit (si le clan a ≥1 maison et pas encore de puit nearby)
    if (entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_ticks_left == 0
            and entity.hunger < 65
            and clans):
        bspec_well = BUILDING_SPECS["well"]
        clan_wells  = [b for b in (buildings or [])
                       if b.clan_id == entity.clan_id and b.btype == "well"]
        clan_houses = [b for b in (buildings or [])
                       if b.clan_id == entity.clan_id and b.btype == "house"]
        if (len(clan_houses) >= 1
                and (bspec_well.max_per_clan == 0 or len(clan_wells) < bspec_well.max_per_clan)):
            donor   = max(clan_houses, key=lambda b: b.wood)   if clan_houses else None
            donor_s = max(clan_houses, key=lambda b: b.stone)  if clan_houses else None
            has_res = (donor   and donor.wood    >= bspec_well.wood_cost
                       and donor_s and donor_s.stone >= bspec_well.stone_cost)
            if has_res:
                clan = clans.get(entity.clan_id)
                if clan:
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
                        donor.wood    -= bspec_well.wood_cost
                        donor_s.stone -= bspec_well.stone_cost
                        entity.building_ticks_left = bspec_well.build_time
                        entity.building_type = "well"
                        entity.target_x = float(wx)
                        entity.target_y = float(wy)
                        return

    # 4.3a Récolte : champ mûr (priorité élevée : affamé OU adjacent à un champ mûr)
    if entity.spec.can_build and entity.clan_id is not None:
        clan_fields = [b for b in (buildings or [])
                       if b.clan_id == entity.clan_id and b.btype == "wheatfield"]
        clan_mills  = [b for b in (buildings or [])
                       if b.clan_id == entity.clan_id and b.btype == "mill"]
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
                events.append({"type": "harvest_wheat", "clan_id": entity.clan_id,
                               "x": ripe_adj.x, "y": ripe_adj.y, "to_mill": True})
            else:
                food = WHEAT_HARVEST_FOOD + (SICKLE_HARVEST_BONUS if entity.sickle else 0.0)
                entity.hunger = max(0.0, entity.hunger - food)
                events.append({"type": "harvest_wheat", "clan_id": entity.clan_id,
                               "x": ripe_adj.x, "y": ripe_adj.y})
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
        clan_fields = [b for b in (buildings or [])
                       if b.clan_id == entity.clan_id and b.btype == "wheatfield"]
        clan_mills  = [b for b in (buildings or [])
                       if b.clan_id == entity.clan_id and b.btype == "mill"]

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
                    events.append({"type": "water_wheat", "clan_id": entity.clan_id,
                                   "x": dry.x, "y": dry.y})
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
                clan_wells = [b for b in (buildings or [])
                              if b.btype == "well" and b.clan_id == entity.clan_id]
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
        if (entity.building_ticks_left == 0
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
                            or world.biome_grid[fy, fx] != int(Biome.GRASS)):
                        continue
                    if any(_dist(fx, fy, b.x, b.y) < bspec_w.min_dist for b in clan_fields):
                        continue
                    entity.building_ticks_left = bspec_w.build_time
                    entity.building_type = "wheatfield"
                    entity.target_x = float(fx)
                    entity.target_y = float(fy)
                    return

    # 4.5 Coupe du bois (entités avec can_chop, si pas trop affamées et portée non pleine)
    if entity.spec.can_chop and entity.wood < MAX_CARRY and entity.hunger < 70:
        # Arbre adjacent → coupe directement
        for adx in range(-1, 2):
            for ady in range(-1, 2):
                if adx == 0 and ady == 0:
                    continue
                tx, ty = entity.ix + adx, entity.iy + ady
                if world.is_choppable(tx, ty):
                    entity.state = State.CHOPPING
                    wood = world.chop_tree(tx, ty)
                    if entity.tool == "stone_axe":
                        wood += STONE_AXE_BONUS
                    elif entity.tool == "axe":
                        wood += AXE_BONUS
                    entity.wood = min(MAX_CARRY, entity.wood + wood)
                    events.append({"type": "chop", "x": tx, "y": ty})
                    return
        # Arbre dans le champ de vision → se déplace vers le plus proche
        r = int(entity.traits["vision"])
        best_tree, best_d = None, float("inf")
        for vdx in range(-r, r + 1):
            for vdy in range(-r, r + 1):
                tx, ty = entity.ix + vdx, entity.iy + vdy
                if world.is_choppable(tx, ty):
                    d = _dist(entity.x, entity.y, tx, ty)
                    if d < best_d:
                        best_d = d
                        best_tree = (tx, ty)
        if best_tree:
            entity.state = State.CHOPPING
            entity.target_x = float(best_tree[0])
            entity.target_y = float(best_tree[1])
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
                    events.append({"type": "mine", "x": tx, "y": ty})
                    return
        # Roche dans le champ de vision → se déplace vers la plus proche
        r = int(entity.traits["vision"])
        best_rock, best_d = None, float("inf")
        for vdx in range(-r, r + 1):
            for vdy in range(-r, r + 1):
                tx, ty = entity.ix + vdx, entity.iy + vdy
                if world.is_mineable(tx, ty):
                    d = _dist(entity.x, entity.y, tx, ty)
                    if d < best_d:
                        best_d = d
                        best_rock = (tx, ty)
        if best_rock:
            entity.state = State.MINING
            entity.target_x = float(best_rock[0])
            entity.target_y = float(best_rock[1])
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed * 0.8, world)
            return

    # 4.7 Exploration du territoire (humain reposé : étend les frontières du clan)
    if (entity.etype == EntityType.HUMAN
            and entity.clan_id is not None
            and entity.hunger < 40
            and entity.thirst < 30
            and clans
            and random.random() < 0.18):
        clan = clans.get(entity.clan_id)
        if clan:
            for _ in range(12):
                angle = random.uniform(0, 2 * math.pi)
                dist  = random.uniform(35, 80)
                tx = int(clan.cx + math.cos(angle) * dist)
                ty = int(clan.cy + math.sin(angle) * dist)
                tx = max(0, min(world.width  - 1, tx))
                ty = max(0, min(world.height - 1, ty))
                if world.is_walkable(tx, ty):
                    entity.target_x = float(tx)
                    entity.target_y = float(ty)
                    entity.state = State.WANDERING
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
            # Plus reposé → explore plus loin autour du campfire
            wander_r = 40 if entity.hunger < 40 else 22
            for _ in range(8):
                tx = clan.cx + random.uniform(-wander_r, wander_r)
                ty = clan.cy + random.uniform(-wander_r, wander_r)
                tx = max(0, min(world.width  - 1, tx))
                ty = max(0, min(world.height - 1, ty))
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
            EntityType.BOAR:         25,
            EntityType.CHICKEN:      30,
            EntityType.HORNED_SHEEP: 15,
            EntityType.HORSE:        10,
            EntityType.PIG:          30,
            EntityType.SHEEP:        30,
            EntityType.FISH:         30,
            EntityType.SHARK:        5,
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
        clan_positions: list[tuple[int, int]] = []
        for _ in range(8000):
            x = random.randint(10, self.world.width  - 10)
            y = random.randint(10, self.world.height - 10)
            if not self.world.is_walkable(x, y):
                continue
            if all(_dist(x, y, cx, cy) >= MIN_CAMPFIRE_DIST for cx, cy in clan_positions):
                clan_positions.append((x, y))
            if len(clan_positions) == N_CLANS:
                break
        # Fallback : compléter sans contrainte de distance si la map est serrée
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
                        member.hunger  = random.uniform(10, 30)
                        member.age     = SPECS[EntityType.HUMAN].max_age * random.uniform(0.1, 0.3)
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

        for e in self.entities:
            tick_entity(e, self.world, self.entities, births, tick_events,
                        self.tick_count, season, clans_dict, self.buildings,
                        temp_c, species_counts, self.raining, self.heatwave)

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
        self.buildings.extend(new_this_tick)

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
                tick_events.append({"type": "bake_bread", "clan_id": b.clan_id,
                                    "x": b.x, "y": b.y})

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
        fire_changes = self.world.step_fire(season, self.raining)

        # Auto-upgrade L1 → L2 si le clan a assez de ressources en stock
        for b in self.buildings:
            bspec = BUILDING_SPECS.get(b.btype)
            if not bspec or bspec.max_level <= 1 or b.level >= bspec.max_level:
                continue
            clan_houses = [x for x in self.buildings
                           if x.clan_id == b.clan_id and x.btype == "house"]
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

        # Log événements
        self.events_log.extend(tick_events)
        if len(self.events_log) > 200:
            self.events_log = self.events_log[-200:]

        # Stats
        stats = self._compute_stats()
        self.stats_history.append({"tick": self.tick_count, **stats})
        if len(self.stats_history) > 300:
            self.stats_history = self.stats_history[-300:]

        # Arbres abattus / roches minées ce tick
        for ev in tick_events:
            if ev["type"] == "chop":
                tree_changes.append({"x": ev["x"], "y": ev["y"], "stump": True})
            elif ev["type"] == "mine":
                rock_changes.append({"x": ev["x"], "y": ev["y"], "depleted": True})

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
