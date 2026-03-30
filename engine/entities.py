"""
BioSim — Entités (animaux)
Chaque entité est un agent autonome avec état interne.
Extensible : ajouter une espèce = définir son EntitySpec.
"""
import random
import math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class EntityType(str, Enum):
    BOAR         = "boar"
    CHICKEN      = "chicken"
    HORNED_SHEEP = "horned_sheep"
    HORSE        = "horse"
    HUMAN        = "human"
    PIG          = "pig"
    SHEEP        = "sheep"
    FISH         = "fish"
    SHARK        = "shark"


class Sex(str, Enum):
    MALE   = "M"
    FEMALE = "F"


class State(str, Enum):
    IDLE         = "idle"
    WANDERING    = "wandering"
    SEEKING_FOOD = "seeking_food"
    EATING       = "eating"
    FLEEING      = "fleeing"
    HUNTING      = "hunting"
    SEEKING_MATE = "seeking_mate"
    CHOPPING      = "chopping"
    MINING        = "mining"
    BUILDING      = "building"
    FARMING       = "farming"
    FISHING       = "fishing"
    RESTING       = "resting"
    SEEKING_WATER = "seeking_water"
    DRINKING      = "drinking"
    DEAD          = "dead"


@dataclass
class EntitySpec:
    """Définition statique d'une espèce. Ajouter ici pour créer une nouvelle espèce."""
    name: str
    color: str           # couleur rendu web (hex)
    is_predator: bool
    is_prey: bool
    prey_types: list     # EntityType que cette espèce peut chasser
    max_age: float       # en ticks
    max_hunger: float    # hunger max avant mort
    hunger_rate: float   # hunger gagné par tick
    eat_amount: float    # nourriture consommée par repas (végétal)
    eat_meat: float      # quantité récupérée en mangeant une proie
    speed: float         # tuiles/tick (max)
    vision: int          # portée de vision en tuiles
    repro_hunger_min: float  # hunger min pour se reproduire
    repro_cooldown: int  # ticks entre deux reproductions
    gestation: int       # ticks avant naissance
    litter_size: tuple   # (min, max) petits par portée
    flee_distance: int   # distance à laquelle fuit les prédateurs
    aquatic: bool = False   # vit dans l'eau (se déplace sur tuiles WATER)
    hitbox_width:  int = 1  # largeur hitbox en tuiles  (numpy distances si > 1)
    hitbox_height: int = 1  # hauteur hitbox en tuiles  (numpy distances si > 1)
    # ── Capacités spéciales ─────────────────────────────────────────────────
    can_chop: bool  = False  # peut abattre des arbres
    can_mine: bool  = False  # peut miner la pierre
    can_build: bool = False  # peut construire des bâtiments
    can_herd: bool  = False  # se déplace en troupeau


@dataclass
class BuildingSpec:
    """Définition statique d'un type de bâtiment. Ajouter ici pour créer un nouveau bâtiment."""
    btype: str
    wood_cost: int          # bois stocké nécessaire (à partir du 2e)
    stone_cost: int
    build_time: int         # ticks de travail pour finir le bâtiment
    pop_bonus: int          # habitants max supplémentaires par bâtiment dans le clan
    first_cost: int  = 0    # bois porté pour la toute première instance du clan
    min_dist: int    = 4    # distance min entre deux bâtiments de ce type
    min_from_fire: int = 3  # distance min depuis le feu de camp
    max_from_fire: int = 14 # distance max depuis le feu de camp
    max_wood: int    = 0    # stock max de bois (0 = illimité)
    consumes_wood: bool = False  # consomme du bois chaque tick
    max_per_clan: int = 0   # nombre max par clan (0 = illimité)
    # ── Amélioration niveau 2 ───────────────────────────────────────────────
    max_level: int    = 1   # niveau max du bâtiment
    upgrade_wood: int = 0   # bois stocké (pool clan) pour L1→L2
    upgrade_stone: int = 0  # pierre stockée (pool clan) pour L1→L2


BUILDING_SPECS: dict[str, BuildingSpec] = {
    "campfire": BuildingSpec(
        btype="campfire",
        wood_cost=0, stone_cost=0, build_time=0, pop_bonus=0,
        max_level=2, upgrade_wood=5, upgrade_stone=20,
        # L2 : territoire +6 tuiles
    ),
    "house": BuildingSpec(
        btype="house",
        wood_cost=15, stone_cost=0,
        build_time=40, pop_bonus=3,
        first_cost=5,
        min_dist=4, min_from_fire=3, max_from_fire=14,
        max_per_clan=12,
        max_level=2, upgrade_wood=10, upgrade_stone=15,
        # L2 : pop_bonus ×2 (6 habitants)
    ),
    "wheatfield": BuildingSpec(
        btype="wheatfield",
        wood_cost=0, stone_cost=0,
        build_time=8, pop_bonus=0,
        min_dist=2, min_from_fire=3, max_from_fire=15,
        max_per_clan=8,
    ),
    "mill": BuildingSpec(
        btype="mill",
        wood_cost=12, stone_cost=8,
        build_time=60, pop_bonus=0,
        min_dist=4, min_from_fire=3, max_from_fire=16,
        max_per_clan=2,
    ),
    "storehouse": BuildingSpec(
        btype="storehouse",
        wood_cost=10, stone_cost=4,
        build_time=45, pop_bonus=0,
        min_dist=3, min_from_fire=2, max_from_fire=12,
        max_per_clan=1,
    ),
    "well": BuildingSpec(
        btype="well",
        wood_cost=8, stone_cost=6,
        build_time=50, pop_bonus=0,
        min_dist=8, min_from_fire=3, max_from_fire=16,
        max_per_clan=2,
    ),
}


# ── Registre des espèces ────────────────────────────────────────────────────
SPECS: dict[EntityType, EntitySpec] = {

    EntityType.BOAR: EntitySpec(
        name="boar", color="#8B4513",
        is_predator=True, is_prey=False,
        prey_types=[EntityType.CHICKEN, EntityType.SHEEP],
        max_age=2000, hunger_rate=0.11, max_hunger=100,
        eat_amount=8, eat_meat=50, speed=1.2, vision=8,
        repro_hunger_min=35, repro_cooldown=140, gestation=50, litter_size=(1, 3),
        flee_distance=0,
    ),
    EntityType.CHICKEN: EntitySpec(
        name="chicken", color="#f39c12",
        is_predator=False, is_prey=True,
        prey_types=[],
        max_age=1000, hunger_rate=0.07, max_hunger=100,
        eat_amount=8, eat_meat=0, speed=0.7, vision=4,
        repro_hunger_min=40, repro_cooldown=180, gestation=80, litter_size=(1, 2),
        flee_distance=3,
    ),
    EntityType.HORNED_SHEEP: EntitySpec(
        name="horned_sheep", color="#c0a080",
        is_predator=False, is_prey=True,
        prey_types=[],
        max_age=1800, hunger_rate=0.08, max_hunger=100,
        eat_amount=18, eat_meat=0, speed=0.85, vision=6,
        repro_hunger_min=45, repro_cooldown=220, gestation=80, litter_size=(1, 2),
        flee_distance=5,
        can_herd=True,
    ),
    EntityType.HUMAN: EntitySpec(
        name="human", color="#f4d03f",
        is_predator=True, is_prey=False,
        prey_types=[EntityType.CHICKEN, EntityType.SHEEP, EntityType.PIG,
                    EntityType.HORNED_SHEEP, EntityType.HORSE],
        max_age=3600, hunger_rate=0.08, max_hunger=100,
        eat_amount=20, eat_meat=40, speed=1.0, vision=8,
        repro_hunger_min=40, repro_cooldown=120, gestation=60, litter_size=(1, 2),
        flee_distance=0,
        can_chop=True, can_mine=True, can_build=True,
    ),
    EntityType.HORSE: EntitySpec(
        name="horse", color="#a0826d",
        is_predator=False, is_prey=False,
        prey_types=[],
        max_age=3000, hunger_rate=0.09, max_hunger=100,
        eat_amount=25, eat_meat=0, speed=1.5, vision=7,
        repro_hunger_min=45, repro_cooldown=200, gestation=90, litter_size=(1, 1),
        flee_distance=6,
        can_herd=True,
    ),
    EntityType.PIG: EntitySpec(
        name="pig", color="#f1948a",
        is_predator=False, is_prey=True,
        prey_types=[],
        max_age=1500, hunger_rate=0.07, max_hunger=100,
        eat_amount=18, eat_meat=0, speed=0.8, vision=5,
        repro_hunger_min=45, repro_cooldown=200, gestation=80, litter_size=(1, 3),
        flee_distance=4,
    ),
    EntityType.SHEEP: EntitySpec(
        name="sheep", color="#ecf0f1",
        is_predator=False, is_prey=True,
        prey_types=[],
        max_age=1800, hunger_rate=0.06, max_hunger=100,
        eat_amount=15, eat_meat=0, speed=0.9, vision=6,
        repro_hunger_min=50, repro_cooldown=200, gestation=80, litter_size=(1, 2),
        flee_distance=5,
        can_herd=True,
    ),
    EntityType.FISH: EntitySpec(
        name="fish", color="#5dade2",
        is_predator=False, is_prey=True,
        prey_types=[],
        max_age=800, hunger_rate=0.05, max_hunger=100,
        eat_amount=12, eat_meat=0, speed=0.8, vision=5,
        repro_hunger_min=40, repro_cooldown=80, gestation=30, litter_size=(2, 4),
        flee_distance=5,
        aquatic=True,
    ),
    EntityType.SHARK: EntitySpec(
        name="shark", color="#1a5276",
        is_predator=True, is_prey=False,
        prey_types=[EntityType.FISH],
        max_age=6000, hunger_rate=0.04, max_hunger=100,
        eat_amount=0, eat_meat=70, speed=1.2, vision=14,
        repro_hunger_min=30, repro_cooldown=300, gestation=150, litter_size=(1, 2),
        flee_distance=0,
        aquatic=True,
        hitbox_width=1, hitbox_height=2,  # tiles (54,5) et (54,6) : colonne centrale, 2 lignes
    ),
}


# ── Compteur global d'IDs ───────────────────────────────────────────────────
_next_id = 0
def _new_id():
    global _next_id
    _next_id += 1
    return _next_id


# ── Classe entité ───────────────────────────────────────────────────────────
class Entity:
    __slots__ = [
        "id", "etype", "spec", "x", "y",
        "age", "hunger", "sex", "state",
        "repro_cooldown_left", "gestation_left", "alive",
        "target_x", "target_y", "target_id",
        "_move_frac_x", "_move_frac_y",
        "clan_id", "traits", "wood", "stone", "meat",
        "building_ticks_left", "building_type",
        "tool", "pick", "sickle", "watering_can", "can_filled", "fishing_rod", "thirst",
        "_stuck_ticks", "chop_cooldown_left",
    ]

    def __init__(self, etype: EntityType, x: float, y: float, sex: Sex = None):
        self.id    = _new_id()
        self.etype = etype
        self.spec  = SPECS[etype]
        self.x     = float(x)
        self.y     = float(y)
        self.age   = random.uniform(0, self.spec.max_age * 0.3)
        self.hunger = random.uniform(20, 60)
        self.sex   = sex or random.choice([Sex.MALE, Sex.FEMALE])
        self.state = State.WANDERING
        self.repro_cooldown_left = random.randint(0, self.spec.repro_cooldown)
        self.gestation_left = 0
        self.alive  = True
        self.target_x: Optional[float] = None
        self.target_y: Optional[float] = None
        self.target_id: Optional[int]  = None
        self._move_frac_x = 0.0
        self._move_frac_y = 0.0
        self.thirst: float = random.uniform(0, 30)
        self.clan_id: Optional[int] = None
        self.wood: int = 0
        self.stone: int = 0
        self.meat: int = 0
        self.building_ticks_left: int = 0
        self.building_type: Optional[str] = None  # "house", "wheatfield", …
        self.tool: Optional[str] = None   # None, "axe", "stone_axe"
        self.pick: Optional[str] = None   # None, "wood_pick", "stone_pick"
        self.sickle: Optional[str] = None        # None, "sickle"
        self.watering_can: Optional[str] = None  # None, "watering_can"
        self.can_filled: bool = False             # arrosoir rempli d'eau
        self.fishing_rod: Optional[str] = None   # None, "fishing_rod"
        self._stuck_ticks: int = 0               # ticks consécutifs sans déplacement
        self.chop_cooldown_left: int = 0
        # Traits héréditaires : valeurs de base ± 5% aléatoire
        self.traits = {
            "speed":       round(self.spec.speed       * random.uniform(0.95, 1.05), 4),
            "vision":      max(1, round(self.spec.vision * random.uniform(0.95, 1.05))),
            "hunger_rate": round(self.spec.hunger_rate * random.uniform(0.95, 1.05), 5),
        }

    @property
    def ix(self): return int(self.x)
    @property
    def iy(self): return int(self.y)

    def to_dict(self):
        d = {
            "id":    self.id,
            "t":     self.etype.value,
            "x":     round(self.x, 1),
            "y":     round(self.y, 1),
            "s":     self.state.value,
            "h":     round(self.hunger, 1),
            "age":   int(self.age),
            "sex":   self.sex.value,
            "th":    round(self.thirst, 1),
        }
        if self.clan_id is not None:
            d["clan"] = self.clan_id
        if self.spec.can_chop or self.spec.can_mine or self.spec.can_build:
            d["inv"]       = self.wood
            d["stone"]     = self.stone
            d["meat"]      = self.meat
            d["tool"]      = self.tool
            d["pick"]      = self.pick
            d["sickle"]    = self.sickle
            d["wcan"]      = self.watering_can   # watering_can
            d["can_filled"]= self.can_filled
            d["rod"]       = self.fishing_rod
        if self.state == State.BUILDING and self.target_x is not None:
            d["tx"] = int(round(self.target_x))
            d["ty"] = int(round(self.target_y))
        # Traits compactés [speed, vision, hunger_rate]
        d["tr"] = [round(self.traits["speed"], 3),
                   int(self.traits["vision"]),
                   round(self.traits["hunger_rate"], 4)]
        return d


# ── Factory ─────────────────────────────────────────────────────────────────
def spawn(etype: EntityType, x: float, y: float, sex: Sex = None) -> Entity:
    return Entity(etype, x, y, sex)
