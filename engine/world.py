"""
BioSim — Moteur de simulation
World: grille, biomes, ressources
"""
import base64
import random
from collections import deque
import numpy as np
from enum import IntEnum

# Bornes défensives de chargement (durcissement I2, audit #16) : un save corrompu ou
# forgé (dimensions aberrantes / grille géante) ne doit pas provoquer d'OOM au boot —
# c'est un risque pour l'invariant « tourne à l'infini » (LOAD_ON_START rejouerait).
_MAX_DIM        = 4096                 # dimension max d'un monde chargé
_MAX_GRID_CELLS = _MAX_DIM * _MAX_DIM  # cellules max d'une grille chargée

# ── ÉCHELLE DE TEMPS (réalisme, 2026-07-15, demande Alexis) ───────────────────
# Ralentit l'HORLOGE BIOLOGIQUE du monde d'un facteur TIME_SCALE, en laissant la
# VITESSE de déplacement inchangée. Effet : une créature vit toujours le même nombre
# de JOURS, mais parcourt TIME_SCALE× plus de terrain pendant ce temps → fini le
# « quelques cases par jour » irréaliste. Les saisons se vivent (5 min au lieu de 50 s).
#
# Règle d'application (tout écart casse l'équilibre — tous les rapports doivent tenir) :
#   × TIME_SCALE  : les DURÉES en ticks (vie, gestation, cooldowns, périodes, chantiers).
#   ÷ TIME_SCALE  : les TAUX par tick (faim, soif, repousse, fertilité, science, probas/tick).
#   INCHANGÉ      : vitesse, vision, distances, seuils, quantités par ACTION, timeouts de
#                   trajet (bornés par la vitesse, qui ne bouge pas).
# Importé par entities.py et simulation.py (world.py = module le plus bas, aucun cycle).
TIME_SCALE = 6

# Rayon (tuiles, Chebyshev) du test « proche de l'eau » pour la cuisson/placement des
# moulins. Pré-calculé une fois en masque booléen (World._near_water_mill) car l'eau est
# immuable ; défini ici pour rester la source unique du rayon partagée par le masque et
# tout appelant. Le changer impose de regénérer le masque (fait au __init__ / au load).
MILL_WATER_RADIUS = 6


def _grid_to_b64(arr: np.ndarray) -> dict:
    """Encode une grille numpy en base64 (compact + sans perte) pour la sauvegarde."""
    return {"b64": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii"),
            "dtype": str(arr.dtype), "shape": list(arr.shape)}


def _grid_from_b64(d: dict) -> np.ndarray:
    """Décode une grille base64. Durci (I2) : valide la forme et borne la taille AVANT
    de décoder (b64decode alloue la mémoire brute) → un save forgé lève proprement au
    lieu de faire exploser la RAM."""
    shape = tuple(int(s) for s in d["shape"])
    cells = 1
    for s in shape:
        if s < 0:
            raise ValueError(f"forme de grille invalide: {shape}")
        cells *= s
    if cells > _MAX_GRID_CELLS:
        raise ValueError(f"grille trop grande: {cells} cellules > {_MAX_GRID_CELLS}")
    b64 = d["b64"]
    if len(b64) > cells * 16 + 64:   # borne la chaîne AVANT le décodage (anti-OOM)
        raise ValueError("chaîne base64 incohérente avec la forme (trop longue)")
    a = np.frombuffer(base64.b64decode(b64), dtype=np.dtype(d["dtype"]))
    if a.size != cells:
        raise ValueError(f"buffer de {a.size} éléments incohérent avec la forme {shape}")
    return a.reshape(shape).copy()

class Biome(IntEnum):
    WATER   = 0
    GRASS   = 1
    FOREST  = 2
    DESERT  = 3
    MOUNTAIN= 4
    RIVER   = 5   # rivières — sprite eau peu profonde, visuellement distinct de la mer
    DIRT    = 6   # herbe surpâturée — terre battue, sans nourriture

# Couleurs hex par biome (pour le rendu web)
BIOME_COLORS = {
    Biome.WATER:    "#3a7bd5",
    Biome.GRASS:    "#5aad5a",
    Biome.FOREST:   "#2d7a2d",
    Biome.DESERT:   "#c8a96e",
    Biome.MOUNTAIN: "#8a8a8a",
    Biome.RIVER:    "#5bb8d4",
    Biome.DIRT:     "#8B7355",
}

# Ressources nourriture max par biome
BIOME_FOOD_MAX = {
    Biome.WATER:    60,   # plancton / algues pour les poissons
    Biome.GRASS:    100,
    Biome.FOREST:   60,
    Biome.DESERT:   10,
    Biome.MOUNTAIN: 5,
    Biome.RIVER:    30,
    Biome.DIRT:     0,    # terre battue : aucune nourriture
}

BIOME_FOOD_REGEN = {   # valeurs de BASE (par tick) — divisées par TIME_SCALE juste après
    Biome.WATER:    2.0,
    Biome.GRASS:    3,
    Biome.FOREST:   2,
    Biome.DESERT:   0.2,
    Biome.MOUNTAIN: 0.1,
    Biome.RIVER:    1.0,
    Biome.DIRT:     0.0,
}
# Taux/tick → ralentis avec l'horloge biologique (cf. TIME_SCALE)
BIOME_FOOD_REGEN = {k: v / TIME_SCALE for k, v in BIOME_FOOD_REGEN.items()}


TREE_STUMP_THRESHOLD  = 50.0   # en-dessous : souche visible
TREE_REGEN_RATE       = 0.08 / TIME_SCALE   # santé/tick (taux → ralenti avec l'horloge)
WOOD_PER_CHOP         = 2      # bois obtenu par abattage (sans outil) — par ACTION, inchangé

# ── Feu de forêt ─────────────────────────────────────────────────────────────
FIRE_INTENSITY_INIT   = 80.0   # intensité initiale quand une tuile s'embrase (seuil, inchangé)
FIRE_BURN_RATE        = 1.2  / TIME_SCALE   # intensité perdue/tick (la tuile brûle TIME_SCALE× plus de ticks)
FIRE_SPREAD_PROB_DRY  = 0.012 / TIME_SCALE  # proba de propagation/tick en été (forêt sèche)
FIRE_SPREAD_PROB_WET  = 0.003 / TIME_SCALE  # proba hors été
FIRE_RAIN_DAMP        = 4.0  / TIME_SCALE   # intensité retirée/tick sous la pluie

STONE_STUMP_THRESHOLD = 50.0   # en-dessous : roche épuisée (visible)
STONE_REGEN_RATE      = 0.021 / TIME_SCALE  # santé/tick
STONE_PER_MINE        = 1      # pierre par minage (sans outil) — par ACTION, inchangé

# ── Fer (bloc B : rare, dans un sous-ensemble des montagnes) ───────────────────
IRON_FRACTION         = 0.16   # part des tuiles montagne portant du fer
IRON_STUMP_THRESHOLD  = 50.0   # en-dessous : gisement de fer épuisé
IRON_REGEN_RATE       = 0.012 / TIME_SCALE  # régen plus lente que la pierre (le fer est rare)
IRON_PER_MINE         = 3      # fer obtenu par minage (veine riche → 1 voyage suffit)

# ── Or (bloc C2 : plus rare que le fer, filons distincts) ──────────────────────
GOLD_FRACTION         = 0.05   # part des tuiles montagne (hors fer) portant de l'or
GOLD_STUMP_THRESHOLD  = 50.0
GOLD_REGEN_RATE       = 0.008 / TIME_SCALE  # la plus lente : l'or est précieux
GOLD_PER_MINE         = 1      # 1 pièce par coup (< MAX_GOLD_CARRY=2 : jamais écrêté)

# ── Fertilité / pâturage ──────────────────────────────────────────────────────
FERTILITY_MAX          = 100.0   # fertilité pleine (seuil, inchangé)
FERTILITY_CONSUME      = 12.0    # retirée par ACTION de broutage → inchangé
FERTILITY_TRAMPLE      = 0.3  / TIME_SCALE   # retirée par TICK de présence → taux
FERTILITY_REGEN_BASE   = 0.05 / TIME_SCALE   # regen/tick
FERTILITY_REGEN_RAIN   = 0.12 / TIME_SCALE   # bonus regen/tick sous la pluie
FERTILITY_REGEN_SPRING = 0.03 / TIME_SCALE   # bonus regen/tick au printemps


class World:
    def __init__(self, width: int = 80, height: int = 60, seed: int = None):
        self.width = width
        self.height = height
        # `seed or …` traiterait seed=0 comme absent (audit #3) → from_state({seed:0}) régénérerait
        # un monde ALÉATOIRE, dont les masques immuables (walkable/eau/forêt) ne correspondraient plus
        # aux grilles chargées = corruption silencieuse. Seul None = « pas de graine ».
        self.seed = seed if seed is not None else random.randint(0, 999999)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.biome_grid = self._generate_biomes()
        self.food_grid  = self._init_food()
        # Grilles pré-calculées pour la régénération (vectorisée)
        self._regen_grid = self._build_regen_grid()
        self._max_grid   = self._build_max_grid()
        # Grille des arbres (santé 0–100, uniquement tuiles forêt)
        self._forest_mask = (self.biome_grid == int(Biome.FOREST))
        self.tree_grid = np.where(self._forest_mask, 100.0, 0.0).astype(np.float32)
        # Grille des roches (santé 0–100, uniquement tuiles montagne)
        self._mountain_mask = (self.biome_grid == int(Biome.MOUNTAIN))
        self.stone_grid = np.where(self._mountain_mask, 100.0, 0.0).astype(np.float32)
        # Gisements de fer (bloc B) : sous-ensemble RARE des montagnes. Tiré d'un
        # RandomState LOCAL (indépendant du flux np.random global) → reproduit à
        # l'identique par from_state (qui re-appelle __init__ avec le même seed).
        _iron_rng = np.random.RandomState((self.seed ^ 0x1A0E5) & 0xFFFFFFFF)
        self._iron_mask = self._mountain_mask & (
            _iron_rng.random((self.height, self.width)) < IRON_FRACTION)
        self.iron_grid = np.where(self._iron_mask, 100.0, 0.0).astype(np.float32)
        # Filons d'OR (bloc C2) : encore plus rares, DISTINCTS du fer. RandomState
        # LOCAL (même pattern que _iron_rng : flux np.random global intact).
        _gold_rng = np.random.RandomState((self.seed ^ 0x901D) & 0xFFFFFFFF)
        self._gold_mask = self._mountain_mask & ~self._iron_mask & (
            _gold_rng.random((self.height, self.width)) < GOLD_FRACTION)
        self.gold_grid = np.where(self._gold_mask, 100.0, 0.0).astype(np.float32)
        # Grille de feu (intensité 0–100, uniquement tuiles forêt avec arbre debout)
        self.fire_grid = np.zeros((self.height, self.width), dtype=np.float32)
        # Fertilité des tuiles herbe (GRASS → DIRT quand épuisée, régénère lentement)
        self._orig_grass_mask = (self.biome_grid == int(Biome.GRASS))
        self.fertility_grid = np.where(
            self._orig_grass_mask, FERTILITY_MAX, 0.0
        ).astype(np.float32)
        self._biome_changes: list = []   # changements GRASS↔DIRT ce tick
        self._chop_changes: list = []    # arbres abattus ce tick [(x, y), ...]
        self._mine_changes: list = []    # roches minées ce tick [(x, y), ...]
        # Métrique d'observabilité (I0) : nombre cumulé de resets de cible pour
        # cause de blocage (_move_toward). NON inclus dans la sortie de step() →
        # neutre pour le hash déterministe. Indicateur n°1 du succès du pathfinding.
        self._stuck_resets: int = 0
        # ── Masques de navigation pré-calculés (invariants : l'eau ne bouge pas) ──
        _water = (self.biome_grid == int(Biome.WATER)) | (self.biome_grid == int(Biome.RIVER))
        self._walkable          = ~_water                     # bool, terrestres
        self._aquatic_walkable  = _water                      # bool, aquatiques
        # Tuiles terrestres adjacentes à l'eau (pour pénaliser nourriture en bord d'eau)
        _nw = np.zeros((self.height, self.width), dtype=bool)
        _nw[:-1, :] |= _water[1:, :]
        _nw[1:,  :] |= _water[:-1, :]
        _nw[:, :-1] |= _water[:, 1:]
        _nw[:, 1:]  |= _water[:, :-1]
        self._near_water = _nw & ~_water
        # Masque « eau dans un rayon de MILL_WATER_RADIUS » (cuisson des moulins,
        # placement C3). Remplace le scan Python O(r²) de _tile_near_water (169 tuiles/
        # appel), qui était le poste CPU n°1 du moteur — d'autant plus après le rescale
        # du temps (÷ faim → 4× plus d'entités atteignent _beh_work et l'appellent).
        # L'eau étant IMMUABLE (seuls GRASS↔DIRT bougent), ce masque reste valide pour
        # toujours. Dilatation carrée séparable, edge-correcte (pas de wrap), byte-exacte
        # avec la boucle « any(... in WATER for dy,dx in [-r,r] if is_valid) » d'origine.
        _r = MILL_WATER_RADIUS
        _horz = _water.copy()
        for _k in range(1, _r + 1):
            _horz[:, _k:]  |= _water[:, :-_k]
            _horz[:, :-_k] |= _water[:, _k:]
        _sq = _horz.copy()
        for _k in range(1, _r + 1):
            _sq[_k:, :]  |= _horz[:-_k, :]
            _sq[:-_k, :] |= _horz[_k:, :]
        self._near_water_mill = _sq
        # Carte BFS : pour chaque tuile terrestre, coordonnées de la tuile bord-eau la plus proche
        # Permet à _find_water_spot de répondre en O(1) au lieu d'O(r²)
        self._nearest_water_tile = self._build_nearest_water()

    def _fbm(self, X: np.ndarray, Y: np.ndarray,
             octaves: int = 6, base_freq: float = 0.025,
             lacunarity: float = 2.05, gain: float = 0.5) -> np.ndarray:
        """Fractional Brownian Motion vectorisé sur grilles numpy."""
        result = np.zeros(np.broadcast(X, Y).shape)
        amp, freq, norm = 1.0, base_freq, 0.0
        for _ in range(octaves):
            ox = random.uniform(0, 500)
            oy = random.uniform(0, 500)
            result += amp * np.sin((X + ox) * freq) * np.cos((Y + oy) * freq)
            norm += amp
            amp   *= gain
            freq  *= lacunarity
        return result / norm

    def _generate_biomes(self) -> np.ndarray:
        """Génération procédurale : fBm + domain warping + moisture + mer + rivières."""
        w, h = self.width, self.height
        X = np.arange(w, dtype=float)[np.newaxis, :]   # (1, w)
        Y = np.arange(h, dtype=float)[:, np.newaxis]   # (h, 1)

        # — Carte de hauteur : fBm 6 octaves + domain warping —
        # Le warp distord les coordonnées → formes organiques, pas de bandes
        warp_strength = max(w, h) * 0.12
        Xw = X + warp_strength * self._fbm(X, Y, octaves=4, base_freq=0.018)
        Yw = Y + warp_strength * self._fbm(X, Y, octaves=4, base_freq=0.018)
        hmap = self._fbm(Xw, Yw, octaves=6, base_freq=0.025)
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())

        # — Mer sur le bord sud (gradient doux) —
        sea_grad = np.linspace(1.0, 0.50, h)[:, np.newaxis]
        hmap *= sea_grad
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())

        # — Carte de moisture : fBm indépendant → biomes organiques —
        mmap = self._fbm(X, Y, octaves=5, base_freq=0.022)
        mmap = (mmap - mmap.min()) / (mmap.max() - mmap.min())

        # — Assignation des biomes : hauteur + moisture —
        # Table inspirée de Whittaker : herbe=sec/bas, forêt=humide/bas, désert=sec/haut
        grid = np.zeros((h, w), dtype=np.int8)   # WATER par défaut
        land = hmap >= 0.22
        low  = land & (hmap < 0.50)
        mid  = land & (hmap >= 0.50) & (hmap < 0.65)
        high = land & (hmap >= 0.65)

        grid[low  & (mmap < 0.80)] = Biome.GRASS
        grid[low  & (mmap >= 0.80)] = Biome.FOREST
        grid[mid  & (mmap < 0.74)] = Biome.DESERT
        grid[mid  & (mmap >= 0.74)] = Biome.FOREST
        grid[high] = Biome.MOUNTAIN

        # Rivières depuis les hautes terres vers la mer
        self._carve_rivers(hmap, grid, n_rivers=4)

        return grid

    def _carve_rivers(self, hmap: np.ndarray, grid: np.ndarray, n_rivers: int = 4):
        """Creuse N rivières depuis les hautes terres vers la mer (sud)."""
        h, w = self.height, self.width

        # Sources dans la moitié nord sur terrain élevé
        ys, xs = np.where((hmap > 0.55) & (np.arange(h)[:, np.newaxis] < h // 2))
        sources = list(zip(ys.tolist(), xs.tolist()))
        if not sources:
            return
        random.shuffle(sources)

        sep = w // (n_rivers + 1)   # espacement horizontal minimal entre sources
        rivers_carved = 0
        used_sx: list[int] = []

        for sy, sx in sources:
            if any(abs(sx - ux) < sep for ux in used_sx):
                continue
            used_sx.append(sx)

            y, x = sy, sx
            path: list[tuple[int, int]] = []
            visited: set[tuple[int, int]] = set()
            reached_water = False

            for _ in range(h * 10):
                if not (0 <= y < h and 0 <= x < w):
                    break
                if grid[y, x] in (Biome.WATER, Biome.RIVER):
                    reached_water = True
                    break

                if (y, x) in visited:
                    # Bloqué dans une dépression : chercher n'importe quel voisin non visité
                    escape = None
                    for dy, dx in [(1, 0), (0, -1), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, -1), (-1, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                            escape = (ny, nx)
                            break
                    if escape:
                        y, x = escape
                        continue
                    break  # vraiment bloqué (toutes les tuiles adjacentes visitées)

                visited.add((y, x))
                path.append((y, x))

                # Descend vers le voisin non-visité le plus bas avec légère perturbation
                best_val = float('inf')
                best_ny, best_nx = min(y + 1, h - 1), x   # biais vers le sud par défaut
                for dy, dx in [(1, 0), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                        val = float(hmap[ny, nx]) + random.uniform(-0.04, 0.04)
                        if val < best_val:
                            best_val = val
                            best_ny, best_nx = ny, nx

                y, x = best_ny, best_nx

            if reached_water and len(path) >= 6:
                # Trace le chemin central
                river_mask = np.zeros((h, w), dtype=bool)
                for py, px in path:
                    river_mask[py, px] = True

                # Dilatation morphologique numpy : rayon 3 → rivières ~6 tuiles larges
                dilated = river_mask.copy()
                for _ in range(3):
                    dilated |= (np.roll(dilated,  1, axis=0) |
                                np.roll(dilated, -1, axis=0) |
                                np.roll(dilated,  1, axis=1) |
                                np.roll(dilated, -1, axis=1))

                # Applique seulement sur les tuiles non-montagne
                apply_mask = dilated & (grid != Biome.MOUNTAIN) & (grid != Biome.WATER)
                grid[apply_mask] = Biome.RIVER
                rivers_carved += 1

            if rivers_carved >= n_rivers:
                break

    def _init_food(self) -> np.ndarray:
        max_lookup = np.array([BIOME_FOOD_MAX[Biome(b)] for b in range(7)], dtype=np.float32)
        max_food   = max_lookup[self.biome_grid.astype(np.int8)]
        rand_f     = np.random.uniform(0.5, 1.0, (self.height, self.width)).astype(np.float32)
        return (max_food * rand_f).astype(np.float32)

    def _build_regen_grid(self) -> np.ndarray:
        regen_lookup = np.array([BIOME_FOOD_REGEN[Biome(b)] for b in range(7)], dtype=np.float32)
        return regen_lookup[self.biome_grid.astype(np.int8)]

    def _build_max_grid(self) -> np.ndarray:
        max_lookup = np.array([BIOME_FOOD_MAX[Biome(b)] for b in range(7)], dtype=np.float32)
        return max_lookup[self.biome_grid.astype(np.int8)]

    def regen_trees(self) -> list[tuple[int, int]]:
        """Régénère les arbres coupés. Retourne les positions qui franchissent le seuil (souche → arbre)."""
        was_stump = (self.tree_grid < TREE_STUMP_THRESHOLD) & self._forest_mask
        self.tree_grid[self._forest_mask] = np.minimum(
            100.0, self.tree_grid[self._forest_mask] + TREE_REGEN_RATE
        )
        regrown = was_stump & (self.tree_grid >= TREE_STUMP_THRESHOLD)
        ys, xs = np.where(regrown)
        return list(zip(xs.tolist(), ys.tolist()))

    def is_choppable(self, x: int, y: int) -> bool:
        return (self.is_valid(x, y)
                and self.biome_grid[y, x] == int(Biome.FOREST)
                and float(self.tree_grid[y, x]) >= TREE_STUMP_THRESHOLD)

    def chop_tree(self, x: int, y: int) -> int:
        """Abat l'arbre en (x, y). Retourne le bois obtenu (0 si non choppable)."""
        if not self.is_choppable(x, y):
            return 0
        self.tree_grid[y, x] = max(0.0, float(self.tree_grid[y, x]) - 50.0)
        self._chop_changes.append((x, y))
        return WOOD_PER_CHOP

    def get_stumps(self) -> list[list[int]]:
        """Retourne la liste [[x, y], …] des tuiles-souche actuelles."""
        ys, xs = np.where((self.tree_grid < TREE_STUMP_THRESHOLD) & self._forest_mask)
        return [[int(x), int(y)] for x, y in zip(xs.tolist(), ys.tolist())]

    def regen_stones(self) -> list[tuple[int, int]]:
        """Régénère les roches épuisées. Retourne les positions qui repassent le seuil."""
        was_depleted = (self.stone_grid < STONE_STUMP_THRESHOLD) & self._mountain_mask
        self.stone_grid[self._mountain_mask] = np.minimum(
            100.0, self.stone_grid[self._mountain_mask] + STONE_REGEN_RATE
        )
        regrown = was_depleted & (self.stone_grid >= STONE_STUMP_THRESHOLD)
        ys, xs = np.where(regrown)
        return list(zip(xs.tolist(), ys.tolist()))

    def is_mineable(self, x: int, y: int) -> bool:
        return (self.is_valid(x, y)
                and self.biome_grid[y, x] == int(Biome.MOUNTAIN)
                and float(self.stone_grid[y, x]) >= STONE_STUMP_THRESHOLD)

    def mine_stone(self, x: int, y: int) -> int:
        """Mine la roche en (x, y). Retourne la pierre obtenue (0 si non minable)."""
        if not self.is_mineable(x, y):
            return 0
        self.stone_grid[y, x] = max(0.0, float(self.stone_grid[y, x]) - 60.0)
        self._mine_changes.append((x, y))
        return STONE_PER_MINE

    def is_iron_mineable(self, x: int, y: int) -> bool:
        """Tuile montagne portant du fer non épuisé (bloc B)."""
        return (self.is_valid(x, y)
                and bool(self._iron_mask[y, x])
                and float(self.iron_grid[y, x]) >= IRON_STUMP_THRESHOLD)

    def mine_iron(self, x: int, y: int) -> int:
        """Mine le fer en (x, y). Retourne le fer obtenu (0 si non minable).
        Ne pousse RIEN dans _mine_changes : ce canal draine en rock_changes
        {"depleted": True} côté front → marquerait comme épuisée une ROCHE dont la
        pierre est intacte (gate-review B). Les gisements de fer n'ont pas encore
        de rendu dédié ; à ajouter avec leur propre canal le jour où ils en ont un."""
        if not self.is_iron_mineable(x, y):
            return 0
        self.iron_grid[y, x] = max(0.0, float(self.iron_grid[y, x]) - 60.0)
        return IRON_PER_MINE

    def regen_iron(self):
        """Régénère lentement les gisements de fer épuisés (invariant infini)."""
        self.iron_grid[self._iron_mask] = np.minimum(
            100.0, self.iron_grid[self._iron_mask] + IRON_REGEN_RATE)

    def is_gold_mineable(self, x: int, y: int) -> bool:
        """Filon d'or non épuisé (bloc C2)."""
        return (self.is_valid(x, y)
                and bool(self._gold_mask[y, x])
                and float(self.gold_grid[y, x]) >= GOLD_STUMP_THRESHOLD)

    def mine_gold(self, x: int, y: int) -> int:
        """Mine l'or en (x, y). NE pousse RIEN dans _mine_changes (même piège que
        mine_iron : ce canal marquerait une ROCHE intacte comme épuisée au front)."""
        if not self.is_gold_mineable(x, y):
            return 0
        self.gold_grid[y, x] = max(0.0, float(self.gold_grid[y, x]) - 60.0)
        return GOLD_PER_MINE

    def regen_gold(self):
        """Régénère (très lentement) les filons d'or épuisés (invariant infini)."""
        self.gold_grid[self._gold_mask] = np.minimum(
            100.0, self.gold_grid[self._gold_mask] + GOLD_REGEN_RATE)

    def get_depleted_rocks(self) -> list[list[int]]:
        """Retourne [[x, y], …] des tuiles-roche épuisées."""
        ys, xs = np.where((self.stone_grid < STONE_STUMP_THRESHOLD) & self._mountain_mask)
        return [[int(x), int(y)] for x, y in zip(xs.tolist(), ys.tolist())]

    def regen_food(self, season_mult: float = 1.0):
        """Régénère la nourriture sur toutes les tuiles (appelé chaque tick)."""
        self.food_grid += self._regen_grid * season_mult
        np.minimum(self.food_grid, self._max_grid, out=self.food_grid)

    def get_food(self, x: int, y: int) -> float:
        return float(self.food_grid[y, x])

    def consume_food(self, x: int, y: int, amount: float) -> float:
        """Consomme de la nourriture, retourne la quantité réellement consommée."""
        available = self.food_grid[y, x]
        consumed = min(available, amount)
        self.food_grid[y, x] -= consumed
        return float(consumed)

    def consume_fertility(self, x: int, y: int, amount: float = None) -> bool:
        """Réduit la fertilité d'une tuile GRASS (broutage ou piétinement).
        Retourne True si la tuile vient de passer à DIRT."""
        if not self.is_valid(x, y):
            return False
        if self.biome_grid[y, x] != int(Biome.GRASS):
            return False
        cost = amount if amount is not None else FERTILITY_CONSUME
        self.fertility_grid[y, x] = max(0.0, float(self.fertility_grid[y, x]) - cost)
        if self.fertility_grid[y, x] <= 0:
            self.biome_grid[y, x]   = int(Biome.DIRT)
            self.food_grid[y, x]    = 0.0
            self._max_grid[y, x]    = 0.0
            self._regen_grid[y, x]  = 0.0
            self._biome_changes.append({"x": x, "y": y, "biome": int(Biome.DIRT)})
            return True
        return False

    def regen_fertility(self, raining: bool = False, season: str = "spring") -> list[dict]:
        """Régénère les tuiles DIRT. Retourne les tuiles redevenues GRASS ce tick."""
        dirt_mask = (self.biome_grid == int(Biome.DIRT)) & self._orig_grass_mask
        if not dirt_mask.any():
            return []
        rate = FERTILITY_REGEN_BASE
        if raining:
            rate += FERTILITY_REGEN_RAIN
        if season == "spring":
            rate += FERTILITY_REGEN_SPRING
        self.fertility_grid[dirt_mask] = np.minimum(
            FERTILITY_MAX, self.fertility_grid[dirt_mask] + rate
        )
        recovered = dirt_mask & (self.fertility_grid >= FERTILITY_MAX)
        if not recovered.any():
            return []
        self.biome_grid[recovered]  = int(Biome.GRASS)
        self._max_grid[recovered]   = float(BIOME_FOOD_MAX[Biome.GRASS])
        self._regen_grid[recovered] = float(BIOME_FOOD_REGEN[Biome.GRASS])
        ys, xs = np.where(recovered)
        return [{"x": int(xi), "y": int(yi), "biome": int(Biome.GRASS)}
                for xi, yi in zip(xs.tolist(), ys.tolist())]

    def drain_biome_changes(self) -> list[dict]:
        """Retourne et vide la liste des changements GRASS→DIRT ce tick."""
        ch = self._biome_changes
        self._biome_changes = []
        return ch

    def _build_nearest_water(self) -> np.ndarray:
        """BFS depuis toutes les tuiles terrestres bord-eau.
        Retourne un flow field (H, W, 2) de dtype int16 : [x, y] du PREMIER PAS vers l'eau.
        Chaque case stocke son parent BFS (le voisin plus proche de l'eau d'un cran).
        Les tuiles bord-eau pointent vers elles-mêmes.  -1 = pas d'eau accessible."""
        h, w = self.height, self.width
        result = np.full((h, w, 2), -1, dtype=np.int16)
        visited = np.zeros((h, w), dtype=bool)
        dq: deque = deque()
        # Initialiser le BFS depuis les tuiles terrestres adjacentes à l'eau
        ys, xs = np.where(self._near_water)
        for x, y in zip(xs.tolist(), ys.tolist()):
            result[y, x] = [x, y]   # déjà au bord : premier pas = soi-même
            visited[y, x] = True
            dq.append((x, y))
        # Propagation 4-directions sur toutes les tuiles marchables terrestres
        dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
        while dq:
            cx, cy = dq.popleft()
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and self._walkable[ny, nx]:
                    result[ny, nx] = [cx, cy]  # premier pas depuis (nx,ny) = revenir vers (cx,cy)
                    visited[ny, nx] = True
                    dq.append((nx, ny))
        return result

    def is_walkable(self, x: int, y: int, aquatic: bool = False) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return bool(self._aquatic_walkable[y, x] if aquatic else self._walkable[y, x])

    def is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, x: int, y: int, walkable_only: bool = True):
        """Retourne les cases adjacentes (8 directions)."""
        dirs = [(-1,-1),( 0,-1),(1,-1),
                (-1, 0),        (1, 0),
                (-1, 1),( 0, 1),(1, 1)]
        result = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                if not walkable_only or self.is_walkable(nx, ny):
                    result.append((nx, ny))
        return result

    def ignite(self, x: int, y: int) -> bool:
        """Allume une tuile forêt avec arbre debout. Retourne True si allumée."""
        if not self.is_valid(x, y):
            return False
        if self.biome_grid[y, x] != int(Biome.FOREST):
            return False
        if float(self.tree_grid[y, x]) < TREE_STUMP_THRESHOLD:
            return False   # déjà une souche, rien à brûler
        if float(self.fire_grid[y, x]) > 0:
            return False   # déjà en feu
        self.fire_grid[y, x] = FIRE_INTENSITY_INIT
        return True

    def step_fire(self, season: str, raining: bool) -> list[dict]:
        """Avance d'un tick : brûle, propage, éteint sous pluie.
        Retourne les changements [{x, y, intensity}] pour le frontend."""
        changes: list[dict] = []
        burning = self.fire_grid > 0

        if not burning.any():
            return changes

        spread_prob = FIRE_SPREAD_PROB_DRY if season == "summer" else FIRE_SPREAD_PROB_WET

        # Propagation vers les 4 voisins (numpy roll)
        spread_mask = burning
        for shift, axis in [(1,0),(-1,0),(1,1),(-1,1)]:
            neighbor_burning = np.roll(spread_mask, shift, axis=axis)
            # Tuiles forêt avec arbre debout, pas encore en feu
            candidate = (neighbor_burning
                         & self._forest_mask
                         & (self.tree_grid >= TREE_STUMP_THRESHOLD)
                         & (~burning))
            if candidate.any():
                rand = np.random.random(candidate.shape).astype(np.float32)
                ignite = candidate & (rand < spread_prob)
                self.fire_grid[ignite] = FIRE_INTENSITY_INIT
                burning |= ignite

        # Bruler les arbres sous le feu (réduit santé arbre = souche quand épuisé)
        self.tree_grid[burning] = np.maximum(
            0.0, self.tree_grid[burning] - FIRE_BURN_RATE * 0.8
        )

        # Décroissance intensité feu
        damp = FIRE_BURN_RATE + (FIRE_RAIN_DAMP if raining else 0.0)
        self.fire_grid[burning] = np.maximum(0.0, self.fire_grid[burning] - damp)

        # Recalcule masque après extinction
        just_out   = burning & (self.fire_grid <= 0)
        still_burn = burning & (self.fire_grid > 0)

        ys_out, xs_out = np.where(just_out)
        for x, y in zip(xs_out.tolist(), ys_out.tolist()):
            changes.append({"x": int(x), "y": int(y), "fire": False})

        ys_on, xs_on = np.where(still_burn)
        for x, y in zip(xs_on.tolist(), ys_on.tolist()):
            changes.append({"x": int(x), "y": int(y),
                            "fire": True,
                            "i": round(float(self.fire_grid[y, x]), 1)})
        return changes

    def get_fires(self) -> list[list[int]]:
        """[[x, y], …] des tuiles actuellement en feu."""
        ys, xs = np.where(self.fire_grid > 0)
        return [[int(x), int(y)] for x, y in zip(xs.tolist(), ys.tolist())]

    def to_dict(self):
        return {
            "width":          self.width,
            "height":         self.height,
            "seed":           self.seed,
            "biomes":         self.biome_grid.tolist(),
            "colors":         {str(k.value): v for k, v in BIOME_COLORS.items()},
            "stumps":         self.get_stumps(),
            "depleted_rocks": self.get_depleted_rocks(),
            "fires":          self.get_fires(),
            "gold_veins":     [[int(x), int(y)] for y, x in zip(*np.where(self._gold_mask))],
        }

    # ── Sauvegarde / reprise ─────────────────────────────────────────────────
    # On ne sérialise QUE les grilles MUTABLES. Les masques d'invariants
    # (_forest/_mountain/_orig_grass/_walkable/_near_water/_nearest_water_tile)
    # ne dépendent que de biomes qui ne changent jamais (forêt/montagne/eau/
    # herbe-d'origine) → régénérés à l'identique par World(seed) au chargement.
    def to_state(self) -> dict:
        return {
            "seed": self.seed, "width": self.width, "height": self.height,
            "biome_grid":     _grid_to_b64(self.biome_grid),
            "food_grid":      _grid_to_b64(self.food_grid),
            "tree_grid":      _grid_to_b64(self.tree_grid),
            "stone_grid":     _grid_to_b64(self.stone_grid),
            "iron_grid":      _grid_to_b64(self.iron_grid),
            "gold_grid":      _grid_to_b64(self.gold_grid),
            "fertility_grid": _grid_to_b64(self.fertility_grid),
            "fire_grid":      _grid_to_b64(self.fire_grid),
        }

    @classmethod
    def from_state(cls, d: dict) -> "World":
        # Durcissement I2 (audit #16) : borner les dimensions AVANT d'allouer quoi que
        # ce soit → un save aux dims aberrantes lève au lieu de saturer la RAM.
        width, height = d["width"], d["height"]
        if not (isinstance(width, int) and isinstance(height, int)
                and 1 <= width <= _MAX_DIM and 1 <= height <= _MAX_DIM):
            raise ValueError(f"dimensions de save hors bornes: {width}x{height}")
        w = cls(width=width, height=height, seed=d["seed"])
        w.biome_grid     = _grid_from_b64(d["biome_grid"])
        w.food_grid      = _grid_from_b64(d["food_grid"])
        w.tree_grid      = _grid_from_b64(d["tree_grid"])
        w.stone_grid     = _grid_from_b64(d["stone_grid"])
        if d.get("iron_grid"):   # compat vieux saves : sinon garde le fer frais de __init__
            w.iron_grid  = _grid_from_b64(d["iron_grid"])
        if d.get("gold_grid"):   # compat vieux saves (C2)
            w.gold_grid  = _grid_from_b64(d["gold_grid"])
        w.fertility_grid = _grid_from_b64(d["fertility_grid"])
        w.fire_grid      = _grid_from_b64(d["fire_grid"])
        # _max_grid / _regen_grid dépendent du biome (0 sur DIRT) → recomputer
        # depuis le biome chargé pour rester cohérents avec l'état surpâturé.
        w._max_grid   = w._build_max_grid()
        w._regen_grid = w._build_regen_grid()
        # Buffers de changements intra-tick : repartent vides.
        w._biome_changes = []; w._chop_changes = []; w._mine_changes = []
        return w
