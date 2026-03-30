"""
BioSim — Moteur de simulation
World: grille, biomes, ressources
"""
import random
import numpy as np
from enum import IntEnum

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

BIOME_FOOD_REGEN = {
    Biome.WATER:    2.0,
    Biome.GRASS:    3,
    Biome.FOREST:   2,
    Biome.DESERT:   0.2,
    Biome.MOUNTAIN: 0.1,
    Biome.RIVER:    1.0,
    Biome.DIRT:     0.0,
}


TREE_STUMP_THRESHOLD  = 50.0   # en-dessous : souche visible
TREE_REGEN_RATE       = 0.08   # santé régénérée par tick (50 → ~625 ticks ≈ 2 saisons)
WOOD_PER_CHOP         = 2      # bois obtenu par abattage (sans outil)

# ── Feu de forêt ─────────────────────────────────────────────────────────────
FIRE_INTENSITY_INIT   = 80.0   # intensité initiale quand une tuile s'embrase
FIRE_BURN_RATE        = 1.2    # intensité perdue par tick (tuile brûle ~67 ticks)
FIRE_SPREAD_PROB_DRY  = 0.012  # probabilité de propagation/tick en été (forêt sèche)
FIRE_SPREAD_PROB_WET  = 0.003  # probabilité hors été
FIRE_RAIN_DAMP        = 4.0    # intensité retirée par tick sous la pluie

STONE_STUMP_THRESHOLD = 50.0   # en-dessous : roche épuisée (visible)
STONE_REGEN_RATE      = 0.021  # santé régénérée par tick (0→50 en ~2400 ticks ≈ 8 saisons)
STONE_PER_MINE        = 1      # pierre obtenue par minage (sans outil)

# ── Fertilité / pâturage ──────────────────────────────────────────────────────
FERTILITY_MAX          = 100.0   # fertilité pleine
FERTILITY_CONSUME      = 12.0    # fertilité retirée par action de broutage (manger)
FERTILITY_TRAMPLE      = 0.3     # fertilité retirée par tick de présence sur herbe
FERTILITY_REGEN_BASE   = 0.05    # regen/tick (~2000 ticks = 6-7 saisons)
FERTILITY_REGEN_RAIN   = 0.12    # bonus regen/tick sous la pluie
FERTILITY_REGEN_SPRING = 0.03    # bonus regen/tick au printemps


class World:
    def __init__(self, width: int = 80, height: int = 60, seed: int = None):
        self.width = width
        self.height = height
        self.seed = seed or random.randint(0, 999999)
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
        # Grille de feu (intensité 0–100, uniquement tuiles forêt avec arbre debout)
        self.fire_grid = np.zeros((self.height, self.width), dtype=np.float32)
        # Fertilité des tuiles herbe (GRASS → DIRT quand épuisée, régénère lentement)
        self._orig_grass_mask = (self.biome_grid == int(Biome.GRASS))
        self.fertility_grid = np.where(
            self._orig_grass_mask, FERTILITY_MAX, 0.0
        ).astype(np.float32)
        self._biome_changes: list = []   # changements GRASS↔DIRT ce tick

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

        grid[low  & (mmap < 0.62)] = Biome.GRASS
        grid[low  & (mmap >= 0.62)] = Biome.FOREST
        grid[mid  & (mmap < 0.55)] = Biome.DESERT
        grid[mid  & (mmap >= 0.55)] = Biome.FOREST
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
        return STONE_PER_MINE

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

    def is_walkable(self, x: int, y: int, aquatic: bool = False) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        b = self.biome_grid[y, x]
        if aquatic:
            return b in (Biome.WATER, Biome.RIVER)
        return b not in (Biome.WATER, Biome.RIVER)

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
        }
