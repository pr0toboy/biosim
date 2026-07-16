#!/usr/bin/env python3
"""Génère les planches dérivées de la refonte visuelle nº7 (forêt WorldBox).

Sortie : static/refonte/*.png (planches composées/remappées consommées par le front).
Reproduit fidèlement la recette validée de demo/build7.py + build8.py :
- floor_dark      : sol forestier = TexturedGrass row1 cols0-2 assombri ×0.84 (3 tuiles)
- forest_floor    : litière (champignons/rondins/souche) composée (4 tuiles)
- trees_soft      : Trees.png, contour noir (23,23,23) remappé vert feuille (72,124,52)
                    — sinon le trait double d'épaisseur au dessin ×2
- forest_trees_soft: [feuillus×3 + pin] composé & remappé (le pin = index 3)
- things_g        : Basic_Grass_Biom_things remappé vers la palette verte du jeu
                    (2×2 = gros arbres feuillus de la canopée)

Idempotent : relancer régénère à l'identique. Aucune dépendance au moteur.
"""
import math
import pathlib
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MWS = ROOT / "static" / "MiniWorldSprites"
OUT = ROOT / "static" / "refonte"
OUT.mkdir(parents=True, exist_ok=True)
S = 16


def load(rel):
    return Image.open(MWS / rel).convert("RGBA")


def tile(im, c, r):
    return im.crop((c * S, r * S, (c + 1) * S, (r + 1) * S))


def compose(tiles):
    out = Image.new("RGBA", (len(tiles) * S, S), (0, 0, 0, 0))
    for i, t in enumerate(tiles):
        out.alpha_composite(t, (i * S, 0))
    return out


def remap(im, table):
    im = im.copy()
    px = im.load()
    for yy in range(im.height):
        for xx in range(im.width):
            r, g, b, a = px[xx, yy]
            if (r, g, b) in table:
                px[xx, yy] = table[(r, g, b)] + (a,)
    return im


def darken(im, k):
    im = im.copy()
    px = im.load()
    for yy in range(im.height):
        for xx in range(im.width):
            r, g, b, a = px[xx, yy]
            px[xx, yy] = (int(r * k), int(g * k), int(b * k), a)
    return im


grass = load("Ground/TexturedGrass.png")
things = load("Nature/Basic_Grass_Biom_things.png")
trees = load("Nature/Trees.png")
pines = load("Nature/PineTrees.png")

REMAP_G = {
    (95, 122, 121): (84, 148, 63),
    (110, 150, 124): (118, 202, 109),
    (151, 187, 142): (142, 205, 101),
    (174, 212, 153): (170, 219, 110),
    (194, 224, 154): (203, 227, 111),
}
SOFT = {(23, 23, 23): (72, 124, 52)}  # contour noir → vert feuille (dessin ×2)
# nº3 : sable texturé — les 3 tuiles mouchetées de TexturedGrass row1 remappées en teintes
# dérivées du sable actuel (Shore col0 = 231,213,147). Même grain, palette sable.
REMAP_SAND = {
    (177, 211, 84): (231, 213, 147),   # base herbe → base sable
    (203, 227, 111): (243, 229, 175),  # rehaut → crème clair
    (142, 205, 101): (203, 183, 118),  # brin → ocre discret
}

things_g = remap(things, REMAP_G)
trees_soft_img = remap(trees, SOFT)


# nº16 saisons : recoloration par luminance (automne or/rouge) + mouchetis de fleurs (printemps).
def _lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def season_ramp(im, ramp):
    """ramp = [(sombre), (moyen), (clair)] appliqué par luminance aux pixels verts (feuillage)."""
    im = im.copy()
    px = im.load()
    for yy in range(im.height):
        for xx in range(im.width):
            r, g, b, a = px[xx, yy]
            if a > 0 and g > r * 1.02 and g > b * 1.08:
                lum = _lum((r, g, b))
                px[xx, yy] = (ramp[0] if lum < 130 else (ramp[1] if lum < 185 else ramp[2])) + (a,)
    return im


def bloom(im, density=0.13):
    """Mouchetis de fleurs roses sur les zones vertes (hash déterministe, printemps)."""
    im = im.copy()
    px = im.load()
    for yy in range(im.height):
        for xx in range(im.width):
            r, g, b, a = px[xx, yy]
            if a > 0 and g > r * 1.02 and g > b * 1.08:
                h = abs(math.sin(xx * 127.1 + yy * 311.7) * 43758.5453) % 1
                if h < density:
                    px[xx, yy] = ((245, 190, 210) if h < density * 0.6 else (250, 225, 235)) + (a,)
    return im


GOLD = [(160, 96, 40), (224, 152, 60), (240, 200, 96)]
RED = [(140, 52, 36), (202, 84, 50), (232, 126, 74)]


def blend_tile(a, b):
    """Tuile de mélange 50/50 : masque speckle déterministe par pixel (nº6)."""
    out = a.copy()
    pa, pb, po = a.load(), b.load(), out.load()
    for yy in range(S):
        for xx in range(S):
            h = abs(math.sin(xx * 127.1 + yy * 311.7) * 43758.5453) % 1
            po[xx, yy] = pb[xx, yy] if h < 0.5 else pa[xx, yy]
    return out


# nº6 transitions : tuiles de mélange herbe↔sable [0] et herbe↔terre-battue [1].
_sand = remap(grass.crop((0, S, 3 * S, 2 * S)), REMAP_SAND)   # même sable que nº3
_dead = load("Ground/DeadGrass.png")
blends_img = compose([
    blend_tile(grass.crop((0, S, S, 2 * S)), _sand.crop((0, 0, S, S))),
    blend_tile(grass.crop((0, S, S, 2 * S)), tile(_dead, 0, 0)),
])

# nº4 : 2 paliers d'eau profonde = Shore col4 (66,172,175) assombri ×0.84 et ×0.68.
shore_img = load("Ground/Shore.png")
deep_water = Image.new("RGBA", (2 * S, S), (0, 0, 0, 0))
for _i, _k in enumerate((0.84, 0.68)):
    for _yy in range(S):
        for _xx in range(S):
            _r, _g, _b, _a = shore_img.getpixel((4 * S + _xx, _yy))
            deep_water.putpixel((_i * S + _xx, _yy), (int(_r * _k), int(_g * _k), int(_b * _k), _a))

outputs = {
    "floor_dark": darken(grass.crop((0, S, 3 * S, 2 * S)), 0.84),
    "forest_floor": compose([tile(things, 5, 0), tile(things, 7, 0),
                             tile(things, 5, 2), tile(trees, 0, 0)]),
    "trees_soft": remap(trees, SOFT),
    "forest_trees_soft": remap(compose([tile(trees, 1, 0), tile(trees, 2, 0),
                                        tile(trees, 3, 0), tile(pines, 1, 0)]), SOFT),
    "things_g": things_g,
    # nº5 plaine : litière = champignon / petit rocher / rondin / buisson à baies / buisson
    # (les 2 buissons reteintés vers la palette du jeu, cf. things_g).
    "plain_extra": compose([tile(things, 6, 0), tile(things, 7, 1), tile(things, 5, 2),
                            tile(things_g, 0, 3), tile(things_g, 1, 3)]),
    # nº3 désert : sable texturé (3 variantes), remplace le sable plat Shore col0.
    "sand_tex": remap(grass.crop((0, S, 3 * S, 2 * S)), REMAP_SAND),
    # nº4 océan : 2 paliers d'eau profonde = Shore col4 assombri ×0.84 et ×0.68.
    "deep_water": deep_water,
    # nº6 transitions : tuiles de mélange 50/50 (herbe↔sable [0], herbe↔terre-battue [1]).
    "blends": blends_img,
    # nº12 hiver : feuillus NUS (branches enneigées) — WinterDeadTrees, contour noir adouci
    # (23,23,23 → brun sombre) pour éviter le trait doublé au dessin ×2. 4 cols (on prend 2-3).
    "winter_dead_soft": remap(load("Nature/WinterDeadTrees.png"), {(23, 23, 23): (74, 64, 56)}),
    # nº16 saisons : feuillages d'automne (or/rouge par luminance) + floraison de printemps.
    "trees_autumn1": season_ramp(trees_soft_img, GOLD),
    "trees_autumn2": season_ramp(trees_soft_img, RED),
    "trees_bloom": bloom(trees_soft_img),
    "things_autumn1": season_ramp(things_g, GOLD),
    "things_autumn2": season_ramp(things_g, RED),
    "things_bloom": bloom(things_g),
}

for name, im in outputs.items():
    p = OUT / f"{name}.png"
    im.save(p, "PNG")
    print(f"  {p.relative_to(ROOT)}  {im.width}x{im.height}")
print(f"{len(outputs)} planches écrites dans {OUT.relative_to(ROOT)}/")
