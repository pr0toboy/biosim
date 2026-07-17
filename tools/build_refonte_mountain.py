#!/usr/bin/env python3
"""Génère les planches de la refonte nº26 (biome montagne minéral).

Sortie : static/refonte/cliff_rock.png, cliff_rock_high.png, gravel.png.
Reproduit fidèlement la recette validée de demo/build26.py :
- cliff_rock      : Cliff.png, verts du plateau → rampe de gris PRISE DANS LA PLANCHE
                    (141,140,117 → 185,188,170) pour les rangées 0-5 ; rampe SABLE
                    (183,150,95 → 217,212,140) pour le bloc grès des mesas (rangées 6-8).
                    Détecteur STRICT (g ≥ r·1.05 ET g > b·1.35) : les gris-verts de la
                    roche (194,204,179…) ne sont PAS happés.
- cliff_rock_high : idem + blanchi 22 % vers (238,240,244) = étage 2 « altitude ».
- gravel          : TexturedGrass.png désaturé vers le gris de la falaise
                    (v = lum·0.42 + 92 → (v+4, v+3, v−12)) — sol minéral sous le massif
                    et frange d'éboulis. Même géométrie 3×2 que l'herbe.

Idempotent : relancer régénère à l'identique. Aucune dépendance au moteur.
"""
import pathlib
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MWS = ROOT / "static" / "MiniWorldSprites"
OUT = ROOT / "static" / "refonte"
OUT.mkdir(parents=True, exist_ok=True)
S = 16

GREY_DARK, GREY_LIGHT = (141, 140, 117), (185, 188, 170)
SAND_DARK, SAND_LIGHT = (183, 150, 95), (217, 212, 140)


def load(rel):
    return Image.open(MWS / rel).convert("RGBA")


def is_green(r, g, b):
    """Vrai vert jaune du plateau — pas les gris-verts de la roche."""
    return g >= r * 1.05 and g > b * 1.35


def green_to_stone(im, bleach=0.0):
    im = im.copy()
    px = im.load()
    for yy in range(im.height):
        dark, light = (GREY_DARK, GREY_LIGHT) if yy < 6 * S else (SAND_DARK, SAND_LIGHT)
        for xx in range(im.width):
            r, g, b, a = px[xx, yy]
            if a == 0:
                continue
            if is_green(r, g, b):
                lum = 0.35 * r + 0.5 * g + 0.15 * b
                t = max(0.0, min(1.0, (lum - 140) / 60))
                r = int(dark[0] + (light[0] - dark[0]) * t)
                g = int(dark[1] + (light[1] - dark[1]) * t)
                b = int(dark[2] + (light[2] - dark[2]) * t)
            if bleach > 0:
                r = int(r + (238 - r) * bleach)
                g = int(g + (240 - g) * bleach)
                b = int(b + (244 - b) * bleach)
            px[xx, yy] = (r, g, b, a)
    return im


def desat_gravel(im):
    im = im.copy()
    px = im.load()
    for yy in range(im.height):
        for xx in range(im.width):
            r, g, b, a = px[xx, yy]
            if a == 0:
                continue
            lum = 0.35 * r + 0.5 * g + 0.15 * b
            v = lum * 0.42 + 92
            px[xx, yy] = (int(v + 4), int(v + 3), int(v - 12), a)
    return im


cliff = load("Ground/Cliff.png")
green_to_stone(cliff).save(OUT / "cliff_rock.png")
green_to_stone(cliff, 0.22).save(OUT / "cliff_rock_high.png")
desat_gravel(load("Ground/TexturedGrass.png")).save(OUT / "gravel.png")
print("cliff_rock / cliff_rock_high / gravel écrits dans static/refonte/")
