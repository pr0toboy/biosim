"""Garde déterministe AUTOMATISÉE (invariant #2) : hash de toute la séquence step()
d'un run seedé, COMPARÉ à un golden versionné. Échoue (exit 1) si le hash diffère.

DEUX scénarios :
  - BASE (défaut) : 140x100, seed 424242, 500 ticks, Âge du Bois. Golden `GOLDEN`.
  - CIV (--civ)   : 140x100, seed 42, 1000 ticks, science pré-amorcée à l'Âge Acier
    → exerce forge / marché / église / or / commerce / pèlerinage EN DÉTERMINISME
    (les ~2000 lignes de moteur qu'un run Âge-Bois n'atteint jamais). Golden `GOLDEN_CIV`.
    Le pré-amorçage est une simple affectation de `clan.science` (RNG-neutre) après
    populate() ; les 8 systèmes avancés tirent tous avant le tick 960 (mesuré).

Lancé en processus frais → _next_id repart de 0, random/np.random seedés par World.
Le smoke exécute CHAQUE scénario en SOUS-PROCESSUS (test_determinism_golden /
test_determinism_civ_golden) pour préserver cette garantie « process frais » : sinon
les tests qui tournent avant polluent le compteur d'id global et le hash diverge à tort.

Ce qui est hashé = le payload WIRE de step() (to_dict, trié). C'est LOSSY (audit #54 :
soudé au format d'affichage) MAIS c'est la base historique du golden d39fb432 et de toute
la lignée d'imputations — la changer imposerait un regold. Hasher l'état interne complet
est une évolution SÉPARÉE (hors J1/J2, qui exigent goldens stables).

REGOLD (quand un changement moteur modifie VOLONTAIREMENT le comportement) :
  1. python3 tests/determinism_guard.py [--civ] --regold   (imprime le hash, ne compare pas)
  2. Rejouer une 2ᵉ fois EN PROCESSUS FRAIS → EXIGER un hash IDENTIQUE
     (sinon non-déterminisme = bug, pas regold)
  3. Remplacer GOLDEN / GOLDEN_CIV ci-dessous + noter l'évolution dans NEXT-STEPS.

⚠ Les budgets (TICKS/CIV_TICKS/PROD_TICKS) sont des DURÉES : ils suivent TIME_SCALE.
  Les figer en ticks bruts ampute la couverture EN SILENCE, sans faire rougir le test
  (mesuré au rescale ×6 : à 1000t bruts le scénario CIV ne voyait plus AUCUN système
  avancé — ni cloche ni troc — alors qu'il existe précisément pour les exercer).
"""
import sys, json, hashlib, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.world import World, TIME_SCALE
from engine.simulation import Simulation

# ── Scénario BASE (Âge du Bois) ──────────────────────────────────────────────
SEED = 424242
TICKS = 500 * TIME_SCALE   # DURÉE → suit TIME_SCALE (= mêmes 500t d'ancien barème)
WIDTH = 140
HEIGHT = 100
# Golden versionné — invariant #2. Historique complet dans ~/torterra/NEXT-STEPS.md.
# Regold bloc PRÉDATION (2026-07-15) : ex-d39fb432 (pop 600) → prédation réelle
# (boar hh14/em30/cap140, ciblage réparti 4 proies, I3 rebond). pop 706.
GOLDEN = "9bfcf6cf086da32d762a505b36704eee826c482e6aa29b31489300def382bad4"  # regold P2 POLITIQUE 2026-07-20 (ex-5c14d2d0 ; relations inter-clans : une guerre déclarée en 500t re-cible via la relation (min) au lieu de la pop max + -60 → rival + wire rivals/chief_trait). pop 821 INCHANGÉE (P2 change les LABELS géopolitiques, pas l'écologie au BASE). Imputation : RELATIONS_OFF=1 → 5c14d2d0 (P2 off, = P2.1) ; JOBS_OFF → 5f244402 ; SOCIETY_OFF → 4ccfdd60 ; POLITICS_OFF → 5c14d2d0.

# ── Scénario CIV (Âge Acier, systèmes avancés) ───────────────────────────────
CIV_SEED = 42
CIV_TICKS = 1000 * TIME_SCALE   # DURÉE → × : à 1000t bruts PLUS AUCUN système avancé
                                # ne tire (cloche 1800 > 1000) → golden vidé de son sens
CIV_SCIENCE = 6500   # > seuil Acier (6000) → tous les clans montent à Acier au 1er tick
GOLDEN_CIV = "c294ac4b31aa8065e44aa71d381de286b8957ecb9fc3363b74a3bdfaf471ee6e"  # regold P2 POLITIQUE 2026-07-20 (ex-c08b6a14 ; personnalité du chef + relations inter-clans : guerres re-ciblées par rancune, alliés jamais ciblés, wire chief_trait/allies/rivals). Imputation RELATIONS_OFF=1 → c08b6a14 exact (P2.1) ; POLITICS_OFF → a85d306b (S2c) ; JOBS_OFF/SOCIETY_OFF/WARBEH inchangés. pop 1155 → 1196. — Historique : c08b6a14 = P2.1 succession ; a85d306b = S2c warriors+plancher.

# ── Scénario PROD (--prod) : gabarit RÉELLEMENT déployé, 220x160 ──────────────
# À LA DEMANDE / nightly (trop lent pour le smoke rapide). Ferme l'angle mort
# « le golden garde le 140x100, pas le 220x160 servi » (tore np.roll du feu, coûts
# de scan, entity_grid tous différents à ce gabarit). Non câblé au runner test_smoke.
PROD_SEED = 424242
PROD_TICKS = 500 * TIME_SCALE   # DURÉE → suit TIME_SCALE
PROD_WIDTH = 220
PROD_HEIGHT = 160
GOLDEN_PROD = "e51f0ad07e712b92e935fbfd1f2ed779e345e64667d3215a86d8c693ea8624c6"  # regold P2 POLITIQUE 2026-07-20 (ex-cb0f4b53 ; comme BASE, une guerre déclarée en 500t re-cible par relation + wire). Imputation RELATIONS_OFF=1 → cb0f4b53 exact. pop 787 inchangée (labels géopolitiques). — INCHANGÉ par S2c/P2.1 avant P2.


def _run(seed, ticks, width=WIDTH, height=HEIGHT, science_boost=None):
    """(hexdigest, pop finale) d'un run seedé. À appeler en PROCESSUS FRAIS
    (_next_id doit repartir de 0 pour reproduire le golden). Si science_boost est
    fourni, on l'affecte à chaque clan APRÈS populate() (mutation RNG-neutre)."""
    world = World(width=width, height=height, seed=seed)
    sim = Simulation(world)
    sim.populate()
    if science_boost is not None:
        for c in sim.clans:
            c.science = float(science_boost)
    h = hashlib.sha256()
    for _ in range(ticks):
        data = sim.step()
        # sérialisation stable (tri des clés) de tout ce que step produit
        h.update(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest(), len(sim.entities)


def run_hash():
    """Scénario BASE (rétro-compat : ancien nom public)."""
    return _run(SEED, TICKS)


def run_civ_hash():
    """Scénario CIV (Âge Acier)."""
    return _run(CIV_SEED, CIV_TICKS, science_boost=CIV_SCIENCE)


def run_prod_hash():
    """Scénario PROD (gabarit 220x160 déployé)."""
    return _run(PROD_SEED, PROD_TICKS, width=PROD_WIDTH, height=PROD_HEIGHT)


if __name__ == "__main__":
    if "--civ" in sys.argv:
        label, digest_pop, golden = "CIV", run_civ_hash(), GOLDEN_CIV
    elif "--prod" in sys.argv:
        label, digest_pop, golden = "PROD", run_prod_hash(), GOLDEN_PROD
    else:
        label, digest_pop, golden = "BASE", run_hash(), GOLDEN
    digest, pop = digest_pop
    if "--regold" in sys.argv:
        print(f"{digest} (pop finale {pop}) [{label} regold — non comparé]")
        sys.exit(0)
    ok = digest == golden
    print(f"{digest} (pop finale {pop}) [{label} {'PASS' if ok else 'FAIL'}]")
    if not ok:
        print(f"  ATTENDU {golden}", file=sys.stderr)
        print("  → dérive déterministe (invariant #2 violé) OU regold volontaire "
              "(cf. procédure en tête de fichier)", file=sys.stderr)
    sys.exit(0 if ok else 1)
