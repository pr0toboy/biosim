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
GOLDEN = "6159136cc76d0079537b6eb86dc408a238888c3d1e5c753637d002d94b6cdf66"  # regold P3 GUERRE 2.0 2026-07-20 (ex-73fd2b25 ; issues de guerre : conquête/tribut/paix blanche + riposte + aide d'allié + mariages). pop 821 inchangée. Imputation : WAR2_OFF=1 → 73fd2b25 (P2) ; RELATIONS_OFF → 5c14d2d0 ; JOBS_OFF → 5f244402 ; SOCIETY_OFF → 4ccfdd60. — Historique : 73fd2b25 = P2 calibration ; 9bfcf6cf = P2 pré-calibration.

# ── Scénario CIV (Âge Acier, systèmes avancés) ───────────────────────────────
CIV_SEED = 42
CIV_TICKS = 1000 * TIME_SCALE   # DURÉE → × : à 1000t bruts PLUS AUCUN système avancé
                                # ne tire (cloche 1800 > 1000) → golden vidé de son sens
CIV_SCIENCE = 6500   # > seuil Acier (6000) → tous les clans montent à Acier au 1er tick
GOLDEN_CIV = "c7f1a86086b06ef85fcb6123ae08ef72a9d7bb7bc9869797485e89d1161d41e1"  # regold P3 GUERRE 2.0 2026-07-20 (ex-acb48bcb ; conquêtes/tributs + mariages transfèrent des membres → démographie change). Imputation WAR2_OFF=1 → acb48bcb (P2) ; RELATIONS_OFF → c08b6a14 (P2.1) ; POLITICS_OFF → a85d306b (S2c). pop 1201 → 1175. — Historique : acb48bcb = P2 calibration ; c08b6a14 = P2.1 ; a85d306b = S2c.

# ── Scénario PROD (--prod) : gabarit RÉELLEMENT déployé, 220x160 ──────────────
# À LA DEMANDE / nightly (trop lent pour le smoke rapide). Ferme l'angle mort
# « le golden garde le 140x100, pas le 220x160 servi » (tore np.roll du feu, coûts
# de scan, entity_grid tous différents à ce gabarit). Non câblé au runner test_smoke.
PROD_SEED = 424242
PROD_TICKS = 500 * TIME_SCALE   # DURÉE → suit TIME_SCALE
PROD_WIDTH = 220
PROD_HEIGHT = 160
GOLDEN_PROD = "5ffb9432c51941559de9c852f9a3de959850c3206430d26120537209477381f4"  # regold P3 GUERRE 2.0 2026-07-20 (ex-e51f0ad0 ; issues de guerre au gabarit prod). pop 787 inchangée. Imputation WAR2_OFF=1 → e51f0ad0 (P2) ; RELATIONS_OFF → cb0f4b53. — Historique : e51f0ad0 = P2 (calibration hash-neutre au PROD) ; cb0f4b53 = P1.


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
