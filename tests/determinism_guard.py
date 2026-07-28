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
GOLDEN = "4954f834c534b2a6e33da0c180a658fed86aaf2cdb4b809e139d8eca239fffaa"  # regold P5 E1 CULTES 2026-07-21 (ex-0e39bd4e ; wire nom du culte au BASE, pas de conversion/schisme à l'Âge Bois → pop 849 inchangée). Imputation CULTURE_OFF=1 → 0e39bd4e (HEAD pré-P5) EXACT ; CULTS_OFF idem. P5 E2 FÊTE : INCHANGÉ (3000t < 1er automne 3600t → fête dormante) ; FEAST_OFF idem. P5 E3 MONUMENTS : INCHANGÉ (Âge Bois, pas d'Acier ni d'église → aucun monument) ; MONUMENT_OFF idem. P5 E4 HÉROS : INCHANGÉ (Âge Bois, pas de héros nommé à 3000t) ; HEROES_OFF idem. P6 F1 OR-MONNAIE : INCHANGÉ (Bois, pas d'église/marché/or à 3000t) ; MONEY_OFF/ECON_OFF idem. REGOLD P6 F2 RICHESSE+ENVIE 2026-07-28 (ex-6f348104) : wire wealth sur tous les clans + l'ENVIE réordonne le choix de cible de guerre (pop 849→839 = comportement réellement modifié). ECON_OFF → 6f348104 (pré-P6) EXACT ; ENVY_OFF → 004a7ded (wire seul, pop 849). P6 F3 SENTIERS : INCHANGÉ (hors payload). REGOLD P6 F4 GRENIERS 2026-07-28 (ex-07a1aedf) : moulins L2 (cap pains ×3) + famine par les réserves (pop 839→864). GRANARY_OFF → 07a1aedf EXACT.

# ── Scénario CIV (Âge Acier, systèmes avancés) ───────────────────────────────
CIV_SEED = 42
CIV_TICKS = 1000 * TIME_SCALE   # DURÉE → × : à 1000t bruts PLUS AUCUN système avancé
                                # ne tire (cloche 1800 > 1000) → golden vidé de son sens
CIV_SCIENCE = 6500   # > seuil Acier (6000) → tous les clans montent à Acier au 1er tick
GOLDEN_CIV = "61988103c6dfeb35a78f0c9bb863b235a3ecb94fa416cb05d8404dd6c71f8795"  # regold P6 F1 OR-MONNAIE 2026-07-23 (ex-2477001a P5 ; l'or CIRCULE dans le CIV Acier — injection demande-marché + paiement en or → flux caravane modifié, pop 1150→1153). Imputation MONEY_OFF → 2477001a (P5) EXACT ; HEROES_OFF → 114dd533 (E3) ; CULTURE_OFF → 4b0c9df8 (pré-P5) ; ECON_OFF → 2477001a (pré-P6). REGOLD P6 F2 RICHESSE+ENVIE 2026-07-28 (ex-8daee48e) : wire wealth + envie (pop 1153→1175). ECON_OFF → 2477001a EXACT ; ENVY_OFF → 96ac30fa (wire seul). P6 F3 SENTIERS : INCHANGÉ (hors payload). REGOLD P6 F4 GRENIERS 2026-07-28 (ex-31f00c5b) : greniers + famine par les réserves (pop 1175→1215). GRANARY_OFF → 31f00c5b EXACT.

# ── Scénario PROD (--prod) : gabarit RÉELLEMENT déployé, 220x160 ──────────────
# À LA DEMANDE / nightly (trop lent pour le smoke rapide). Ferme l'angle mort
# « le golden garde le 140x100, pas le 220x160 servi » (tore np.roll du feu, coûts
# de scan, entity_grid tous différents à ce gabarit). Non câblé au runner test_smoke.
PROD_SEED = 424242
PROD_TICKS = 500 * TIME_SCALE   # DURÉE → suit TIME_SCALE
PROD_WIDTH = 220
PROD_HEIGHT = 160
GOLDEN_PROD = "8eff7ae1e5963e7d4d242dacd02a951efb69e10c534ba4c57d8f62715e9aa340"  # regold P5 E1 CULTES 2026-07-21 (ex-d9294cd1). Imputation CULTURE_OFF → d9294cd1 EXACT. P5 E2 FÊTE : INCHANGÉ (3000t < automne 3600t) ; FEAST_OFF idem. P5 E3 MONUMENTS : INCHANGÉ (pas d'Acier @3000t) ; MONUMENT_OFF idem. P5 E4 HÉROS : INCHANGÉ (pas de héros @3000t) ; HEROES_OFF idem. P6 F1 OR-MONNAIE : INCHANGÉ (pas d'or @3000t) ; MONEY_OFF idem. REGOLD P6 F2 2026-07-28 (ex-3d70b047) : wire wealth (pop 757 inchangée). ECON_OFF → 3d70b047 EXACT. P6 F3 SENTIERS + F4 GRENIERS : INCHANGÉ (pas de moulin L2 @3000t) ; TRAILS_OFF/GRANARY_OFF idem.


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
