"""
Tests smoke BioSim — filet de sécurité (aucune dépendance à pytest).

Lancer :  python3 tests/test_smoke.py   (depuis le dossier biosim/)
Ou :      python3 -m pytest tests/test_smoke.py

Couvre :
- test_smoke_runs : N ticks seedés sans exception + invariants (positions in-grid,
  cap MAX_PER_SPECIES par espèce, clés attendues de to_dict).
- test_deposit_no_crash_when_houses_full : régression du crash H1
  (UnboundLocalError: nearest) — un mineur porteur de pierre alors que toutes
  les maisons du clan sont pleines de bois ne doit PAS faire planter tick_entity.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.world import World
from engine.simulation import (
    Simulation, Building, tick_entity, MAX_PER_SPECIES, MAX_WOOD_PER_HOUSE,
    MAX_STONE_CARRY, _dist,
)
from engine.entities import EntityType, spawn, Sex

REQUIRED_KEYS = {"id", "t", "x", "y", "s", "h", "age", "sex", "th", "tr"}


def test_smoke_runs(ticks: int = 400, seed: int = 12345):
    """N ticks sans exception + invariants vérifiés périodiquement."""
    world = World(width=90, height=70, seed=seed)
    sim = Simulation(world)
    sim.populate()

    for t in range(ticks):
        sim.step()  # ne swallow rien : une exception fait échouer le test
        if t % 50 == 0:
            # positions in-grid
            for e in sim.entities:
                assert 0 <= e.x < world.width, f"x hors grille: {e.x}"
                assert 0 <= e.y < world.height, f"y hors grille: {e.y}"
            # cap par espèce
            counts = {}
            for e in sim.entities:
                counts[e.etype] = counts.get(e.etype, 0) + 1
            for et, c in counts.items():
                assert c <= MAX_PER_SPECIES, f"{et} dépasse le cap: {c} > {MAX_PER_SPECIES}"
            # clés to_dict
            if sim.entities:
                d = sim.entities[0].to_dict()
                assert REQUIRED_KEYS <= set(d.keys()), f"clés manquantes: {REQUIRED_KEYS - set(d.keys())}"
    # full_state sérialisable de bout en bout
    fs = sim.full_state()
    assert "entities" in fs and "world" in fs
    print(f"  test_smoke_runs OK ({ticks} ticks, {len(sim.entities)} entités finales, seed={seed})")


def _find_land_tile(world):
    for y in range(world.height):
        for x in range(world.width):
            if world.is_walkable(x, y, aquatic=False):
                return x, y
    raise RuntimeError("aucune tuile terrestre")


def test_deposit_no_crash_when_houses_full():
    """Régression H1 : mineur avec pierre + maisons pleines de bois → pas de crash,
    et la pierre est bien déposée."""
    world = World(width=60, height=45, seed=7)
    lx, ly = _find_land_tile(world)

    human = spawn(EntityType.HUMAN, lx, ly, Sex.MALE)
    human.clan_id = 1
    human.hunger = 10.0   # sous tous les seuils → tombe jusqu'au bloc de dépôt
    human.thirst = 10.0
    human.wood = 0
    human.stone = MAX_STONE_CARRY   # porte de la pierre

    # une maison du clan, PLEINE de bois, placée loin (dist >= 1.5) pour viser
    # la branche "portée pleine → aller déposer" (la ligne du crash)
    house = Building(id=1, clan_id=1, x=min(lx + 10, world.width - 1),
                     y=ly, btype="house", wood=MAX_WOOD_PER_HOUSE, stone=0)
    clan_bldg = {1: {"house": [house]}}

    births, events = [], []
    # Ne doit PAS lever UnboundLocalError
    tick_entity(human, world, [human], births, events, tick=1,
                clan_bldg=clan_bldg, species_counts={"human": 5})

    # La pierre doit avoir été déposée si adjacent, sinon l'entité cible la maison.
    # Dans les 2 cas : pas de crash. Vérifions aussi le cas adjacent (dépôt pierre).
    human2 = spawn(EntityType.HUMAN, lx, ly, Sex.MALE)
    human2.clan_id = 1
    human2.hunger = 10.0
    human2.thirst = 10.0
    human2.wood = 0
    human2.stone = MAX_STONE_CARRY
    house_adj = Building(id=2, clan_id=1, x=lx, y=ly, btype="house",
                         wood=MAX_WOOD_PER_HOUSE, stone=0)
    tick_entity(human2, world, [human2], [], [], tick=1,
                clan_bldg={1: {"house": [house_adj]}}, species_counts={"human": 5})
    assert house_adj.stone == MAX_STONE_CARRY, (
        f"la pierre n'a pas été déposée (maison pleine de bois): {house_adj.stone}")
    assert human2.stone == 0, f"le mineur porte encore de la pierre: {human2.stone}"
    print("  test_deposit_no_crash_when_houses_full OK (pierre déposée, pas de crash)")


def _find_water_tile(world):
    for y in range(world.height):
        for x in range(world.width):
            if not world.is_walkable(x, y, aquatic=False):
                return x, y
    raise RuntimeError("aucune tuile d'eau")


def test_water_stranded_entity_rescued():
    """Régression bug de l'eau : une entité terrestre posée sur une tuile d'eau
    doit être renvoyée sur la terre ferme en UN tick (filet _teleport_to_nearest_
    walkable à anneaux complets). Auparavant elle pouvait rester coincée à vie."""
    world = World(width=90, height=70, seed=12345)
    wx, wy = _find_water_tile(world)

    sheep = spawn(EntityType.SHEEP, wx + 0.5, wy + 0.5)
    sheep.hunger = 20.0
    sheep.thirst = 10.0
    assert not world.is_walkable(sheep.ix, sheep.iy, False), "départ censé être sur l'eau"

    tick_entity(sheep, world, [sheep], [], [], tick=1, species_counts={"sheep": 10})

    assert sheep.alive, "le mouton ne devrait pas être mort en 1 tick"
    assert world.is_walkable(sheep.ix, sheep.iy, False), (
        f"mouton toujours sur une tuile non-terrestre après 1 tick: ({sheep.ix},{sheep.iy})")
    print("  test_water_stranded_entity_rescued OK (échoué secouru en 1 tick)")


def test_preservation_live_counter():
    """Régression M1 : à 30 proies, 2 prédateurs affamés dans le même tick ne
    doivent PAS tuer 2 proies (le compteur vivant passe sous 30 après le 1er kill
    → le 2e prédateur préserve l'espèce)."""
    world = World(width=60, height=45, seed=3)
    lx, ly = _find_land_tile(world)

    sheep_a = spawn(EntityType.SHEEP, lx, ly)
    sheep_b = spawn(EntityType.SHEEP, lx, ly)
    boar1 = spawn(EntityType.BOAR, lx, ly)
    boar2 = spawn(EntityType.BOAR, lx, ly)
    for b in (boar1, boar2):
        b.hunger = 80.0   # > 55 → chasse
        b.thirst = 0.0
    all_e = [boar1, boar2, sheep_a, sheep_b]
    species_counts = {"sheep": 30, "boar": 2}

    tick_entity(boar1, world, all_e, [], [], tick=1, species_counts=species_counts)
    tick_entity(boar2, world, all_e, [], [], tick=1, species_counts=species_counts)

    dead = sum(1 for s in (sheep_a, sheep_b) if not s.alive)
    assert dead == 1, f"préservation d'espèce contournée : {dead} moutons tués (attendu 1)"
    assert species_counts["sheep"] == 29, f"compteur vivant faux: {species_counts['sheep']}"
    print("  test_preservation_live_counter OK (1 seul mouton tué, seuil respecté)")


def test_c1bis_toolless_human_crafts_without_depositing():
    """Régression C1bis : un humain sans outil, adjacent à une maison du clan qui a
    du bois en stock, DOIT fabriquer une pioche/hache même s'il ne transporte rien
    à déposer. Sans le découplage craft↔dépôt, la chaîne d'outils se fige quand le
    bois du clan est capé (plus personne ne dépose → plus personne ne fabrique)."""
    world = World(width=60, height=45, seed=7)
    lx, ly = _find_land_tile(world)
    h = spawn(EntityType.HUMAN, lx, ly, Sex.MALE)
    h.clan_id = 1
    h.hunger = 10.0
    h.thirst = 10.0
    h.pick = None
    h.tool = None
    h.wood = 0
    h.stone = 0
    house = Building(id=1, clan_id=1, x=lx, y=ly, btype="house", wood=50, stone=0)
    tick_entity(h, world, [h], [], [], tick=1,
                clan_bldg={1: {"house": [house]}}, species_counts={"human": 5})
    assert (h.pick is not None or h.tool is not None), (
        "l'humain sans outil n'a rien fabriqué alors qu'une maison du clan avait le bois")
    assert house.wood < 50, "aucune ressource consommée → pas de fabrication"
    print(f"  test_c1bis_toolless_human_crafts_without_depositing OK "
          f"(pick={h.pick}, tool={h.tool}, bois maison {house.wood})")


def test_c2_hungry_harvester_eats_not_feeds_mill():
    """Régression C2 : un récolteur AFFAMÉ mange sa part au lieu de tout livrer au
    moulin (avant, le blé partait intégralement au moulin et le récolteur mourait
    de faim = trou noir alimentaire). Un récolteur RASSASIÉ, lui, livre le surplus."""
    world = World(width=60, height=45, seed=7)
    lx, ly = _find_land_tile(world)
    world.food_grid[:] = 0.0   # pas de nourriture sauvage → force le passage par la récolte

    def _setup(hunger):
        h = spawn(EntityType.HUMAN, lx, ly, Sex.MALE)  # MALE → saute la repro
        h.clan_id = 1; h.hunger = hunger; h.thirst = 10.0
        h.wood = 0; h.stone = 0; h.pick = "stone_pick"; h.tool = "axe"  # pas de craft/dépôt
        field = Building(id=1, clan_id=1, x=lx, y=ly, btype="wheatfield")
        field.stage = 4  # mûr, adjacent
        mill = Building(id=2, clan_id=1, x=lx, y=ly, btype="mill")
        mill.wheat = 0; mill.bread = 0  # capacité libre, pas de pain (sinon il mange le pain)
        cb = {1: {"wheatfield": [field], "mill": [mill]}}  # 1 champ → pas de build-moulin ; pas de maison → pas de build-champ
        return h, field, mill, cb

    # Affamé → mange, ne livre rien
    h, field, mill, cb = _setup(50.0)
    tick_entity(h, world, [h], [], [], tick=1, clan_bldg=cb, species_counts={"human": 5})
    assert field.stage == 1, f"le champ n'a pas été récolté (bloc 4.3a non atteint, stage={field.stage})"
    assert mill.wheat == 0, f"le récolteur affamé a quand même nourri le moulin: wheat={mill.wheat}"
    assert h.hunger < 50.0, f"le récolteur affamé n'a pas mangé: hunger={h.hunger}"

    # Rassasié → livre le surplus au moulin, ne mange pas
    h2, field2, mill2, cb2 = _setup(10.0)
    tick_entity(h2, world, [h2], [], [], tick=1, clan_bldg=cb2, species_counts={"human": 5})
    assert field2.stage == 1, f"le champ n'a pas été récolté (rassasié), stage={field2.stage}"
    assert mill2.wheat == 1, f"le récolteur rassasié n'a pas livré au moulin: wheat={mill2.wheat}"
    print(f"  test_c2_hungry_harvester_eats_not_feeds_mill OK "
          f"(affamé: hunger 50→{h.hunger:.0f}, moulin {mill.wheat} ; rassasié: moulin {mill2.wheat})")


def test_a1_clan_gains_science_and_ages_up():
    """Bloc A1 (âges/tech) : un clan accumule de la science (bâtiments + pop) et
    franchit un âge au seuil, avec un événement clan_age_up visible."""
    from engine.simulation import Clan, AGE_SCIENCE_THRESHOLDS, AGE_NAMES
    world = World(width=60, height=45, seed=7)
    sim = Simulation(world)
    lx, ly = _find_land_tile(world)
    h = spawn(EntityType.HUMAN, lx, ly, Sex.MALE)
    h.clan_id = 0
    sim.entities = [h]
    sim.clans = [Clan(id=0, cx=float(lx), cy=float(ly), color="#fff", chief_id=h.id)]
    sim.buildings = [Building(id=1, clan_id=0, x=lx, y=ly, btype="house")]

    # (1) la science augmente (1 bâtiment + 1 humain vivant)
    sim.step()
    assert sim.clans[0].science > 0, "la science du clan n'augmente pas"
    assert sim.clans[0].age == 0, "le clan ne devrait pas encore avoir changé d'âge"

    # (2) au franchissement du seuil → âge +1 + événement clan_age_up
    sim.clans[0].science = AGE_SCIENCE_THRESHOLDS[1] - 0.01
    data = sim.step()
    assert sim.clans[0].age == 1, f"le clan n'est pas passé à l'Âge 1 (age={sim.clans[0].age})"
    assert any(ev.get("type") == "clan_age_up" and ev.get("age_name") == AGE_NAMES[1]
               for ev in data["events"]), "événement clan_age_up manquant"
    print(f"  test_a1_clan_gains_science_and_ages_up OK "
          f"(science {sim.clans[0].science:.1f}, Âge → {AGE_NAMES[sim.clans[0].age]})")


def test_b_forge_upgrades_stone_tools_to_iron():
    """Bloc B : un humain avec des outils PIERRE, adjacent à la forge du clan qui a
    du fer en stock, les met à niveau vers le FER (découplé du transport de fer,
    comme C1bis). Et un mineur adjacent à la forge y dépose son fer porté."""
    world = World(width=60, height=45, seed=7)
    lx, ly = _find_land_tile(world)

    # (1) upgrade outil pierre → fer à la forge
    h = spawn(EntityType.HUMAN, lx, ly, Sex.MALE)
    h.clan_id = 1; h.hunger = 10.0; h.thirst = 10.0
    h.pick = "stone_pick"; h.tool = "stone_axe"; h.wood = 0; h.stone = 0; h.iron = 0
    forge = Building(id=1, clan_id=1, x=lx, y=ly, btype="forge", iron=10)
    cb = {1: {"forge": [forge]}}
    events = []
    tick_entity(h, world, [h], [], events, tick=1, clan_bldg=cb,
                species_counts={"human": 5})
    assert h.pick == "iron_pick" or h.tool == "iron_axe", (
        f"aucun outil upgradé fer à la forge (pick={h.pick}, tool={h.tool})")
    assert forge.iron < 10, "la forge n'a pas consommé de fer"
    assert any(ev.get("type", "").startswith("craft_iron") for ev in events), \
        "événement craft_iron_* manquant"

    # (2) dépôt du fer porté à la forge
    m = spawn(EntityType.HUMAN, lx, ly, Sex.MALE)
    m.clan_id = 1; m.hunger = 10.0; m.thirst = 10.0
    m.pick = "iron_pick"; m.tool = "iron_axe"   # déjà équipé → pas de craft
    m.iron = 3; m.wood = 0; m.stone = 0
    forge2 = Building(id=2, clan_id=1, x=lx, y=ly, btype="forge", iron=0)
    tick_entity(m, world, [m], [], [], tick=1, clan_bldg={1: {"forge": [forge2]}},
                species_counts={"human": 5})
    assert forge2.iron == 3, f"le fer n'a pas été déposé à la forge: {forge2.iron}"
    assert m.iron == 0, f"le mineur porte encore du fer: {m.iron}"
    print(f"  test_b_forge_upgrades_stone_tools_to_iron OK "
          f"(pick={h.pick}, tool={h.tool}, forge {10}→{forge.iron} ; dépôt 3 fer OK)")


def test_e8_dead_clan_leaves_ruins_then_fade():
    """Régression E8 : à l'extinction d'un clan, ses structures durables deviennent
    des RUINES (btype=ruin, clan_id=-1) + un événement clan_extinct est émis — au
    lieu de disparaître en silence. Les ruines s'effacent après RUIN_LIFETIME ticks
    (borne la mémoire → invariant infini)."""
    from engine.simulation import Clan, RUIN_LIFETIME
    world = World(width=60, height=45, seed=7)
    sim = Simulation(world)
    lx, ly = _find_land_tile(world)

    chief = spawn(EntityType.HUMAN, lx, ly, Sex.MALE)
    chief.clan_id = 7
    sim.entities = [chief]
    sim.clans = [Clan(id=7, cx=float(lx), cy=float(ly), color="#fff", chief_id=chief.id)]
    house = Building(id=1, clan_id=7, x=lx, y=ly, btype="house", wood=10)
    well  = Building(id=2, clan_id=7, x=min(lx + 1, world.width - 1), y=ly, btype="well")
    sim.buildings = [house, well]

    chief.alive = False   # le clan n'a plus aucun humain vivant → éteint ce tick
    data = sim.step()

    ruins = [b for b in sim.buildings if b.btype == "ruin"]
    assert len(ruins) == 2, f"maison+puit auraient dû devenir des ruines: {[b.btype for b in sim.buildings]}"
    assert all(r.clan_id == -1 for r in ruins), "les ruines devraient être orphelines (clan_id=-1)"
    assert not any(c.id == 7 for c in sim.clans), "le clan éteint n'a pas été retiré"
    ext = [ev for ev in data["events"] if ev.get("type") == "clan_extinct"]
    assert ext and ext[0]["ruins"] == 2, f"événement clan_extinct manquant/incorrect: {ext}"

    # Les ruines s'effacent avec le temps (la nature les reprend)
    for _ in range(RUIN_LIFETIME + 5):
        sim.step()
    assert not any(b.btype == "ruin" for b in sim.buildings), "les ruines ne se sont pas effacées"
    print(f"  test_e8_dead_clan_leaves_ruins_then_fade OK "
          f"(2 ruines + event, effacées après {RUIN_LIFETIME} ticks)")


def _find_pair_land_tiles(world, sep=4):
    """Deux tuiles terrestres alignées, distantes de `sep` (∈ ]3, vision])."""
    for y in range(world.height):
        for x in range(world.width - sep):
            if world.is_walkable(x, y, False) and world.is_walkable(x + sep, y, False):
                return (x, y), (x + sep, y)
    raise RuntimeError("aucune paire de tuiles terrestres alignées")


def test_e2_female_seeks_distant_mate():
    """Régression E2 : une femelle éligible SANS mâle adjacent (<3) mais avec un
    mâle éligible EN VISION (ici à 4 tuiles) doit passer en SEEKING_MATE et s'en
    RAPPROCHER — au lieu de rester à attendre un croisement au hasard. Sans E2, la
    repro n'arrive qu'à faible probabilité → goulot fatal à basse densité."""
    from engine.entities import State
    world = World(width=120, height=90, seed=7)
    (fx, fy), (mx, my) = _find_pair_land_tiles(world, sep=4)

    female = spawn(EntityType.SHEEP, fx, fy, Sex.FEMALE)
    male   = spawn(EntityType.SHEEP, mx, my, Sex.MALE)
    for s in (female, male):
        s.hunger = 20.0            # bien nourri : < repro_hunger_min (50), pas de "va manger"
        s.thirst = 10.0
        s.repro_cooldown_left = 0
        s.age = female.spec.max_age * 0.5   # > 20% du max_age → adulte fertile
    d0 = _dist(female.x, female.y, male.x, male.y)

    tick_entity(female, world, [female, male], [], [], tick=1, season="spring",
                species_counts={"sheep": 20})

    assert female.state == State.SEEKING_MATE, (
        f"la femelle n'a pas cherché de partenaire (état {female.state})")
    assert female.gestation_left == 0, "accouplement à distance interdit (doit d'abord se rapprocher)"
    d1 = _dist(female.x, female.y, male.x, male.y)
    assert d1 < d0, f"la femelle ne s'est pas rapprochée du mâle: {d0:.2f} → {d1:.2f}"
    print(f"  test_e2_female_seeks_distant_mate OK (SEEKING_MATE, dist {d0:.2f} → {d1:.2f})")


def test_save_load_roundtrip_and_resume(ticks: int = 200, resume: int = 120, seed: int = 555):
    """Persistance : (1) round-trip sans perte (état identique après load) ;
    (2) reprise EXACTE — un sim rechargé rejoue byte-à-byte la suite de l'original
    (état + RNG restaurés). Le RNG étant global au process, on enregistre la trace
    de référence AVANT le load, puis load (qui restaure le RNG au point de save)
    et on rejoue : les deux traces doivent coïncider."""
    import tempfile

    world = World(width=90, height=70, seed=seed)
    sim = Simulation(world)
    sim.populate()
    for _ in range(ticks):
        sim.step()

    path = tempfile.mktemp(suffix=".json")
    try:
        sim.save(path)

        # (1) round-trip immédiat dans un sim neuf
        sim2 = Simulation(World(width=10, height=10, seed=1))
        sim2.load(path)
        a0, b0 = sim.full_state(), sim2.full_state()
        assert a0["entities"] == b0["entities"], "entités différentes après load"
        assert a0["buildings"] == b0["buildings"], "bâtiments différents après load"
        assert a0["clans"] == b0["clans"], "clans différents après load"
        assert a0["tick"] == b0["tick"], "tick différent après load"

        # (2) reprise exacte (séquentiel car RNG global)
        ref = [sim.step() for _ in range(resume)]
        sim3 = Simulation(World(width=10, height=10, seed=1))
        sim3.load(path)
        rep = [sim3.step() for _ in range(resume)]
        assert ref == rep, "la reprise post-load diverge de l'original"
    finally:
        if os.path.exists(path):
            os.remove(path)
    print(f"  test_save_load_roundtrip_and_resume OK "
          f"(round-trip {len(a0['entities'])} entités + {resume} ticks rejoués identiques)")


if __name__ == "__main__":
    failures = 0
    for fn in (test_deposit_no_crash_when_houses_full,
               test_preservation_live_counter, test_water_stranded_entity_rescued,
               test_c1bis_toolless_human_crafts_without_depositing,
               test_e2_female_seeks_distant_mate,
               test_c2_hungry_harvester_eats_not_feeds_mill,
               test_a1_clan_gains_science_and_ages_up,
               test_b_forge_upgrades_stone_tools_to_iron,
               test_e8_dead_clan_leaves_ruins_then_fade,
               test_save_load_roundtrip_and_resume,
               test_smoke_runs):
        try:
            fn()
        except Exception as e:
            failures += 1
            import traceback
            print(f"  ÉCHEC {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("RÉSULTAT:", "TOUS OK" if failures == 0 else f"{failures} ÉCHEC(S)")
    sys.exit(1 if failures else 0)
