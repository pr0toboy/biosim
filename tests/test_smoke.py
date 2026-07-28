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
from engine.entities import EntityType, spawn, Sex, SPECS

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
            # cap par espèce (I1 : cap différencié spec.max_pop, prédateurs plus bas)
            counts = {}
            for e in sim.entities:
                counts[e.etype] = counts.get(e.etype, 0) + 1
            for et, c in counts.items():
                cap = SPECS[et].max_pop
                assert c <= cap, f"{et} dépasse son cap: {c} > {cap}"
            # Invariant persistance : AUCUN buffer d'événements en attente à la
            # frontière du tick (sinon un save perd des événements → replay divergent,
            # bug bisecté à t=5081 sur l'endurance 20k)
            assert sim.world._biome_changes == [], f"biome buffer non drainé: {sim.world._biome_changes}"
            assert sim.world._chop_changes == [], "chop buffer non drainé"
            assert sim.world._mine_changes == [], "mine buffer non drainé"
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


def test_e_boar_hunts_and_captures_prey():
    """Régression bloc E / PRÉDATION : un sanglier affamé (hunger 40, au-dessus de son
    hunt_hunger abaissé) avec une proie EN VISION la traque (HUNTING) et la capture si
    adjacente — au lieu de brouter avant d'avoir jamais assez faim pour chasser. C'est
    ce qui rend la prédation terrestre RÉELLE et visible (cosmétique en baseline : ~13
    kills/8000 ticks → ~200 kills/1000t avec le bloc PRÉDATION)."""
    from engine.entities import State
    world = World(width=120, height=90, seed=7)
    tx, ty = _find_land_tile(world)
    boar  = spawn(EntityType.BOAR, tx + 0.1, ty + 0.1, Sex.MALE)
    sheep = spawn(EntityType.SHEEP, tx + 0.4, ty + 0.4, Sex.FEMALE)   # dist ~0.42 < catch_r 0.8
    assert boar.spec.hunt_hunger < 30, "le sanglier doit chasser AVANT de brouter (seuil sous ~30)"
    boar.hunger = 40.0    # > hunt_hunger (chasse) ET > 30 (broutage) : sans le bloc E il brouterait
    boar.thirst = 10.0
    counts = {"sheep": 100}   # bien au-dessus du plancher de préservation (30)

    tick_entity(boar, world, [boar, sheep], [], [], tick=1, season="spring",
                species_counts=counts)

    assert boar.state == State.HUNTING, f"le sanglier n'a pas chassé (état {boar.state})"
    assert not sheep.alive, "la proie adjacente aurait dû être capturée"
    assert counts["sheep"] == 99, "le compteur vivant doit décrémenter au kill (anti-sur-chasse)"
    print("  test_e_boar_hunts_and_captures_prey OK (HUNTING + capture, compteur décrémenté)")


def test_e_hunt_preserves_prey_below_floor():
    """Régression bloc E : le plancher anti-extinction PAR LA CHASSE (garde
    species_counts[proie] < 30) tient malgré le seuil de chasse abaissé. Une proie dont
    l'espèce est sous 30 n'est jamais chassée → la prédation ne peut pas éteindre une
    espèce déjà rare (le rebond démographique reste au filet E2)."""
    from engine.entities import State
    world = World(width=120, height=90, seed=7)
    tx, ty = _find_land_tile(world)
    boar  = spawn(EntityType.BOAR, tx + 0.1, ty + 0.1, Sex.MALE)
    sheep = spawn(EntityType.SHEEP, tx + 0.4, ty + 0.4, Sex.FEMALE)
    boar.hunger = 40.0
    boar.thirst = 10.0
    counts = {"sheep": 29}   # sous le plancher → chasse interdite

    tick_entity(boar, world, [boar, sheep], [], [], tick=1, season="spring",
                species_counts=counts)

    assert sheep.alive, "une proie sous le plancher (29 < 30) ne doit pas être chassée"
    assert boar.state != State.HUNTING, f"le sanglier ne devait pas chasser (état {boar.state})"
    assert counts["sheep"] == 29, "le compteur ne doit pas bouger (aucun kill)"
    print("  test_e_hunt_preserves_prey_below_floor OK (plancher <30 tenu)")


def _find_walkable_row_segment(world, length):
    """Un segment horizontal de `length+1` tuiles terrestres consécutives →
    route de caravane garantie marchable en ligne droite."""
    for y in range(world.height):
        run = 0
        for x in range(world.width):
            run = run + 1 if world.is_walkable(x, y, False) else 0
            if run >= length + 1:
                return (x - length, y), (x, y)
    raise RuntimeError("aucun segment terrestre assez long")


def _d1_rig(a_stone=22, b_stone=100):
    """Rig caravane (D1) : 2 clans à l'Âge de Pierre, marchés finis, route droite
    marchable. Pièges neutralisés : pool bois=100 (=CLAN_WOOD_CAP → pas de coupe),
    stone_grid=0 (pas de minage), pick+tool posés (pas de craft C1bis),
    tick_count=119 (dispatch au 1er step, avant tout autre comportement)."""
    from engine.simulation import Clan, TRADE_CHECK_PERIOD
    world = World(width=60, height=45, seed=7)
    world.stone_grid[:] = 0.0
    sim = Simulation(world)
    (ax, ay), (bx, by) = _find_walkable_row_segment(world, 20)

    def mk_clan(cid, x, y, stone):
        h = spawn(EntityType.HUMAN, x, y, Sex.MALE)
        h.clan_id = cid; h.hunger = 10.0; h.thirst = 10.0
        h.wood = 0; h.stone = 0; h.iron = 0
        h.pick = "stone_pick"; h.tool = "stone_axe"   # bloque le craft découplé
        house = Building(id=cid * 10 + 1, clan_id=cid, x=x, y=y, btype="house",
                         wood=100, stone=stone)
        market = Building(id=cid * 10 + 2, clan_id=cid, x=x, y=y, btype="market")
        return h, house, market

    ha, house_a, mkt_a = mk_clan(0, ax, ay, a_stone)
    hb, house_b, mkt_b = mk_clan(1, bx, by, b_stone)
    sim.entities = [ha, hb]
    sim.clans = [Clan(id=0, cx=float(ax), cy=float(ay), color="#f00", chief_id=ha.id, age=1),
                 Clan(id=1, cx=float(bx), cy=float(by), color="#00f", chief_id=hb.id, age=1)]
    sim.buildings = [house_a, mkt_a, house_b, mkt_b]
    sim.tick_count = TRADE_CHECK_PERIOD - 1
    return sim, ha, hb, house_a, mkt_a, house_b, mkt_b


def test_d1_caravan_roundtrip_conserves_resources():
    """D1 : cycle caravane complet (deal → chargement −12 bois pool A → échange
    −6 pierre pool B / +12 bois étal B → retour +pierre étal A → drain vers les
    maisons), avec CONSERVATION de Σbois et Σpierre vérifiée à chaque tick."""
    from engine.simulation import TRADE_WOOD_LOT
    TRADE_STONE_PRICE = 6   # D2 : taux spot — vendeur rigé à 100 → palier 6 (sémantique D1 intacte)
    sim, ha, hb, house_a, mkt_a, house_b, mkt_b = _d1_rig()

    def totals():
        w = house_a.wood + house_b.wood + mkt_a.wood + mkt_b.wood + \
            ha.cargo_wood + hb.cargo_wood + ha.wood + hb.wood
        s = house_a.stone + house_b.stone + mkt_a.stone + mkt_b.stone + \
            ha.cargo_stone + hb.cargo_stone + ha.stone + hb.stone
        return w, s

    w0, s0 = totals()
    seen = set()
    for _ in range(800):
        data = sim.step()
        for ev in data["events"]:
            if ev.get("type", "").startswith("trade_"):
                seen.add(ev["type"])
        w, s = totals()
        assert (w, s) == (w0, s0), f"conservation violée: bois {w0}->{w}, pierre {s0}->{s}"
        if "trade_complete" in seen:
            break
    assert {"trade_deal", "trade_depart", "trade_exchange", "trade_complete"} <= seen, \
        f"cycle incomplet: {seen}"
    assert mkt_b.wood == TRADE_WOOD_LOT, f"étal B: {mkt_b.wood} bois (attendu {TRADE_WOOD_LOT})"
    assert house_b.stone == 100 - TRADE_STONE_PRICE, f"pool B: {house_b.stone}"
    # la pierre importée est chez A (étal, en cours de drain, ou déjà en maison)
    assert mkt_a.stone + house_a.stone == 22 + TRADE_STONE_PRICE, \
        f"pierre A: étal {mkt_a.stone} + maison {house_a.stone}"
    assert ha.trade_phase is None and ha.cargo_wood == 0 and ha.cargo_stone == 0
    print(f"  test_d1_caravan_roundtrip_conserves_resources OK "
          f"(cycle complet, Σ conservées, +{TRADE_STONE_PRICE} pierre importée)")


def test_d1_no_trade_without_complementary_surplus():
    """D1 : pas d'excédents complémentaires → ZÉRO caravane (anti-ping-pong).
    (a) personne n'est pauvre en pierre ; (b) personne ne peut vendre."""
    for a_stone, b_stone, label in ((100, 100, "A riche"), (10, 10, "B pauvre")):
        sim, *_ = _d1_rig(a_stone=a_stone, b_stone=b_stone)
        deals = 0
        for _ in range(400):
            data = sim.step()
            deals += sum(1 for ev in data["events"] if ev.get("type") == "trade_deal")
        assert deals == 0, f"{label}: {deals} deal(s) émis sans complémentarité"
    print("  test_d1_no_trade_without_complementary_surplus OK (0 deal dans les 2 rigs)")


def test_d1_dest_ruined_merchant_returns_and_replay():
    """D1 : clan destination éteint en cours de route → demi-tour cargaison
    intacte + bois recyclé chez A ; puis save/load EN PLEINE MISSION → replay
    byte-à-byte (les 5 champs trade suffisent)."""
    sim, ha, hb, house_a, mkt_a, house_b, mkt_b = _d1_rig()
    for _ in range(400):
        sim.step()
        if ha.trade_phase == "out" and ha.cargo_wood > 0:
            break
    assert ha.trade_phase == "out", f"jamais parti (phase={ha.trade_phase})"
    hb.alive = False   # clan B s'éteint → E8 ruine son marché au tick suivant
    for _ in range(400):
        sim.step()
        if ha.trade_phase is None:
            break
    assert ha.trade_phase is None, "mission jamais terminée après ruine de la destination"
    assert mkt_b.btype == "ruin", f"marché B pas ruiné: {mkt_b.btype}"
    assert ha.cargo_wood == 0 and ha.cargo_stone == 0
    # le lot de bois est revenu chez A (étal en drain ou maisons), rien créé ni perdu
    assert house_a.wood + mkt_a.wood == 100, \
        f"bois A: maison {house_a.wood} + étal {mkt_a.wood} (attendu 100 au total)"

    # Replay byte-à-byte depuis une mission EN TRANSIT (rig neuf)
    sim2, ha2, *_ = _d1_rig()
    for _ in range(400):
        sim2.step()
        if ha2.trade_phase == "out":
            break
    assert ha2.trade_phase == "out"
    snap = sim2.save_state()
    ref = [sim2.step() for _ in range(100)]
    sim3 = Simulation(World(width=10, height=10, seed=1))
    sim3.load_state(snap)
    rep = [sim3.step() for _ in range(100)]
    assert ref == rep, "replay divergent avec une caravane en mission"
    print("  test_d1_dest_ruined_merchant_returns_and_replay OK "
          "(demi-tour propre + replay 100 ticks identiques en mission)")


def _d2_rig(a_stone=200, a_iron=2, b_stone=40, b_iron=18):
    """Rig D2 : comme _d1_rig + une FORGE par clan. Outils FER posés (bloque
    l'upgrade forge qui consommerait le fer → conservation Σ testable). Âge 1
    (pas de minage de fer, gated âge 2)."""
    from engine.simulation import Clan, TRADE_CHECK_PERIOD
    world = World(width=60, height=45, seed=7)
    world.stone_grid[:] = 0.0
    sim = Simulation(world)
    (ax, ay), (bx, by) = _find_walkable_row_segment(world, 20)

    def mk_clan(cid, x, y, stone, iron):
        h = spawn(EntityType.HUMAN, x, y, Sex.MALE)
        h.clan_id = cid; h.hunger = 10.0; h.thirst = 10.0
        h.wood = 0; h.stone = 0; h.iron = 0
        h.pick = "iron_pick"; h.tool = "iron_axe"
        house = Building(id=cid * 10 + 1, clan_id=cid, x=x, y=y, btype="house",
                         wood=100, stone=stone)
        market = Building(id=cid * 10 + 2, clan_id=cid, x=x, y=y, btype="market")
        forge = Building(id=cid * 10 + 3, clan_id=cid, x=x, y=y, btype="forge",
                         iron=iron)
        return h, house, market, forge

    ha, house_a, mkt_a, forge_a = mk_clan(0, ax, ay, a_stone, a_iron)
    hb, house_b, mkt_b, forge_b = mk_clan(1, bx, by, b_stone, b_iron)
    sim.entities = [ha, hb]
    sim.clans = [Clan(id=0, cx=float(ax), cy=float(ay), color="#f00", chief_id=ha.id, age=1),
                 Clan(id=1, cx=float(bx), cy=float(by), color="#00f", chief_id=hb.id, age=1)]
    sim.buildings = [house_a, mkt_a, forge_a, house_b, mkt_b, forge_b]
    sim.tick_count = TRADE_CHECK_PERIOD - 1
    return sim, ha, hb, (house_a, mkt_a, forge_a), (house_b, mkt_b, forge_b)


def test_d2_iron_for_stone_roundtrip_conserves():
    """D2 : route FER payée en PIERRE (le vendeur valorise la pierre rare > le bois
    en glut), taux spot avec plancher, conservation Σ bois/pierre/fer par tick."""
    sim, ha, hb, (house_a, mkt_a, forge_a), (house_b, mkt_b, forge_b) = _d2_rig()

    def totals():
        w = house_a.wood + house_b.wood + mkt_a.wood + mkt_b.wood + \
            ha.cargo_wood + hb.cargo_wood + ha.wood + hb.wood
        s = house_a.stone + house_b.stone + mkt_a.stone + mkt_b.stone + \
            ha.cargo_stone + hb.cargo_stone + ha.stone + hb.stone
        i = forge_a.iron + forge_b.iron + mkt_a.iron + mkt_b.iron + \
            ha.cargo_iron + hb.cargo_iron + ha.iron + hb.iron
        return w, s, i

    t0 = totals()
    deals = []
    for _ in range(800):
        data = sim.step()
        for ev in data["events"]:
            if ev.get("type", "").startswith("trade_"):
                deals.append(ev)
        assert totals() == t0, f"conservation violée: {t0} -> {totals()}"
        if any(ev.get("type") == "trade_complete" for ev in deals):
            break
    kinds = {ev["type"] for ev in deals}
    assert "trade_complete" in kinds, f"cycle incomplet: {kinds}"
    deal = next(ev for ev in deals if ev["type"] == "trade_deal")
    assert deal["good"] == "iron" and deal["pay"] == "stone", f"deal inattendu: {deal}"
    assert forge_b.iron == 15, f"forge B: {forge_b.iron} (attendu 18-3=15, plancher 8 respecté)"
    # le drain (1/4 ticks) a pu déjà déplacer l'étal vers les maisons → somme
    assert mkt_b.stone + house_b.stone == 40 + 6, \
        f"pierre B: étal {mkt_b.stone} + maison {house_b.stone} (attendu 46 au total)"
    assert forge_a.iron + mkt_a.iron == 2 + 3, \
        f"fer A: forge {forge_a.iron} + étal {mkt_a.iron} (attendu 5 au total)"
    print("  test_d2_iron_for_stone_roundtrip_conserves OK "
          "(fer contre pierre, spot 3, Σ w/s/i conservées)")


def test_d2_no_flip_and_refusal():
    """D2 : (a) anti-flip — l'acheteur enrichi sort de la fenêtre, le vendeur
    appauvri sort du palier → plus AUCUN deal fer ensuite ; (b) cours tombé en
    route → trade_refused, cargaison revenue intégralement (Σ conservées)."""
    # (a) A fer=4, B fer=11 (rate 2) → 1 échange de 2, puis fenêtre refermée
    sim, ha, hb, (house_a, mkt_a, forge_a), (house_b, mkt_b, forge_b) = \
        _d2_rig(a_stone=200, a_iron=4, b_stone=40, b_iron=11)
    got_exchange = False; deals_after = 0
    for k in range(1200):
        data = sim.step()
        for ev in data["events"]:
            if ev.get("type") == "trade_exchange":
                got_exchange = True
                assert ev["qty"] == 2, f"qty: {ev['qty']} (attendu spot 2)"
            elif ev.get("type") == "trade_deal" and got_exchange:
                deals_after += 1
    assert got_exchange, "aucun échange fer dans le rig (a)"
    assert deals_after == 0, f"{deals_after} deal(s) après l'échange (flip/ping-pong)"
    assert forge_b.iron == 9, f"forge B: {forge_b.iron} (11-2, sous le palier 10)"

    # (b) refus : cours effondré pendant le trajet
    sim, ha, hb, (house_a, mkt_a, forge_a), (house_b, mkt_b, forge_b) = _d2_rig()
    for _ in range(400):
        sim.step()
        if ha.trade_phase == "out":
            break
    assert ha.trade_phase == "out"
    forge_b.iron = 9   # sous le palier 10 → rate 0 à l'arrivée
    refused = False
    for _ in range(600):
        data = sim.step()
        refused = refused or any(ev.get("type") == "trade_refused" for ev in data["events"])
        if ha.trade_phase is None:
            break
    assert refused, "trade_refused jamais émis"
    assert ha.trade_phase is None and ha.cargo_stone == 0
    # le paiement (6 pierre) est revenu côté A : étal (en drain) ou maisons
    assert house_a.stone + mkt_a.stone == 200, \
        f"pierre A: {house_a.stone}+{mkt_a.stone} (attendu 200 au total)"
    print("  test_d2_no_flip_and_refusal OK (1 seul échange puis silence ; refus + retour intégral)")


def test_d2_price_board_and_replay():
    """D2 : le board affiche le cours spot (une seule émission d'event, pas de
    re-spam) ; save/load en phase out d'une mission fer-contre-pierre → replay
    100 ticks byte-à-byte (3 champs Entity + 4 champs Building suffisent)."""
    from engine.simulation import TRADE_CHECK_PERIOD
    sim, ha, hb, (house_a, mkt_a, forge_a), (house_b, mkt_b, forge_b) = \
        _d2_rig(b_stone=600)
    price_events = 0
    for _ in range(2 * TRADE_CHECK_PERIOD + 2):
        data = sim.step()
        price_events += sum(1 for ev in data["events"]
                            if ev.get("type") == "market_price"
                            and ev.get("clan_id") == 1 and ev.get("good") == "stone")
    assert mkt_b.rate_stone == 12, f"board B: {mkt_b.rate_stone} (600 → palier 12)"
    assert price_events == 1, f"{price_events} events market_price (attendu 1, pas de re-spam)"

    sim2, ha2, *_ = _d2_rig()
    for _ in range(400):
        sim2.step()
        if ha2.trade_phase == "out":
            break
    assert ha2.trade_phase == "out" and ha2.trade_good == "iron"
    snap = sim2.save_state()
    ref = [sim2.step() for _ in range(100)]
    sim3 = Simulation(World(width=10, height=10, seed=1))
    sim3.load_state(snap)
    rep = [sim3.step() for _ in range(100)]
    assert ref == rep, "replay divergent (mission D2 en vol)"
    print("  test_d2_price_board_and_replay OK (board 12 doré, 1 event, replay identique)")


def _c1_rig(a_age=0, a_church=False, b_age=3, tick0=None):
    """Rig C1 : clan A (pèlerin potentiel) + clan B (église, Acier). Pools bois à
    100 (=CLAN_WOOD_CAP → pas de coupe → Σ bois testable), pierre/minage/craft
    neutralisés comme aux rigs D1/D2."""
    from engine.simulation import Clan, PILGRIM_CHECK_PERIOD
    world = World(width=60, height=45, seed=7)
    world.stone_grid[:] = 0.0
    sim = Simulation(world)
    (ax, ay), (bx, by) = _find_walkable_row_segment(world, 20)

    def mk(cid, x, y, age, church):
        h = spawn(EntityType.HUMAN, x, y, Sex.MALE)
        h.clan_id = cid; h.hunger = 10.0; h.thirst = 10.0
        h.wood = 0; h.stone = 0; h.iron = 0
        h.pick = "iron_pick"; h.tool = "iron_axe"
        houses = [Building(id=cid * 10 + 1, clan_id=cid, x=x, y=y, btype="house", wood=34),
                  Building(id=cid * 10 + 2, clan_id=cid, x=x, y=y, btype="house", wood=33),
                  Building(id=cid * 10 + 3, clan_id=cid, x=x, y=y, btype="house", wood=33)]
        blds = list(houses)
        # marché/puits/forge pré-construits : sinon un clan éligible bâtit
        # (déduction bois/pierre au chantier) et casse la conservation Σ du test
        blds.append(Building(id=cid * 10 + 4, clan_id=cid, x=x, y=y, btype="market"))
        blds.append(Building(id=cid * 10 + 6, clan_id=cid, x=x, y=y, btype="well"))
        blds.append(Building(id=cid * 10 + 7, clan_id=cid, x=x, y=y, btype="well"))
        blds.append(Building(id=cid * 10 + 8, clan_id=cid, x=x, y=y, btype="forge"))
        # 8 champs pré-construits (= cap/clan) : sinon l'humain passe sa vie à
        # PLANTER (coût 0 mais _build_target_type posé → jamais candidat aux missions)
        for k in range(8):
            blds.append(Building(id=cid * 100 + 20 + k, clan_id=cid, x=x, y=y,
                                 btype="wheatfield"))
        if church:
            blds.append(Building(id=cid * 10 + 5, clan_id=cid, x=x, y=y, btype="church"))
        return h, houses, blds

    ha, houses_a, blds_a = mk(0, ax, ay, a_age, a_church)
    hb, houses_b, blds_b = mk(1, bx, by, b_age, True)
    sim.entities = [ha, hb]
    sim.clans = [Clan(id=0, cx=float(ax), cy=float(ay), color="#f00", chief_id=ha.id, age=a_age),
                 Clan(id=1, cx=float(bx), cy=float(by), color="#00f", chief_id=hb.id, age=b_age)]
    sim.buildings = blds_a + blds_b
    sim.tick_count = (PILGRIM_CHECK_PERIOD - 1) if tick0 is None else tick0
    return sim, ha, hb, houses_a, houses_b, blds_b[-1]


def test_c1_office_procession_blessing_and_hunger():
    """C1 office : à la cloche, les fidèles processionnent vers l'église (distances
    décroissantes), s'agenouillent et sont bénis ; l'affamé est exclu ; l'effet de
    la bénédiction sur la faim est RÉEL (vs contrôle non-béni à traits égaux)."""
    from engine.simulation import (Clan, State, CHURCH_SERVICE_PERIOD,
                                   PRAY_DURATION, CHURCH_CALL_RADIUS)
    world = World(width=60, height=45, seed=7)
    world.stone_grid[:] = 0.0
    world.food_grid[:] = 0.0   # personne ne mange → faim comparable
    sim = Simulation(world)
    # Rangée garantie marchable : l'église au bout, les fidèles alignés dessus
    # (sinon _find_land_tile peut rendre un coin borde d'eau → fidèles noyés/coupés)
    (lx, ly), _ = _find_walkable_row_segment(world, 16)
    church = Building(id=1, clan_id=0, x=lx, y=ly, btype="church")
    house = Building(id=2, clan_id=0, x=lx, y=ly, btype="house", wood=100)
    fideles = []
    for i in range(4):
        h = spawn(EntityType.HUMAN, min(lx + 8 + i * 2, world.width - 1), ly, Sex.MALE)
        h.clan_id = 0; h.hunger = 5.0; h.thirst = 5.0
        h.pick = "iron_pick"; h.tool = "iron_axe"
        h.traits = {"speed": 1.0, "vision": 8, "hunger_rate": 0.08}
        fideles.append(h)
    gourmand = spawn(EntityType.HUMAN, min(lx + 10, world.width - 1), ly, Sex.MALE)
    gourmand.clan_id = 0; gourmand.hunger = 60.0; gourmand.thirst = 5.0   # > PRAY_HUNGER_MAX
    gourmand.pick = "iron_pick"; gourmand.tool = "iron_axe"
    gourmand.traits = dict(fideles[0].traits)
    # Témoin : n'importe quelle tuile marchable à >28 de l'église (hors rayon d'appel)
    tx = ty = None
    for yy in range(world.height):
        for xx in range(world.width):
            if world.is_walkable(xx, yy, False) and _dist(xx, yy, lx, ly) > 28:
                tx, ty = xx, yy
                break
        if tx is not None:
            break
    # Témoin : contrôle non-béni, traits identiques, pour prouver l'effet RÉEL sur la faim.
    # Il doit être inéligible par STRUCTURE, pas par distance : un humain du clan 0 dérive
    # vers son centre de clan (= la tuile de l'église) et finit dans le rayon d'appel dès
    # que la fenêtre d'office est longue. On le met donc dans un clan SANS église : son
    # _cb.get("church") est vide ⇒ il ne prie jamais, où qu'il soit et quelle que soit la
    # fenêtre. (Loin quand même, pour ne pas interférer avec le clan 0.)
    temoin = spawn(EntityType.HUMAN, tx, ty, Sex.MALE)
    temoin.clan_id = 1; temoin.hunger = 5.0; temoin.thirst = 5.0
    temoin.pick = "iron_pick"; temoin.tool = "iron_axe"
    temoin.traits = dict(fideles[0].traits)
    sim.entities = fideles + [gourmand, temoin]
    sim.clans = [Clan(id=0, cx=float(lx), cy=float(ly), color="#f00",
                      chief_id=fideles[0].id, age=3),
                 Clan(id=1, cx=float(tx), cy=float(ty), color="#00f",
                      chief_id=temoin.id, age=3)]
    sim.buildings = [church, house]
    # Dérivé de la constante (jamais figé) : 1er step → fenêtre d'office ouverte (cid 0)
    sim.tick_count = CHURCH_SERVICE_PERIOD - 1

    d0 = [_dist(h.x, h.y, church.x, church.y) for h in fideles]
    dmin = list(d0)   # distance min atteinte PENDANT la fenêtre (après, ils repartent)
    prayed = set(); gourmand_prayed = False
    # Budget = approche (bornée par le rayon d'appel, vitesse inchangée) + prière + marge.
    # Suit TIME_SCALE tout seul.
    for _ in range(CHURCH_CALL_RADIUS + 2 * PRAY_DURATION):
        sim.step()
        for i, h in enumerate(fideles):
            if h.state == State.PRAYING:
                prayed.add(i)
            dmin[i] = min(dmin[i], _dist(h.x, h.y, church.x, church.y))
        gourmand_prayed = gourmand_prayed or gourmand.state == State.PRAYING
    blessed = sum(1 for h in fideles if h.blessed_ticks > 0)
    assert len(prayed) >= 3, f"seulement {len(prayed)} fidèles en PRAYING"
    assert sum(1 for a, b in zip(d0, dmin) if b < a - 1.0) >= 3, \
        f"la procession ne converge pas: {[f'{a:.0f}->min{b:.0f}' for a, b in zip(d0, dmin)]}"
    assert blessed >= 3, f"seulement {blessed} bénis"
    assert not gourmand_prayed, "l'affamé (hunger 60 > 55) a participé à l'office"
    assert temoin.blessed_ticks == 0, "le témoin (clan sans église) a été béni"
    b0 = next(h for h in fideles if h.blessed_ticks > 0)
    assert b0.hunger < temoin.hunger, \
        f"bénédiction sans effet réel: béni {b0.hunger:.1f} vs témoin {temoin.hunger:.1f}"
    print(f"  test_c1_office_procession_blessing_and_hunger OK "
          f"({len(prayed)} fidèles, {blessed} bénis, faim {b0.hunger:.1f}<{temoin.hunger:.1f})")


def test_c1_pilgrimage_pays_offering_and_conserves():
    """C1 pèlerinage : A (sans église) envoie un pèlerin déposer 8 bois sur l'autel
    de B contre bénédiction ; Σ bois conservée à CHAQUE tick (burn suspendu) ;
    exclusivité trade/pilgrim ; puis le burn consume l'autel (le puits)."""
    import engine.simulation as S
    sim, ha, hb, houses_a, houses_b, church_b = _c1_rig()
    # Opportunité caravane PENDANT la mission (gate-review C1), via le FER (la
    # pierre déverrouillerait puits/moulins/upgrades → bruit de construction dans le
    # ledger bois) : A (forge presque vide) achète du fer à B (forge pleine), payé
    # en BOIS → le troc déplace du bois DANS le ledger (étals comptés). Sans la
    # garde croisée pilgrim_phase du dispatch caravanes, le pèlerin serait recruté
    # marchand et son offrande écrasée (Σ violée + bénédiction gratuite).
    for b in sim.buildings:
        if b.btype == "forge":
            b.iron = 2 if b.clan_id == 0 else 18
    old_burn = S.ALTAR_BURN_PERIOD
    S.ALTAR_BURN_PERIOD = 10 ** 9   # burn suspendu pour la conservation pure
    try:
        def wood_total():
            # ledger GÉNÉRAL : tout le bois du monde clos (bâtiments porteurs de
            # stock + cargaisons + inventaires) — le troc D1/D2 peut légitimement
            # déplacer du bois vers les étals pendant le pèlerinage
            return (sum(b.wood for b in sim.buildings
                        if b.btype in ("house", "market", "church"))
                    + ha.cargo_wood + hb.cargo_wood + ha.wood + hb.wood)
        w0 = wood_total()
        seen = set()
        # Budget DÉRIVÉ (jamais figé) : le 1er contrôle pèlerin (t = PILGRIM_CHECK_PERIOD)
        # est capté par la caravane — les deux périodes coïncident et les caravanes sont
        # dispatchées d'abord. Le pèlerin ne part donc qu'au contrôle SUIVANT, une fois la
        # mission de troc rendue. Il faut couvrir ce 2e contrôle + l'aller-retour (~900
        # ticks, borné par le TRAJET → vitesse inchangée → indépendant de TIME_SCALE).
        for _ in range(S.PILGRIM_CHECK_PERIOD + 900):
            data = sim.step()
            for ev in data["events"]:
                if ev.get("type", "").startswith("pilgrim"):
                    seen.add(ev["type"])
            assert wood_total() == w0, f"Σ bois violée: {w0} -> {wood_total()}"
            assert not (ha.trade_phase is not None and ha.pilgrim_phase is not None), \
                "exclusivité trade/pilgrim violée (les DEUX missions actives)"
            if "pilgrim_home" in seen:
                break
        assert {"pilgrim_depart", "pilgrim_blessed", "pilgrim_home"} <= seen, \
            f"cycle incomplet: {seen}"
        assert church_b.wood == 8, f"autel B: {church_b.wood} (attendu 8)"
        assert church_b.pilgrims_served == 1
        assert ha.blessed_ticks > 0, "pas béni après le dépôt (BLESS_DURATION=600 > trajet retour)"
        assert ha.pilgrim_phase is None and ha.cargo_wood == 0
    finally:
        S.ALTAR_BURN_PERIOD = old_burn
    for _ in range(8 * old_burn + 10):
        sim.step()
        if church_b.wood == 0:
            break
    assert church_b.wood == 0, f"l'autel ne se consume pas: {church_b.wood}"
    print("  test_c1_pilgrimage_pays_offering_and_conserves OK "
          "(8 bois offerts, Σ conservée, autel consumé)")


def test_c1_dest_ruined_and_replay():
    """C1 : église détruite (clan B éteint) pendant le trajet → demi-tour, offrande
    re-créditée, AUCUNE bénédiction ; puis replay byte-à-byte en plein pèlerinage
    ET en pleine prière."""
    sim, ha, hb, houses_a, houses_b, church_b = _c1_rig()
    for _ in range(500):
        sim.step()
        if ha.pilgrim_phase == "out" and ha.cargo_wood > 0:
            break
    assert ha.pilgrim_phase == "out", f"jamais parti ({ha.pilgrim_phase})"
    hb.alive = False   # clan B s'éteint → E8 ruine son église
    for _ in range(600):
        sim.step()
        if ha.pilgrim_phase is None:
            break
    assert ha.pilgrim_phase is None, "mission jamais close après ruine"
    assert church_b.btype == "ruin"
    assert ha.blessed_ticks == 0, "béni sans avoir déposé l'offrande"
    assert sum(h.wood for h in houses_a) == 100, \
        f"offrande non re-créditée: {sum(h.wood for h in houses_a)}"

    # Replay en plein pèlerinage
    sim2, ha2, *_ = _c1_rig()
    for _ in range(500):
        sim2.step()
        if ha2.pilgrim_phase == "out":
            break
    assert ha2.pilgrim_phase == "out"
    snap = sim2.save_state()
    ref = [sim2.step() for _ in range(100)]
    sim3 = Simulation(World(width=10, height=10, seed=1))
    sim3.load_state(snap)
    rep = [sim3.step() for _ in range(100)]
    assert ref == rep, "replay divergent en plein pèlerinage"

    # Replay en pleine prière (pray_ticks > 0)
    sim4, ha4, hb4, _, _, church4 = _c1_rig(a_age=3, a_church=True, tick0=299)
    from engine.simulation import State as St
    for _ in range(120):
        sim4.step()
        if ha4.pray_ticks > 0:
            break
    assert ha4.pray_ticks > 0, "jamais en prière"
    snap4 = sim4.save_state()
    ref4 = [sim4.step() for _ in range(80)]
    sim5 = Simulation(World(width=10, height=10, seed=1))
    sim5.load_state(snap4)
    rep4 = [sim5.step() for _ in range(80)]
    assert ref4 == rep4, "replay divergent en pleine prière"
    print("  test_c1_dest_ruined_and_replay OK "
          "(demi-tour + re-crédit sans bénédiction ; replay OK en mission et en prière)")


def _c2_rig(specs):
    """Rig C2 : N clans sur une rangée marchable (espacés de 20), chacun avec église
    (trésor/renom configurables) + tout le constructible pré-construit. FER et
    PIERRE neutralisés (amendement contre-vérif : 161 tuiles de fer sur cette carte
    détourneraient les mineurs, et un porteur de fer n'est plus candidat pèlerin).
    specs = [(age, treasury, renom), ...]"""
    from engine.simulation import Clan, PILGRIM_CHECK_PERIOD
    world = World(width=60, height=45, seed=7)
    world.stone_grid[:] = 0.0
    world.iron_grid[:] = 0.0
    world.gold_grid[:] = 0.0   # monde clos en or par défaut (les tests peignent leurs filons)
    sim = Simulation(world)
    (x0, y0), _ = _find_walkable_row_segment(world, 20 * (len(specs) - 1) + 1)
    humans, churches = [], []
    blds = []
    for cid, (age, treasury, renom) in enumerate(specs):
        x = x0 + cid * 20
        h = spawn(EntityType.HUMAN, x, y0, Sex.MALE)
        h.clan_id = cid; h.hunger = 10.0; h.thirst = 10.0
        h.wood = 0; h.stone = 0; h.iron = 0; h.gold = 0
        h.pick = "iron_pick"; h.tool = "iron_axe"
        humans.append(h)
        for k in range(3):
            blds.append(Building(id=cid * 100 + k, clan_id=cid, x=x, y=y0,
                                 btype="house", wood=[34, 33, 33][k]))
        ch = Building(id=cid * 100 + 5, clan_id=cid, x=x, y=y0, btype="church",
                      gold=treasury, pilgrims_served=renom)
        churches.append(ch); blds.append(ch)
        blds.append(Building(id=cid * 100 + 4, clan_id=cid, x=x, y=y0, btype="market"))
        blds.append(Building(id=cid * 100 + 6, clan_id=cid, x=x, y=y0, btype="well"))
        blds.append(Building(id=cid * 100 + 7, clan_id=cid, x=x, y=y0, btype="well"))
        blds.append(Building(id=cid * 100 + 8, clan_id=cid, x=x, y=y0, btype="forge"))
        for k in range(8):
            blds.append(Building(id=cid * 100 + 20 + k, clan_id=cid, x=x, y=y0,
                                 btype="wheatfield"))
    sim.entities = list(humans)
    sim.clans = [Clan(id=cid, cx=float(x0 + cid * 20), cy=float(y0), color="#fff",
                      chief_id=humans[cid].id, age=specs[cid][0])
                 for cid in range(len(specs))]
    sim.buildings = blds
    sim.tick_count = PILGRIM_CHECK_PERIOD - 1
    return sim, humans, churches


def test_c2_gold_mine_deposit_hysteresis():
    """C2 source bornée : trésor sous l'hystérésis → expédition, minage (jamais
    écrêté), dépôt exact, puis la vanne fermée n'envoie PLUS personne."""
    from engine.simulation import GOLD_RESTOCK_THRESHOLD, MAX_GOLD_CARRY
    sim, humans, churches = _c2_rig([(3, GOLD_RESTOCK_THRESHOLD - 1, 0)])
    world = sim.world; h = humans[0]; ch = churches[0]
    # peindre 2 filons marchables adjacents près du camp
    vx, vy = int(ch.x) + 3, int(ch.y)
    for dx in (0, 1):
        world._gold_mask[vy, vx + dx] = True
        world.gold_grid[vy, vx + dx] = 100.0
    deposited = False
    for _ in range(400):
        data = sim.step()
        assert h.gold <= MAX_GOLD_CARRY
        if any(ev.get("type") == "gold_deposit" for ev in data["events"]):
            deposited = True
            break
    assert deposited, "aucun dépôt d'or"
    assert ch.gold == GOLD_RESTOCK_THRESHOLD - 1 + MAX_GOLD_CARRY, \
        f"trésor: {ch.gold} (attendu 3+2=5)"
    assert h.gold == 0
    # hystérésis : trésor 5 >= 4 → plus AUCUNE expédition (les filons ne baissent plus)
    g_before = float(world.gold_grid[vy, vx]) + float(world.gold_grid[vy, vx + 1])
    for _ in range(400):
        sim.step()
    g_after = float(world.gold_grid[vy, vx]) + float(world.gold_grid[vy, vx + 1])
    assert g_after >= g_before - 0.001, f"ré-expédition malgré l'hystérésis: {g_before}->{g_after}"
    assert ch.gold == GOLD_RESTOCK_THRESHOLD - 1 + MAX_GOLD_CARRY
    print(f"  test_c2_gold_mine_deposit_hysteresis OK (trésor 3→{ch.gold}, vanne refermée)")


def test_c2_gold_offering_circulates_and_conserves():
    """C2 circulation (LA gate du bloc) : A (trésor 1) offre sa pièce à B ; B
    RE-DÉPENSE la pièce reçue vers C (plein) → dorure (le puits). Σ or EXISTANT+gilt
    constante à chaque tick en monde clos. Exclusivité trade préservée."""
    import engine.simulation as S
    from engine.simulation import GOLD_TREASURY_MAX
    sim, humans, churches = _c2_rig([(3, 1, 0), (3, 0, 4), (3, GOLD_TREASURY_MAX, 8)])
    cha, chb, chc = churches
    old_min = S.PILGRIM_WOOD_MIN
    S.PILGRIM_WOOD_MIN = 10 ** 9   # coupe les pèlerinages bois : seuls les chemins OR
    try:
        def gold_total():
            return (sum(c.gold + c.gilt for c in churches)
                    + sum(h.cargo_gold + h.gold for h in humans))
        g0 = gold_total()
        assert g0 == 1 + 0 + GOLD_TREASURY_MAX
        seen = []
        # Budget DÉRIVÉ : la chaîne A→B→C demande DEUX départs, donc deux contrôles
        # pèlerins distincts (B ne peut re-dépenser qu'après avoir reçu la pièce de A).
        # On offre ~6 contrôles comme l'ancien barème (1400/240) ; la boucle sort dès la
        # dorure de C, donc cette borne ne coûte rien quand tout va bien.
        for _ in range(6 * S.PILGRIM_CHECK_PERIOD):
            data = sim.step()
            for ev in data["events"]:
                if ev.get("type") in ("pilgrim_depart", "pilgrim_blessed", "church_gilt"):
                    seen.append((ev["type"], ev.get("clan_id"), ev.get("dest_clan_id")))
            assert gold_total() == g0, f"Σ or violée: {g0} -> {gold_total()}"
            for h in humans:
                assert not (h.trade_phase and h.pilgrim_phase), "exclusivité violée"
            if ("church_gilt", 2, None) in [(s[0], s[1], None) for s in seen]:
                break
        assert ("pilgrim_depart", 0, 1) in seen, f"A→B manquant: {seen}"
        assert ("pilgrim_blessed", 0, 1) in seen
        assert ("pilgrim_depart", 1, 2) in seen, f"B ne re-dépense pas: {seen}"
        assert chc.gilt == 1, f"dorure C: {chc.gilt} (le puits)"
        assert chc.gold == GOLD_TREASURY_MAX
        assert chb.pilgrims_served == 5
    finally:
        S.PILGRIM_WOOD_MIN = old_min
    print("  test_c2_gold_offering_circulates_and_conserves OK "
          "(A→B→C, re-dépense, dorure, Σ=13 constante)")


def test_c2_dest_ruined_gold_recredit_and_replay():
    """C2 : destination ruinée en route → la pièce revient AU TRÉSOR de A, aucune
    bénédiction ; replays byte-à-byte en mission-or ET avec mineur chargé."""
    import engine.simulation as S
    sim, humans, churches = _c2_rig([(3, 1, 0), (3, 0, 4)])
    ha, hb = humans; cha, chb = churches
    old_min = S.PILGRIM_WOOD_MIN
    S.PILGRIM_WOOD_MIN = 10 ** 9
    try:
        for _ in range(500):
            sim.step()
            if ha.pilgrim_phase == "out" and ha.cargo_gold > 0:
                break
        assert ha.pilgrim_phase == "out", f"jamais parti ({ha.pilgrim_phase})"
        hb.alive = False   # clan B s'éteint → E8 ruine son église
        for _ in range(600):
            sim.step()
            if ha.pilgrim_phase is None:
                break
        assert ha.pilgrim_phase is None
        assert chb.btype == "ruin"
        assert ha.blessed_ticks == 0, "béni sans dépôt"
        assert cha.gold == 1, f"pièce non re-créditée au trésor: {cha.gold}"

        # replay en mission-or
        sim2, humans2, _ = _c2_rig([(3, 1, 0), (3, 0, 4)])
        ha2 = humans2[0]
        for _ in range(500):
            sim2.step()
            if ha2.pilgrim_phase == "out":
                break
        assert ha2.pilgrim_phase == "out"
        snap = sim2.save_state()
        ref = [sim2.step() for _ in range(100)]
        sim3 = Simulation(World(width=10, height=10, seed=1))
        sim3.load_state(snap)
        rep = [sim3.step() for _ in range(100)]
        assert ref == rep, "replay divergent en mission-or"
    finally:
        S.PILGRIM_WOOD_MIN = old_min

    # replay avec mineur chargé (rig test 1)
    from engine.simulation import GOLD_RESTOCK_THRESHOLD
    sim4, humans4, churches4 = _c2_rig([(3, GOLD_RESTOCK_THRESHOLD - 1, 0)])
    w4 = sim4.world; ch4 = churches4[0]
    vx, vy = int(ch4.x) + 3, int(ch4.y)
    w4._gold_mask[vy, vx] = True; w4.gold_grid[vy, vx] = 100.0
    h4 = humans4[0]
    for _ in range(300):
        sim4.step()
        if h4.gold > 0:
            break
    assert h4.gold > 0, "mineur jamais chargé"
    snap4 = sim4.save_state()
    ref4 = [sim4.step() for _ in range(80)]
    sim5 = Simulation(World(width=10, height=10, seed=1))
    sim5.load_state(snap4)
    rep4 = [sim5.step() for _ in range(80)]
    assert ref4 == rep4, "replay divergent avec or porté"
    print("  test_c2_dest_ruined_gold_recredit_and_replay OK "
          "(re-crédit trésor + replays or en mission et en mine)")


def test_k_chronicle_records_and_persists():
    """Bloc K : les annales enregistrent les jalons (dérivées des tick_events, sans
    toucher la sortie de step() → guard intact), dédupliquent les « premières fois »
    par clan, et survivent au save/load."""
    import tempfile
    from engine.simulation import Clan, AGE_SCIENCE_THRESHOLDS
    world = World(width=60, height=45, seed=7)
    sim = Simulation(world)
    lx, ly = _find_land_tile(world)
    h = spawn(EntityType.HUMAN, lx, ly, Sex.MALE)
    h.clan_id = 0
    sim.entities = [h]
    sim.clans = [Clan(id=0, cx=float(lx), cy=float(ly), color="#fff", chief_id=h.id)]
    sim.buildings = [Building(id=1, clan_id=0, x=lx, y=ly, btype="house")]

    sim.clans[0].science = AGE_SCIENCE_THRESHOLDS[1] - 0.01
    sim.step()   # franchit l'Âge de Pierre → entrée d'annales
    assert any(c["kind"] == "age" for c in sim.chronicle), \
        f"le passage d'âge n'est pas dans les annales: {sim.chronicle}"

    path = tempfile.mktemp(suffix=".json")
    try:
        sim.save(path)
        sim2 = Simulation(World(width=10, height=10, seed=1))
        sim2.load(path)
        assert sim2.chronicle == sim.chronicle, "annales perdues au load"
        assert sim2._chronicle_seen == sim._chronicle_seen, "jalons 'première fois' perdus au load"
    finally:
        if os.path.exists(path):
            os.remove(path)
    print(f"  test_k_chronicle_records_and_persists OK "
          f"({len(sim.chronicle)} entrée(s), round-trip identique)")


def test_p6_f3_trails_grid():
    """P6 F3 : la grille de sentiers s'incrémente sous les pas humains, décroît périodiquement
    (plancher 0), sature sans wrap, survit au save/load, et reste HORS du payload step()."""
    import numpy as np
    from engine.simulation import TRAIL_DECAY_PERIOD, TRAIL_MAX
    sim = Simulation(World(width=60, height=45, seed=7)); sim.populate()
    g = sim.world.trail_grid
    assert g.dtype == np.uint16 and g.shape == (45, 60), "grille uint16 H×W"
    assert int(g.sum()) == 0, "monde neuf → sentiers vierges"
    data = sim.step()
    assert "trail" not in str(data.keys()), "la grille N'ENTRE PAS dans le payload step (hors hash)"
    for _ in range(200):
        sim.step()
    assert int(sim.world.trail_grid.sum()) > 0, "les pas humains marquent le sol"
    # décroissance : −1 plancher 0, jamais de valeur négative ni de wrap
    sim.world.trail_grid[0, 0] = 5; sim.world.trail_grid[0, 1] = 0
    sim.tick_count = TRAIL_DECAY_PERIOD - 1      # le prochain step tombe sur la période
    sim.step()
    assert sim.world.trail_grid[0, 1] == 0, "plancher 0 (pas de wrap uint16)"
    # saturation : une tuile au max ne déborde pas
    sim.world.trail_grid[2, 2] = TRAIL_MAX
    for _ in range(3):
        sim.step()
    assert sim.world.trail_grid[2, 2] >= TRAIL_MAX - 1, "saturation sans wrap à 0"
    # save/load : grille non dérivable → sérialisée
    sim.world.trail_grid[5, 5] = 1234
    sim2 = Simulation(World(width=60, height=45, seed=7)); sim2.load_state(sim.save_state())
    assert int(sim2.world.trail_grid[5, 5]) == 1234, "sentiers restaurés au load"
    # vieux save (grille absente) → sentiers vierges, pas de crash
    st = sim.save_state(); st["world"].pop("trail_grid", None)
    sim3 = Simulation(World(width=60, height=45, seed=7)); sim3.load_state(st)
    assert int(sim3.world.trail_grid.sum()) == 0, "vieux save → grille vierge"
    print("  test_p6_f3_trails_grid OK (incrément, décroissance plancher 0, saturation, save/load, hors payload)")


def test_harden_load_state_transactional():
    """I1 durcissement persistance : un load_state qui ÉCHOUE (save corrompu) laisse
    la sim vivante ET les RNG globaux INTACTS. Avant, load_state mutait self champ par
    champ → une erreur en cours laissait un world neuf collé à de vieux clans + RNG
    perturbé (état hybride corrompu, replay cassé)."""
    import random as _r
    import numpy as _np
    import copy as _copy
    sim = Simulation(World(width=60, height=45, seed=3))
    sim.populate()
    for _ in range(30):
        sim.step()
    good = sim.save_state()
    w0, t0, n0, eid0 = sim.world, sim.tick_count, len(sim.entities), id(sim.entities)
    py_before = _r.getstate()
    np_before = _np.random.get_state()

    bad = dict(good); del bad["np_random_state"]   # save corrompu → erreur en phase 1
    raised = False
    try:
        sim.load_state(bad)
    except KeyError:
        raised = True
    assert raised, "load_state aurait dû lever sur un save corrompu"

    # état vivant intact (identité d'objet → aucune mutation partielle)
    assert sim.world is w0, "world muté malgré l'échec du load"
    assert sim.tick_count == t0, "tick muté malgré l'échec"
    assert len(sim.entities) == n0 and id(sim.entities) == eid0, "entities mutées malgré l'échec"
    # RNG globaux restaurés → reprise exacte préservée
    py_after = _r.getstate()
    np_after = _np.random.get_state()
    assert py_after == py_before, "RNG python perturbé par un load raté"
    assert (np_after[1].tolist() == np_before[1].tolist() and np_after[2] == np_before[2]), \
        "RNG numpy perturbé par un load raté"

    # F2 (gate) : état RNG de mauvaise longueur → rejet en PHASE 1 (dry-run setstate sur un
    # générateur jetable) → pas de commit partiel. Avant, setstate levait en phase 2 (après
    # le remplacement de self) = RNG hybride.
    bad_rng = _copy.deepcopy(good)
    bad_rng["py_random_state"] = [3, [1, 2, 3], None]
    raised = False
    try:
        sim.load_state(bad_rng)
    except (ValueError, TypeError):
        raised = True
    assert raised, "état RNG invalide non rejeté (F2)"
    assert sim.world is w0 and sim.tick_count == t0, "sim mutée par un RNG invalide (F2)"

    # F3 (gate) : compteur d'id non entier → rejet en phase 1 (sinon TypeError différé au 1er spawn)
    bad_ctr = _copy.deepcopy(good)
    bad_ctr["entity_id_counter"] = "1000"
    raised = False
    try:
        sim.load_state(bad_ctr)
    except ValueError:
        raised = True
    assert raised, "compteur d'id non entier non rejeté (F3)"
    assert sim.world is w0 and sim.tick_count == t0, "sim mutée par un compteur invalide (F3)"

    # un load VALIDE marche toujours (non-régression)
    sim.load_state(good)
    assert sim.tick_count == t0
    print("  test_harden_load_state_transactional OK (sim + RNG intacts ; F2/F3 rejetés)")


def test_harden_from_state_bounds():
    """I2 durcissement (audit #16) : un save aux dimensions aberrantes lève PROPREMENT
    au chargement au lieu de saturer la RAM (OOM au boot = crash-loop LOAD_ON_START)."""
    import copy as _copy
    sim = Simulation(World(width=50, height=40, seed=1))
    sim.populate()
    good = sim.save_state()
    bad = _copy.deepcopy(good)
    bad["world"]["width"] = 10 ** 9          # dimension géante
    raised = False
    try:
        Simulation(World(width=10, height=10, seed=1)).load_state(bad)
    except ValueError:
        raised = True
    assert raised, "dimensions géantes non rejetées (risque OOM)"
    print("  test_harden_from_state_bounds OK (dims aberrantes rejetées)")


def test_harden_load_rejects_nan():
    """I2 durcissement (audit #64) : json.load accepte NaN/Infinity par défaut → un save
    empoisonné passerait puis crasherait step() plus tard. On rejette au parsing."""
    import tempfile, os as _os
    path = tempfile.mktemp(suffix=".json")
    try:
        with open(path, "w") as f:
            f.write('{"tick_count": NaN, "world": {}}')
        raised = False
        try:
            Simulation(World(width=10, height=10, seed=1)).load(path)
        except ValueError:
            raised = True
        assert raised, "NaN non rejeté au chargement"
    finally:
        if _os.path.exists(path):
            _os.remove(path)
    print("  test_harden_load_rejects_nan OK (NaN rejeté au parsing)")


def test_harden_save_state_chronicle_copy():
    """I5 durcissement (audit #89/#101) : save_state COPIE le chronicle → une mutation
    après le snapshot ne le change pas (plus d'aliasing de la liste vivante, qui déchirait
    le snapshot quand la sérialisation JSON avait lieu hors state_lock)."""
    sim = Simulation(World(width=40, height=30, seed=5))
    sim.populate()
    sim.chronicle.append({"kind": "test", "tick": 1, "msg": "avant"})
    snap = sim.save_state()
    n_before = len(snap["chronicle"])
    sim.chronicle.append({"kind": "test", "tick": 2, "msg": "apres"})   # mutation post-snapshot
    sim.chronicle[0]["msg"] = "modifie"                                  # mutation d'un event
    assert len(snap["chronicle"]) == n_before, "liste du snapshot aliasée (allongée)"
    assert snap["chronicle"][0]["msg"] == "avant", "event du snapshot aliasé (muté)"
    print("  test_harden_save_state_chronicle_copy OK (snapshot isolé des mutations)")


def test_harden_entity_traits_copy():
    """F1 durcissement (gate Arceus) : le dict traits est COPIÉ aux deux bouts de la
    persistance. to_state() ne doit pas aliaser self.traits (muté en place par le clamp
    anti-dérive à chaque tick → snapshot déchiré pendant le json.dump hors state_lock,
    même classe que le chronicle I5). from_state() ne doit pas aliaser le dict de la save
    (deux entités rechargées partageraient sinon le même dict traits)."""
    from engine.entities import Entity, EntityType, Sex
    e = spawn(EntityType.HUMAN, 5.0, 5.0, Sex.MALE)
    e.clan_id = 0
    snap = e.to_state()
    speed_before = snap["traits"]["speed"]
    e.traits["speed"] += 1.0                         # mutation en place (comme le clamp)
    assert snap["traits"]["speed"] == speed_before, "to_state a aliasé self.traits (F1)"

    # from_state : deux entités bâties depuis le MÊME dict ne partagent pas leurs traits
    d = e.to_state()
    a = Entity.from_state(d)
    b = Entity.from_state(d)
    a.traits["speed"] += 5.0
    assert b.traits["speed"] != a.traits["speed"], "from_state partage le dict traits (F1)"
    assert d["traits"]["speed"] != a.traits["speed"], "from_state aliase le dict de la save (F1)"
    print("  test_harden_entity_traits_copy OK (traits copiés to_state + from_state)")


def test_infinite_run(ticks: int = 1500, seed: int = 424242):
    """J3 durcissement (audit #6) : le SEUL test automatisé de l'invariant #1
    (« tourne à l'infini »). Run headless moyen avec assertions à échantillons :
      - 0 exception (le monde ne crashe pas) ;
      - AUCUNE espèce TERRESTRE éteinte (les 7 hors eau restent >= 1 ; requins/poissons
        aquatiques exclus = fluctuation préexistante hors scope, cf. bloc E) ;
      - pas d'explosion (total <= 9 espèces x MAX_PER_SPECIES) ;
      - le monde reste vivable (arbres debout > 0) ;
      - le monde est DYNAMIQUE (pop finale > pop initiale → ni gelé ni effondré).
    L'endurance réelle (20k+) reste détachée (wake-on-exit) pour les gros changements ;
    ici on veut un filet rapide qui casse le build si l'invariant #1 saute franchement.

    NON-FLAKY malgré l'exécution IN-PROCESS (pas de sous-processus, contrairement aux
    goldens) : les ids d'entités sont pollués par les tests d'avant (compteur _next_id
    global), MAIS (a) les assertions ne portent QUE sur des comptes (pop, espèces, arbres),
    jamais sur des valeurs d'id ; (b) la trajectoire de pop est invariante à ce décalage —
    les usages comportementaux de l'id sont des tie-breaks d'ORDRE (min(…, (dist, e.id)))
    préservés par un décalage uniforme, et World(seed) re-seede les RNG. ⚠ Si un futur
    moteur rendait le comportement dépendant de la VALEUR d'un id (pas juste de l'ordre),
    cette invariance tomberait → il faudrait alors isoler ce test en sous-processus."""
    from engine.world import TREE_STUMP_THRESHOLD
    from engine.entities import EntityType
    aquatic = {EntityType.FISH.value, EntityType.SHARK.value}
    terrestrial = [t.value for t in EntityType if t.value not in aquatic]
    ceiling = len(list(EntityType)) * MAX_PER_SPECIES

    world = World(width=140, height=100, seed=seed)
    sim = Simulation(world)
    sim.populate()
    pop0 = len(sim.entities)

    last = None
    for i in range(ticks):
        sim.step()                                   # une exception ici = échec du test
        if i % 50 == 0:
            st = sim._compute_stats()["populations"]
            for sp in terrestrial:
                assert st[sp] >= 1, f"espèce terrestre {sp} ÉTEINTE au tick {i} (invariant #1)"
            total = sum(st.values())
            assert total <= ceiling, f"explosion pop {total} > {ceiling} au tick {i}"
            standing = int(((world.tree_grid >= TREE_STUMP_THRESHOLD)
                            & world._forest_mask).sum())
            assert standing > 0, f"plus aucun arbre debout au tick {i} (monde stérile)"
            last = total
    assert last is not None and last > pop0, \
        f"monde gelé/effondré : pop finale {last} <= initiale {pop0}"
    print(f"  test_infinite_run OK ({ticks} ticks, pop {pop0}→{last}, "
          f"7 espèces terrestres vivantes, 0 crash)")


def test_determinism_golden():
    """J1 durcissement (audit #4) : l'invariant #2 (déterminisme seedé) devient une
    FAILLE DE TEST, plus un print à l'œil non automatisé. Exécute le guard en
    SOUS-PROCESSUS (process frais → _next_id=0, exactement le contexte sous lequel le
    golden a été produit) et exige exit 0 (hash == golden versionné dans le guard)."""
    import subprocess
    guard = os.path.join(os.path.dirname(os.path.abspath(__file__)), "determinism_guard.py")
    r = subprocess.run([sys.executable, guard], capture_output=True, text=True)
    assert r.returncode == 0, (
        "golden déterministe cassé (invariant #2) :\n"
        f"  stdout: {r.stdout.strip()}\n  stderr: {r.stderr.strip()}")
    print(f"  test_determinism_golden OK ({r.stdout.strip()})")


def test_determinism_civ_golden():
    """J2 durcissement (audit #5/#18/#25/#31) : golden CIV (Âge Acier) sous le gate
    déterministe → forge / marché / église / or / commerce / pèlerinage (les ~2000
    lignes de moteur qu'un run Âge-Bois n'atteint jamais) deviennent REPRODUCTIBLES.
    Exécute le guard `--civ` en SOUS-PROCESSUS (process frais) et exige exit 0."""
    import subprocess
    guard = os.path.join(os.path.dirname(os.path.abspath(__file__)), "determinism_guard.py")
    r = subprocess.run([sys.executable, guard, "--civ"], capture_output=True, text=True)
    assert r.returncode == 0, (
        "golden CIV cassé (invariant #2, systèmes avancés) :\n"
        f"  stdout: {r.stdout.strip()}\n  stderr: {r.stderr.strip()}")
    print(f"  test_determinism_civ_golden OK ({r.stdout.strip()})")


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


def _land_tile(world):
    for y in range(world.height):
        for x in range(world.width):
            if world.is_walkable(x, y, False):
                return x, y
    raise RuntimeError("no land")


def test_p3_tribute_and_surplus_lost():
    """P3 §2 : le perdant verse 20% entier de chaque ressource (caps respectés, surplus perdu)."""
    from engine.simulation import Clan, FORGE_MAX_IRON
    w = World(width=60, height=45, seed=7); sim = Simulation(w)
    lx, ly = _land_tile(w)
    xh = Building(id=1, clan_id=0, x=lx, y=ly, btype="house")
    xf = Building(id=2, clan_id=0, x=lx, y=ly, btype="forge")
    xm = Building(id=3, clan_id=0, x=lx, y=ly, btype="mill")
    yh = Building(id=4, clan_id=1, x=lx, y=ly, btype="house", wood=100, stone=50)
    yf = Building(id=5, clan_id=1, x=lx, y=ly, btype="forge", iron=20)
    ym = Building(id=6, clan_id=1, x=lx, y=ly, btype="mill", bread=5)
    sim.buildings = [xh, xf, xm, yh, yf, ym]
    cb = {0: {"house": [xh], "forge": [xf], "mill": [xm]},
          1: {"house": [yh], "forge": [yf], "mill": [ym]}}
    ev = []
    sim._tribute(0, 1, cb, ev)
    assert (yh.wood, yh.stone, yf.iron, ym.bread) == (80, 40, 16, 4)
    assert (xh.wood, xh.stone, xf.iron, xm.bread) == (20, 10, 4, 1)
    t = [e for e in ev if e["type"] == "clan_tribute"]
    assert t and t[0] == {"type": "clan_tribute", "from_clan": 1, "to_clan": 0,
                          "wood": 20, "stone": 10, "iron": 4, "bread": 1}, t
    # surplus perdu : forge X déjà quasi-pleine → seul ce qui rentre est crédité, event=prélevé
    xf2 = Building(id=7, clan_id=0, x=lx, y=ly, btype="forge", iron=FORGE_MAX_IRON - 1)
    yf2 = Building(id=8, clan_id=1, x=lx, y=ly, btype="forge", iron=20)
    sim.buildings = [xf2, yf2]
    ev2 = []
    sim._tribute(0, 1, {0: {"forge": [xf2]}, 1: {"forge": [yf2]}}, ev2)
    assert yf2.iron == 16 and xf2.iron == FORGE_MAX_IRON  # 3 perdus
    assert [e for e in ev2 if e["type"] == "clan_tribute"][0]["iron"] == 4
    print("  test_p3_tribute_and_surplus_lost OK (20% entier, caps, surplus perdu, event=prélevé)")


def test_p3_conquest_absorption():
    """P3 §3 : conquête — membres+bâtiments durables → X, feu+chantiers effacés, Y retiré, purge."""
    from engine.simulation import Clan
    w = World(width=60, height=45, seed=7); sim = Simulation(w)
    lx, ly = _land_tile(w)
    xs = [spawn(EntityType.HUMAN, lx, ly, Sex.MALE) for _ in range(5)]
    ys = [spawn(EntityType.HUMAN, lx, ly, Sex.MALE) for _ in range(3)]
    for e in xs: e.clan_id = 0
    for e in ys: e.clan_id = 1
    sim.entities = xs + ys
    xh = Building(id=1, clan_id=0, x=lx, y=ly, btype="house")
    yh = Building(id=2, clan_id=1, x=lx, y=ly, btype="house")
    yfire = Building(id=3, clan_id=1, x=lx, y=ly, btype="campfire")
    ysite = Building(id=4, clan_id=1, x=lx, y=ly, btype="site_mill")
    sim.buildings = [xh, yh, yfire, ysite]
    sim.clans = [Clan(id=0, cx=float(lx), cy=float(ly), color="#f00", chief_id=xs[0].id),
                 Clan(id=1, cx=float(lx), cy=float(ly), color="#00f", chief_id=ys[0].id)]
    sim.relations = {(0, 1): -50}; sim._rival_state = {(0, 1)}; sim._ally_state = set()
    ev = []
    sim._absorb_clan(0, 1, ev)
    assert all(e.clan_id == 0 for e in ys), "membres de Y → X"
    assert yh.clan_id == 0 and yh in sim.buildings, "maison de Y → X"
    assert yfire not in sim.buildings and ysite not in sim.buildings, "feu+chantier effacés"
    assert not any(c.id == 1 for c in sim.clans), "clan Y retiré"
    assert sim.relations == {} and sim._rival_state == set(), "relations de Y purgées"
    a = [e for e in ev if e["type"] == "clan_absorbed"]
    assert a and a[0] == {"type": "clan_absorbed", "clan_id": 1, "by": 0, "members": 3}, a
    print("  test_p3_conquest_absorption OK (membres+maison→X, feu/chantier effacés, Y retiré, purge)")


def test_p3_purge_e8_and_save_counters():
    """P3 fix E8 : purge des clés relations d'un clan mort ; save/load des compteurs de guerre."""
    from engine.simulation import Clan
    w = World(width=60, height=45, seed=7); sim = Simulation(w)
    sim.relations = {(0, 1): -50, (1, 2): 30, (0, 2): 45}
    sim._ally_state = {(0, 2)}; sim._rival_state = {(0, 1)}
    sim._purge_clan_relations(1)
    assert sim.relations == {(0, 2): 45}, sim.relations
    assert sim._ally_state == {(0, 2)} and sim._rival_state == set()
    # save/load compteurs + relations
    sim2 = Simulation(World(width=60, height=45, seed=7)); sim2.populate()
    sim2.clans[0].war_kills_for = 7; sim2.clans[0].war_kills_against = 3
    sim2.relations = {(0, 1): -55}
    sim3 = Simulation(World(width=60, height=45, seed=7))
    sim3.load_state(sim2.save_state())
    c0 = next(c for c in sim3.clans if c.id == 0)
    assert (c0.war_kills_for, c0.war_kills_against) == (7, 3), (c0.war_kills_for, c0.war_kills_against)
    assert sim3.relations.get((0, 1)) == -55
    print("  test_p3_purge_e8_and_save_counters OK (purge E8 + compteurs/relations round-trip)")


def _adult_age(frac=0.5):
    return SPECS[EntityType.HUMAN].max_age * frac


def test_p4_coup():
    """P4 §2.2 : le + jeune adulte non-chef renverse le chef, tension −40 ; requiert un challenger."""
    from engine.simulation import Clan
    w = World(width=60, height=45, seed=7); sim = Simulation(w)
    lx, ly = _land_tile(w)
    chief = spawn(EntityType.HUMAN, lx, ly, Sex.MALE); chief.age = _adult_age(0.6)
    young = spawn(EntityType.HUMAN, lx, ly, Sex.MALE); young.age = _adult_age(0.25)
    old   = spawn(EntityType.HUMAN, lx, ly, Sex.MALE); old.age = _adult_age(0.8)
    for e in (chief, young, old): e.clan_id = 0
    sim.entities = [chief, young, old]
    c = Clan(id=0, cx=float(lx), cy=float(ly), color="#f00", chief_id=chief.id); c.tension = 75
    sim.clans = [c]
    ev = []
    sim._coup(c, ev)
    assert c.chief_id == young.id and c.tension == 35, (c.chief_id, c.tension)
    assert [e for e in ev if e["type"] == "clan_coup"] == [{"type": "clan_coup", "clan_id": 0, "chief_id": young.id}]
    # sans challenger adulte non-chef → rien
    c2 = Clan(id=1, cx=float(lx), cy=float(ly), color="#00f", chief_id=chief.id); c2.tension = 80
    baby = spawn(EntityType.HUMAN, lx, ly, Sex.MALE); baby.age = _adult_age(0.05); baby.clan_id = 1
    sim.entities = [chief, baby]; chief.clan_id = 1
    sim.clans = [c2]
    ev2 = []
    sim._coup(c2, ev2)
    assert c2.chief_id == chief.id and c2.tension == 80 and not ev2, "0 adulte non-chef → pas de coup"
    print("  test_p4_coup OK (jeune loup 75→35 + event ; sans challenger → rien)")


def test_p4_rebel_split():
    """P4 §2.1 : K=pop//3 les + éloignés du feu font sécession, leader=chef, rel −50 rival, tension −60."""
    from engine.simulation import Clan, N_CLANS, CLAN_COLORS, REL_D_REBELLION
    w = World(width=120, height=90, seed=7); sim = Simulation(w)
    cx, cy = 20.0, 20.0
    ents = []
    for i in range(8):
        e = spawn(EntityType.HUMAN, int(cx) + (i % 2), int(cy) + (i // 2), Sex.MALE)
        e.age = _adult_age(0.5); e.clan_id = 0; ents.append(e)
    far = []
    for i in range(4):
        e = spawn(EntityType.HUMAN, int(cx) + 20 + i * 5, int(cy) + 20, Sex.MALE)
        e.age = _adult_age(0.5); e.clan_id = 0; ents.append(e); far.append(e)
    sim.entities = ents
    mother = Clan(id=0, cx=cx, cy=cy, color="#f00", chief_id=ents[0].id); mother.tension = 95
    sim.clans = [mother]; sim._next_clan_id = N_CLANS
    ev = []
    sim._rebel_split(0, ev)
    nc = next(c for c in sim.clans if c.id == N_CLANS)
    seceded = [e for e in ents if e.clan_id == nc.id]
    assert len(seceded) == 4 and set(seceded) == set(far), "K=4 = les 4 plus éloignés"
    assert nc.chief_id == far[-1].id and nc.color == CLAN_COLORS[N_CLANS % 4]
    assert sim.relations.get((0, nc.id)) == REL_D_REBELLION and (0, nc.id) in sim._rival_state
    assert mother.tension == 35
    assert any(b.btype == "campfire" and b.clan_id == nc.id for b in sim.buildings)
    r = [e for e in ev if e["type"] == "clan_rebellion"]
    assert r == [{"type": "clan_rebellion", "clan_id": 0, "new_clan": nc.id, "members": 4, "chief_id": far[-1].id}]
    print("  test_p4_rebel_split OK (K=4 éloignés, leader chef, rel −50 rival, tension −60, feu, event)")


def test_p4_found_clan_deterministic():
    """P4 §3 : _found_clan sans RNG (2 fondations → mêmes id/couleur/position/compteur)."""
    from engine.simulation import Clan, N_CLANS, CLAN_COLORS
    outs = []
    for _ in range(2):
        sim = Simulation(World(width=60, height=45, seed=7))
        lx, ly = _land_tile(sim.world)
        leader = spawn(EntityType.HUMAN, lx, ly, Sex.MALE); leader.clan_id = 0
        m2 = spawn(EntityType.HUMAN, lx, ly, Sex.MALE); m2.clan_id = 0
        sim.entities = [leader, m2]
        sim.clans = [Clan(id=0, cx=0.0, cy=0.0, color="#f00", chief_id=leader.id)]
        sim._next_clan_id = N_CLANS
        nc = sim._found_clan(leader, [leader, m2], 5, 7)
        outs.append((nc.id, nc.color, nc.cx, nc.cy, sim._next_clan_id))
    assert outs[0] == outs[1] == (N_CLANS, CLAN_COLORS[N_CLANS % 4], 5.0, 7.0, N_CLANS + 1), outs
    print("  test_p4_found_clan_deterministic OK (id/couleur/position stables, compteur ++)")


def test_p5_cult_founding_and_names_deterministic():
    """P5 E1 : populate fonde 1 culte/clan ; noms déterministes (2 process → mêmes noms)."""
    from engine.simulation import _cult_name
    sim = Simulation(World(width=90, height=70, seed=333)); sim.populate()
    assert all(c.cult_id >= 0 for c in sim.clans), "chaque clan fonde son culte"
    assert len(set(c.cult_id for c in sim.clans)) == len(sim.clans), "cultes initiaux distincts"
    # nom déterministe = pure fonction (seed, cult_id)
    assert sim.cults[0].name == _cult_name(sim.world.seed, 0)
    sim2 = Simulation(World(width=90, height=70, seed=333)); sim2.populate()
    assert [sim.cults[i].name for i in sim.cults] == [sim2.cults[i].name for i in sim2.cults]
    print("  test_p5_cult_founding_and_names_deterministic OK (1 culte/clan, noms stables)")


def test_p5_conversion_and_schism():
    """P5 E1 : conversion (petit → culte du gros, rel≥60, verrou 1×) ; schisme (rival du gardien)."""
    from engine.simulation import Clan, Cult, CULT_CONVERT_REL, _rel_key
    sim = Simulation(World(width=90, height=70, seed=7))
    # 2 clans, cultes distincts, A petit (pop 3) / B gros (pop 8), rel 60
    sim.clans = [Clan(id=0, cx=10.0, cy=10.0, color="#f00", chief_id=0, cult_id=0),
                 Clan(id=1, cx=30.0, cy=10.0, color="#00f", chief_id=1, cult_id=1)]
    sim.cults = {0: Cult(0, "Foi A", 0, 0), 1: Cult(1, "Foi B", 1, 0)}; sim._next_cult_id = 2
    sim.relations = {(0, 1): CULT_CONVERT_REL}; sim._ally_state = {(0, 1)}
    pop = {0: 3, 1: 8}
    ev = []
    sim._update_cults([sim.clans[0]], pop, ev)   # A(petit) due
    assert sim.clans[0].cult_id == 1 and sim.clans[0].cult_converted, "A converti au culte de B"
    assert [e for e in ev if e["type"] == "cult_converted"], "event conversion"
    # rel 59 → pas de conversion (nouveau clan C)
    sim.clans.append(Clan(id=2, cx=50.0, cy=10.0, color="#0f0", chief_id=2, cult_id=1))
    sim.relations[(1, 2)] = 59
    ev2 = []; sim._update_cults([sim.clans[2]], {0: 3, 1: 8, 2: 2}, ev2)
    assert not [e for e in ev2 if e["type"] == "cult_converted"], "rel 59 < 60 → pas de conversion"
    # SCHISME : D rival du gardien de son culte
    sim.clans.append(Clan(id=3, cx=70.0, cy=10.0, color="#ff0", chief_id=3, cult_id=1))
    sim._rival_state = {(1, 3)}   # D rival du fondateur/gardien (clan 1) de son culte
    ev3 = []; sim._update_cults([sim.clans[3]], {0: 3, 1: 8, 2: 2, 3: 5}, ev3)
    s = [e for e in ev3 if e["type"] == "cult_schism"]
    assert s and sim.clans[3].cult_id not in (1,) and not sim.clans[3].cult_converted, "schisme : D fonde son culte, verrou ré-armé"
    print("  test_p5_conversion_and_schism OK (conversion verrou, seuil 60, schisme rival du gardien)")


def test_p5_guardian_and_save_load():
    """P5 E1 : gardien dérivé (fondateur vivant sinon +peuplé) ; cults round-trip + vieux save refondé."""
    from engine.simulation import Clan, Cult
    sim = Simulation(World(width=90, height=70, seed=7))
    sim.clans = [Clan(id=0, cx=0., cy=0., color="#f00", chief_id=0, cult_id=5),
                 Clan(id=1, cx=0., cy=0., color="#00f", chief_id=1, cult_id=5)]
    sim.cults = {5: Cult(5, "X", founder_clan=0, founded_tick=0)}
    assert sim._cult_guardian(5, {0: 2, 1: 9}) == 0, "fondateur vivant = gardien"
    # fondateur (clan 0) éteint → le + peuplé (clan 1)
    sim.clans = [c for c in sim.clans if c.id != 0]
    assert sim._cult_guardian(5, {1: 9}) is None, "1 seul clan du culte → pas de gardien (pas de schisme)"
    # save/load round-trip du registre
    sim2 = Simulation(World(width=90, height=70, seed=7)); sim2.populate()
    sim2.cults[0].name = "Culte Test"; sim2._next_cult_id = 42
    sim3 = Simulation(World(width=90, height=70, seed=7)); sim3.load_state(sim2.save_state())
    assert sim3.cults[0].name == "Culte Test" and sim3._next_cult_id == 42
    # vieux save (sans cults) → refondation à froid déterministe
    st = sim2.save_state(); del st["cults"]; del st["next_cult_id"]
    for c in st["clans"]: c.pop("cult_id", None); c.pop("cult_converted", None)
    sim4 = Simulation(World(width=90, height=70, seed=7)); sim4.load_state(st)
    assert len(sim4.cults) == len(sim4.clans) and all(c.cult_id >= 0 for c in sim4.clans)
    print("  test_p5_guardian_and_save_load OK (gardien dérivé, cults round-trip, vieux save refondé)")


def test_p5_feast_trigger_and_bounds():
    """P5 E2 : la fête se déclenche au 1er tick d'automne (clan en paix, ≥4 champs de blé MÛRS),
    ne consomme RIEN, 1×/an ; guerre/famine l'interrompt ; <4 mûrs ou hors automne → rien ; déterministe."""
    from engine.simulation import (Clan, Building, get_season, TICKS_PER_SEASON,
                                    FEAST_TICKS, FEAST_FIELDS_MIN)
    assert FEAST_FIELDS_MIN == 4
    autumn = 2 * TICKS_PER_SEASON            # 1er tick d'automne (SEASON_NAMES index 2)
    assert get_season(autumn) == "autumn" and get_season(autumn - 1) != "autumn"
    year = autumn // (4 * TICKS_PER_SEASON)

    def _fields(n_ripe, n_unripe=0):
        bs = [Building(id=i, clan_id=0, x=i, y=0, btype="wheatfield", stage=4) for i in range(n_ripe)]
        bs += [Building(id=100 + i, clan_id=0, x=i, y=1, btype="wheatfield", stage=2) for i in range(n_unripe)]
        return {0: {"wheatfield": bs}}

    sim = Simulation(World(width=90, height=70, seed=7))
    sim.clans = [Clan(id=0, cx=20.0, cy=20.0, color="#f00", chief_id=0)]   # mode="peace" par défaut
    sim.tick_count = autumn
    cb = _fields(4, n_unripe=1)              # 4 mûrs (+1 vert) = seuil pile
    ev = []; sim._update_feasts(cb, ev)
    assert sim.clans[0].feast_ticks == FEAST_TICKS, "fête lancée pour FEAST_TICKS"
    assert sim.clans[0].feast_year == year
    fe = [e for e in ev if e["type"] == "feast_start"]
    assert fe and fe[0].get("fields") == 4, "event feast_start avec le compte de champs mûrs"
    # ne CONSOMME RIEN : les champs restent stage 4
    assert all(f.stage == 4 for f in cb[0]["wheatfield"] if f.id < 100), "champs mûrs intacts (rien consommé)"
    # 1×/an : re-appel au même tick → décompte seulement, PAS de 2e fête
    ev2 = []; sim._update_feasts(cb, ev2)
    assert not [e for e in ev2 if e["type"] == "feast_start"], "pas de 2e fête la même année"
    assert sim.clans[0].feast_ticks == FEAST_TICKS - 1, "décompte d'1 tick"
    # guerre interrompt la fête (feast_ticks → 0, silencieux)
    sim.clans[0].mode = "war"
    ev3 = []; sim._update_feasts(cb, ev3)
    assert sim.clans[0].feast_ticks == 0 and not [e for e in ev3 if e["type"] == "feast_start"], "guerre interrompt (silence)"
    # helper de déclenchement isolé
    def _mk(n_ripe, tick):
        s = Simulation(World(width=90, height=70, seed=7))
        s.clans = [Clan(id=0, cx=20.0, cy=20.0, color="#f00", chief_id=0)]
        s.tick_count = tick
        e = []; s._update_feasts(_fields(n_ripe), e)
        return s.clans[0].feast_ticks, s.clans[0].feast_year, [x["type"] for x in e]
    assert _mk(3, autumn) == (0, -1, []), "3 mûrs < 4 → pas de fête"
    # hors automne (printemps, tick 0) → rien même avec la moisson
    assert _mk(9, 0) == (0, -1, []), "hors automne → pas de fête"
    # déterminisme : 2 setups identiques → résultat identique
    assert _mk(6, autumn) == _mk(6, autumn), "déclenchement déterministe"
    print("  test_p5_feast_trigger_and_bounds OK (auto+paix+≥4 champs mûrs → fête 1×/an, rien consommé, guerre interrompt)")


def test_p5_feast_save_load():
    """P5 E2 : feast_ticks / feast_year survivent au save/load (via asdict / Clan(**c)) ; vieux save → défauts."""
    sim = Simulation(World(width=60, height=45, seed=7)); sim.populate()
    sim.clans[0].feast_ticks = 200; sim.clans[0].feast_year = 3
    sim2 = Simulation(World(width=60, height=45, seed=7)); sim2.load_state(sim.save_state())
    assert sim2.clans[0].feast_ticks == 200 and sim2.clans[0].feast_year == 3, "fête restaurée au load"
    # vieux save (sans feast_*) → défauts (0 / -1), pas de crash
    st = sim.save_state()
    for c in st["clans"]:
        c.pop("feast_ticks", None); c.pop("feast_year", None)
    sim3 = Simulation(World(width=60, height=45, seed=7)); sim3.load_state(st)
    assert sim3.clans[0].feast_ticks == 0 and sim3.clans[0].feast_year == -1, "vieux save → défauts"
    print("  test_p5_feast_save_load OK (feast_ticks/feast_year round-trip + défauts vieux save)")


def test_p5_monument_completion_and_ruin():
    """P5 E3 : un chantier monument terminé → bâtiment `monument` + tension −MONUMENT_TENSION +
    event monument_built{dedication} ; un monument d'un clan ÉTEINT → ruine durable ×MONUMENT_RUIN_MULT."""
    from engine.simulation import (Building, MONUMENT_TENSION, MONUMENT_RUIN_MULT,
                                    RUIN_LIFETIME)
    from engine.entities import EntityType
    # ── Complétion + effets ──
    sim = Simulation(World(width=90, height=70, seed=7)); sim.populate()
    c0 = sim.clans[0]; c0.tension = 80
    sim.buildings.append(Building(id=9000, clan_id=c0.id, x=int(c0.cx) + 3, y=int(c0.cy),
                                  btype="site_monument", work_done=999, work_needed=1,
                                  dedication="sa victoire sur le clan 2"))
    data = sim.step()
    mon = next((b for b in sim.buildings if b.id == 9000), None)
    assert mon is not None and mon.btype == "monument", "chantier promu en monument"
    mb = [e for e in data["events"] if e["type"] == "monument_built"]
    assert mb and mb[0]["dedication"] == "sa victoire sur le clan 2", "event + dédicace portée"
    assert c0.tension < 80, "tension apaisée à l'achèvement (−MONUMENT_TENSION)"
    # ── Ruine ×MULT d'un clan éteint ──
    sim2 = Simulation(World(width=90, height=70, seed=7)); sim2.populate()
    c = sim2.clans[0]
    sim2.buildings.append(Building(id=9001, clan_id=c.id, x=int(c.cx) + 2, y=int(c.cy), btype="monument"))
    for e in sim2.entities:                       # éteindre le clan (aucun humain vivant)
        if e.clan_id == c.id and e.etype == EntityType.HUMAN:
            e.alive = False
    sim2.step()
    ruin = next((b for b in sim2.buildings if b.id == 9001), None)
    assert ruin is not None and ruin.btype == "ruin", "monument du clan mort → ruine"
    assert ruin.ruin_ticks >= RUIN_LIFETIME * MONUMENT_RUIN_MULT - 1, "ruine de monument durable ×MULT"
    assert ruin.ruin_ticks > RUIN_LIFETIME, "nettement plus durable qu'une ruine ordinaire (×1)"
    print("  test_p5_monument_completion_and_ruin OK (promotion + tension + dédicace + ruine ×4)")


def test_p5_monument_save_load():
    """P5 E3 : Building.dedication + Clan.last_deed survivent au save/load ; vieux save → défauts."""
    from engine.simulation import Building
    sim = Simulation(World(width=60, height=45, seed=7)); sim.populate()
    sim.clans[0].last_deed = "sa victoire sur le clan 3"
    sim.buildings.append(Building(id=9002, clan_id=sim.clans[0].id, x=5, y=5,
                                  btype="monument", dedication="sa fondation"))
    sim2 = Simulation(World(width=60, height=45, seed=7)); sim2.load_state(sim.save_state())
    assert sim2.clans[0].last_deed == "sa victoire sur le clan 3", "last_deed restauré"
    mon = next((b for b in sim2.buildings if b.id == 9002), None)
    assert mon is not None and mon.dedication == "sa fondation", "dédicace restaurée"
    # vieux save (sans dedication / last_deed) → défauts
    st = sim.save_state()
    for b in st["buildings"]:
        b.pop("dedication", None)
    for c in st["clans"]:
        c.pop("last_deed", None)
    sim3 = Simulation(World(width=60, height=45, seed=7)); sim3.load_state(st)
    assert sim3.clans[0].last_deed == "sa fondation", "vieux save → last_deed défaut"
    assert all(b.dedication == "" for b in sim3.buildings), "vieux save → dedication défaut"
    print("  test_p5_monument_save_load OK (dedication/last_deed round-trip + défauts vieux save)")


def test_p5_hero_naming_and_wire():
    """P5 E4 : nomination (helper + voie founder via _found_clan), noms déterministes, wire `hero`, idempotence."""
    from engine.simulation import _name_hero, _hero_name
    from engine.entities import spawn, EntityType
    sim = Simulation(World(width=80, height=60, seed=42)); w = sim.world
    e = spawn(EntityType.HUMAN, 10, 10); e.clan_id = 0
    assert e.hero_name is None and "hero" not in e.to_dict(), "pas de nom → pas de clé wire"
    ev = []; _name_hero(e, "kills", w, ev)
    assert e.hero_name == _hero_name(w.seed, e.id, "kills"), "nom déterministe (voie kills)"
    assert [x for x in ev if x["type"] == "hero_named" and x["via"] == "kills"], "event hero_named"
    assert e.to_dict()["hero"] == e.hero_name, "wire `hero` présent une fois nommé"
    ev2 = []; _name_hero(e, "builds", w, ev2)     # déjà nommé → idempotent
    assert not ev2 and e.hero_name == _hero_name(w.seed, e.id, "kills"), "idempotent (déjà nommé)"
    # voie FONDATEUR (vrai chemin _found_clan)
    sim.populate()
    leader = next(x for x in sim.entities if x.etype == EntityType.HUMAN and x.hero_name is None)
    ev3 = []; sim._found_clan(leader, [leader], 20, 20, tick_events=ev3)
    assert leader.hero_name is not None, "chef fondateur nommé"
    assert [x for x in ev3 if x["type"] == "hero_named" and x["via"] == "founder"], "event founder"
    print("  test_p5_hero_naming_and_wire OK (kills/builds/founder, déterministe, wire, idempotent)")


def test_p5_hero_fallen_and_save_load():
    """P5 E4 : mort d'un héros → hero_fallen + chronique annals ; save/load compteurs+nom ; vieux save→défauts."""
    from engine.entities import EntityType
    sim = Simulation(World(width=80, height=60, seed=42)); sim.populate()
    hero = next(e for e in sim.entities if e.etype == EntityType.HUMAN)
    hero.hero_name = "Kargh le Sanglant"; hero.war_kills = 7
    hid = hero.id
    hero.alive = False
    data = sim.step()
    hf = [e for e in data["events"] if e["type"] == "hero_fallen"]
    assert hf and hf[0]["entity_id"] == hid and hf[0]["name"] == "Kargh le Sanglant", "event hero_fallen"
    assert any(c.get("cat") == "annals" and "Kargh" in c["msg"] for c in sim.chronicle), "chronique cat=annals"
    # save/load compteurs + nom
    sim2 = Simulation(World(width=80, height=60, seed=42)); sim2.populate()
    h2 = next(e for e in sim2.entities if e.etype == EntityType.HUMAN)
    h2.war_kills = 4; h2.built_count = 9; h2.hero_name = "Doru le Bâtisseur"
    sim3 = Simulation(World(width=80, height=60, seed=42)); sim3.load_state(sim2.save_state())
    r = next(e for e in sim3.entities if e.id == h2.id)
    assert r.war_kills == 4 and r.built_count == 9 and r.hero_name == "Doru le Bâtisseur", "compteurs+nom restaurés"
    # vieux save (sans les champs) → défauts
    st = sim2.save_state()
    for e in st["entities"]:
        e.pop("war_kills", None); e.pop("built_count", None); e.pop("hero_name", None)
    sim4 = Simulation(World(width=80, height=60, seed=42)); sim4.load_state(st)
    r4 = next(e for e in sim4.entities if e.id == h2.id)
    assert r4.war_kills == 0 and r4.built_count == 0 and r4.hero_name is None, "vieux save → défauts 0/0/None"
    print("  test_p5_hero_fallen_and_save_load OK (hero_fallen+annals, compteurs+nom round-trip, défauts)")


def test_p6_f1_money_dawn_and_save_load():
    """P6 F1 : money_dawn (flag + annale cat=annals) + save/load money_dawn/gold_dest ; vieux save → défauts."""
    from engine.entities import EntityType
    sim = Simulation(World(width=60, height=45, seed=7)); sim.populate()
    assert sim.money_dawn is False, "money_dawn faux au départ"
    # annale money_dawn depuis l'event
    sim._update_chronicle([{"type": "money_dawn"}])
    assert any(c.get("cat") == "annals" and "monnaie" in c["msg"] for c in sim.chronicle), "annale money_dawn cat=annals"
    # save/load : money_dawn (Simulation) + gold_dest/gold (Entity)
    sim.money_dawn = True
    e0 = next(x for x in sim.entities if x.etype == EntityType.HUMAN)
    e0.gold_dest = "market"; e0.gold = 3
    sim2 = Simulation(World(width=60, height=45, seed=7)); sim2.load_state(sim.save_state())
    assert sim2.money_dawn is True, "money_dawn restauré"
    r = next(x for x in sim2.entities if x.id == e0.id)
    assert r.gold_dest == "market" and r.gold == 3, "gold_dest + or portés restaurés"
    # vieux save (champs absents) → défauts
    st = sim.save_state(); st.pop("money_dawn", None)
    for e in st["entities"]:
        e.pop("gold_dest", None)
    sim3 = Simulation(World(width=60, height=45, seed=7)); sim3.load_state(st)
    assert sim3.money_dawn is False, "vieux save → money_dawn False"
    assert all(x.gold_dest == "church" for x in sim3.entities), "vieux save → gold_dest church"
    print("  test_p6_f1_money_dawn_and_save_load OK (money_dawn flag+annale, gold_dest round-trip, défauts)")


def test_p6_f1_gold_leak_refund():
    """P6 F1 (correctif fuite) : une pièce d'or INVENDUE (refus/timeout) est re-créditée au coffre
    du marché maison (cap → dorure église), repli au trésor si marché disparu ; _clear_trade nettoie
    cargo_gold en dernier filet → aucune pièce fantôme, atomicité préservée (finding Regigigas T7/T8)."""
    from engine.simulation import _refund_cargo_gold, _clear_trade, Building, MARKET_GOLD_MAX
    from engine.entities import spawn, EntityType
    # marché présent : coffre + cap → débordement en dorure d'église
    e = spawn(EntityType.HUMAN, 5, 5); e.cargo_gold = 3
    mkt = Building(id=1, clan_id=0, x=1, y=1, btype="market", gold=7)
    ch = Building(id=2, clan_id=0, x=2, y=2, btype="church", gold=0, gilt=0)
    _refund_cargo_gold(e, {"market": [mkt], "church": [ch]})
    assert mkt.gold == MARKET_GOLD_MAX and ch.gilt == 2 and e.cargo_gold == 0, "refund marché cap(8)→dorure(2)"
    # marché disparu → repli au trésor de l'église
    e2 = spawn(EntityType.HUMAN, 5, 5); e2.cargo_gold = 1
    ch2 = Building(id=3, clan_id=0, x=2, y=2, btype="church", gold=1, gilt=0)
    _refund_cargo_gold(e2, {"church": [ch2]})
    assert ch2.gold == 2 and e2.cargo_gold == 0, "refund repli trésor église (marché disparu)"
    # _clear_trade nettoie cargo_gold (dernier filet, après un éventuel refund)
    e3 = spawn(EntityType.HUMAN, 5, 5); e3.cargo_gold = 1; e3.trade_phase = "home"
    _clear_trade(e3)
    assert e3.cargo_gold == 0 and e3.trade_phase is None, "_clear_trade dernier filet cargo_gold"
    print("  test_p6_f1_gold_leak_refund OK (refund marché/église, cap→dorure, _clear_trade net)")


def test_p6_f2_wealth_formula_and_wire():
    """P6 F2 : formule de richesse exacte (bois 1 / pierre 3 / fer 6 / pain 2 / or+dorure 12) ;
    wire `wealth` présent sur chaque clan (0 accepté) ; DÉRIVÉE — survit au save/load sans être
    sérialisée (recalcul à la volée : un load ne doit pas la figer)."""
    from engine.simulation import _clan_wealth, Building
    cb = {"house": [Building(id=1, clan_id=0, x=0, y=0, btype="house", wood=10, stone=2)],
          "forge": [Building(id=2, clan_id=0, x=1, y=0, btype="forge", iron=3)],
          "mill":  [Building(id=3, clan_id=0, x=2, y=0, btype="mill", bread=4)],
          "church": [Building(id=4, clan_id=0, x=3, y=0, btype="church", gold=5, gilt=1)]}
    # 10 + 3*2 + 6*3 + 2*4 + 12*(5+1) = 10+6+18+8+72 = 114
    assert _clan_wealth(cb) == 114, f"formule richesse exacte (attendu 114, obtenu {_clan_wealth(cb)})"
    assert _clan_wealth({}) == 0, "clan sans bâtiment → 0"
    # wire : clé présente sur chaque clan, valeur = richesse dérivée de l'état courant
    sim = Simulation(World(width=60, height=45, seed=7)); sim.populate()
    w = sim._clans_wire()
    assert all("wealth" in d for d in w), "wire wealth sur tous les clans (0 accepté)"
    # dérivée : après save/load, la valeur est RECALCULÉE (pas figée à 0 ni sérialisée)
    sim.buildings.append(Building(id=9100, clan_id=sim.clans[0].id, x=4, y=4, btype="house", wood=30))
    w_before = next(d["wealth"] for d in sim._clans_wire() if d["id"] == sim.clans[0].id)
    sim2 = Simulation(World(width=60, height=45, seed=7)); sim2.load_state(sim.save_state())
    w_after = next(d["wealth"] for d in sim2._clans_wire() if d["id"] == sim2.clans[0].id)
    assert w_after == w_before, f"richesse recalculée à l'identique après load ({w_before} vs {w_after})"
    assert "wealth" not in sim.save_state()["clans"][0], "richesse JAMAIS sérialisée (dérivable)"
    print("  test_p6_f2_wealth_formula_and_wire OK (formule exacte, wire 0-accepté, dérivée save/load-safe)")


def test_audit_ally_hysteresis_survives_save_load():
    """Audit #1 : l'hystérésis allié/rival (entrée ±40, sortie ±35) doit survivre au save/load.
    Une paire alliée décayée dans [35,40) est encore alliée en run continu ; la recalculer au
    seuil dur au load la déclasserait → guerre/aide/mariages divergents. Fix : sets sérialisés."""
    from engine.simulation import Clan
    sim = Simulation(World(width=60, height=45, seed=7)); sim.populate()
    sim.relations[(0, 1)] = 37; sim._ally_state = {(0, 1)}   # alliée à 40 puis decay → 37 (fenêtre)
    sim.relations[(1, 2)] = -37; sim._rival_state = {(1, 2)}
    sim2 = Simulation(World(width=60, height=45, seed=7)); sim2.load_state(sim.save_state())
    assert (0, 1) in sim2._ally_state, "alliance en fenêtre [35,40) perdue au load"
    assert (1, 2) in sim2._rival_state, "rivalité en fenêtre perdue au load"
    assert sim2.relations.get((0, 1)) == 37
    # vieux save (sans les sets) → fallback recalcul au seuil dur (best-effort, documenté)
    st_old = sim.save_state(); del st_old["ally_state"]; del st_old["rival_state"]
    sim3 = Simulation(World(width=60, height=45, seed=7)); sim3.load_state(st_old)
    assert (0, 1) not in sim3._ally_state, "vieux save → fallback recalcul au seuil"
    print("  test_audit_ally_hysteresis_survives_save_load OK (hystérésis sérialisée ; fallback vieux save)")


def test_p41_swarm_recolonizes_ruin():
    """P4.1 essaimage : un clan prospère à l'étroit envoie K=pop//4 colons fonder une colonie SUR
    la ruine la plus proche (consommée), alliée +50 SANS event clan_allies (pas de mariage auto)."""
    from engine.simulation import Clan, N_CLANS, REL_D_COLONY, REL_ALLY
    w = World(width=120, height=90, seed=7); sim = Simulation(w)
    cx, cy = 20.0, 20.0
    ents = []
    for i in range(24):   # pop 24 → K = 24//4 = 6 colons ; les 6 plus éloignés partent
        far = i >= 18
        e = spawn(EntityType.HUMAN, int(cx) + (20 if far else (i % 3)), int(cy) + (i // 3), Sex.MALE)
        e.age = _adult_age(0.5); e.clan_id = 0; ents.append(e)
    sim.entities = ents
    mother = Clan(id=0, cx=cx, cy=cy, color="#f00", chief_id=ents[0].id); mother.tension = 10
    sim.clans = [mother]; sim._next_clan_id = N_CLANS
    ruin = Building(id=99, clan_id=-1, x=int(cx) + 25, y=int(cy), btype="ruin", ruin_ticks=1000)
    other = Building(id=98, clan_id=-1, x=int(cx) + 60, y=int(cy) + 40, btype="ruin", ruin_ticks=1000)
    sim.buildings = [ruin, other]
    ev = []
    sim._swarm_split(0, ev)
    nc = next(c for c in sim.clans if c.id == N_CLANS)
    colonists = [e for e in ents if e.clan_id == nc.id]
    assert len(colonists) == 6, f"K=pop//4=6 colons: {len(colonists)}"
    # colonie sur la ruine la plus proche (ruin, pas other) → ruine consommée, feu à sa place
    assert ruin not in sim.buildings, "la ruine la plus proche devrait être consommée"
    assert other in sim.buildings, "la ruine lointaine reste"
    assert any(b.btype == "campfire" and b.clan_id == nc.id and (b.x, b.y) == (ruin.x, ruin.y)
               for b in sim.buildings), "feu de la colonie sur l'emplacement de la ruine"
    # alliée +50, dans _ally_state, SANS event clan_allies (donc pas de mariage)
    assert sim.relations.get((0, nc.id)) == REL_D_COLONY and (0, nc.id) in sim._ally_state
    assert not [e for e in ev if e["type"] == "clan_allies"], "pas d'event clan_allies (pas de mariage auto)"
    s = [e for e in ev if e["type"] == "clan_swarm"]
    assert s and s[0]["on_ruin"] is True and s[0]["members"] == 6 and s[0]["new_clan"] == nc.id, s
    print("  test_p41_swarm_recolonizes_ruin OK (6 colons, ruine consommée, allié +50 sans mariage)")


def test_p4_save_load_tension_counter():
    """P4 §8 : tension + _next_clan_id round-trip ; vieux save → tension 0 + compteur reconstruit."""
    sim = Simulation(World(width=60, height=45, seed=7)); sim.populate()
    sim.clans[0].tension = 66; sim._next_clan_id = 11
    sim2 = Simulation(World(width=60, height=45, seed=7))
    sim2.load_state(sim.save_state())
    assert sim2.clans[0].tension == 66 and sim2._next_clan_id == 11
    st = sim.save_state(); del st["next_clan_id"]
    for c in st["clans"]: c.pop("tension", None)
    sim3 = Simulation(World(width=60, height=45, seed=7)); sim3.load_state(st)
    assert sim3.clans[0].tension == 0
    assert sim3._next_clan_id == max(c.id for c in sim3.clans) + 1
    print("  test_p4_save_load_tension_counter OK (round-trip + vieux save reconstruit)")


if __name__ == "__main__":
    # Tests unitaires rapides (comportement) + le golden BASE : toujours joués.
    FAST = (test_deposit_no_crash_when_houses_full,
            test_preservation_live_counter, test_water_stranded_entity_rescued,
            test_c1bis_toolless_human_crafts_without_depositing,
            test_e2_female_seeks_distant_mate,
            test_e_boar_hunts_and_captures_prey,
            test_e_hunt_preserves_prey_below_floor,
            test_c2_hungry_harvester_eats_not_feeds_mill,
            test_a1_clan_gains_science_and_ages_up,
            test_b_forge_upgrades_stone_tools_to_iron,
            test_d1_caravan_roundtrip_conserves_resources,
            test_d1_no_trade_without_complementary_surplus,
            test_d1_dest_ruined_merchant_returns_and_replay,
            test_d2_iron_for_stone_roundtrip_conserves,
            test_d2_no_flip_and_refusal,
            test_d2_price_board_and_replay,
            test_c1_office_procession_blessing_and_hunger,
            test_c1_pilgrimage_pays_offering_and_conserves,
            test_c1_dest_ruined_and_replay,
            test_c2_gold_mine_deposit_hysteresis,
            test_c2_gold_offering_circulates_and_conserves,
            test_c2_dest_ruined_gold_recredit_and_replay,
            test_k_chronicle_records_and_persists,
            test_e8_dead_clan_leaves_ruins_then_fade,
            test_p3_tribute_and_surplus_lost,
            test_p3_conquest_absorption,
            test_p3_purge_e8_and_save_counters,
            test_p4_coup,
            test_p4_rebel_split,
            test_p4_found_clan_deterministic,
            test_p41_swarm_recolonizes_ruin,
            test_p4_save_load_tension_counter,
            test_audit_ally_hysteresis_survives_save_load,
            test_p5_cult_founding_and_names_deterministic,
            test_p5_conversion_and_schism,
            test_p5_guardian_and_save_load,
            test_p5_feast_trigger_and_bounds,
            test_p5_feast_save_load,
            test_p5_monument_completion_and_ruin,
            test_p5_monument_save_load,
            test_p5_hero_naming_and_wire,
            test_p5_hero_fallen_and_save_load,
            test_p6_f1_money_dawn_and_save_load,
            test_p6_f1_gold_leak_refund,
            test_p6_f2_wealth_formula_and_wire,
            test_p6_f3_trails_grid,
            test_harden_load_state_transactional,
            test_harden_from_state_bounds,
            test_harden_load_rejects_nan,
            test_harden_save_state_chronicle_copy,
            test_harden_entity_traits_copy,
            test_determinism_golden,
            test_save_load_roundtrip_and_resume,
            test_smoke_runs)
    # Tests lourds (~90 s cumulés) : endurance invariant #1 + golden CIV Âge-Acier.
    # Sautés en `--fast` (itération rapide) ; joués par défaut (filet complet).
    HEAVY = (test_infinite_run, test_determinism_civ_golden)

    fast = "--fast" in sys.argv
    tests = FAST if fast else FAST + HEAVY
    if fast:
        print(f"[--fast] {len(HEAVY)} test(s) lourd(s) sautés (endurance + golden CIV)")
    failures = 0
    for fn in tests:
        try:
            fn()
        except Exception as e:
            failures += 1
            import traceback
            print(f"  ÉCHEC {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("RÉSULTAT:", "TOUS OK" if failures == 0 else f"{failures} ÉCHEC(S)")
    sys.exit(1 if failures else 0)
