"""
BioSim — Boucle de simulation
Appelée à chaque tick. Gère IA, déplacements, reproduction, mort.
"""
import random
import math
import json
import os
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, asdict, field, fields as _dc_fields
from typing import TYPE_CHECKING
from .entities import (Entity, EntityType, Sex, State, SPECS, spawn, BUILDING_SPECS,
                       get_id_counter, set_id_counter, _JOBS_ON)
from .world import (Biome, TREE_STUMP_THRESHOLD, STONE_STUMP_THRESHOLD,
                    IRON_STUMP_THRESHOLD, GOLD_STUMP_THRESHOLD, FERTILITY_TRAMPLE,
                    TIME_SCALE)   # échelle de temps biologique (contrat dans world.py)

# ── Clans ─────────────────────────────────────────────────────────────────────
N_CLANS = 4
CLAN_COLORS = ["#e74c3c", "#3498db", "#27ae60", "#9b59b6"]

# ── Territoire (bloc CONFLIT & TERRITOIRE, T1) ────────────────────────────────
# Owner-grid : chaque tuile terrestre appartient au clan dont un bâtiment est le plus
# proche, dans la limite d'un rayon d'influence. Au-delà = terre sauvage (-1).
# Le calcul est PUR (aucun RNG) → n'affecte pas le hash déterministe ; il n'est PAS mis
# dans le payload de step() (le front l'obtient via /api/territory) → T1 hash-neutre,
# pas de regold. Recalculé périodiquement (le territoire bouge lentement, au rythme des
# constructions/destructions) : ~8 ms sur 220×160, inutile de le refaire chaque tick.
TERRITORY_MAX_DIST         = 22    # rayon d'influence max d'une ancre (tuiles)
TERRITORY_RECOMPUTE_PERIOD = 30    # recalcul tous les N ticks (≈ un demi-jour)

# ── Âges technologiques (bloc civilisation A1) ─────────────────────────────────
# Colonne vertébrale de la progression : un clan accumule de la SCIENCE (selon ses
# bâtiments + sa population) et franchit des âges à des seuils. L'âge est visible
# (badge feu de camp, événement, infobulle) et donne un bonus de capacité de
# population → un clan avancé devient une plus grande cité. Les futurs blocs
# (forge/fer, économie…) se brancheront sur `clan.age >= X`.
AGE_NAMES = ["Bois", "Pierre", "Fer", "Acier"]
AGE_SCIENCE_THRESHOLDS = [0, 700, 2500, 6000]   # science cumulée requise pour l'âge i
SCIENCE_PER_BUILDING   = 0.20 / TIME_SCALE  # science/TICK par bâtiment durable → taux
SCIENCE_PER_POP        = 0.05 / TIME_SCALE  # science/TICK par humain vivant → taux
# (les seuils AGE_SCIENCE_THRESHOLDS restent inchangés : les taux ralentis suffisent à
#  conserver la même progression PAR JOUR — un clan monte d'âge au même rythme vécu.)
AGE_POP_BONUS          = 2      # capacité de population supplémentaire par âge franchi

@dataclass
class Clan:
    id: int
    cx: float        # centre du territoire (feu de camp)
    cy: float
    color: str
    chief_id: int
    science: float = 0.0   # savoir accumulé (bâtiments + population) — pilote les âges
    age: int = 0           # âge technologique : index dans AGE_NAMES (0 = Bois)
    mode: str = "peace"    # gouvernement (société) : "peace" | "war" | "famine"
    war_target: int = -1   # clan_id ciblé en guerre (-1 = aucun ; int → asdict/JSON stable)
    mode_ticks: int = 0    # ticks écoulés dans le mode courant (hystérésis)
    war_kills_for: int = 0     # P3 : kills infligés à la cible depuis la déclaration (issue de guerre)
    war_kills_against: int = 0 # P3 : pertes subies de la cible depuis la déclaration
    tension: int = 0           # P4 : pression interne 0-100 (coup d'État ≥70, scission ≥90)
    cult_id: int = -1          # P5 : culte du clan (-1 = aucun avant fondation)
    cult_converted: bool = False  # P5 : verrou de conversion (1× irréversible ; ré-armé par un schisme)
    feast_ticks: int = 0       # P5 E2 : ticks de fête restants (>0 = fête en cours ; convergence + natalité×2)
    feast_year: int = -1       # P5 E2 : dernière année où une fête a eu lieu (verrou 1×/an)
    last_deed: str = "sa fondation"  # P5 E3 : dernier jalon majeur (dédicace d'un futur monument)
    known_sites: list = field(default_factory=list)  # P7 G1 : site_id connus, triés, bornés KNOWN_SITES_MAX
    failed_sites: list = field(default_factory=list)  # P7 G1 : sites où un éclaireur a échoué (renoncement)
    # P7 G3 : l'état d'une migration en cours. Champs de dataclass → sérialisés gratuitement par
    # asdict, et un vieux save les reconstruit aux défauts ci-dessous (pattern G1 known_sites).
    migrating_to: int = -1        # site_id visé, chantier en cours (-1 = aucune migration)
    migration_t0: int = -1        # tick de POSE du chantier — timeout ABSOLU, jamais un compteur
                                  # incrémenté dans un bloc comportemental (leçon G1 : un compteur
                                  # cesse de tourner dès que le porteur est figé par la faim ou le
                                  # combat, et le clan resterait « en migration » à vie)
    push_evals: int = 0           # évals consécutives en terroir épuisé (hystérésis MIGRATE_PUSH_EVALS)
    last_migration_tick: int = -1  # -1 = n'a JAMAIS migré (le cooldown ne s'applique pas). Un
                                   # sentinelle plutôt qu'un `-MIGRATION_COOLDOWN` : la constante
                                   # est définie bien plus bas dans le module, et surtout « jamais
                                   # migré » n'est pas « a migré il y a longtemps » — sans ça, un
                                   # clan fondé tard hériterait d'un cooldown déjà purgé par le
                                   # seul écoulement du monde, et un clan fondé tôt non.
    last_coup_tick: int = -1       # P8 H2 — sentinelle « jamais de coup », sérialisée. Slot PROPRE
                                   # et non un champ recyclé (leçon A10 : deux délais aux sémantiques
                                   # distinctes ne tiennent pas dans un seul champ). Vieux save → -1
                                   # = libre, aucun clan existant n'est puni rétroactivement.
    founded_tick: int = 0          # A10 — tick de FONDATION, distinct de last_migration_tick.
                                   # « Délai après une fondation » et « délai après une migration »
                                   # sont deux sémantiques : les faire tenir dans un seul champ
                                   # écrasait la sentinelle « jamais migré » et rendait les deux
                                   # délais indiscernables. Défaut 0 : les clans de la genèse
                                   # sédentarisent aussi (un monde ne commence pas par un exode),
                                   # et un vieux save — donc un monde déjà mûr — est libre d'emblée.

    def to_dict(self):
        d = {"id": self.id, "cx": self.cx, "cy": self.cy,
             "color": self.color, "chief_id": self.chief_id,
             "science": round(self.science, 1), "age": self.age,
             "age_name": AGE_NAMES[min(self.age, len(AGE_NAMES) - 1)]}
        if _SOCIETY_ON:   # kill-switch d'imputation : off → payload identique au pré-bloc
            d["mode"] = self.mode
            d["war_target"] = self.war_target
        return d


_next_building_id = 0

@dataclass
class Building:
    id: int
    clan_id: int
    x: int
    y: int
    btype: str = "house"
    wood: int  = 0          # bois stocké dans ce bâtiment
    stone: int = 0          # pierre stockée dans ce bâtiment
    level: int = 1          # niveau du bâtiment (1 ou 2)
    stage: int = 1          # wheatfield : stade de croissance 1-4 (4 = mûr)
    grow_ticks: int = 0     # ticks accumulés dans le stade courant
    watered_ticks: int = 0  # ticks restants de croissance accélérée (arrosoir)
    wheat: int = 0          # mill : blé stocké (livré par les fermiers)
    bread: int = 0          # mill : pains prêts à consommer
    mill_ticks: int = 0     # mill : ticks de cuisson en cours
    work_done: int = 0      # ticks de travail déjà effectués sur un chantier
    work_needed: int = 0    # ticks de travail total nécessaires pour terminer
    ruin_ticks: int = 0     # ruine : ticks restants avant que la nature la reprenne (0 = pas une ruine)
    iron: int = 0           # forge : fer stocké (bloc B) ; market : fer d'étal (D2)
    pilgrims_served: int = 0   # church : pèlerins bénis (renommée) — bloc C1
    gold: int = 0           # church : trésor de pièces (bloc C2)
    gilt: int = 0           # church : dorure cumulée = or FONDU, le puits (bloc C2)
    rate_stone: int = 0     # market : cours affiché pierre/lot (D2, 0 = ne vend pas)
    rate_iron: int = 0      # market : cours affiché fer/lot (D2)
    wants_stone: int = 0    # market : 1 si le clan cherche de la pierre (D2)
    wants_iron: int = 0     # market : 1 si le clan cherche du fer (D2)
    dedication: str = ""    # monument (P5 E3) : jalon commémoré, fixé à la pose (asdict ; hors wire)

    def to_dict(self):
        d = {"id": self.id, "clan_id": self.clan_id,
             "x": self.x, "y": self.y, "btype": self.btype,
             "wood": self.wood, "stone": self.stone, "level": self.level}
        if self.work_needed > 0:
            d["work_done"] = self.work_done
            d["work_needed"] = self.work_needed
        if self.btype == "wheatfield":
            d["stage"]   = self.stage
            d["watered"] = self.watered_ticks > 0
        elif self.btype == "mill":
            d["wheat"] = self.wheat
            d["bread"] = self.bread
        elif self.btype == "ruin":
            d["ruin_ticks"] = self.ruin_ticks   # front : fondu quand la ruine s'efface
        elif self.btype == "forge":
            d["iron"] = self.iron               # fer stocké à la forge (bloc B)
        elif self.btype == "market":
            d["iron"] = self.iron               # étal fer (D2)
            d["rs"] = self.rate_stone; d["ri"] = self.rate_iron   # cours affichés
            d["ws"] = self.wants_stone; d["wi"] = self.wants_iron # recherche
            if self.gold:                       # F1 : coffre d'or du marché (0-gardé → absent sous MONEY_OFF)
                d["gold"] = self.gold
        elif self.btype == "church":
            d["pil"] = self.pilgrims_served     # renommée du sanctuaire (C1)
            d["gold"] = self.gold               # trésor (C2)
            if self.gilt:
                d["gilt"] = self.gilt           # dorure (le puits visible)
        return d


# ── Bâtiments & outils ────────────────────────────────────────────────────────
AXE_CRAFT_COST      = 5    # bois stocké pour fabriquer une hache en bois
AXE_BONUS           = 3    # bois bonus par coup avec hache bois (total 2+3=5)
STONE_AXE_COST      = 5    # pierre stockée pour upgrader → hache en pierre
STONE_AXE_BONUS     = 5    # bois bonus par coup avec hache pierre (total 2+5=7)
PICK_WOOD_COST      = 8    # bois stocké pour fabriquer une pioche en bois
STONE_PICK_COST     = 5    # pierre stockée pour upgrader → pioche en pierre
STONE_PICK_BONUS    = 1    # pierre bonus par minage avec pioche pierre (total 1+1=2)
MAX_STONE_CARRY     = 3    # pierre max transportable par un humain
# En dessous de ce stock de pierre TOTAL dans le clan, un mineur cherche la montagne
# minable la plus proche sur TOUTE la carte (pas seulement en vision) — sinon un clan
# sans montagne en vue ne mine jamais (→ ni puit, ni moulin, ni niveau 2). Au-dessus,
# on n'envoie plus les mineurs traverser la carte en continu (garde-fou perf). (C1)
CLAN_STONE_BOOTSTRAP = 20

# ── Fer & forge (bloc B) — réservé aux clans de l'Âge du Fer ───────────────────
FER_AGE            = 2     # index d'âge « Fer » dans AGE_NAMES (Bois=0, Pierre=1, Fer=2)
MAX_IRON_CARRY     = 3     # fer max transportable par un mineur
IRON_PICK_COST     = 3     # fer (stocké à la forge) pour upgrader pioche pierre → fer
IRON_AXE_COST      = 3     # fer pour upgrader hache pierre → fer
IRON_PICK_BONUS    = 2     # pierre bonus par minage roche avec pioche fer (total 1+2=3
                           # = portée pleine en UN coup). Sur le FER : aucun bonus —
                           # IRON_PER_MINE=3 sature déjà MAX_IRON_CARRY (gate-review B).
# Hache fer : le bonus de rendement serait mangé par MAX_CARRY=5 (2+8 capé → identique
# pierre, prouvé au gate-review). Son vrai effet = COUPER PLUS VITE (cooldown réduit).
IRON_AXE_COOLDOWN  = 6 * TIME_SCALE   # ticks entre 2 coupes (hache fer) → durée
FORGE_MAX_IRON     = 20    # stock de fer max dans une forge (cap de dépôt)
# Hystérésis anti-pendule (gate-review B) : les mineurs ne partent en expédition fer
# que si la forge passe SOUS ce seuil (l'upgrade en consomme 3 → si on rouvrait la
# vanne dès iron < MAX, tout clan Fer enverrait ses mineurs au fer en permanence et
# affamerait la chaîne PIERRE). Entre RESTOCK et MAX, seul le dépôt remplit.
IRON_RESTOCK_THRESHOLD = 8

# ── Économie / troc (bloc D1 « Marchés & caravanes ») ──────────────────────────
# Route unique : A (riche-bois, pauvre-pierre) ACHÈTE de la pierre à B
# (thésauriseur) en payant en bois. Seuils calibrés sur pools MESURÉS (sonde
# 2×3500 ticks, scratchpad/probe_s*.jsonl) : bois de clan mûr = 100-165 (sature),
# pierre = 0-19 chronique chez les pauvres vs 300-6000 chez le thésauriseur.
MARKET_AGE           = 1     # Âge de Pierre requis pour bâtir un marché
TRADE_CHECK_PERIOD   = 120 * TIME_SCALE   # période d'évaluation des routes → ×
TRADE_WOOD_LOT       = 12    # bois emporté par caravane (cargo dédié, hors MAX_CARRY)
TRADE_WOOD_SURPLUS   = 60    # pool bois min de l'acheteur A (peut payer en bois)
# (TRADE_STONE_PRICE / TRADE_STONE_SURPLUS / TRADE_STONE_DEFICIT : subsumés par D2 —
#  le taux devient SPOT par paliers sur le pool du vendeur, la fenêtre d'achat par
#  STONE_WANT_FULL, l'éligibilité vendeur par STONE_RATE_TIERS/_stone_rate.)

# ── Économie v2 (bloc D2 « Loi de l'offre ») — paliers calibrés sonde d2probe_judge (200×150) ──
GOODS_ORDER      = ("stone", "iron")   # ordre d'éval déterministe par acheteur
STONE_RATE_TIERS = ((500, 12), (150, 9), (60, 6))  # pool vendeur → pierre/lot ; 12 = parité
IRON_RATE_TIERS  = ((16, 3), (10, 2))              # pool forge vendeur → fer/lot
STONE_SELL_FLOOR = 30    # réserve incessible du vendeur (vente partielle au-delà)
IRON_SELL_FLOOR  = 8     # = IRON_RESTOCK_THRESHOLD : la vente seule ne rouvre PAS la
                         # vanne de minage (il faut <8) ; les upgrades (−3) la rouvrent
STONE_WANT_FULL  = 30    # fenêtre d'achat pierre (ex-TRADE_STONE_DEFICIT, inchangée)
IRON_WANT_FULL   = 5     # forge < 5 : plus de quoi payer 2 upgrades (coût 3)
IRON_WANT_BARGAIN = 9    # forge < 9 SI un vendeur affiche 3 (anti-flip : 8+3=11 < 16 →
                         # jamais vendeur rate-3 ; vendeur rate-2 = relais borné, voulu)
PAY_STONE_LOT    = 6     # lot de paiement pierre (équivalent du lot de 12 bois)
PAY_STONE_MIN    = 150   # pierre min du payeur pour payer en pierre (glut seulement)
WOOD_VALUE_TIERS  = (20, 45, 90)   # rareté 3/2/1/0 chez le vendeur (choix du paiement)
STONE_VALUE_TIERS = (12, 30, 60)

# ── Société v1 (bloc C1 « Le Sanctuaire ») — constantes ratifiées panel C1 ──────
CHURCH_AGE            = 3     # Acier : MESURÉ t1503-2106 (s7) / t1901-2070 (s42)
CHURCH_SERVICE_PERIOD = 300 * TIME_SCALE   # 1 cloche/saison → doit suivre TICKS_PER_SEASON
CHURCH_SERVICE_WINDOW = 70 * TIME_SCALE   # durée d'office → × (contient approche +
                              # prière : 26 tuiles @ speed 1 + PRAY_DURATION 96 < 420)
CHURCH_CALL_RADIUS    = 26    # éligibles mesurés 7-44 par clan → processions garanties
PRAY_RADIUS           = 2.5   # distance au parvis pour prier
PRAY_DURATION         = 16 * TIME_SCALE   # durée agenouillé → ×
BLESS_DURATION        = 600 * TIME_SCALE   # durée de bénédiction (2 saisons) → suit les saisons
BLESS_HUNGER_MULT     = 0.85  # SEUL effet : faim ralentie 15 % (quasi neutre en glut mesuré)
PRAY_HUNGER_MAX       = 55    # éligibilité office (jamais au détriment de la survie)
PRAY_THIRST_MAX       = 50
OFFERING_WOOD         = 8     # offrande du pèlerin (lot fixe, tout-ou-rien au pool)
PILGRIM_WOOD_MIN      = 30    # pool bois min du clan pèlerin (< SURPLUS=60 : les pauvres accèdent)
PILGRIM_CHECK_PERIOD  = 240 * TIME_SCALE   # période de dispatch → ×
PILGRIM_TIMEOUT       = 1600  # pire trajet mesuré 179 tuiles, couvert même à eff_speed 0.5
CHURCH_FAME_GAP       = 3     # hystérésis (amendé 5→3 par le juge : fenêtre Acier serrée ~600 ticks)
ALTAR_BURN_PERIOD     = 6 * TIME_SCALE   # 1 offrande / N ticks → durée
ALTAR_MAX             = 30    # cap de pile d'autel (surplus consumé immédiatement)
CHURCH_FAME_MILESTONE = 10    # pèlerins reçus → chronique « le sanctuaire rayonne »

# ── Or des offrandes (bloc C2) — source bornée + puits + circulation (verdict D2) ──
GOLD_AGE               = 3    # = CHURCH_AGE : l'or naît avec le sanctuaire (kill-switch : 99)
MAX_GOLD_CARRY         = 2    # pièces portées par expédition (> GOLD_PER_MINE=1 : jamais écrêté)
GOLD_TREASURY_MAX      = 12   # cap du trésor d'église ; au-delà : la pièce fond en DORURE (puits)
GOLD_RESTOCK_THRESHOLD = 4    # hystérésis type IRON_RESTOCK : expédition ssi trésor < 4 ;
                              # entre 4 et 12, seuls dépôts/offrandes remplissent — et l'or REÇU
                              # ferme la vanne (substitution : la circulation éteint la source)
OFFERING_GOLD          = 1    # 1 pièce = 1 offrande (même renom qu'une offrande bois : +1)
GILT_MILESTONE         = 3    # dorure cumulée → chronique « resplendit au soleil »

# ── Société : gouvernements de clan (PAIX/GUERRE/FAMINE) + conflit gouverné ──────────
# Le chef choisit le mode selon l'état du clan, réévalué périodiquement (déphasé par clan),
# 100 % déterministe (thresholds, zéro RNG). Kill-switch d'imputation : SOCIETY_OFF=1 → mode
# figé "peace", décision et conflit-de-guerre coupés, `mode` absent du payload → hash pré-bloc.
_SOCIETY_ON     = os.environ.get("SOCIETY_OFF") != "1"
MODE_PERIOD     = 120 * TIME_SCALE   # ré-évaluation du mode par le chef (déphasée par clan)
FAMINE_HUNGER   = 55                 # faim moyenne du clan → mode FAMINE (survie prime tout)
WAR_MIN_POP     = 4                  # pop humaine mini du clan pour déclarer la guerre
# Périodicité (une guerre est un ÉVÉNEMENT, pas un état permanent) : elle dure au plus
# WAR_MAX_TICKS puis retombe en paix, et un cooldown PEACE_MIN_TICKS empêche d'en redéclarer
# aussitôt → cycle guerre→paix→(éventuelle reprise). Sans ça les clans restent en guerre ~95 %.
WAR_MAX_TICKS   = 240 * TIME_SCALE   # durée max d'une guerre → paix forcée
PEACE_MIN_TICKS = 480 * TIME_SCALE   # cooldown de paix avant de pouvoir redéclarer la guerre
# Défection (S2a) : en guerre, une victime dont le clan perd NETTEMENT (pop < ce ratio × pop de
# l'attaquant) ABANDONNE et rejoint le vainqueur au lieu de mourir → conquête par absorption.
DEFECT_RATIO    = 0.5
# Plancher anti-anéantissement (S2c équilibrage) : une war-kill (acte de guerre) ne réduit JAMAIS
# un clan sous ce seuil → un clan battu survit en « rump » au lieu de disparaître. Sans ça, la
# marche de guerre (assaut fiable) broie le war_target jusqu'à l'extinction → le monde converge
# vers 1 clan à long horizon (attracteur « dernier debout » que Société II combat ; mesuré : 4/6
# seeds → 1 clan à 18000t sans ce plancher). Symétrique du plancher de défection (>3). La force
# régénératrice complète = P4 rébellions (plus tard) ; P3 raffinera l'absorption du rump. Le raid
# de SURVIE (faim) et le pré-société gardent le seul plancher d'espèce (un affamé mange l'ennemi).
WAR_MIN_CLAN_POP = 3

# ── Métiers (P1, chantier A du plan civilisation) ────────────────────────────────────
# Champ Entity.role assigné par clan selon la pop, 100 % déterministe (seuils + tri id, zéro
# RNG), ré-évalué périodiquement (déphasé). Un rôle = biais de la cascade _beh_work (guards) +
# petit bonus. La survie n'est JAMAIS modifiée. versatile = cascade actuelle intacte (fallback).
# _JOBS_ON importé d'entities.py (kill-switch JOBS_OFF → tout versatile, clé job absente du wire).
JOB_PERIOD         = 120 * TIME_SCALE   # ré-évaluation, déphasée par clan (tick + clan_id*41)
JOB_BONUS_COOLDOWN = 0.85               # cooldown de travail du spécialiste (×, arrondi int min 1)
_JOB_ROLES = ("farmer", "woodcutter", "miner", "builder", "warrior", "merchant", "priest", "scout")
# Sections de _beh_work réservées par métier (P1) : versatile partout ; un rôle absent de la
# valeur saute la section. Sections non listées (dépôts, crafts, office, repro) = tous autorisés.
_ROLE_SECTIONS = {
    "farm":  ("farmer", "versatile"),
    "chop":  ("woodcutter", "versatile"),
    "mine":  ("miner", "versatile"),
    "build": ("builder", "versatile"),
    "scout": ("scout", "versatile"),
}
# ── Comportements guerriers (S2c) ────────────────────────────────────────────────────
# Sans ça, la guerre ne se déclenche qu'au CONTACT FORTUIT : un warrior sans ennemi dans son
# voisinage vaque et la guerre reste lettre morte. S2c donne deux comportements au warrior :
#   • MARCHE DE GUERRE (_beh_survival) : en guerre, pas d'ennemi engagé et pas affamé → converge
#     vers le feu du clan ciblé (`war_target`) et attaque à vue en route (via le scan de conflit).
#   • GARDE EN PAIX (_beh_wander) : en paix, errance ancrée au rayon WARBAND_GUARD_R du feu du clan.
# Gated _WARBEH_ON (+ _JOBS_ON + _SOCIETY_ON aux points d'usage) : kill-switch d'imputation
# WARBEH_OFF=1 → aucune marche ni garde → hash P1 (5c14d2d0…) restauré ; JOBS_OFF/SOCIETY_OFF
# inchangés (pas de warriors / pas de mode war → jamais déclenché).
_WARBEH_ON     = os.environ.get("WARBEH_OFF") != "1"
WARBAND_GUARD_R = 10                     # rayon d'errance d'un warrior en paix (garnison au feu)

# ── Politique (P2) ────────────────────────────────────────────────────────────────────
# Succession du chef : si `chief_id` pointe une entité morte OU qui a changé de clan (chef
# défecté via S2a), le clan promeut un nouveau chef = son membre le plus ÂGÉ (loi du plus fort,
# tie-break id min), 100 % déterministe. Sans ça, un clan reste « sans chef valide » (chief_id
# pointe un mort ou un membre ennemi) → gouvernance/UI incohérentes, et rien sur quoi bâtir la
# politique P2. Kill-switch d'imputation POLITICS_OFF=1 → pas de succession → hash S2c restauré.
_POLITICS_ON   = os.environ.get("POLITICS_OFF") != "1"

# ── Politique P2 : personnalité du chef + relations inter-clans (chantier B) ───────────
# Substrat géopolitique : deux clans qui se sont fait la guerre restent RIVAUX (et se re-ciblent),
# deux clans qui commercent/voisinent deviennent ALLIÉS (et ne s'attaquent plus) ; le caractère du
# chef (dérivé de son id, ères politiques via la succession P2.1) penche la balance. Seul branchement
# moteur = le choix de cible de guerre. Kill-switch RELATIONS_OFF=1 → aucun delta/décay/influence/
# wire → hash c08b6a14 (P2.1) EXACT. Gated _POLITICS_ON AUSSI (POLITICS_OFF → a85d306b inchangé).
_RELATIONS_ON  = os.environ.get("RELATIONS_OFF") != "1"
# P2 est le SOMMET de la pile d'imputation → gated par TOUS les étages inférieurs, sinon les
# deltas événementiels (raids de survie sous SOCIETY_OFF, fights I1 sous JOBS_OFF) feraient
# dériver les baselines 4ccfdd60/5f244402 que la spec exige inchangés. RELATIONS_OFF isole P2.
_POL_ON        = _SOCIETY_ON and _JOBS_ON and _POLITICS_ON and _RELATIONS_ON
REL_MIN, REL_MAX = -100, 100
REL_CAUSE_DEFAULT = "decay"   # cause par défaut d'une sortie d'alliance : l'usure ordinaire
REL_CAUSE_ENVY    = "envy"    # H1 : l'injustice de richesse — SEULE cause qui se déclare
REL_ALLY, REL_RIVAL = 40, -40         # seuils allié/rival (PLAN §10)
REL_ALLY_OFF, REL_RIVAL_OFF = 35, -35 # hystérésis : on CESSE d'être allié <35 / rival >-35
REL_NEIGHBOR_DIST = 80                 # voisinage pacifique : bonus/éval si feux plus proches que ça
                                       # (80 : sur 220×160 les feux sont espacés 32-125 tuiles ; à 40
                                       #  quasi aucune paire ne qualifiait → alliances inatteignables)
_KNUTH = 2654435761                    # mixage entier de Knuth (personnalité dérivée de chief_id)
WAR_TEMPER_STEP = 48 * TIME_SCALE      # temper ±2 → cooldown de paix ∓96×TS (~±20 % fréq. guerre)
# Deltas événementiels (aux points d'émission existants, O(1) chacun) :
REL_D_WAR, REL_D_FIGHT, REL_D_DEFECT, REL_D_TRADE = -60, -2, -5, 2
REL_D_NEIGHBOR = 2                     # bonus voisinage (EXCLUSIF du décay) : +1 mettrait 14400t à
                                       # atteindre +40 (une paire évaluée 2×/MODE_PERIOD) ; +2 → ~7200t

# ── Guerre 2.0 : issues de guerre, tribut, conquête, aide d'allié, mariages (P3, chantier C) ──
# Une guerre a enfin des CONSÉQUENCES (avant : timeout → paix, rien). Jugée chez son déclarant :
# conquête-absorption si la cible touche le plancher S2c, sinon tribut 20 % au timeout si le
# déclarant a plus tué que perdu, sinon paix blanche. + aide d'allié, mariages sur alliance.
# Consomme l'existant (relations P2, plancher S2c, warrior-only P1, ruines E8). Kill-switch
# WAR2_OFF au SOMMET de la pile (_WAR2_ON ⊂ _POL_ON) : off → hashes P2 (73fd2b25…) exacts.
_WAR2_ON       = _POL_ON and os.environ.get("WAR2_OFF") != "1"
TRIBUTE_PCT    = 20                    # % de chaque ressource stockée prélevé au perdant (div. entière)
REL_D_MARRIAGE = 10                    # un mariage CIMENTE l'alliance (+10 sur la paire, clampé)

# ── Vie interne : tension, coups d'État, scissions (P4, chantier D + S2b) ──────────────
# P3 CONSOLIDE (conquêtes en cascade → 1 clan). P4 FRAGMENTE par la pression interne : une TENSION
# causale 0-100 par clan (famine/guerre/surpopulation/tribut la nourrissent, la paix prospère
# l'apaise) déclenche à l'éval `due` un COUP D'ÉTAT (le pouvoir change) ou une SCISSION (le clan
# éclate en un clan neuf). Cerise : les sécessionnistes = les + éloignés du feu = les conquis P3
# (qui vivent près de leur ancien feu) → re-scission le long des anciennes frontières, gratuit.
# Kill-switch UNREST_OFF au sommet de la pile → hashes P3 (6159136c…) exacts.
_UNREST_ON      = _WAR2_ON and os.environ.get("UNREST_OFF") != "1"
TENSION_SPLIT   = 90    # ≥ → SCISSION (pop ≥ REBEL_MIN_POP, clans < MAX_CLANS)
TENSION_COUP    = 70    # ≥ (et < SPLIT) → COUP D'ÉTAT (le + jeune adulte renverse le chef)
REBEL_MIN_POP   = 12    # pop mini d'un clan pour pouvoir se scinder
MAX_CLANS       = 8     # plafond de clans vivants (pas de scission au-delà — PLAN §10)
OVERPOP_TENSION_MAX = 20  # tension d'overpop = min(ce cap, pop − cap logement) : un clan gavé
                          # de conquis (pop ≫ cap) bout vite, un clan juste plein monte à peine
REL_D_REBELLION = -50  # rancune fondatrice : la née est rivale de la mère d'entrée
# Tension ÉVÉNEMENTIELLE (spec §1 + fix calibration) : le tribut PAYÉ humilie le perdant ; et un
# empire qui s'ÉTEND (assimile des défecteurs, absorbe des vaincus) se déstabilise politiquement —
# c'est la seule force qui monte la tension d'un méga-clan CONFORTABLEMENT logé (il a hérité des
# maisons des conquis via P3 → jamais en surpop → mesuré : sans ça, seed 7 pique à 35, 0 scission).
TRIBUTE_TENSION  = 15  # tribut payé → +15 au perdant (spec §1)
DEFECT_TENSION   = 2   # assimiler un défecteur ennemi → +2 (over-extension continue)
CONQUEST_TENSION = 8   # absorber un clan vaincu → +8 (acte politique d'annexion)
# Surextension STRUCTURELLE (spec §1 amendée, reco Regigigas) : pression PERSISTANTE du span-of-
# control, +max(0,(pop−SPAN)//10)/éval (cap OVEREXTEND_MAX). Les chocs événementiels ci-dessus
# s'estompent (contentement −2) ; la surextension garantit qu'un MONOCLAN unifié continue à monter
# en tension jusqu'à la scission → pas d'état stable monoclan à tension nulle (le cycle des empires).
OVEREXTEND_SPAN = 30   # pop à partir de laquelle le span-of-control devient ingouvernable
OVEREXTEND_MAX  = 10   # cap du delta structurel par éval
# Essaimage pacifique (P4.1) : le pendant POSITIF de la scission — un clan prospère (tension basse)
# mais à l'étroit (surpop) envoie une colonie, de préférence RECOLONISER la ruine la plus proche
# (referme la boucle E8 : les gravats revivent). Colonie ALLIÉE de la mère d'entrée. Kill-switch
# SWARM_OFF (sous-bloc de P4) → hashes P4 (ae52bd38…) exacts ; _UNREST_ON coupe aussi l'essaimage.
_SWARM_ON       = _UNREST_ON and os.environ.get("SWARM_OFF") != "1"
SWARM_TENSION_MAX = 30   # tension < → clan serein (condition d'essaimage, pas de scission/coup)
SWARM_MIN_POP     = 20   # pop mini pour essaimer (au-delà de REBEL_MIN_POP : la prospérité déborde)
REL_D_COLONY      = 50   # colonie alliée d'entrée (POSÉE direct, sans event → pas de mariage auto)

def _chief_personality(chief_id):
    """(temper, diplo) dérivés de chief_id — zéro état, zéro RNG, stable inter-process/save.
    temper ∈ −2..+2 (belliqueux+/pacifique−) ; diplo ∈ −1..+1 (rancunier−/conciliant+)."""
    h = (chief_id * _KNUTH) % (2**32)
    return (h % 5) - 2, ((h >> 8) % 3) - 1

def _rel_key(a, b):
    return (a, b) if a < b else (b, a)

def _rel_apply(relations, ally_state, rival_state, a, b, delta, events, cause=REL_CAUSE_DEFAULT):
    """Applique `delta` à la relation (a,b), clampé [REL_MIN,REL_MAX], symétrique (clé triée).
    Gère les franchissements d'hystérésis : émet clan_allies/clan_rivals à l'ENTRÉE dans l'état
    (≥REL_ALLY / ≤REL_RIVAL) et clan_ally_break à la SORTIE d'alliance (<REL_ALLY_OFF). Une
    relation retombée à 0 est retirée (= neutre, save minimal). Appelant responsable de _POL_ON.

    LA SORTIE D'ALLIANCE ÉTAIT MUETTE, et ça rendait H1 INVISIBLE : l'entrée avait son annale, la
    sortie n'était qu'un `discard`. Le bloc « l'envie éteint l'alliance » produisait donc un effet
    réel que le spectateur ne voyait jamais — or le jeu est 100 % visuel, un système invisible
    n'existe pas pour lui. La symétrie manquait, elle est rétablie ici.

    `cause` est un PARAMÈTRE EXPLICITE porté par le site d'appel (exigence de conception : on ne
    DEVINE pas une cause après coup, on la TRANSPORTE). Seule l'érosion d'envie se déclare ; tous
    les autres appels laissent le défaut, et l'annale reste sobre. Une inférence a posteriori
    (« la relation a baissé de 4, donc c'est l'envie ») serait fausse dès qu'un autre delta vaudra
    4 — et personne ne le verrait avant longtemps."""
    if a == b or a is None or b is None:
        return
    k = _rel_key(a, b)
    new = max(REL_MIN, min(REL_MAX, relations.get(k, 0) + delta))
    if new == 0:
        relations.pop(k, None)
    else:
        relations[k] = new
    if k not in ally_state and new >= REL_ALLY:
        ally_state.add(k); events.append({"type": "clan_allies", "a": k[0], "b": k[1]})
    elif k in ally_state and new < REL_ALLY_OFF:
        ally_state.discard(k)
        events.append({"type": "clan_ally_break", "a": k[0], "b": k[1], "cause": cause})
    if k not in rival_state and new <= REL_RIVAL:
        rival_state.add(k); events.append({"type": "clan_rivals", "a": k[0], "b": k[1]})
    elif k in rival_state and new > REL_RIVAL_OFF:
        rival_state.discard(k)
def _role_ok(role, section):
    if not _JOBS_ON:
        return True                      # kill-switch : aucune restriction de métier
    allowed = _ROLE_SECTIONS.get(section)
    return allowed is None or role in allowed

def _job_quotas(pop, clan, at_war, has_market, has_church, has_site):
    """Quotas de rôles pour un clan de `pop` humains (hors chef), selon la table de la spec P1.
    Pourcentages = floor(pop*p/100). Modificateurs appliqués DANS L'ORDRE (déterministe)."""
    q = {r: 0 for r in _JOB_ROLES}
    stone = clan.age >= 1
    if pop <= 5:
        pass                                   # tous versatile
    elif pop <= 9:
        q["farmer"] = 2; q["woodcutter"] = 1; q["builder"] = 1
    elif pop <= 15:
        q["farmer"] = 3; q["woodcutter"] = 2; q["builder"] = 1
        if stone: q["miner"] = 1
        if at_war: q["warrior"] = 1
        if has_market: q["merchant"] = 1        # 0-1 → 1 si marché
    elif pop <= 24:
        q["farmer"] = pop * 30 // 100; q["woodcutter"] = pop * 15 // 100
        q["miner"] = (pop * 10 // 100) if stone else 0
        q["builder"] = pop * 10 // 100; q["warrior"] = pop * 10 // 100
        if has_market: q["merchant"] = 1
        if has_church: q["priest"] = 1
    else:                                       # >= 25
        q["farmer"] = pop * 30 // 100; q["woodcutter"] = pop * 12 // 100
        q["miner"] = (pop * 10 // 100) if stone else 0
        q["builder"] = pop * 10 // 100; q["warrior"] = pop * 15 // 100
        if has_market: q["merchant"] = 2
        if has_church: q["priest"] = 2
        q["scout"] = 1
    # 1. famine → +2 farmer, retirés du quota miner puis woodcutter (plancher 0)
    if clan.mode == "famine":
        q["farmer"] += 2
        rm = 2
        for src in ("miner", "woodcutter"):
            take = min(rm, q[src]); q[src] -= take; rm -= take
    # 2. guerre → warrior ×2
    if clan.mode == "war":
        q["warrior"] *= 2
    # 3. chantier actif → +2 builder
    if has_site:
        q["builder"] += 2
    return q
TRADE_TIMEOUT        = 1200  # ticks max d'une mission → abort propre (trajets ≤ ~230)
MARKET_MAX_STOCK     = 40    # cap de stock d'étal (borne mémoire/économie)
MARKET_DRAIN_PERIOD  = 4 * TIME_SCALE   # 1 ressource / N ticks → durée
MERCHANT_HUNGER_MAX  = 50    # éligibilité au recrutement d'un marchand
MERCHANT_THIRST_MAX  = 45

# ── Champ de blé ──────────────────────────────────────────────────────────────
WHEAT_TICKS_PER_STAGE  = 180 * TIME_SCALE   # croissance du blé → biologique, ×
WHEAT_HARVEST_FOOD     = 45.0  # réduction de faim lors de la récolte
WHEAT_HUNGER_THRESH    = 35    # faim min pour décider de récolter (chercher de la nourriture)
WHEAT_WORK_THRESH      = 65   # faim max pour travailler les champs (planter/arroser quand reposé)
SICKLE_STONE_COST      = 5    # pierre pour fabriquer une faucille
SICKLE_HARVEST_BONUS   = 25.0 # nourriture bonus récupérée avec la faucille
WATERING_CAN_WOOD_COST = 6    # bois pour fabriquer un arrosoir
WATERED_TICKS          = 25 * TIME_SCALE   # durée d'accélération après arrosage → ×
FISHING_ROD_WOOD_COST  = 4    # bois pour fabriquer une canne à pêche
FISHING_CATCH_PROB     = 0.10 / TIME_SCALE   # proba de prise par TICK → taux
FISHING_FOOD           = 35.0 # réduction de faim par prise
FISHING_HUNGER_THRESH  = 40   # faim min pour décider de pêcher

# ── Météo / Pluie ─────────────────────────────────────────────────────────────
RAIN_PROB = {k: v / TIME_SCALE for k, v in {   # proba PAR TICK de démarrer un épisode → taux
    "spring": 0.004,   # ~1 épisode / 250 ticks (échelle de base)
    "summer": 0.001,
    "autumn": 0.006,
    "winter": 0.002,
}.items()}
STORM_PROB = {        # proba qu'un épisode pluvieux SOIT un orage → RATIO, inchangé
    "spring": 0.25,
    "summer": 0.45,   # orages d'été fréquents
    "autumn": 0.20,
    "winter": 0.05,
}
RAIN_DURATION_MIN  = 40  * TIME_SCALE   # durée d'un épisode → ×
RAIN_DURATION_MAX  = 120 * TIME_SCALE
RAIN_THIRST_REDUCE = 0.06 / TIME_SCALE  # soif retirée / TICK sous la pluie → taux
RAIN_SPEED_MULT    = 0.65  # vitesse humains sous la pluie → VITESSE, inchangée
RAIN_WHEAT_BONUS   = 1     # grow_ticks bonus/tick — ratio vs croissance (+1/tick) → inchangé
LIGHTNING_PROB     = 0.015 / TIME_SCALE  # proba de foudre par TICK d'orage → taux

# ── Canicule ──────────────────────────────────────────────────────────────────
HEATWAVE_PROB         = 0.0015 / TIME_SCALE  # proba de déclenchement par TICK (été) → taux
HEATWAVE_DURATION_MIN = 60  * TIME_SCALE     # durée → ×
HEATWAVE_DURATION_MAX = 180 * TIME_SCALE
HEATWAVE_THIRST_MULT  = 2.2     # multiplicateur de soif → RATIO, inchangé
HEATWAVE_FIRE_PROB    = 0.008 / TIME_SCALE   # proba d'incendie par TICK de canicule → taux

# ── Moulin ────────────────────────────────────────────────────────────────────
MILL_BREAD_COST_WHEAT  = 1    # 1 récolte de blé (stockée) par pain
MILL_BREAD_TICKS       = 80 * TIME_SCALE   # durée de production d'un pain → ×
MILL_BREAD_FOOD        = 65.0 # réduction de faim en mangeant un pain
MILL_MAX_BREAD         = 5    # stock max de pains dans un moulin

# ── Ruines ─────────────────────────────────────────────────────────────────────
RUIN_LIFETIME = 2500 * TIME_SCALE   # durée de visibilité d'une ruine → ×
                       # avant que la nature la reprenne (borne la mémoire = invariant infini)

# ── Saisons ──────────────────────────────────────────────────────────────────
TICKS_PER_SEASON = 300 * TIME_SCALE   # durée d'une saison → × (4 saisons = 1 année)

SEASON_NAMES = ["spring", "summer", "autumn", "winter"]

# Multiplicateur de régénération de nourriture par saison
SEASON_REGEN_MULT = {
    "spring": 1.6,
    "summer": 1.1,
    "autumn": 0.7,
    "winter": 0.32,
}

def get_season(tick: int) -> str:
    return SEASON_NAMES[(tick // TICKS_PER_SEASON) % 4]

# Températures moyennes par saison (°C) avec amplitude ±
SEASON_TEMP = {
    "spring": (12.0, 6.0),   # moy, amplitude
    "summer": (28.0, 6.0),
    "autumn": (10.0, 5.0),
    "winter": (-3.0, 5.0),
}

def get_temperature(tick: int) -> float:
    """Retourne la température en °C (sinusoïde lisse dans la saison)."""
    season = get_season(tick)
    tick_in_season = tick % TICKS_PER_SEASON
    phase = (tick_in_season / TICKS_PER_SEASON) * math.pi  # 0 → π
    mean, amp = SEASON_TEMP[season]
    return round(mean + amp * math.sin(phase), 1)

def _hunger_mult(temp_c: float) -> float:
    """Froid → faim accélérée. < 5°C : ×1.5 max, > 20°C : ×0.85 (chaleur coupe l'appétit)."""
    if temp_c <= 5.0:
        return 1.0 + (5.0 - temp_c) * 0.033   # ≈ ×1.5 à -10°C
    elif temp_c >= 20.0:
        return max(0.85, 1.0 - (temp_c - 20.0) * 0.01)
    return 1.0

def _thirst_mult(temp_c: float) -> float:
    """Chaud → soif accélérée. > 20°C : ×2.0 max, < 5°C : ×0.5."""
    if temp_c >= 20.0:
        return 1.0 + (temp_c - 20.0) * 0.067   # ≈ ×2.0 à 35°C
    elif temp_c <= 5.0:
        return max(0.5, 1.0 - (5.0 - temp_c) * 0.033)
    return 1.0

# La reproduction n'est possible qu'au printemps, été et automne (début)
SEASON_REPRO_ALLOWED = {
    "spring": True,
    "summer": True,
    "autumn": True,   # une dernière portée avant l'hiver
    "winter": False,
}


# ── Puit ──────────────────────────────────────────────────────────────────────
WELL_DRINK_AMOUNT = 85.0   # soif retirée en buvant au puit (même valeur qu'une tuile eau)


# ── Soif ─────────────────────────────────────────────────────────────────────
THIRST_RATE = 0.04 / TIME_SCALE    # soif gagnée par TICK → taux
MAX_THIRST  = 100.0   # mort au-delà
DRINK_AMOUNT = 85.0   # soif retirée en buvant une fois

# Biomes considérés comme sources d'eau potable
WATER_BIOMES = {int(Biome.WATER), int(Biome.RIVER)}

if TYPE_CHECKING:
    from .world import World


def _dist(ax, ay, bx, by) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _terroir_epuise(frais: int, fige: int) -> bool:
    """G3 PUSH — le terroir s'est-il VRAIMENT effondré ? (arbitrage A8, constantes plus haut)
    `frais` = conv d'AUJOURD'HUI au feu (seule capable de voir un terroir s'épuiser),
    `fige`  = conv du monde VIERGE au même point (ce que cette terre valait à l'origine).
    Les DEUX gardes sont exigées, et chacune veto une famille que l'autre laisserait passer :
    la chute relative écarte le clan NÉ pauvre (2 sur une terre qui vaut 2 : rien ne s'est
    épuisé), le plancher écarte le clan RICHE qui se dégrade (21 après 34 : encore le double
    de la médiane du catalogue). Produit croisé ENTIER — aucun flottant dans un seuil."""
    return (frais * MIGRATE_PUSH_RATIO_D <= fige * MIGRATE_PUSH_RATIO_N
            and frais <= MIGRATE_PUSH_FLOOR)


def _tile_near_water(world, x: int, y: int) -> bool:
    """True s'il existe une tuile d'eau dans un rayon de MILL_WATER_RADIUS autour de
    (x,y). Condition de cuisson d'un moulin (production) — réutilisée au placement (C3)
    pour ne pas bâtir de moulin qui ne cuira jamais. Lecture O(1) d'un masque pré-calculé
    (World._near_water_mill) : l'eau étant immuable, il vaut pour toute la partie. Byte-
    exact avec l'ancien scan carré ; c'était le poste CPU n°1 (169 tuiles Python/appel)."""
    return bool(world._near_water_mill[y, x])


def _dist_hitbox(entity: Entity, target: Entity) -> float:
    """Distance minimale entre les tuiles du hitbox de entity et target.
    Utilise numpy pour les entités multi-tuiles (hitbox_width > 1 ou hitbox_height > 1).
    Exemple requin : hitbox_width=1, hitbox_height=2 → grille 1×2 centrée sur entity."""
    hw = entity.spec.hitbox_width
    hh = entity.spec.hitbox_height
    if hw <= 1 and hh <= 1:
        return _dist(entity.x, entity.y, target.x, target.y)
    x_off = np.linspace(-(hw - 1) / 2.0, (hw - 1) / 2.0, hw)  # shape (hw,)
    y_off = np.linspace(-(hh - 1) / 2.0, (hh - 1) / 2.0, hh)  # shape (hh,)
    # Grille cartésienne hw × hh de points du hitbox
    gx = entity.x + x_off[:, None]          # (hw, 1)  → broadcast → (hw, hh)
    gy = entity.y + y_off[None, :]          # (1,  hh) → broadcast → (hw, hh)
    dists = np.hypot(gx - target.x, gy - target.y)
    return float(dists.min())


STUCK_TICKS_RESET = 25   # ticks sans déplacement avant reset de la cible

# Anti-empilement (espace personnel) : la répulsion historique ne tournait QUE dans _beh_wander →
# les humains OCCUPÉS (agriculture, coupe, chantier…) s'entassaient jusqu'à ~12 sur une tuile (bug
# visuel révélé par les sprites nets). Cette séparation POSITIONNELLE s'applique à TOUS les humains
# chaque tick, à très courte portée (quasi-chevauchement seulement) : elle NUDGE la position sans
# toucher la CIBLE de travail → le regroupement au travail (à distance 0.6+) est préservé.
_SEPARATE_ON = os.environ.get("SEPARATE_OFF") != "1"   # kill-switch d'imputation → hash pré-fix exact
SEPARATE_DIST = 0.75     # deux humains plus proches que ça se poussent (anti-chevauchement)
SEPARATE_PUSH = 0.5      # amplitude du nudge : assez forte pour SORTIR de la tuile partagée
                         # (0.35 était trop faible → ils restaient empilés ; 0.5 → max ~2/tuile)

# ── Religion & culture (P5, chantier E) ────────────────────────────────────────────────
# Cultes (E1), fête des moissons (E2), monuments (E3), héros & annales (E4). Kill-switch MASTER
# CULTURE_OFF → hashes HEAD exacts (toute la culture éteinte) ; sous-switches par bloc (chacun
# ⊂ CULTURE, convention SWARM ⊂ UNREST). AUCUN tirage `random` neuf : les NOMS sont générés par
# arithmétique pure (hash entier), les effets probabilistes (fête) ne bougent que des SEUILS.
_CULTURE_ON  = os.environ.get("CULTURE_OFF") != "1"
_CULTS_ON    = _CULTURE_ON and os.environ.get("CULTS_OFF")    != "1"
_FEAST_ON    = _CULTURE_ON and os.environ.get("FEAST_OFF")    != "1"
_MONUMENT_ON = _CULTURE_ON and os.environ.get("MONUMENT_OFF") != "1"
_HEROES_ON   = _CULTURE_ON and os.environ.get("HEROES_OFF")   != "1"
# ── Économie profonde (P6, chantier F) — master ECON_OFF → hashes pré-P6 exacts ; sous-switches
# par bloc ⊂ ECON (convention CULTURE). Zéro tirage neuf : décisions par seuils + tris.
_ECON_ON  = os.environ.get("ECON_OFF") != "1"
_MONEY_ON = _ECON_ON and os.environ.get("MONEY_OFF") != "1"   # F1 : or-monnaie au marché
_ENVY_ON  = _ECON_ON and os.environ.get("ENVY_OFF")  != "1"   # F2 : l'envie réordonne le choix de cible
# ── Migration & exploration (P7, chantier G) — master EXPLO_OFF → hashes P6 exacts ;
# _CARTO_ON (G1) est le SOCLE : G2/G3/G4 consomment ses données, donc sans carto tout P7
# est inerte par construction. Zéro tirage neuf : les missionnés sont pilotés hors flux RNG
# (pattern caravanes/pèlerins), le bloc scout 4.7 existant garde son flux intact.
_EXPLO_ON  = os.environ.get("EXPLO_OFF")  != "1"
_CARTO_ON  = _EXPLO_ON and os.environ.get("CARTO_OFF")  != "1"   # G1 catalogue + expéditions
_COLONY_ON = _CARTO_ON and os.environ.get("COLONY_OFF") != "1"   # G2 colonies lointaines
_MIGRATE_ON= _CARTO_ON and os.environ.get("MIGRATE_OFF")!= "1"   # G3 migration de village
_TOPO_ON   = _CARTO_ON and os.environ.get("TOPO_OFF")   != "1"   # G4 toponymes & annales
# ── P8 « la relance de l'Histoire » : casser les verrous de la fin de partie ────────────
_P8_ON     = os.environ.get("P8_OFF") != "1"                      # master
_COUPCD_ON = _P8_ON and os.environ.get("COUPCD_OFF") != "1"       # H2 cooldown du coup d'État
_ENVYALLY_ON = _P8_ON and os.environ.get("ENVYALLY_OFF") != "1"   # H1 l'envie éteint l'alliance
# H1 — une alliance ne survit pas à l'INJUSTICE DURABLE. L'envie n'a toujours pas le droit de
# CRÉER une guerre (principe F2 conservé) : elle ronge la RELATION, et la guerre redevient
# simplement possible par les canaux existants (sortie d'hystérésis alliée, ciblage P2b inchangé).
# PRODUIT CROISÉ, sens vérifié : « le riche vaut au moins D/N fois le pauvre » s'écrit
# `pauvre * D <= riche * N`. La spec l'avait écrit dans l'autre sens (`pauvre * N <= riche * D`,
# déclenchement sur FAUX) — or c'est un TRUISME dès que riche >= pauvre, donc l'érosion n'aurait
# JAMAIS tiré et H1 serait né mort. Le bloc censé relancer l'Histoire aurait livré du contenu
# inerte, invisible autrement qu'à la sonde d'acceptation.
ENVY_ALLY_RATIO_N = 1    # \  pauvre * D <= riche * N  <=>  riche >= (D/N) * pauvre
ENVY_ALLY_RATIO_D = 3    # /   ici : le riche vaut au moins le TRIPLE du pauvre
# Plancher absolu, CALIBRÉ sur la distribution et non a priori : deux clans misérables ne se
# jalousent pas, il n'y a rien à envier. Mesuré sur les 3 seeds (paires alliées, 24000 t) : la
# richesse du RICHE a un p10 de 74-110, donc un plancher à 60 — la valeur de départ de la spec —
# passait SOUS toute la distribution et ne filtrait rien (29->28, 10->10, 6->6 : garde morte).
# À 200 il mord sur le vrai bruit (des paires à 74 contre 0) et laisse tranquilles les mondes
# jeunes déjà turbulents ; la cible réelle, le duopole installé du monde servi, le franchit d'un
# facteur 363 (riche 72604).
ENVY_ALLY_MIN     = 200  # richesse minimale du RICHE pour qu'il y ait matière à envie
ENVY_ALLY_EROSION = 4    # points de relation rongés par éval — SEUL levier si c'est trop LENT
                         # (le ratio et le plancher répondent à d'autres questions : leçon A10)
# H2 — le coup renverse un chef, il ne dissout pas la pression STRUCTURELLE. Sans garde, le coup
# à 70 préempte éternellement la scission à 90 dès n>1 : la surextension pompe, le coup purge −40,
# et le cycle 30↔70 tourne à vide (69 coups, 0 scission sur le monde live en 1,77 M de ticks).
# VALEUR CALIBRÉE, PAS CHOISIE. Sur le save live, le clan pauvre et peuplé monte de +9 de tension
# par éval : après un coup sa tension repart de 30 et fait 39-48-57-66-75-84-93, donc elle repasse
# 70 à la 5ᵉ éval (3600 t) et n'atteint 90 qu'à la 7ᵉ (5040 t). Une garde de 500·TS = 3000 t serait
# EXPIRÉE avant le retour à 70 → le coup retirerait, purgerait, et la scission n'arriverait jamais :
# on aurait reconstruit le verrou avec une étape de plus. 800·TS = 4800 reste SOUS les 5040 requis
# — le piège du seuil qu'on croit franchi. 1000·TS couvre dt≥7 avec une éval et demie de marge.
# La sensibilité est ASSUMÉE : un clan à faible pression (dt≤5) garde son churn de coups, c'est le
# grand clan surétendu qui doit se fracturer.
COUP_COOLDOWN = 1000 * TIME_SCALE   # période de CALENDRIER (règle A4) : on ne renverse pas un
                                    # chef deux fois dans la même saison
SCOUT_PERIOD     = 150 * TIME_SCALE  # période de dispatch d'expédition par clan (déphasée) → durée
SCOUT_PHASE      = 37   # déphasage par clan (premier avec SCOUT_PERIOD → les clans ne partent pas ensemble)
KNOWN_SITES_MAX  = 12   # sites mémorisés par clan (on garde les mieux scorés)
SITE_DISCOVER_R  = 4.0  # un site passé à cette distance d'un humain en mission est DÉCOUVERT
EXPEDITION_HOME_R = 6.0 # mission close quand l'éclaireur est rentré à cette distance du feu
# Anti-zombie. C'est un budget de TRAJET (aller + retour), pas une durée biologique : il ne
# suit donc PAS TIME_SCALE, exactement comme TRADE_TIMEOUT=1200 dont il est le frère (la
# vitesse de marche, elle, n'a pas été rescalée). 3× ce budget parce que la mission fait
# l'aller-retour sur une carte 2× plus grande — soit ~690 tuiles parcourues, au-delà de la
# diagonale du gabarit servi. Court exprès : une mission qui s'éternise gèle l'exploration
# de son clan (une seule en vol), donc le filet doit tomber en 4 périodes de dispatch max.
EXPEDITION_TIMEOUT = 3600
SITE_MIN_AGE     = 1    # âge technologique minimal pour explorer (Pierre)
SITE_MIN_POP     = 6    # pop minimale du clan pour détacher un éclaireur
SCOUT_HUNGER_MAX = 65   # seuils d'éligibilité — MÊMES valeurs que le bloc d'exploration 4.7
SCOUT_THIRST_MAX = 55
EXPLORE_MARK_PERIOD = 25       # ticks entre deux marquages de la carte explorée (cosmétique)
EXPLORE_MARK_R      = 2        # rayon de marquage autour d'un humain
EXPLORE_MARK_R_SCOUT = 4       # rayon de marquage autour d'un éclaireur en mission (il regarde loin)
# G2 colonies lointaines. Le seuil de distance est RELATIF au monde (arbitrage A6) : « lointain »
# n'a pas le même sens sur un gabarit de test et sur la carte servie, et un absolu ne peut pas être
# juste sur les deux. Mesuré avant de câbler (leçon des contenus morts E2/E3) : l'éclaireur vise
# toujours le site inconnu LE PLUS PROCHE, donc la carte d'un clan se remplit du proche vers le
# lointain (médiane 22, max 41 sur 220x160 en 12000 t) pendant que l'essaimage part TÔT — un seuil
# absolu à 60 n'aurait jamais firé. Division ENTIÈRE : aucun flottant dans un seuil.
COLONY_MIN_SCORE = 9    # une colonie lointaine se fonde sur une BONNE terre (médiane du catalogue)
SITE_OCCUPIED_R  = 6    # un site est OCCUPÉ si un feu de camp vivant est à cette distance
COLONIST_TIMEOUT = 3600 # budget de marche des colons (même échelle que l'expédition, plat)
# A7 : la terre lointaine ne l'emporte sur la ruine d'à côté que si elle est NETTEMENT
# meilleure. Sans cette marge, il suffit qu'UNE terre lointaine soit connue — condition
# vraie tôt et durablement — pour que TOUS les essaimages partent au loin : mesuré 6 sur 6
# sur deux gabarits, et la recolonisation des ruines (E8) devenait du contenu mort, alors
# que le cycle des empires a besoin que les ruines se repeuplent.
SWARM_DIRECT_MARGIN = 4  # provisoire, calibré par sonde : il faut les DEUX contenus vivants
# G3 migration de village. DEUX exodes qui racontent deux histoires différentes (arbitrage A8) :
# le PULL est l'appel d'une terre bien meilleure, le PUSH la fuite d'une terre épuisée.
# PULL — gain ABSOLU, pas de ratio (arbitrage a2eaf725) : le coût d'une migration est FIXE (tout
# rebâtir, matériaux perdus, cooldown), donc son déclencheur doit l'être. Un ratio comparerait un
# gain proportionnel à un coût qui ne l'est pas — un clan à 6 déménagerait pour +4 points.
# Le site est jugé sur la conv FIGÉE (le potentiel que l'éclaireur a rapporté), le local sur la
# conv FRAÎCHE (la réalité vécue) : asymétrie ASSUMÉE et MESURÉE — sur le monde live à 64800 t,
# les 23 sites inoccupés n'ont perdu que 1,65 point en moyenne (pire cas 4), donc la promesse du
# figé est surestimée d'environ 2 points et la marge de 10 tient largement.
MIGRATE_MIN_DIST = 40   # en deçà, ce n'est pas une migration : le village bougerait pour rien
MIGRATE_MIN_GAIN = 10   # le site connu doit valoir CELA de plus que le terroir vécu
# PUSH — un COUPLE, pas un seuil (arbitrage A8) : chute relative ET plancher de misère, les DEUX
# exigés. Chaque garde veto une famille que l'autre laisserait passer, et les deux sont mesurées :
#  - sans le PLANCHER, un clan RICHE qui se dégrade fuirait une terre encore excellente
#    (sonde seed 7 : c5 tombe de 34 à 21 — 21 reste le double de la médiane du catalogue) ;
#  - sans le RATIO, un clan NÉ pauvre fuirait sa naissance et non son épuisement
#    (sonde déployée : c1 et c2 vivent à 2 sur une terre qui VAUT 2 — rien ne s'est épuisé).
# Les clans nés pauvres relèvent du PULL, pas du PUSH : le PUSH réserve l'exode-nécessité aux
# terres qui se sont VRAIMENT effondrées.
# ARITHMÉTIQUE ENTIÈRE OBLIGATOIRE (même discipline que le //4 d'A6) : le ratio s'écrit en PRODUIT
# CROISÉ, jamais en flottant — `frais <= figé x N/D` devient `frais * D <= figé * N`.
# CALIBRÉ sur 48000 ticks-monde (sonde probe_depletion.py, 24000 t x 2 gabarits, relevé /1500 t).
# Le couple retenu se lit en une phrase : « une terre qui valait au moins le DOUBLE, et qui est
# tombée au niveau du PIRE site du monde ». Il isole l'unique effondrement authentique des deux
# sondes (seed 7 c2 : 10 -> 4, soit -60 %) et n'attrape rien d'autre.
# Pourquoi 1/2 et pas 2/3, qui donnerait pourtant un taux plus proche de la cible : à 2/3 le
# gabarit déployé fait déclencher c6 (6/9), un clan NÉ sur une terre à 9 — le minimum mesuré du
# catalogue — et qui a perdu 3 points. Rien ne s'y est effondré, il n'a jamais rien eu ; sur une
# base figée aussi basse, une perte absolue minuscule suffit à tromper un ratio lâche. C'est
# précisément la famille qu'A8 réserve au PULL. Le ratio serré préfère MANQUER un cas douteux
# plutôt que d'exiler un clan de sa naissance.
MIGRATE_PUSH_RATIO_N = 1   # numérateur   \  frais * D <= figé * N
MIGRATE_PUSH_RATIO_D = 2   # dénominateur /  (1/2 : le terroir a perdu au moins la moitié)
MIGRATE_PUSH_FLOOR   = 6   # ... ET il faut être tombé dans la misère (min du catalogue mesuré : 6-9)
MIGRATE_PUSH_EVALS   = 2   # évals consécutives (hystérésis P2) : un creux passager n'exile personne
MIGRATION_COOLDOWN   = 3000  # budget de DISTANCE/rythme, PLAT (règle A4 : seules les périodes de
                             # calendrier suivent TIME_SCALE) — délai APRÈS UNE MIGRATION
# A10 — délai APRÈS UNE FONDATION, sémantique DIFFÉRENTE du cooldown (les confondre écrasait la
# sentinelle « jamais migré » et rendait les deux délais indiscernables). Calibré, pas inventé :
# le churn mesuré part à 240-480 ticks de vie, les départs légitimes à 3120-5040. On CALE BAS dans
# le trou, pas au milieu : le coût est ASYMÉTRIQUE — trop long stérilise un monde entier (mesuré :
# 0 départ sur 40000 t du gabarit déployé), trop court ne coûte qu'un départ précoce de plus.
# FIGÉ à 500 = la plus petite valeur ronde STRICTEMENT au-dessus du paquet churn mesuré (max 480),
# et STRICTEMENT en dessous de la première opportunité RÉELLE du gabarit déployé (720 : c6 né
# t=15840 parti t=16560, c8 né t=20880 parti t=21600). Le trou utile est [480, 720] — étroit, d'où
# le calage bas. 750 aurait re-stérilisé le déployé en emportant ses seules opportunités.
# NB : un clan ne peut PAS migrer le jour de sa naissance — `due` est calculé avant les scissions,
# donc la première éval d'un clan neuf tombe au plus tôt un déphasage plus tard.
MIGRATION_SETTLE     = 500
# G4 — une découverte ne fait date que si elle est LOINTAINE. Mesuré en G1 : la carte d'un clan se
# remplit du proche vers le lointain (médiane 22 tuiles) ; sans ce filtre les annales se noieraient
# sous des « découvertes » situées à une promenade du feu.
TOPO_ANNAL_DIST   = 100
# Le feu de camp a build_time=0 (il est POSÉ à la fondation, jamais bâti) : la spec fixe donc le
# coût du chantier au temps d'une MAISON x2 — le village doit peiner pour renaître.
# CORRIGÉ (finding d'ultra-audit) : je lisais BUILDING_SPECS["house"].build_time à la DÉFINITION
# (40), or entities.py le scale par TIME_SCALE APRÈS — la valeur réelle est 240. Mon 80 valait
# donc un SIXIÈME de la friction voulue, et la constante avait été calibrée sur cet écart.
MIGRATION_WORK    = 480   # = 2 x build_time RÉEL de la maison (240), vérifié à l'exécution
MIGRATE_MIN_POP   = 8     # un hameau ne déménage pas : il n'a pas les bras pour rebâtir
# Budget de trajet, donc PLAT (règle A4, précédent TRADE_TIMEOUT/EXPEDITION_TIMEOUT). Il doit
# couvrir la marche gloutonne d'un village entier PUIS 80 ticks de travail ; mesuré côté G1 : des
# trajets de 237 tuiles aboutissent, et 20,5 % échouent pour cause de TERRAIN, pas de distance.
# Sans ce timeout, un clan visant une terre greedy-inatteignable resterait paralysé À VIE — tous
# ses bâtisseurs happés par le bloc 4.1 vers un chantier qu'ils n'atteindront jamais.
MIGRATION_TIMEOUT = 5400
_TRAILS_ON = _ECON_ON and os.environ.get("TRAILS_OFF") != "1"  # F3 : sentiers d'usure (cosmétique pur)
_GRANARY_ON = _ECON_ON and os.environ.get("GRANARY_OFF") != "1"  # F4 : moulin L2 = grenier + famine par les réserves
MILL_L2_BREAD_MULT = 3   # F4 : cap de pains du moulin L2 (5 → 15) — il STOCKE, il n'accélère pas
FAMINE_EXIT_BREAD  = 2   # F4 : réserve minimale pour SORTIR de la famine (anti-clignotement)
TRAIL_DECAY_PERIOD = 60 * TIME_SCALE   # période d'estompage des sentiers → durée (un sentier vit des saisons)
TRAIL_MAX = 65535                      # saturation uint16 (jamais de wrap : un chemin très fréquenté plafonne)
MARKET_GOLD_MAX = 8   # F1 : cap du coffre d'or du marché (débordement → dorure de l'église du clan)
MONEY_RESTOCK   = 3   # F1 : le marché tire une expédition d'or si son coffre < ce seuil (demande-pull)
MONEY_PAY_VALUE = 4   # F1 : préférence de l'or en paiement (> max _scarcity bois/pierre = 3 → toujours préféré)


_CHAMPS_CONNUS: dict = {}          # cls -> frozenset des noms de champs (mémorisé)


def _depuis_dict(cls, d: dict, ignorees: set):
    """Construit une dataclass depuis un save en ÉCARTANT les clés qu'elle ne connaît plus.
    Sans ce filtre, retirer un champ casse le chargement des parties existantes : `cls(**d)`
    lève TypeError sur la clé résiduelle et le monde ne rouvre plus. C'est une garde GÉNÉRALE
    (elle vaut pour tout retrait futur), pas un pansement pour un champ précis.
    Les clés écartées sont REMONTÉES à l'appelant : le filtre avale aussi les typos et les
    renommages, et sans trace un champ renommé se réinitialiserait aux défauts EN SILENCE."""
    champs = _CHAMPS_CONNUS.get(cls)
    if champs is None:
        champs = frozenset(f.name for f in _dc_fields(cls))
        _CHAMPS_CONNUS[cls] = champs
    inconnues = [k for k in d if k not in champs]
    if inconnues:
        ignorees.update(inconnues)
        d = {k: v for k, v in d.items() if k in champs}
    return cls(**d)


def _clan_wealth(cb: dict) -> int:
    """P6 F2 — richesse d'un clan depuis SES bâtiments déjà groupés (clan_bldg[cid]) : entiers purs,
    pondérations = rareté relative (bois 1, pierre 3, fer 6, pain 2, or/dorure 12). Pure fonction de
    l'état courant → DÉRIVABLE : jamais sérialisée, jamais mémorisée (recalcul au besoin ; le wire la
    recalcule chaque tick, sinon un load la figerait à 0 jusqu'à la prochaine éval due = replay divergent)."""
    w = 0
    for _bl in cb.values():
        for b in _bl:
            w += b.wood + 3 * b.stone + 6 * b.iron + 2 * b.bread + 12 * (b.gold + b.gilt)
    return w
CULT_CONVERT_REL = 60   # rel ≥ ce seuil (+ pop stricte) → le petit clan se convertit au culte du gros
HOLY_REL_PENALTY = -10  # guerre sainte (cultes ≠) : pénalité relationnelle à la déclaration
HOLY_TENSION     = 2    # guerre sainte : tension additionnelle par éval en mode war (+3 → +5)
# Fête des moissons (E2) : au 1er tick d'automne, un clan EN PAIX avec ≥ FEAST_FIELDS_MIN champs de
# blé MÛRS (wheatfield stage 4 = la moisson sur pied) tient une fête (1×/an). Elle ne CONSOMME rien
# (l'abondance est la porte, pas une dépense — calibration Regigigas : le trigger pain d'origine était
# du contenu mort, moulins Fer+ tardifs + pain mangé dès cuisson → réserve jamais ≥6 ; les champs mûrs
# s'accumulent dès le Bronze). Pendant FEAST_TICKS : les membres CONVERGENT vers le feu (rayon
# FEAST_RADIUS) et la natalité est facilitée (le SEUIL — cap de pop — est ×FEAST_BIRTH_MULT).
# Guerre/famine interrompt. Zéro RNG neuf : déclenchement/convergence 100 % déterministes ; seul un
# SEUIL change (la natalité produit alors plus de naissances via les tirages de portée DÉJÀ existants).
FEAST_FIELDS_MIN = 4           # champs de blé mûrs (stage 4) requis pour déclencher la fête
FEAST_TICKS      = 60 * TIME_SCALE   # durée de la fête → suit TIME_SCALE
FEAST_RADIUS     = 8           # rayon de convergence des membres autour du feu pendant la fête
FEAST_BIRTH_MULT = 2.0         # multiplicateur du cap de population pendant la fête (natalité facilitée)
# Monuments (E3) : à l'Âge Acier (age ≥ CHURCH_AGE), un clan érige UN monument de PRESTIGE (gloire du
# culte) — 20 pierre du POOL + 4 or du TRÉSOR d'église (comme l'offrande C2). À l'achèvement : tension
# one-shot −MONUMENT_TENSION (fierté apaisante). Ruine ×MULT (vestige durable = cible de recolonisation
# P4.1 gratuite). Porte une DÉDICACE (dernier jalon du clan, figée à la pose). Gated _MONUMENT_ON →
# MONUMENT_OFF = zéro planner/tirage → E2 hash exact. (MONUMENT_SCIENCE droppé : la science ne sert
# qu'aux âges et le monument exige déjà l'âge max → bonus mort ; 5e âge en réserve P6, cf. spec §10.)
MONUMENT_STONE     = 20        # pierre du pool clan pour le chantier
MONUMENT_GOLD      = 4         # or prélevé au trésor d'église (Building.gold) à la pose
MONUMENT_TENSION   = 10        # apaisement one-shot de la tension à l'achèvement (clampé ≥ 0)
MONUMENT_RUIN_MULT = 4         # ruin_ticks ×MULT pour une ruine de monument (vestige durable)
# Héros & annales (E4) : une entité est NOMMÉE (au fil de l'eau, pas d'éval dédiée) dès qu'elle
# franchit HERO_KILLS kills de guerre OU HERO_BUILDS bâtiments achevés, OU devient chef FONDATEUR.
# Nom = _hero_name(seed, entity.id, voie) — arithmétique pure. Sa mort entre aux ANNALES (hero_fallen).
HERO_KILLS  = 5                # kills de guerre → « le Sanglant/la Lame/… »
HERO_BUILDS = 10               # bâtiments achevés → « le Bâtisseur/l'Architecte/… »

@dataclass
class Cult:
    id: int
    name: str
    founder_clan: int
    founded_tick: int

# Générateur de noms déterministe (zéro RNG) : hash entier de (seed, id) indexe des tables de
# syllabes → nom stable inter-process ET après load. Deux styles (culte / héros via épithète).
_CULT_FORMS = ["Culte de {r}", "Voie d'{r}", "Ordre de {r}", "Foi de {r}", "Communion d'{r}", "Cercle de {r}"]
_NAME_ROOTS = ["Vornak", "Ithélis", "Karûn", "Sélene", "Drakh", "Ombrelin", "Vael", "Thoros", "Ysgar",
               "Miréthil", "Corvak", "Nûl", "Azhen", "Brûm", "Célios", "Dûnmar", "Éphra", "Faelis",
               "Gorlan", "Haldûr", "Ivrel", "Jorven", "Kûmara", "Lûthien", "Morvane", "Nyx", "Ophir",
               "Pyrelis", "Quorin", "Rhael", "Sûldis", "Tûvok", "Ûrsel", "Vhalor", "Wyrn", "Xareth",
               "Ythar", "Zûl", "Alvra", "Belok"]
_HERO_EPITHETS = {"kills": ["le Sanglant", "la Lame", "l'Implacable", "le Boucher", "la Furie"],
                  "builds": ["le Bâtisseur", "l'Architecte", "la Maçonne", "le Fondateur d'œuvres", "la Pilière"],
                  "founder": ["la Fondatrice", "le Fondateur", "l'Émancipé", "la Sécessionniste", "le Rebelle"]}

def _hash2(seed: int, x: int) -> int:
    return ((seed & 0xFFFFFFFF) * 2654435761 + x * 40503) & 0xFFFFFFFF

def _cult_name(seed: int, cult_id: int) -> str:
    h = _hash2(seed, cult_id)
    return _CULT_FORMS[h % len(_CULT_FORMS)].format(r=_NAME_ROOTS[(h >> 8) % len(_NAME_ROOTS)])

# P7 G4 — toponymes. Un lieu se nomme par sa FORME et par ce qui y pousse : le relief vient du
# nom, la nature de ce que la conv() a effectivement trouvé sur place. Pure arithmétique de
# (seed, site_id) comme les cultes E1 — AUCUN état, donc rien à sérialiser, rien à faire dériver
# au rechargement, et deux processus qui ouvrent la même partie lisent les mêmes noms.
_SITE_FORMS = ["les Hauts de {r}", "la Combe de {r}", "le Val de {r}", "les Rives de {r}",
               "le Plateau de {r}", "la Clairière de {r}", "les Terres de {r}", "le Passage de {r}",
               "la Butte de {r}", "les Sources de {r}", "le Creux de {r}", "la Pointe de {r}"]
_SITE_ROOTS = ["l'Aube", "l'Ombre", "la Brume", "l'Écho", "la Pierre-Grise", "l'Orme", "la Sente",
               "la Corneille", "l'Ambre", "la Bruyère", "le Silence", "la Loutre", "l'Ardoise",
               "la Fougère", "le Vent", "la Marge", "l'Aulne", "la Cendre", "le Renard", "la Mousse",
               "l'Ajonc", "la Genèse", "le Cerf", "la Ronce"]


def site_name(seed: int, site_id: int) -> str:
    """P7 G4 — nom d'un site, DÉRIVÉ de (seed, site_id) et de rien d'autre. Même patron que
    `_cult_name` : aucun état, donc aucun risque qu'un save porte un nom qui ne correspond plus
    à son lieu (le défaut qu'on avait déjà corrigé sur le catalogue lui-même en G1)."""
    h = _hash2(seed, site_id * 7 + 1)
    nom = _SITE_FORMS[h % len(_SITE_FORMS)].format(r=_SITE_ROOTS[(h >> 8) % len(_SITE_ROOTS)])
    # Contraction : « de le Silence » n'existe pas en français. Les racines féminines (« la ») et
    # élidées (« l' ») se composent seules, seul le masculin demande « du ».
    return nom.replace(" de le ", " du ")


def a_lieu(nom: str) -> str:
    """« à » + un toponyme, contracté. Les noms de lieux portent leur article (« les Rives »,
    « le Val », « la Combe », « l'Aube ») : sans ça les annales écriraient « à les Rives ».
    Séparé de `site_name` parce que TOUTES les tournures ne contractent pas — « quitte ses terres
    POUR les Rives » est correct tel quel, et seule la tournure en « à » doit être traitée."""
    if nom.startswith("les "):
        return "aux " + nom[4:]
    if nom.startswith("le "):
        return "au " + nom[3:]
    return "à " + nom                     # « la … » et « l'… » se composent seuls


def _hero_name(seed: int, entity_id: int, via: str) -> str:
    h = _hash2(seed, entity_id)
    epis = _HERO_EPITHETS.get(via, _HERO_EPITHETS["kills"])
    return f"{_NAME_ROOTS[h % len(_NAME_ROOTS)]} {epis[(h >> 8) % len(epis)]}"

def _name_hero(entity, via: str, world, events: list) -> None:
    """P5 E4 — NOMME `entity` héros (voie kills/builds/founder) si pas déjà nommé. Idempotent.
    Nom déterministe (seed, id) → stable inter-process + après load. Émet hero_named ; la chronique
    d'annales est distillée depuis l'event (pas ici). Gated _HEROES_ON par l'appelant."""
    if entity.hero_name is not None:
        return
    entity.hero_name = _hero_name(world.seed, entity.id, via)
    events.append({"type": "hero_named", "entity_id": entity.id, "clan_id": entity.clan_id,
                   "name": entity.hero_name, "via": via})

def _separate_human(entity: Entity, entity_grid, all_entities, world: "World"):
    """Écarte `entity` (humain) de ses voisins humains en quasi-chevauchement. Répulsion SOMMÉE
    (somme des directions inverses, pondérée par la proximité) : dé-tasse les amas sans casser le
    regroupement au travail (la cible n'est jamais touchée). Nudge de POSITION seul, déterministe
    (positions + tie-break id pour le cas superposé, zéro RNG)."""
    scan = (_grid_neighbors(entity_grid, entity.ix, entity.iy, reach=1)
            if entity_grid is not None else all_entities)
    px = py = 0.0; n = 0
    for other in scan:
        if other is entity or not other.alive or other.etype != EntityType.HUMAN:
            continue
        dx = entity.x - other.x; dy = entity.y - other.y
        d2 = dx * dx + dy * dy
        if d2 < SEPARATE_DIST * SEPARATE_DIST:
            if d2 < 1e-6:   # exactement superposés : direction déterministe par id (pas de RNG)
                ang = (entity.id & 7) * 0.7853981633974483   # id%8 × π/4
                px += math.cos(ang); py += math.sin(ang)
            else:
                # pondération par la proximité (1/d) : le voisin le plus proche pèse le plus →
                # une paire serrée se sépare franchement, un amas symétrique ne s'annule pas tout à fait.
                d = math.sqrt(d2); w = 1.0 / d
                px += (dx / d) * w; py += (dy / d) * w
            n += 1
    if not n:
        return
    norm = math.sqrt(px * px + py * py)
    if norm < 1e-6:
        return
    nx = max(0.0, min(world.width  - 0.01, entity.x + (px / norm) * SEPARATE_PUSH))
    ny = max(0.0, min(world.height - 0.01, entity.y + (py / norm) * SEPARATE_PUSH))
    if world.is_walkable(int(nx), int(ny)):
        entity.x = nx; entity.y = ny

def _move_toward(entity: Entity, tx: float, ty: float, speed: float, world: "World"):
    """Déplace l'entité d'un pas vers (tx, ty), sans dépasser la vitesse max.
    Si le chemin direct est bloqué (eau), glisse le long de l'obstacle.
    Si vraiment bloqué (STUCK_TICKS_RESET ticks), reset la cible et essaie une échappée."""
    dx = tx - entity.x
    dy = ty - entity.y
    d  = math.sqrt(dx * dx + dy * dy)
    if d < 0.01:
        entity._stuck_ticks = 0
        return
    step = min(speed, d)
    nx = entity.x + (dx / d) * step
    ny = entity.y + (dy / d) * step
    aq = entity.spec.aquatic
    ox, oy = entity.x, entity.y   # position avant mouvement
    moved = False
    if world.is_walkable(int(nx), int(ny), aq):
        entity.x = max(0, min(world.width  - 0.01, nx))
        entity.y = max(0, min(world.height - 0.01, ny))
        moved = True
    elif world.is_walkable(int(nx), int(entity.y), aq):
        entity.x = max(0, min(world.width  - 0.01, nx))
        moved = True
    elif world.is_walkable(int(entity.x), int(ny), aq):
        entity.y = max(0, min(world.height - 0.01, ny))
        moved = True

    # Micro-mouvement (< 0.01 tile/tick) : traité comme bloqué pour éviter les boucles
    # de virgule flottante (ex : entité à x=42.999 glissant vers x=43.0 au rythme de 0.0001/tick)
    really_moved = moved and ((entity.x - ox)**2 + (entity.y - oy)**2 >= 0.0001)
    if really_moved:
        entity._stuck_ticks = 0
        if _TRAILS_ON and entity.etype == EntityType.HUMAN:
            # F3 SENTIER : la tuile foulée s'use (uint16 saturant). O(1), COSMÉTIQUE — aucune
            # décision ne relit cette grille (pas de rétroaction mouvement→grille→mouvement) et
            # elle sort hors payload step() (/api/trails) → hash déterministe intact.
            _tg = world.trail_grid
            _ty, _tx = entity.iy, entity.ix
            if _tg[_ty, _tx] < TRAIL_MAX:
                _tg[_ty, _tx] += 1
    else:
        entity._stuck_ticks += 1
    if entity._stuck_ticks >= STUCK_TICKS_RESET:
        # Cible inaccessible : reset + annule chantier planifié + échappée aléatoire
        world._stuck_resets += 1   # métrique I0 (hors sortie step → hash neutre)
        entity.target_x = None
        entity.target_y = None
        entity._build_target_type = None
        entity._build_target_x    = None
        entity._build_target_y    = None
        entity._stuck_ticks = 0
        dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        random.shuffle(dirs)
        for ddx, ddy in dirs:
            ex = max(0.0, min(world.width  - 0.01, entity.x + ddx))
            ey = max(0.0, min(world.height - 0.01, entity.y + ddy))
            if world.is_walkable(int(ex), int(ey), aq):
                entity.x, entity.y = ex, ey
                break


def _teleport_to_nearest_walkable(entity: Entity, world: "World") -> bool:
    """Renvoie une entité échouée sur une tuile non-franchissable (terrestre sur
    l'eau, ou l'inverse) vers la tuile franchissable la PLUS PROCHE. Balaie des
    anneaux complets à rayon croissant : l'ancien filet ne testait que 8 offsets
    par anneau, ratant la plupart des tuiles → une entité pouvait rester coincée
    à vie. Sort au premier anneau contenant une tuile valide. Retourne True si
    déplacée."""
    aq = entity.spec.aquatic
    ix, iy = entity.ix, entity.iy
    best = None
    best_d = None
    for r in range(1, max(world.width, world.height) + 1):
        x_lo, x_hi, y_lo, y_hi = ix - r, ix + r, iy - r, iy + r
        # rangées haute/basse de l'anneau, puis colonnes gauche/droite (hors coins)
        candidates = [(nx, y) for nx in range(x_lo, x_hi + 1) for y in (y_lo, y_hi)]
        candidates += [(x, ny) for ny in range(y_lo + 1, y_hi) for x in (x_lo, x_hi)]
        for nx, ny in candidates:
            if world.is_valid(nx, ny) and world.is_walkable(nx, ny, aq):
                d = (nx - ix) ** 2 + (ny - iy) ** 2
                if best_d is None or d < best_d:
                    best, best_d = (nx, ny), d
        if best is not None:      # premier anneau non vide = le plus proche
            entity.x = float(best[0]) + 0.5
            entity.y = float(best[1]) + 0.5
            entity.target_x = None
            entity.target_y = None
            return True
    return False


def _random_walk(entity: Entity, world: "World"):
    """Déambule aléatoirement."""
    tx, ty = entity.target_x, entity.target_y
    if tx is None or (
        (entity.x - tx)**2 + (entity.y - ty)**2 < 0.25
    ):
        entity.target_x = None
        entity.target_y = None
        for radius in (5, 12, 25):
            for _ in range(12):
                tx = entity.x + random.uniform(-radius, radius)
                ty = entity.y + random.uniform(-radius, radius)
                tx = max(0, min(world.width  - 1, tx))
                ty = max(0, min(world.height - 1, ty))
                if world.is_walkable(int(tx), int(ty), entity.spec.aquatic):
                    entity.target_x = tx
                    entity.target_y = ty
                    break
            if entity.target_x is not None:
                break
    if entity.target_x is not None:
        _move_toward(entity, entity.target_x, entity.target_y,
                     entity.traits["speed"] * 0.6, world)


MAX_CARRY = 5          # bois max transportable par un humain
MAX_WOOD_PER_HOUSE = 50  # bois max stocké par maison individuelle
CLAN_WOOD_CAP = 100      # total bois clan → arrête de couper (laisse de la place aux autres tâches)
MAX_PER_SPECIES = 200  # plafond max par espèce (défaut EntitySpec.max_pop ; caps réels
                       # PAR espèce dans entities.py — prédateurs plus bas, cf. bloc PRÉDATION I1).
                       # Reste le majorant global (tous les caps <= 200) : sert de plafond de test.
REBOUND_FLOOR = 50     # I3 : sous ce seuil, une espèce ANIMALE recolonise 2× plus vite
                       # (cooldown repro halvé) → anti-cascade. Complète le plancher <30 de
                       # préservation de proie (qui coupe la prédation). Humains exclus.

HERD_RADIUS = 8    # tuiles de détection des congénères
CHOP_COOLDOWN = 12 * TIME_SCALE   # ticks entre 2 coupes → durée
HERD_MIN    = 2    # membres min pour déclencher le comportement de troupeau


def _herd_move(entity: Entity, all_entities: list[Entity], world: "World",
               entity_grid: dict = None) -> bool:
    """Déplace vers le centroïde du troupeau proche. Retourne True si appliqué.
    `entity_grid` (grille spatiale) évite de scanner toutes les entités ; fallback
    sur `all_entities` si non fourni (ex. tests unitaires)."""
    sx, sy, count = 0.0, 0.0, 0
    herd_r_sq = HERD_RADIUS * HERD_RADIUS
    ex, ey = entity.x, entity.y
    etype = entity.etype
    _scan = (_grid_neighbors(entity_grid, entity.ix, entity.iy, reach=1)
             if entity_grid is not None else all_entities)
    for e in _scan:
        if e is entity or not e.alive or e.etype != etype:
            continue
        dx = ex - e.x; dy = ey - e.y
        if dx*dx + dy*dy <= herd_r_sq:
            sx += e.x
            sy += e.y
            count += 1
            if count >= 15:   # cap pour ne pas scanner tout le monde
                break
    if count < HERD_MIN:
        return False
    cx, cy = sx / count, sy / count
    dx = ex - cx; dy = ey - cy
    if dx*dx + dy*dy < 6.25:
        # Déjà au cœur du troupeau → disperser activement
        angle = random.uniform(0, 2 * math.pi)
        spread_r = random.uniform(3, 7)
        tx = max(0, min(world.width  - 1, entity.x + math.cos(angle) * spread_r))
        ty = max(0, min(world.height - 1, entity.y + math.sin(angle) * spread_r))
        if world.is_walkable(int(tx), int(ty), entity.spec.aquatic):
            entity.target_x = tx
            entity.target_y = ty
            _move_toward(entity, tx, ty, entity.traits["speed"] * 0.7, world)
        return True
    # Léger bruit pour éviter les trajectoires rigides
    tx = max(0, min(world.width  - 1, cx + random.uniform(-2, 2)))
    ty = max(0, min(world.height - 1, cy + random.uniform(-2, 2)))
    if not world.is_walkable(int(tx), int(ty), entity.spec.aquatic):
        return False
    entity.target_x = tx
    entity.target_y = ty
    _move_toward(entity, tx, ty, entity.traits["speed"] * 0.5, world)
    return True


def _find_nearest(entity: Entity, entities: list[Entity],
                  etype: EntityType, max_dist: float) -> "Entity|None":
    best, best_d_sq = None, max_dist * max_dist
    ex, ey = entity.x, entity.y
    for e in entities:
        if e is entity or not e.alive or e.etype != etype:
            continue
        dx = ex - e.x; dy = ey - e.y
        d_sq = dx*dx + dy*dy
        if d_sq < best_d_sq:
            best_d_sq = d_sq
            best = e
    return best


def _find_predator_nearby(entity: Entity, entities: list[Entity],
                          predators: list = None,
                          predator_grid: dict = None) -> "Entity|None":
    fd = entity.spec.flee_distance
    fd_sq = fd * fd
    ex, ey = entity.x, entity.y
    etype = entity.etype
    # Grille spatiale : ne cherche que dans les cellules proches (8×8 tuiles)
    if predator_grid is not None:
        cx, cy = entity.ix >> 3, entity.iy >> 3
        for gcx in range(cx - 1, cx + 2):
            for gcy in range(cy - 1, cy + 2):
                for e in predator_grid.get((gcx, gcy), ()):
                    if not e.alive: continue
                    if etype in e.spec.prey_types:
                        dx = ex - e.x; dy = ey - e.y
                        if dx*dx + dy*dy <= fd_sq:
                            return e
        return None
    scan = predators if predators is not None else entities
    for e in scan:
        if not e.alive:
            continue
        if etype in e.spec.prey_types:
            dx = ex - e.x; dy = ey - e.y
            if dx*dx + dy*dy <= fd_sq:
                return e
    return None


def _grid_neighbors(grid: dict, ix: int, iy: int, reach: int = 1):
    """Itère les entités des cellules 8×8 autour de (ix,iy), sur `reach` cellules de
    rayon (3×3 par défaut ≈ 24×24 tuiles). Sert aux voisinages (troupeau, répulsion)
    pour éviter les scans O(n) par entité → complexité ~O(n) au lieu de O(n²)."""
    cx, cy = ix >> 3, iy >> 3
    for gcx in range(cx - reach, cx + reach + 1):
        for gcy in range(cy - reach, cy + reach + 1):
            yield from grid.get((gcx, gcy), ())


def _find_water_spot(entity: Entity, world: "World", max_r: int = 40) -> "tuple[float,float]|None":
    """Retourne le prochain pas (flow field) vers la tuile bord-eau la plus proche.
    L'entité appelle cette fonction à chaque tick : elle suit le flow field case par case,
    sans jamais tenter de traverser l'eau en ligne droite."""
    ex, ey = entity.ix, entity.iy
    if not world.is_valid(ex, ey):
        return None
    pos = world._nearest_water_tile[ey, ex]
    if int(pos[0]) >= 0:
        return (float(pos[0]), float(pos[1]))
    return None


def _drink_or_seek_water(entity: Entity, world: "World", events: list[dict],
                         buildings: list = None, cb: dict = None) -> bool:
    """Essaie de boire si adjacent à de l'eau ou à un puit, sinon cherche et va vers la source.
    Retourne True si l'action soif a été traitée (l'entité ne doit pas faire autre chose)."""
    # Vérifie si une tuile adjacente est de l'eau
    for adx in range(-1, 2):
        for ady in range(-1, 2):
            if adx == 0 and ady == 0:
                continue
            wx, wy = entity.ix + adx, entity.iy + ady
            if world.is_valid(wx, wy) and int(world.biome_grid[wy, wx]) in WATER_BIOMES:
                entity.state = State.DRINKING
                entity.thirst = max(0.0, entity.thirst - DRINK_AMOUNT)
                entity.target_x = None
                entity.target_y = None
                return True

    # Vérifie si un puit est adjacent (utilise l'index cb si disponible)
    _wells = (cb.get("well", []) if cb is not None
              else ([b for b in buildings if b.btype == "well"] if buildings else []))
    ex, ey = entity.x, entity.y
    for b in _wells:
        dx = ex - b.x; dy = ey - b.y
        if dx*dx + dy*dy < 2.25:
            entity.state = State.DRINKING
            entity.thirst = max(0.0, entity.thirst - WELL_DRINK_AMOUNT)
            entity.target_x = None
            entity.target_y = None
            return True

    # Pas d'eau adjacente : cherche un puit proche d'abord, sinon suit le flow field
    entity.state = State.SEEKING_WATER

    # Puit du clan en priorité (cible fixe jusqu'à arrivée)
    if _wells:
        nearest = min(_wells, key=lambda b: (ex - b.x)**2 + (ey - b.y)**2)
        entity.target_x = float(nearest.x)
        entity.target_y = float(nearest.y)
        _move_toward(entity, entity.target_x, entity.target_y,
                     entity.traits["speed"] * 1.2, world)
        return True

    # Flow field : recalcule le prochain pas à chaque tick depuis la position actuelle.
    # L'entité avance case par case sans jamais tenter de traverser l'eau en droite ligne.
    spot = _find_water_spot(entity, world)
    if spot:
        entity.target_x, entity.target_y = spot
        _move_toward(entity, entity.target_x, entity.target_y,
                     entity.traits["speed"] * 1.2, world)
        return True
    return False


def _try_craft(entity: Entity, house, events: list) -> bool:
    """Fabrique le prochain outil prioritaire finançable par `house` (bois/pierre
    stockés) : pioche bois > hache bois > pioche pierre > hache pierre > arrosoir >
    faucille > canne. Retourne True si un outil a été fabriqué. Partagé par le dépôt
    (3.5) et la fabrication découplée (C1bis) — sans ce découplage, quand le bois du
    clan est capé plus personne ne dépose donc plus personne ne fabrique, et les
    humains nés ensuite restent sans outil (chaîne pierre/bois figée)."""
    if entity.pick is None and house.wood >= PICK_WOOD_COST:
        house.wood -= PICK_WOOD_COST; entity.pick = "wood_pick"
        events.append({"type": "craft_pick", "pick": "wood_pick", "clan_id": entity.clan_id})
    elif entity.tool is None and house.wood >= AXE_CRAFT_COST:
        house.wood -= AXE_CRAFT_COST; entity.tool = "axe"
        events.append({"type": "craft_axe", "tool": "axe", "clan_id": entity.clan_id})
    elif entity.pick == "wood_pick" and house.stone >= STONE_PICK_COST:
        house.stone -= STONE_PICK_COST; entity.pick = "stone_pick"
        events.append({"type": "craft_pick", "pick": "stone_pick", "clan_id": entity.clan_id})
    elif entity.tool == "axe" and house.stone >= STONE_AXE_COST:
        house.stone -= STONE_AXE_COST; entity.tool = "stone_axe"
        events.append({"type": "craft_axe", "tool": "stone_axe", "clan_id": entity.clan_id})
    elif entity.watering_can is None and house.wood >= WATERING_CAN_WOOD_COST:
        house.wood -= WATERING_CAN_WOOD_COST; entity.watering_can = "watering_can"
        events.append({"type": "craft_watering_can", "clan_id": entity.clan_id})
    elif entity.sickle is None and house.stone >= SICKLE_STONE_COST:
        house.stone -= SICKLE_STONE_COST; entity.sickle = "sickle"
        events.append({"type": "craft_sickle", "clan_id": entity.clan_id})
    elif entity.fishing_rod is None and house.wood >= FISHING_ROD_WOOD_COST:
        house.wood -= FISHING_ROD_WOOD_COST; entity.fishing_rod = "fishing_rod"
        events.append({"type": "craft_fishing_rod", "clan_id": entity.clan_id})
    else:
        return False
    return True


def _spend_pool_stone(houses, amount: int) -> bool:
    """Dépense `amount` de pierre depuis le POOL des maisons du clan (greedy). La
    pierre étant rare et étalée (≤3/maison), un gros bâtiment comme la forge doit
    puiser dans plusieurs maisons. True (et déduit) si le total suffit, sinon False."""
    if sum(h.stone for h in houses) < amount:
        return False
    left = amount
    for h in houses:
        take = min(h.stone, left)
        h.stone -= take; left -= take
        if left <= 0:
            break
    return True


def _spend_pool_wood(houses, amount: int) -> bool:
    """Dépense `amount` de bois depuis le POOL des maisons (tout-ou-rien, greedy).
    Miroir exact de _spend_pool_stone. (D1 : chargement de caravane.)"""
    if sum(h.wood for h in houses) < amount:
        return False
    left = amount
    for h in houses:
        take = min(h.wood, left)
        h.wood -= take; left -= take
        if left <= 0:
            break
    return True


def _take_pool_stone(houses, want: int) -> int:
    """Prélève JUSQU'À `want` de pierre du pool (partiel autorisé). Retourne le
    prélevé réel. (D1 : le vendeur donne ce qu'il a au moment de l'échange.)"""
    got = 0
    for h in houses:
        take = min(h.stone, want - got)
        h.stone -= take; got += take
        if got >= want:
            break
    return got


def _tiered(pool: int, tiers) -> int:
    """((seuil, val), …) décroissant → val du 1er palier atteint, sinon 0. (D2)"""
    for th, v in tiers:
        if pool >= th:
            return v
    return 0


def _stone_rate(pool: int) -> int:
    return _tiered(pool, STONE_RATE_TIERS)


def _iron_rate(pool: int) -> int:
    return _tiered(pool, IRON_RATE_TIERS)


def _scarcity(pool: int, tiers) -> int:
    """3 disette / 2 rare / 1 aisance / 0 glut — valorisation d'un paiement par le
    vendeur (D2 : à deux paiements possibles, il prend le bien qui lui manque)."""
    a, b, c = tiers
    return 3 if pool < a else 2 if pool < b else 1 if pool < c else 0


def _take_forge_iron(forges, want: int) -> int:
    """Prélève JUSQU'À `want` de fer des forges (partiel). Miroir _take_pool_stone."""
    got = 0
    for f in forges:
        take = min(f.iron, want - got)
        f.iron -= take; got += take
        if got >= want:
            break
    return got


def _add_pool_wood(houses, n: int):
    """Répartit `n` bois dans les maisons en respectant MAX_WOOD_PER_HOUSE
    (surplus perdu = comportement d'origine du dépôt). (D1 : drain d'étal, abort.)"""
    for h in houses:
        space = MAX_WOOD_PER_HOUSE - h.wood
        if space <= 0:
            continue
        put = min(space, n)
        h.wood += put; n -= put
        if n <= 0:
            break


def _add_pool_stone(houses, n: int):
    """Ajoute `n` pierre à la 1re maison (la pierre n'est pas capée, cf. dépôt 3.5)."""
    if houses and n > 0:
        houses[0].stone += n


def _refund_cargo_gold(entity: Entity, _cb: dict) -> None:
    """P6 F1 : re-crédite une pièce d'or INVENDUE (refus/timeout) au coffre du MARCHÉ maison
    (cap MARKET_GOLD_MAX → dorure de l'église maison), repli au TRÉSOR de l'église si le marché a
    disparu, sinon perte BORNÉE (1 pièce). Symétrique du re-crédit bois/pierre invendu → aucune fuite,
    l'atomicité est préservée. À appeler AVANT _clear_trade. No-op si pas de pièce (sous MONEY_OFF)."""
    if not entity.cargo_gold:
        return
    _mk = _cb.get("market", [])
    if _mk:
        _g = _mk[0].gold + entity.cargo_gold
        _mk[0].gold = min(MARKET_GOLD_MAX, _g)
        _spill = _g - _mk[0].gold
        if _spill:
            _ch = _cb.get("church", [])
            if _ch:
                _ch[0].gilt += _spill
    else:                                    # marché disparu → repli au trésor de l'église
        _ch = _cb.get("church", [])
        if _ch:
            _g = _ch[0].gold + entity.cargo_gold
            _ch[0].gold = min(GOLD_TREASURY_MAX, _g)
            _ch[0].gilt += _g - _ch[0].gold
    entity.cargo_gold = 0


def _clear_trade(entity: Entity):
    """Efface proprement une mission caravane (D1)."""
    entity.trade_phase = None
    entity.trade_dest_cid = None
    entity.trade_ticks = 0
    entity.cargo_wood = 0
    entity.cargo_stone = 0
    entity.cargo_iron = 0
    entity.cargo_gold = 0    # P6 F1 : dernier filet (les chemins d'échec re-créditent AVANT via _refund_cargo_gold)
    entity.trade_good = None
    entity.trade_pay = None


def _beh_trade(entity: Entity, ctx, _cb, _eff_speed) -> bool:
    """Mission caravane (D1) : load (charge le bois au marché maison, atomique
    pool→cargo) → out (échange au marché destination : bois → étal B, pierre du
    pool B → cargo) → home (dépose sur l'étal maison, drainé ensuite vers les
    maisons). Un bien n'existe qu'à UN endroit ; chaque transfert est atomique en
    un tick → aucun chemin de duplication/fuite. La survie (_vitals/_beh_survival)
    reste PRIORITAIRE dans la cascade : la mission se met en pause, ne s'annule
    pas. Filet : timeout → abort qui re-crédite la cargaison au pool."""
    world = ctx.world; events = ctx.events
    entity.trade_ticks += 1
    if entity.trade_ticks > TRADE_TIMEOUT:
        houses = _cb.get("house", [])
        if houses:   # re-crédit (perte bornée à un lot si plus de maisons)
            _add_pool_wood(houses, entity.cargo_wood)
            _add_pool_stone(houses, entity.cargo_stone)
            if entity.cargo_iron:
                _f = next((f for f in _cb.get("forge", [])
                           if f.iron < FORGE_MAX_IRON), None)
                if _f is not None:
                    _f.iron = min(FORGE_MAX_IRON, _f.iron + entity.cargo_iron)
        _refund_cargo_gold(entity, _cb)   # F1 : la pièce invendue rentre au coffre maison (pas de fuite)
        events.append({"type": "trade_aborted", "clan_id": entity.clan_id})
        _clear_trade(entity)
        return False
    phase = entity.trade_phase
    if phase == "load":
        mkts = _cb.get("market", [])
        if not mkts:   # marché maison disparu avant chargement → rien n'a bougé
            _clear_trade(entity)
            return False
        mkt = mkts[0]
        if _dist(entity.x, entity.y, mkt.x, mkt.y) < 1.5:
            houses = _cb.get("house", [])
            if entity.trade_pay == "gold":                # F1 : une pièce du coffre du MARCHÉ acheteur
                ok = mkt.gold >= 1
                if ok:
                    mkt.gold -= 1
                    entity.cargo_gold = 1
            elif entity.trade_pay == "stone":
                ok = _spend_pool_stone(houses, PAY_STONE_LOT)
                if ok:
                    entity.cargo_stone = PAY_STONE_LOT
            else:
                ok = _spend_pool_wood(houses, TRADE_WOOD_LOT)
                if ok:
                    entity.cargo_wood = TRADE_WOOD_LOT
            if ok:
                entity.trade_phase = "out"
                events.append({"type": "trade_depart", "clan_id": entity.clan_id,
                               "dest_clan_id": entity.trade_dest_cid,
                               "good": entity.trade_good, "pay": entity.trade_pay,
                               "pay_qty": entity.cargo_wood or entity.cargo_stone or entity.cargo_gold,
                               "wood": entity.cargo_wood})
            else:   # pool fondu entre-temps : mission annulée, rien n'a été déduit
                _clear_trade(entity)
            return True
        entity.state = State.TRADING
        entity.target_x = float(mkt.x); entity.target_y = float(mkt.y)
        _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
        return True
    if phase == "out":
        dest_cb = (ctx.clan_bldg or {}).get(entity.trade_dest_cid, {})
        dmkts = dest_cb.get("market", [])
        if not dmkts:   # clan B éteint (E8 a ruiné son marché) → demi-tour, cargo intact
            entity.trade_phase = "home"
            return True
        dmkt = dmkts[0]
        if _dist(entity.x, entity.y, dmkt.x, dmkt.y) < 1.5:
            # Taux SPOT recalculé MAINTENANT (loi de l'offre : le cours a pu bouger
            # en route) + plancher incessible du vendeur.
            houses_b = dest_cb.get("house", [])
            forges_b = dest_cb.get("forge", [])
            if entity.trade_good == "iron":
                pool_b = sum(f.iron for f in forges_b)
                rate = _iron_rate(pool_b)
                take = min(rate, max(0, pool_b - IRON_SELL_FLOOR)) if rate else 0
            else:
                pool_b = sum(h.stone for h in houses_b)
                rate = _stone_rate(pool_b)
                take = min(rate, max(0, pool_b - STONE_SELL_FLOOR)) if rate else 0
            if take > 0:   # échange atomique : paiement → étal B, bien → cargo
                dmkt.wood = min(MARKET_MAX_STOCK, dmkt.wood + entity.cargo_wood)
                dmkt.stone = min(MARKET_MAX_STOCK, dmkt.stone + entity.cargo_stone)
                pay_qty = entity.cargo_wood or entity.cargo_stone
                pay_good = "wood" if entity.cargo_wood else "stone"
                if entity.cargo_gold:            # F1 : la pièce entre au coffre du VENDEUR (cap → dorure)
                    _g = dmkt.gold + entity.cargo_gold
                    dmkt.gold = min(MARKET_GOLD_MAX, _g)
                    _spill = _g - dmkt.gold
                    if _spill:                   # débordement → dorure de l'église du vendeur (le puits)
                        _chs_v = dest_cb.get("church", [])
                        if _chs_v:
                            _chs_v[0].gilt += _spill
                    pay_qty = entity.cargo_gold; pay_good = "gold"
                    entity.cargo_gold = 0
                entity.cargo_wood = 0
                entity.cargo_stone = 0
                if entity.trade_good == "iron":
                    entity.cargo_iron = _take_forge_iron(forges_b, take)
                else:
                    entity.cargo_stone = _take_pool_stone(houses_b, take)
                ev = {"type": "trade_exchange", "clan_id": entity.clan_id,
                      "dest_clan_id": entity.trade_dest_cid,
                      "good": entity.trade_good, "qty": take,
                      "pay_good": pay_good, "pay_qty": pay_qty,
                      "x": dmkt.x, "y": dmkt.y}
                if entity.trade_good == "stone" and pay_good == "wood":  # legacy D1
                    ev["wood"] = pay_qty; ev["stone"] = take
                events.append(ev)
            else:   # cours tombé en route / plancher atteint → refus narré (D2)
                events.append({"type": "trade_refused", "clan_id": entity.clan_id,
                               "dest_clan_id": entity.trade_dest_cid,
                               "good": entity.trade_good})
            entity.trade_phase = "home"
            return True
        entity.state = State.TRADING
        entity.target_x = float(dmkt.x); entity.target_y = float(dmkt.y)
        _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
        return True
    if phase == "home":
        mkts = _cb.get("market", [])
        if not mkts:   # défensif (impossible tant que le clan vit) → re-crédit direct
            houses = _cb.get("house", [])
            if houses:
                _add_pool_wood(houses, entity.cargo_wood)
                _add_pool_stone(houses, entity.cargo_stone)
                if entity.cargo_iron:
                    _f = next((f for f in _cb.get("forge", [])
                               if f.iron < FORGE_MAX_IRON), None)
                    if _f is not None:
                        _f.iron = min(FORGE_MAX_IRON, _f.iron + entity.cargo_iron)
            _refund_cargo_gold(entity, _cb)   # F1 : pièce invendue → trésor église (marché disparu)
            _clear_trade(entity)
            return False
        mkt = mkts[0]
        if _dist(entity.x, entity.y, mkt.x, mkt.y) < 1.5:
            mkt.stone = min(MARKET_MAX_STOCK, mkt.stone + entity.cargo_stone)
            mkt.wood = min(MARKET_MAX_STOCK, mkt.wood + entity.cargo_wood)  # invendu visible
            mkt.iron = min(MARKET_MAX_STOCK, mkt.iron + entity.cargo_iron)  # étal fer (D2)
            _refund_cargo_gold(entity, _cb)   # F1 : pièce invendue (refus) → coffre du marché maison
            events.append({"type": "trade_complete", "clan_id": entity.clan_id,
                           "dest_clan_id": entity.trade_dest_cid,
                           "good": entity.trade_good,
                           "stone": entity.cargo_stone, "iron": entity.cargo_iron,
                           "wood_back": entity.cargo_wood})
            _clear_trade(entity)
            return True
        entity.state = State.TRADING
        entity.target_x = float(mkt.x); entity.target_y = float(mkt.y)
        _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
        return True
    _clear_trade(entity)   # phase inconnue (save corrompu) → reset défensif
    return False


def _clear_pilgrim(entity: Entity):
    """Efface proprement une mission de pèlerinage (C1)."""
    entity.pilgrim_phase = None
    entity.pilgrim_dest_cid = None
    entity.pilgrim_ticks = 0
    entity.pilgrim_pay = None


def _beh_pilgrim(entity: Entity, ctx, _cb, _eff_speed) -> bool:
    """Pèlerinage (C1) : load (charge l'offrande de bois au village) → out (la
    dépose sur l'AUTEL de l'église étrangère : elle y BRÛLERA — puits économique —
    contre une bénédiction, le 1er service acheté) → home (retour au feu de camp).
    Miroir du pattern caravane D1 : phases atomiques, timeout re-créditant,
    destination ruinée → demi-tour cargaison intacte SANS bénédiction."""
    world = ctx.world; events = ctx.events
    entity.pilgrim_ticks += 1
    if entity.pilgrim_ticks > PILGRIM_TIMEOUT:
        houses = _cb.get("house", [])
        if houses and entity.cargo_wood:
            _add_pool_wood(houses, entity.cargo_wood)
        entity.cargo_wood = 0
        if entity.cargo_gold:   # C2 : la pièce revient au trésor (même règle de débordement)
            _chs_t = _cb.get("church", [])
            if _chs_t:
                _g = _chs_t[0].gold + entity.cargo_gold
                _chs_t[0].gold = min(GOLD_TREASURY_MAX, _g)
                _chs_t[0].gilt += _g - _chs_t[0].gold
            entity.cargo_gold = 0
        events.append({"type": "pilgrim_aborted", "clan_id": entity.clan_id})
        _clear_pilgrim(entity)
        return False
    phase = entity.pilgrim_phase
    if phase == "load":
        if entity.pilgrim_pay == "gold":   # C2 : la pièce se prend AU trésor
            _chs_l = _cb.get("church", [])
            if not _chs_l:
                _clear_pilgrim(entity)
                return False
            src = _chs_l[0]
            if _dist(entity.x, entity.y, src.x, src.y) < 1.5:
                if src.gold >= OFFERING_GOLD:
                    src.gold -= OFFERING_GOLD
                    entity.cargo_gold = OFFERING_GOLD
                    entity.pilgrim_phase = "out"
                else:   # trésor fondu entre-temps : tout-ou-rien, rien déduit
                    _clear_pilgrim(entity)
                return True
            entity.state = State.PILGRIMAGE
            entity.target_x = float(src.x); entity.target_y = float(src.y)
            _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
            return True
        houses = _cb.get("house", [])
        if not houses:
            _clear_pilgrim(entity)
            return False
        h0 = houses[0]
        if _dist(entity.x, entity.y, h0.x, h0.y) < 1.5:
            if _spend_pool_wood(houses, OFFERING_WOOD):
                entity.cargo_wood = OFFERING_WOOD
                entity.pilgrim_phase = "out"
            else:   # pool fondu : mission annulée, rien déduit
                _clear_pilgrim(entity)
            return True
        entity.state = State.PILGRIMAGE
        entity.target_x = float(h0.x); entity.target_y = float(h0.y)
        _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
        return True
    if phase == "out":
        dest_cb = (ctx.clan_bldg or {}).get(entity.pilgrim_dest_cid, {})
        churches = dest_cb.get("church", [])
        if not churches:   # clan éteint → E8 a ruiné l'église : demi-tour, cargo intact
            entity.pilgrim_phase = "home"
            return True
        ch = churches[0]
        if _dist(entity.x, entity.y, ch.x, ch.y) < 1.5:
            if entity.cargo_gold:   # C2 : la pièce rejoint le TRÉSOR hôte (débordement → dorure)
                _g = ch.gold + entity.cargo_gold
                ch.gold = min(GOLD_TREASURY_MAX, _g)
                _spill = _g - ch.gold
                if _spill:
                    ch.gilt += _spill
                    events.append({"type": "church_gilt",
                                   "clan_id": entity.pilgrim_dest_cid,
                                   "gilt": ch.gilt, "x": ch.x, "y": ch.y})
                entity.cargo_gold = 0
            else:   # offrande de bois sur l'autel (C1 inchangé)
                ch.wood = min(ALTAR_MAX, ch.wood + entity.cargo_wood)
                entity.cargo_wood = 0
            ch.pilgrims_served += 1   # même renom, or ou bois (+1 strict : jalon ==10)
            entity.blessed_ticks = BLESS_DURATION
            events.append({"type": "pilgrim_blessed", "clan_id": entity.clan_id,
                           "dest_clan_id": entity.pilgrim_dest_cid,
                           "served": ch.pilgrims_served, "pay": entity.pilgrim_pay,
                           "x": ch.x, "y": ch.y})
            entity.pilgrim_phase = "home"
            return True
        entity.state = State.PILGRIMAGE
        entity.target_x = float(ch.x); entity.target_y = float(ch.y)
        _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
        return True
    if phase == "home":
        clan = (ctx.clans or {}).get(entity.clan_id)
        if clan is None:   # défensif
            _clear_pilgrim(entity)
            return False
        if _dist(entity.x, entity.y, clan.cx, clan.cy) < 2.0:
            if entity.cargo_wood > 0:   # destination ruinée en route → re-crédit
                _add_pool_wood(_cb.get("house", []), entity.cargo_wood)
                entity.cargo_wood = 0
            if entity.cargo_gold > 0:   # C2 : la pièce revient au trésor propre
                _chs_h = _cb.get("church", [])
                if _chs_h:
                    _g = _chs_h[0].gold + entity.cargo_gold
                    _chs_h[0].gold = min(GOLD_TREASURY_MAX, _g)
                    _chs_h[0].gilt += _g - _chs_h[0].gold
                entity.cargo_gold = 0   # église propre disparue : perte bornée à 1, défensif
            events.append({"type": "pilgrim_home", "clan_id": entity.clan_id})
            _clear_pilgrim(entity)
            return True
        entity.state = State.PILGRIMAGE
        entity.target_x = float(clan.cx); entity.target_y = float(clan.cy)
        _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
        return True
    _clear_pilgrim(entity)   # phase inconnue → reset défensif
    return False


def _clear_expedition(entity: Entity):
    """Efface proprement une mission d'exploration (P7 G1). Appelée à TOUS les chemins de
    sortie (arrivée, retour, timeout, clan disparu, site volatilisé) : un slot qui traîne
    bloquerait le clan pour toujours (une expédition en vol max) — le piège d'état."""
    entity.expedition_site = None
    entity.expedition_phase = None
    entity.expedition_t0 = 0


def _learn_site(clan, site_id: int) -> bool:
    """Mémorise un site dans `clan.known_sites` : liste TRIÉE, dédupliquée, bornée à
    KNOWN_SITES_MAX. Le site_id EST le rang du site au tri par score décroissant (cf.
    World.site_catalogue) → garder les plus petits ids, c'est garder les meilleures terres.
    True si le savoir est neuf (l'appelant narre), False si le clan connaissait déjà."""
    if site_id in clan.known_sites:
        return False
    ks = sorted(set(clan.known_sites) | {site_id})[:KNOWN_SITES_MAX]
    clan.known_sites = ks
    return site_id in ks   # borne atteinte et site moins bon que les 12 connus → rien appris


def _beh_expedition(entity: Entity, ctx, _cb, _eff_speed) -> bool:
    """Mission d'exploration (P7 G1) : « out » = marche vers le site visé, « home » = retour
    au feu. La cible est RE-POSÉE à chaque tick tant que la mission court — target_x est
    écrasé par les détours vitaux en 54 endroits du moteur, donc c'est la PHASE persistante
    qui fait la mission, pas la cible (même discipline que trade_phase/pilgrim_phase). La
    survie reste prioritaire : les blocs amont de la cascade mettent la mission en PAUSE,
    ils ne l'annulent pas. Aucun tirage aléatoire. Découvre tout site qui passe à portée,
    même celui qu'il ne visait pas — c'est le hasard des chemins, pas celui d'un dé."""
    world = ctx.world; events = ctx.events
    clan = (ctx.clans or {}).get(entity.clan_id)
    if clan is None:            # clan éteint sous ses pieds → la mission n'a plus de commanditaire
        _clear_expedition(entity)
        return False
    # Timeout ABSOLU (tick courant − tick de départ) et non un compteur incrémenté ici : un
    # éclaireur figé par la faim ou par un combat ne traverse plus ce bloc, son compteur ne
    # tournerait plus et son clan resterait « expédition en vol » à vie. Le dispatch balaie
    # les missions périmées avec la même horloge.
    if ctx.tick - entity.expedition_t0 > EXPEDITION_TIMEOUT:
        # RENONCEMENT. Sans ça le clan re-viserait éternellement la même terre : la cible est
        # toujours « inconnue » et toujours la plus proche, donc le prochain éclaireur repartirait
        # buter au même endroit — un clan estropié à vie (mesuré sur seed 424242 : le site 15 est
        # à 21 tuiles et sur la MÊME masse terrestre, mais la marche gloutonne — il n'y a pas de
        # pathfinding — plafonne à 13 tuiles ; le clan 0 est resté à 1 seul site connu de t=3000
        # à t=12000). On ne le déclare PAS « connu » : il ne doit jamais devenir une cible de
        # colonie ou de migration. On acte seulement que ses hommes n'y arrivent pas.
        if entity.expedition_site is not None and entity.expedition_site not in clan.failed_sites:
            clan.failed_sites = sorted(set(clan.failed_sites) | {entity.expedition_site})[:KNOWN_SITES_MAX]
        events.append({"type": "expedition_lost", "clan_id": entity.clan_id,
                       "site": entity.expedition_site})
        _clear_expedition(entity)
        return False
    # DÉCOUVERTE : tout site à portée de vue est relevé (le sien comme un autre).
    sites = world.site_catalogue()
    target = None
    for sid, sx, sy, _score in sites:
        if sid == entity.expedition_site:
            target = (sx, sy)
        if _dist(entity.x, entity.y, sx, sy) <= SITE_DISCOVER_R and _learn_site(clan, sid):
            events.append({"type": "site_discovered", "clan_id": clan.id, "site": sid,
                           "x": sx, "y": sy,
                           "dist": int(_dist(clan.cx, clan.cy, sx, sy))})
    if entity.expedition_phase == "out":
        # La bascule retour tient à l'ARRIVÉE, pas à l'apprentissage : quand les 12 sites déjà
        # connus sont tous meilleurs, `_learn_site` refuse d'enregistrer celui-ci (borne), et
        # une condition fondée sur `known_sites` ne tomberait JAMAIS — l'éclaireur piétinerait
        # sur sa cible jusqu'au timeout, et cette terre qu'il a POURTANT ATTEINTE finirait
        # classée « inatteignable » (finding de gate Regigigas 978a6812). Rare, borné, mais
        # sémantiquement faux : il est arrivé, il rentre.
        arrived = (target is not None
                   and _dist(entity.x, entity.y, target[0], target[1]) <= SITE_DISCOVER_R)
        if target is None or arrived or entity.expedition_site in clan.known_sites:
            entity.expedition_phase = "home"   # relevé fait (ou site introuvable) → il rentre
        else:
            entity.state = State.EXPLORING
            entity.target_x = float(target[0]); entity.target_y = float(target[1])
            _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
            return True
    # phase "home" (ou bascule à l'instant) : le savoir rentre au village avec l'homme.
    if _dist(entity.x, entity.y, clan.cx, clan.cy) <= EXPEDITION_HOME_R:
        _clear_expedition(entity)
        return False        # rendu au feu : il reprend une vie normale dès ce tick
    entity.state = State.EXPLORING
    entity.target_x = float(clan.cx); entity.target_y = float(clan.cy)
    _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
    return True


def _beh_colonist(entity: Entity, ctx, _cb, _eff_speed) -> bool:
    """Marche des colons (P7 G2). Le drapeau est planté au départ — la colonie EXISTE dès le
    tick de l'essaimage — mais ses hommes, eux, doivent traverser la carte : sans ça le clan
    serait téléporté (`_found_clan` réassigne clan_id sans déplacer personne). La destination
    vit dans un slot PERSISTANT re-posé à chaque tick, comme la caravane et le pèlerin : une
    cible posée une seule fois ne survivrait pas au premier arrêt pour boire. La survie reste
    prioritaire — le colon mange en chemin, puis REPREND sa route, c'est le comportement voulu.
    Aucun tirage. Rien n'est porté, donc aucun chemin d'échec ne peut faire fuir de ressource."""
    world = ctx.world
    if ctx.tick - entity.colonist_t0 > COLONIST_TIMEOUT:
        entity.colonist_dest = None      # filet anti-zombie : il s'installe où il en est
        entity.colonist_t0 = 0
        return False
    dx, dy = entity.colonist_dest
    if _dist(entity.x, entity.y, dx, dy) <= 2.0:
        entity.colonist_dest = None      # arrivé sur sa terre : il reprend une vie normale
        entity.colonist_t0 = 0
        return False
    entity.state = State.EXPLORING
    entity.target_x = float(dx); entity.target_y = float(dy)
    _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
    return True


def _try_forge_upgrade(entity: Entity, forge, events: list) -> bool:
    """Bloc B : à la forge, upgrade un outil PIERRE → FER en consommant le fer
    stocké. Pioche pierre → pioche fer, puis hache pierre → hache fer."""
    if entity.pick == "stone_pick" and forge.iron >= IRON_PICK_COST:
        forge.iron -= IRON_PICK_COST; entity.pick = "iron_pick"
        events.append({"type": "craft_iron_pick", "clan_id": entity.clan_id})
        return True
    if entity.tool == "stone_axe" and forge.iron >= IRON_AXE_COST:
        forge.iron -= IRON_AXE_COST; entity.tool = "iron_axe"
        events.append({"type": "craft_iron_axe", "clan_id": entity.clan_id})
        return True
    return False


class _TickCtx:
    """Contexte tick-global partagé par TOUTES les entités d'un même tick.
    Construit une seule fois par step() (au lieu de repasser 16 arguments à chaque
    appel). Ne contient QUE de l'état commun au tick : rien de propre à une entité."""
    __slots__ = ("world", "all_entities", "births", "events", "tick", "season",
                 "clans", "buildings", "temp_c", "species_counts", "raining",
                 "heatwave", "clan_bldg", "predators", "predator_grid", "entity_grid",
                 "iron_tiles", "iron_nearest", "gold_tiles", "gold_nearest",
                 "clan_human_pop")

    def __init__(self, world, all_entities, births, events, tick,
                 season="spring", clans=None, buildings=None, temp_c=12.0,
                 species_counts=None, raining=False, heatwave=False,
                 clan_bldg=None, predators=None, predator_grid=None, entity_grid=None):
        self.world = world; self.all_entities = all_entities
        self.births = births; self.events = events; self.tick = tick
        self.season = season; self.clans = clans; self.buildings = buildings
        self.temp_c = temp_c; self.species_counts = species_counts
        self.raining = raining; self.heatwave = heatwave; self.clan_bldg = clan_bldg
        self.predators = predators; self.predator_grid = predator_grid
        self.entity_grid = entity_grid
        self.iron_tiles = None    # bloc B : tuiles de fer minable, mémoïsées par tick
        self.iron_nearest = {}    # bloc B : nearest fer par bucket 8×8, mémoïsé par tick
        self.gold_tiles = None    # bloc C2 : filons d'or minables, mémoïsés par tick
        self.gold_nearest = {}    # bloc C2 : nearest or par bucket 8×8
        # Perf P1 : population humaine par clan, SNAPSHOT du début de tick (le _TickCtx
        # est construit une seule fois par step(), avant la boucle entités). Remplace les
        # sum(1 for e in all_entities ...) O(N) recalculés PAR humain (repro + build maison)
        # → O(N) une fois par tick. Sémantique = snapshot début-de-tick (vs l'ancien compte
        # « live » qui variait avec les morts intra-tick) : divergence possible → à valider
        # au golden (imputable si un humain meurt mid-tick dans le run seedé).
        chp: dict = {}
        for _e in (all_entities or ()):
            if _e.alive and _e.etype is EntityType.HUMAN and _e.clan_id is not None:
                chp[_e.clan_id] = chp.get(_e.clan_id, 0) + 1
        self.clan_human_pop = chp


def tick_entity(entity: Entity, world: "World", all_entities: list[Entity],
                births: list[Entity], events: list[dict], tick: int,
                season: str = "spring", clans: dict = None,
                buildings: list = None,
                temp_c: float = 12.0, species_counts: dict = None,
                raining: bool = False, heatwave: bool = False,
                clan_bldg: dict = None, predators: list = None,
                predator_grid: dict = None, entity_grid: dict = None):
    """Point d'entrée rétro-compatible (tests, appels externes) : emballe les
    paramètres tick-globaux dans un _TickCtx puis délègue à _tick_entity. La boucle
    step() construit le ctx UNE fois et appelle _tick_entity directement (perf)."""
    ctx = _TickCtx(world, all_entities, births, events, tick, season, clans,
                   buildings, temp_c, species_counts, raining, heatwave,
                   clan_bldg, predators, predator_grid, entity_grid)
    _tick_entity(entity, ctx)


def _tick_entity(entity: Entity, ctx: "_TickCtx"):
    if not entity.alive:
        return
    # Dépaquetage du ctx dans les noms locaux utilisés par le corps ci-dessous
    # (garde le code des comportements verbatim → comportement inchangé).
    world = ctx.world; all_entities = ctx.all_entities
    births = ctx.births; events = ctx.events; tick = ctx.tick
    season = ctx.season; clans = ctx.clans; buildings = ctx.buildings
    temp_c = ctx.temp_c; species_counts = ctx.species_counts
    raining = ctx.raining; heatwave = ctx.heatwave; clan_bldg = ctx.clan_bldg
    predators = ctx.predators; predator_grid = ctx.predator_grid
    entity_grid = ctx.entity_grid

    # Reset du verrou intra-tick de construction
    entity.building_type = None
    # Décrémente cooldown de coupe
    if entity.chop_cooldown_left > 0:
        entity.chop_cooldown_left -= 1

    spec = entity.spec

    # Index de bâtiments du clan (évite les list comprehensions répétées)
    _cb = clan_bldg.get(entity.clan_id, {}) if clan_bldg and entity.clan_id is not None else {}

    # Sanity clamp : recadre un trait qui a dérivé hors de [0.4×, 1.8×] de la spec
    # de base (la mutation héréditaire ±5%/naissance n'a qu'un plancher → sans borne
    # haute, vision et hunger_rate dérivent sur des milliers de générations).
    _spec_spd = entity.spec.speed
    if not (_spec_spd * 0.40 <= entity.traits["speed"] <= _spec_spd * 1.80):
        entity.traits["speed"] = round(_spec_spd * random.uniform(0.95, 1.05), 4)
    _spec_vis = entity.spec.vision
    if not (max(1, _spec_vis * 0.40) <= entity.traits["vision"] <= _spec_vis * 1.80):
        entity.traits["vision"] = max(1, round(_spec_vis * random.uniform(0.95, 1.05)))
    _spec_hr = entity.spec.hunger_rate
    if not (_spec_hr * 0.40 <= entity.traits["hunger_rate"] <= _spec_hr * 1.80):
        entity.traits["hunger_rate"] = round(_spec_hr * random.uniform(0.95, 1.05), 5)

    # Vitesse effective ce tick (pluie ralentit les humains sans modifier le trait permanent)
    _eff_speed = (round(entity.traits["speed"] * RAIN_SPEED_MULT, 4)
                  if raining and entity.etype == EntityType.HUMAN
                  else entity.traits["speed"])

    # ── Récupération si sur une tuile non-franchissable (eau) ─────────────
    if not world.is_walkable(entity.ix, entity.iy, spec.aquatic):
        _teleport_to_nearest_walkable(entity, world)

    # ── Sous-systèmes (ordre de priorité préservé — le `or` court-circuite comme les early-return) ──
    if not (_vitals(entity, ctx)
            or _beh_survival(entity, ctx, _cb, _eff_speed)
            or _beh_work(entity, ctx, _cb, _eff_speed)):
        _beh_wander(entity, ctx)
    # Anti-empilement : espace personnel appliqué à TOUS les humains vivants, quel que soit l'état
    # (agriculture, coupe, chantier…) → plus de tas de 12 sur une tuile. Nudge de position seul.
    if _SEPARATE_ON and entity.alive and entity.etype == EntityType.HUMAN:
        _separate_human(entity, entity_grid, all_entities, world)


def _vitals(entity, ctx):
    """Vieillissement, faim, soif, piétinement, morts et gestation. True si l'entité doit s'arrêter (morte ou enfante)."""
    world = ctx.world; all_entities = ctx.all_entities
    births = ctx.births; events = ctx.events; tick = ctx.tick
    season = ctx.season; clans = ctx.clans; buildings = ctx.buildings
    temp_c = ctx.temp_c; species_counts = ctx.species_counts
    raining = ctx.raining; heatwave = ctx.heatwave; clan_bldg = ctx.clan_bldg
    predators = ctx.predators; predator_grid = ctx.predator_grid
    entity_grid = ctx.entity_grid
    spec = entity.spec
    # ── Vieillissement, faim & soif ───────────────────────────────────────
    entity.age    += 1
    entity.hunger += (entity.traits["hunger_rate"] * _hunger_mult(temp_c)
                      * (BLESS_HUNGER_MULT if entity.blessed_ticks > 0 else 1.0))
    if entity.blessed_ticks > 0:   # la grâce s'estompe (jamais cumulée, rafraîchie)
        entity.blessed_ticks -= 1
    if not spec.aquatic:
        entity.thirst += THIRST_RATE * _thirst_mult(temp_c)
        if heatwave:
            entity.thirst += THIRST_RATE * (HEATWAVE_THIRST_MULT - 1.0)
        if raining:
            entity.thirst = max(0.0, entity.thirst - RAIN_THIRST_REDUCE)
        # Piétinement : présence sur herbe → fertilité réduite chaque tick
        world.consume_fertility(entity.ix, entity.iy, FERTILITY_TRAMPLE)

    if entity.repro_cooldown_left > 0:
        entity.repro_cooldown_left -= 1

    # Mort de vieillesse
    if entity.age > spec.max_age:
        entity.alive = False
        entity.state = State.DEAD
        events.append({"type": "death", "cause": "age",
                       "etype": entity.etype.value, "x": entity.ix, "y": entity.iy})
        return True
    # Mort de faim
    if entity.hunger >= spec.max_hunger:
        entity.alive = False
        entity.state = State.DEAD
        events.append({"type": "death", "cause": "hunger",
                       "etype": entity.etype.value, "x": entity.ix, "y": entity.iy})
        return True
    # Mort de soif
    if not spec.aquatic and entity.thirst >= MAX_THIRST:
        entity.alive = False
        entity.state = State.DEAD
        events.append({"type": "death", "cause": "thirst",
                       "etype": entity.etype.value, "x": entity.ix, "y": entity.iy})
        return True
    # ── Gestation ─────────────────────────────────────────────────────────
    if entity.gestation_left > 0:
        entity.gestation_left -= 1
        if entity.gestation_left == 0:
            # Cap PAR espèce (I1 : spec.max_pop, différencié par rôle trophique) :
            # inclut les naissances déjà prévues ce tick.
            cap = spec.max_pop
            current = (species_counts or {}).get(entity.etype.value, 0)
            births_same = sum(1 for b in births if b.etype == entity.etype)
            if current + births_same >= cap:
                entity.state = State.RESTING
                return True
            n = random.randint(*spec.litter_size)
            for _ in range(n):
                if current + births_same >= cap:
                    break
                jx = entity.x + random.uniform(-1, 1)
                jy = entity.y + random.uniform(-1, 1)
                jx = max(0, min(world.width  - 1, jx))
                jy = max(0, min(world.height - 1, jy))
                if world.is_walkable(int(jx), int(jy), spec.aquatic):
                    baby = spawn(entity.etype, jx, jy)
                    baby.age = 0
                    baby.hunger = 20
                    baby.clan_id = entity.clan_id
                    # Hérite des traits de la mère + mutation ±5%
                    baby.traits = {
                        "speed":       max(0.1,  round(entity.traits["speed"]       * random.uniform(0.95, 1.05), 4)),
                        "vision":      max(1,     round(entity.traits["vision"]      * random.uniform(0.95, 1.05))),
                        "hunger_rate": max(0.005, round(entity.traits["hunger_rate"] * random.uniform(0.95, 1.05), 5)),
                    }
                    births.append(baby)
                    births_same += 1
            events.append({"type": "birth", "etype": entity.etype.value,
                           "count": n, "x": entity.ix, "y": entity.iy,
                           "clan": entity.clan_id})
        entity.state = State.RESTING
        return True
    return False


def _beh_survival(entity, ctx, _cb, _eff_speed):
    """Besoins vitaux immédiats : fuite, soif, chasse, conflit de clan, repas, pêche. True si l'entité a agi ce tick."""
    world = ctx.world; all_entities = ctx.all_entities
    births = ctx.births; events = ctx.events; tick = ctx.tick
    season = ctx.season; clans = ctx.clans; buildings = ctx.buildings
    temp_c = ctx.temp_c; species_counts = ctx.species_counts
    raining = ctx.raining; heatwave = ctx.heatwave; clan_bldg = ctx.clan_bldg
    predators = ctx.predators; predator_grid = ctx.predator_grid
    entity_grid = ctx.entity_grid
    spec = entity.spec
    # 1. Fuite (priorité max pour les proies)
    if spec.flee_distance > 0:
        predator = _find_predator_nearby(entity, all_entities, predators, predator_grid)
        if predator:
            entity.state = State.FLEEING
            dx = entity.x - predator.x
            dy = entity.y - predator.y
            d  = math.sqrt(dx*dx + dy*dy) or 1
            # Essaie plusieurs angles de fuite (direct puis ±45° puis ±90°)
            fled = False
            for angle_offset in (0, 0.785, -0.785, 1.571, -1.571):
                ca, sa = math.cos(angle_offset), math.sin(angle_offset)
                fdx = (dx/d) * ca - (dy/d) * sa
                fdy = (dx/d) * sa + (dy/d) * ca
                tx = max(0, min(world.width  - 1, entity.x + fdx * 6))
                ty = max(0, min(world.height - 1, entity.y + fdy * 6))
                if world.is_walkable(int(tx), int(ty), spec.aquatic):
                    entity.target_x = tx
                    entity.target_y = ty
                    fled = True
                    break
            if fled:
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed * 1.4, world)
            return True
    # 2.5 Soif urgente (prioritaire sur la chasse)
    if not spec.aquatic and entity.thirst > 70:
        if _drink_or_seek_water(entity, world, events, buildings, cb=_cb):
            return True
    # 2. Chasse (prédateurs) — seuil par espèce (bloc E). Défaut 55 (comportement
    # historique) ; le sanglier chasse dès ~28 (< broutage 30) → prédation réelle
    # et VISIBLE, au lieu de brouter avant d'avoir jamais assez faim pour chasser.
    if spec.is_predator and entity.hunger > spec.hunt_hunger:
        # I2 : chasse la proie VIVANTE la plus PROCHE parmi TOUS les prey_types, en
        # UNE passe (fin de la préférence ordonnée qui concentrait la prédation sur
        # la 1re espèce et laissait les autres au plafond). Répartit la pression →
        # toutes les proies régulées ensemble. Plancher par espèce (<30) préservé :
        # une proie rare est ignorée tant qu'elle n'a pas recolonisé. Déterministe
        # (1er au d² minimal dans l'ordre stable de all_entities). Aussi + rapide
        # (1 scan O(N) au lieu de n_types scans).
        vis_sq = entity.traits["vision"] ** 2
        ex, ey = entity.x, entity.y
        prey = None
        best_d_sq = vis_sq
        for e in all_entities:
            if e is entity or not e.alive:
                continue
            if e.etype not in spec.prey_types:
                continue
            if (species_counts or {}).get(e.etype.value, 0) < 30:
                continue
            dx = ex - e.x; dy = ey - e.y
            d_sq = dx*dx + dy*dy
            if d_sq < best_d_sq:
                best_d_sq = d_sq
                prey = e
        if prey is not None:
            entity.state  = State.HUNTING
            entity.target_id = prey.id
            _move_toward(entity, prey.x, prey.y, _eff_speed, world)
            # Capture si assez proche (hitbox étendu pour entités larges)
            catch_r = 0.8 + (entity.spec.hitbox_width - 1) * 0.5
            if _dist_hitbox(entity, prey) < catch_r:
                prey.alive = False
                prey.state = State.DEAD
                # Compteur vivant : décrémente pour que la préservation d'espèce
                # (<30) et le cap de naissances voient les morts DE CE TICK →
                # empêche N prédateurs de tuer N proies sous le seuil en un tick.
                if species_counts is not None:
                    species_counts[prey.etype.value] = species_counts.get(prey.etype.value, 0) - 1
                entity.hunger = max(0, entity.hunger - spec.eat_meat)
                events.append({"type": "kill",
                               "predator": entity.etype.value,
                               "prey": prey.etype.value,
                               "x": entity.ix, "y": entity.iy})
            return True
    # 2b. Conflit inter-clan. Deux moteurs (trêve marchand/pèlerin/béni dans les deux) :
    #   - SURVIE (existant) : humain très affamé (hunger>65) attaque un ennemi pour se nourrir.
    #   - GUERRE (société) : clan en mode "war" → ses membres attaquent le CLAN-CIBLE même sans
    #     faim (acte de guerre, pas repas). Plancher d'espèce (<30, comme la prédation) respecté
    #     → anti-cascade. SOCIETY_OFF : `_at_war`=False + plancher désactivé → comportement pré-bloc.
    ec = clans.get(entity.clan_id) if (clans and entity.clan_id is not None) else None
    # Warrior-only (P1) : en guerre, seuls les WARRIORS agressent (gated _JOBS_ON) ; les autres
    # vaquent. Le raid de survie (faim>65) reste universel. JOBS_OFF → comportement société I1.
    _at_war = (_SOCIETY_ON and ec is not None and ec.mode == "war"
               and (not _JOBS_ON or entity.role == "warrior"))
    _hungry = entity.hunger > 65
    # D3 : en guerre pure (pas affamé), si le plancher d'espèce bloque déjà les kills, ne pas
    # scanner ni monopoliser le tick → le membre "en guerre" vaque à ses autres tâches.
    if (_at_war and not _hungry
            and (species_counts or {}).get(EntityType.HUMAN.value, 0) <= 30):
        _at_war = False
    if (entity.etype == EntityType.HUMAN and entity.clan_id is not None
            and (_hungry or _at_war) and clans
            and entity.trade_phase is None
            and entity.pilgrim_phase is None
            and entity.blessed_ticks == 0):   # Trêve de Dieu : un béni n'attaque pas (C1)
        # D2 : scanner le VOISINAGE (grille spatiale) et non tout le monde — une guerre longue
        # ferait sinon un O(N²) soutenu. Gated _SOCIETY_ON : sous SOCIETY_OFF le raid-faim garde
        # le scan all_entities d'origine → l'imputation (hash pré-bloc) reste exacte.
        _scan = (_grid_neighbors(entity_grid, entity.ix, entity.iy, reach=1)
                 if (_SOCIETY_ON and entity_grid is not None) else all_entities)
        for e in _scan:
            if (e is not entity and e.alive
                    and e.etype == EntityType.HUMAN
                    and e.clan_id is not None
                    and e.clan_id != entity.clan_id
                    and e.trade_phase is None
                    and e.pilgrim_phase is None):
                # En guerre (pas affamé) : ne cible QUE le clan désigné (les autres restent en paix).
                if _at_war and not _hungry and ec.war_target >= 0 and e.clan_id != ec.war_target:
                    continue
                d = _dist(entity.x, entity.y, e.x, e.y)
                if d < entity.traits["vision"]:
                    entity.state = State.HUNTING
                    _move_toward(entity, e.x, e.y, _eff_speed * 1.1, world)
                    if d < 0.8:
                        # Défection (S2a) : en guerre, si le clan de la victime perd nettement
                        # (pop < DEFECT_RATIO × pop attaquant, attaquant ≥3), elle ABANDONNE et
                        # rejoint le vainqueur au lieu de mourir → SAIGNE le perdant pendant la
                        # guerre. Plancher perdant >3 (reco Regigigas) : à ≤3 c'est l'absorption-
                        # coup-de-grâce de fin de guerre (P3, distincte) — sans ce plancher la
                        # défection viderait le clan avant que l'absorption existe.
                        _apop = ctx.clan_human_pop.get(entity.clan_id, 0)
                        _vpop = ctx.clan_human_pop.get(e.clan_id, 0)
                        if (_SOCIETY_ON and _at_war and not _hungry and _apop >= 3
                                and _vpop > 3 and _vpop < _apop * DEFECT_RATIO):
                            _old = e.clan_id
                            e.clan_id = entity.clan_id
                            e.state = State.RESTING
                            # FIX (Regigigas) : décrémenter le snapshot EN DIRECT — sinon plusieurs
                            # défections du même tick lisent la même pop et franchissent le plancher
                            # >3 (même correctif que species_counts au kill : N acteurs, 1 tick).
                            ctx.clan_human_pop[_old] = _vpop - 1
                            ctx.clan_human_pop[entity.clan_id] = _apop + 1
                            events.append({"type": "clan_defect", "from_clan": _old,
                                           "to_clan": entity.clan_id, "x": e.ix, "y": e.iy})
                        else:
                            # Plancher anti-anéantissement (S2c) : une war-kill (acte de guerre)
                            # ne peut pas exterminer le dernier carré du clan-cible → un rump
                            # ≥ WAR_MIN_CLAN_POP survit, le monde reste multi-clan à long horizon.
                            # Gated _WARBEH_ON : sous WARBEH_OFF, war-kill non planchée = P1 exact.
                            _warkill = (_WARBEH_ON and _JOBS_ON and _SOCIETY_ON
                                        and _at_war and not _hungry)
                            if ((not _warkill or _vpop > WAR_MIN_CLAN_POP)
                                    and ((not _SOCIETY_ON)
                                         or (species_counts or {}).get(e.etype.value, 0) > 30)):
                                e.alive = False
                                e.state = State.DEAD
                                if species_counts is not None:
                                    species_counts[e.etype.value] = species_counts.get(e.etype.value, 0) - 1
                                if _warkill:
                                    # décrément direct du snapshot (comme la défection) : sinon
                                    # N war-kills du même tick lisent la même pop et franchissent
                                    # le plancher (bug snapshot identifié par Regigigas sur S2a).
                                    ctx.clan_human_pop[e.clan_id] = _vpop - 1
                                if _hungry:
                                    entity.hunger = max(0, entity.hunger - 25)   # repas de survie
                                events.append({"type": "clan_fight",
                                               "attacker_clan": entity.clan_id,
                                               "victim_clan":   e.clan_id,
                                               "x": entity.ix, "y": entity.iy})
                                if (_HEROES_ON and entity.etype == EntityType.HUMAN
                                        and e.etype == EntityType.HUMAN):   # P5 E4 : kill de guerre humain
                                    entity.war_kills += 1
                                    if entity.war_kills >= HERO_KILLS:
                                        _name_hero(entity, "kills", world, events)
                    return True
        # S2c — Marche de guerre : le scan ci-dessus n'a engagé personne (aucun ennemi du
        # clan-cible en vision). Un warrior en guerre pas affamé CONVERGE vers le feu du clan
        # ciblé au lieu de vaquer → il finit par croiser l'ennemi (attaque à vue au prochain
        # tick via le même scan). Sans ça la guerre n'éclate qu'au contact fortuit. Gated
        # _WARBEH_ON + _JOBS_ON : JOBS_OFF (pas de warriors) et SOCIETY_OFF (pas de mode war,
        # _at_war=False) → jamais déclenché → hash de ces switches inchangé.
        if (_WARBEH_ON and _JOBS_ON and _at_war and not _hungry
                and ec is not None and ec.war_target >= 0):
            _wtgt = clans.get(ec.war_target)
            if _wtgt is not None:
                entity.state = State.HUNTING
                _move_toward(entity, _wtgt.cx, _wtgt.cy, _eff_speed * 1.1, world)
                return True
    # 3.0 Mange le pain au moulin (humains affamés, moulin adjacent avec du pain)
    if entity.spec.can_build and entity.clan_id is not None and entity.hunger > 35:
        clan_mills = _cb.get("mill", [])
        mill_adj = next((m for m in clan_mills
                         if m.bread > 0 and _dist(entity.x, entity.y, m.x, m.y) < 1.5), None)
        if mill_adj is not None:
            mill_adj.bread -= 1
            entity.hunger   = max(0.0, entity.hunger - MILL_BREAD_FOOD)
            entity.state    = State.EATING
            return True
        # Moulin avec pain mais pas adjacent → se déplace
        mill_ripe = next((m for m in clan_mills if m.bread > 0), None)
        if mill_ripe is not None and entity.hunger > 50:
            entity.state    = State.SEEKING_FOOD
            entity.target_x = float(mill_ripe.x)
            entity.target_y = float(mill_ripe.y)
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed * 1.1, world)
            return True
    # 3. Mange ou cherche de la nourriture
    if spec.eat_amount > 0 and entity.hunger > 30:
        food = world.get_food(entity.ix, entity.iy)
        # Un non-aquatique ne broute PAS le plancton d'une tuile d'eau : sinon une
        # entité échouée mange sur place et survit indéfiniment (l'état ne se dénoue
        # jamais). Sur l'eau → pas de repas → elle finit par mourir de faim.
        _tile_edible = spec.aquatic or world.is_walkable(entity.ix, entity.iy, False)
        if food >= 5 and _tile_edible:
            entity.state = State.EATING
            consumed = world.consume_food(entity.ix, entity.iy, spec.eat_amount)
            entity.hunger = max(0, entity.hunger - consumed * 1.5)
            if not spec.aquatic:
                world.consume_fertility(entity.ix, entity.iy)
            # Efface la cible après manger → force à réévaluer au lieu de revenir
            entity.target_x = None
            entity.target_y = None
            return True
        # Soif modérée interrompt la recherche de nourriture (mais pas le repas en cours)
        if not spec.aquatic and entity.thirst > 50:
            if _drink_or_seek_water(entity, world, events, buildings, cb=_cb):
                return True
        # Cherche une tuile avec de la nourriture dans son champ de vision (numpy)
        entity.state = State.SEEKING_FOOD
        r = int(entity.traits["vision"])
        ex_i, ey_i = entity.ix, entity.iy
        x0 = max(0, ex_i - r); x1 = min(world.width,  ex_i + r + 1)
        y0 = max(0, ey_i - r); y1 = min(world.height, ey_i + r + 1)
        food_sub = world.food_grid[y0:y1, x0:x1].copy()
        walk_mask = (world._aquatic_walkable if spec.aquatic else world._walkable)[y0:y1, x0:x1]
        food_sub[~walk_mask] = 0.0
        if not spec.aquatic:
            food_sub[world._near_water[y0:y1, x0:x1]] *= 0.35
        best_idx = int(np.argmax(food_sub))
        best_f = float(food_sub.flat[best_idx])
        if best_f > 5:
            ly, lx = np.unravel_index(best_idx, food_sub.shape)
            entity.target_x = float(x0 + int(lx))
            entity.target_y = float(y0 + int(ly))
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed, world)
            return True
        # Aucune nourriture dans le champ de vision → scan global (comme pour les arbres)
        if not spec.aquatic:
            _food_tiles = np.argwhere(world.food_grid > 8)
            if len(_food_tiles):
                _wk = world._walkable
                _food_tiles = _food_tiles[_wk[_food_tiles[:, 0], _food_tiles[:, 1]]]
            if len(_food_tiles):
                _dists = (_food_tiles[:, 1] - entity.x)**2 + (_food_tiles[:, 0] - entity.y)**2
                _best = _food_tiles[int(np.argmin(_dists))]
                entity.target_x = float(_best[1])
                entity.target_y = float(_best[0])
                entity.state    = State.EXPLORING
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed, world)
                return True
    # 3.7 Soif légère (après manger, avant bois/construction)
    if not spec.aquatic and entity.thirst > 35:
        if _drink_or_seek_water(entity, world, events, buildings, cb=_cb):
            return True
    # 3.8 Pêche (canne à pêche + affamé, après échec de la recherche de nourriture)
    if (entity.spec.can_build and entity.fishing_rod is not None
            and entity.hunger > FISHING_HUNGER_THRESH):
        # Adjacent à une tuile eau → pêche
        fishing_spot = None
        for adx in range(-1, 2):
            for ady in range(-1, 2):
                wx, wy = entity.ix + adx, entity.iy + ady
                if world.is_valid(wx, wy) and int(world.biome_grid[wy, wx]) in WATER_BIOMES:
                    fishing_spot = (wx, wy)
                    break
            if fishing_spot:
                break
        if fishing_spot:
            entity.state = State.FISHING
            if random.random() < FISHING_CATCH_PROB:
                entity.hunger = max(0.0, entity.hunger - FISHING_FOOD)
                pass  # fish_catch — événement interne silencieux
            return True
        # Pas adjacent : chercher une berge accessible
        spot = _find_water_spot(entity, world)
        if spot:
            entity.state = State.FISHING
            entity.target_x, entity.target_y = spot
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed, world)
            return True
    return False


def _beh_work(entity, ctx, _cb, _eff_speed):
    """Activités : dépôt de ressources, chantiers, reproduction, construction, agriculture, coupe/minage, exploration. True si agi."""
    world = ctx.world; all_entities = ctx.all_entities
    births = ctx.births; events = ctx.events; tick = ctx.tick
    season = ctx.season; clans = ctx.clans; buildings = ctx.buildings
    temp_c = ctx.temp_c; species_counts = ctx.species_counts
    raining = ctx.raining; heatwave = ctx.heatwave; clan_bldg = ctx.clan_bldg
    predators = ctx.predators; predator_grid = ctx.predator_grid
    entity_grid = ctx.entity_grid
    spec = entity.spec
    # 3.4 Mission caravane (D1) — en tête : un marchand en mission n'est jamais
    # happé par un chantier/craft/repro. La survie (blocs amont de la cascade)
    # reste prioritaire : la mission se met en pause, ne s'annule pas.
    if entity.trade_phase is not None:
        if _beh_trade(entity, ctx, _cb, _eff_speed):
            return True
    # 3.4bis Mission de pèlerinage (C1) — même priorité que la caravane
    if entity.pilgrim_phase is not None:
        if _beh_pilgrim(entity, ctx, _cb, _eff_speed):
            return True
    # 3.4ter Mission d'exploration (P7 G1) — même priorité : un éclaireur en mission n'est
    # pas happé par un chantier. Slot toujours None sous CARTO_OFF → garde gratuite.
    if entity.expedition_phase is not None:
        if _beh_expedition(entity, ctx, _cb, _eff_speed):
            return True
    # 3.4quater Marche des colons (P7 G2) — la colonne qui traverse la carte vers sa terre.
    if entity.colonist_dest is not None:
        if _beh_colonist(entity, ctx, _cb, _eff_speed):
            return True
    # 3.45 OFFICE (C1) : la cloche a sonné → procession vers l'église du clan puis
    # prière (PRAY_DURATION agenouillé) → bénédiction. Collectif, gratuit, borné à
    # la fenêtre. La survie (blocs amont) reste prioritaire : un affamé saute l'office.
    if (entity.spec.can_build and entity.clan_id is not None
            and entity._build_target_type is None
            and entity.trade_phase is None and entity.pilgrim_phase is None):
        _churches = _cb.get("church", [])
        if _churches:
            _in_window = ((ctx.tick + entity.clan_id * 37) % CHURCH_SERVICE_PERIOD
                          < CHURCH_SERVICE_WINDOW)
            if not _in_window:
                entity.pray_ticks = 0   # reset paresseux hors fenêtre
            elif (entity.hunger < PRAY_HUNGER_MAX and entity.thirst < PRAY_THIRST_MAX):
                _ch = _churches[0]
                _d_ch = _dist(entity.x, entity.y, _ch.x, _ch.y)
                if _d_ch <= CHURCH_CALL_RADIUS:
                    if _d_ch > PRAY_RADIUS:
                        entity.state = State.PRAYING   # la PROCESSION (théâtre vrai)
                        entity.target_x = float(_ch.x); entity.target_y = float(_ch.y)
                        _move_toward(entity, entity.target_x, entity.target_y,
                                     _eff_speed, world)
                        return True
                    entity.state = State.PRAYING       # agenouillé au parvis
                    entity.pray_ticks += 1
                    if entity.pray_ticks >= PRAY_DURATION:
                        entity.blessed_ticks = BLESS_DURATION   # rafraîchie, jamais cumulée
                        entity.pray_ticks = 0
                    return True
    # 3.5 Dépôt des ressources à la maison du clan
    if ((entity.spec.can_chop or entity.spec.can_mine)
            and (entity.wood > 0 or entity.stone > 0)
            and entity.clan_id is not None):
        clan_houses = _cb.get("house", [])
        if clan_houses:
            # Cible : pour déposer du bois, préférer une maison avec de la place ;
            # pour la pierre (non capée) ou si tout est plein, n'importe quelle maison.
            # `nearest` est TOUJOURS défini ici (l'ancien code laissait un chemin où il
            # ne l'était pas → UnboundLocalError quand un mineur portait de la pierre et
            # que toutes les maisons étaient pleines de bois).
            _houses_with_space = [h for h in clan_houses if h.wood < MAX_WOOD_PER_HOUSE]
            if entity.wood > 0 and not _houses_with_space:
                entity.wood = 0   # bois plein partout → on l'abandonne (comportement d'origine)
            _target_pool = _houses_with_space if (entity.wood > 0 and _houses_with_space) else clan_houses
            nearest = min(_target_pool, key=lambda b: _dist(entity.x, entity.y, b.x, b.y))
            dist_to_house = _dist(entity.x, entity.y, nearest.x, nearest.y)
            if dist_to_house < 1.5:
                # Adjacent : dépose (bois capé, pierre non capée)
                _wood_space = max(0, MAX_WOOD_PER_HOUSE - nearest.wood)
                nearest.wood  += min(entity.wood, _wood_space)
                entity.wood   = max(0, entity.wood - _wood_space)
                nearest.stone += entity.stone; entity.stone = 0
                # Fabrication d'outils (cascade partagée avec C1bis, cf. _try_craft)
                _try_craft(entity, nearest, events)
            elif entity.wood >= MAX_CARRY or entity.stone >= MAX_STONE_CARRY:
                # Portée pleine → aller déposer
                entity.state = State.WANDERING
                entity.target_x = float(nearest.x)
                entity.target_y = float(nearest.y)
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed, world)
                return True
    # 3.6 Dépôt du FER à la forge du clan (bloc B) + upgrade d'outil fer sur place.
    # On ne cible QUE les forges avec de la place : voyager vers une forge pleine
    # pour y jeter son fer était un aller-retour pour rien (gate-review B).
    if (entity.spec.can_mine and entity.iron > 0 and entity.clan_id is not None):
        _forges = _cb.get("forge", [])
        _forge = next((f for f in _forges if f.iron < FORGE_MAX_IRON), None)
        if _forge is not None:
            if _dist(entity.x, entity.y, _forge.x, _forge.y) < 1.5:
                _space = max(0, FORGE_MAX_IRON - _forge.iron)
                _forge.iron += min(entity.iron, _space)
                entity.iron = 0   # surplus abandonné si la forge s'est remplie entre-temps
                _try_forge_upgrade(entity, _forge, events)
                return True
            elif entity.iron >= MAX_IRON_CARRY:
                entity.state = State.WANDERING
                entity.target_x = float(_forge.x); entity.target_y = float(_forge.y)
                _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
                return True
        elif _forges and entity.iron >= MAX_IRON_CARRY:
            entity.iron = 0   # toutes les forges pleines → on lâche sur place (symétrie bois)
    # 3.64 F1 : dépôt de l'OR au MARCHÉ (si gold_dest=market). AVANT le dépôt église INCHANGÉ ci-dessous
    # → sous MONEY_OFF ce bloc est mort (gold_dest toujours "church") → dépôt église exact. Débordement
    # du coffre → dorure de l'ÉGLISE (le puits ; l'église est requise à la pose de l'expédition marché).
    # Repli défensif (perte bornée) : marché disparu en route → on RETOMBE sur le bloc église ci-dessous.
    if (_MONEY_ON and entity.spec.can_mine and entity.gold > 0
            and entity.clan_id is not None and entity.gold_dest == "market"):
        _mkt_dep = _cb.get("market", [])
        if _mkt_dep:
            _md = _mkt_dep[0]
            if _dist(entity.x, entity.y, _md.x, _md.y) < 1.5:
                _g = _md.gold + entity.gold
                _md.gold = min(MARKET_GOLD_MAX, _g)
                _spill = _g - _md.gold
                if _spill:
                    _chs_sp = _cb.get("church", [])
                    if _chs_sp:
                        _chs_sp[0].gilt += _spill
                        events.append({"type": "church_gilt", "clan_id": entity.clan_id,
                                       "gilt": _chs_sp[0].gilt, "x": _chs_sp[0].x, "y": _chs_sp[0].y})
                entity.gold = 0
                events.append({"type": "gold_deposit", "clan_id": entity.clan_id,
                               "total": _md.gold, "x": _md.x, "y": _md.y})
                return True
            else:                                    # livraison engagée : converge vers l'étal
                entity.state = State.WANDERING
                entity.target_x = float(_md.x); entity.target_y = float(_md.y)
                _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
                return True
        # marché disparu → chute vers le bloc église (repli défensif, perte bornée)
    # 3.65 Dépôt de l'OR au trésor de l'église (bloc C2). Débordement → DORURE
    # (le puits). Condition de marche amendée contre-vérif : rapatrie aussi la
    # pièce orpheline quand la vanne s'est fermée en cours d'expédition.
    if entity.spec.can_mine and entity.gold > 0 and entity.clan_id is not None:
        _chs_dep = _cb.get("church", [])
        if _chs_dep:
            _chd = _chs_dep[0]
            if _dist(entity.x, entity.y, _chd.x, _chd.y) < 1.5:
                _g = _chd.gold + entity.gold
                _chd.gold = min(GOLD_TREASURY_MAX, _g)
                _spill = _g - _chd.gold
                if _spill:
                    _chd.gilt += _spill
                    events.append({"type": "church_gilt", "clan_id": entity.clan_id,
                                   "gilt": _chd.gilt, "x": _chd.x, "y": _chd.y})
                entity.gold = 0
                events.append({"type": "gold_deposit", "clan_id": entity.clan_id,
                               "total": _chd.gold, "x": _chd.x, "y": _chd.y})
                return True
            elif (entity.gold >= MAX_GOLD_CARRY
                  or _chd.gold >= GOLD_RESTOCK_THRESHOLD):
                entity.state = State.WANDERING
                entity.target_x = float(_chd.x); entity.target_y = float(_chd.y)
                _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
                return True
    # C1bis : fabrication d'outils DÉCOUPLÉE du dépôt. Un humain qui manque un outil
    # de base (pioche OU hache) va à la maison du clan qui a le stock et le fabrique,
    # même sans rien à déposer. Sinon, une fois le bois du clan capé (plus de dépôt),
    # les humains nés ensuite restent sans outil → coupe/minage qui s'éteint (chaîne
    # figée même avec de la pierre en stock).
    if (entity.spec.can_build and entity.clan_id is not None
            and entity.hunger < 70
            and (entity.pick is None or entity.tool is None)):
        _craftable = [h for h in _cb.get("house", [])
                      if (entity.pick is None and h.wood >= PICK_WOOD_COST)
                      or (entity.tool is None and h.wood >= AXE_CRAFT_COST)]
        if _craftable:
            _hc = min(_craftable, key=lambda b: _dist(entity.x, entity.y, b.x, b.y))
            if _dist(entity.x, entity.y, _hc.x, _hc.y) < 1.5:
                if _try_craft(entity, _hc, events):
                    entity.state = State.BUILDING
                    return True
            else:
                entity.state = State.BUILDING
                entity.target_x = float(_hc.x)
                entity.target_y = float(_hc.y)
                _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
                return True
    # Bloc B : upgrade d'outil PIERRE → FER à la forge, DÉCOUPLÉ (comme C1bis). Un
    # humain avec une pioche/hache pierre, près de la forge du clan qui a du fer, va
    # la mettre à niveau — même sans transporter de fer. Auto-limité (condition fausse
    # une fois l'outil en fer). Gate de distance (gate-review B) : seuls les humains
    # DÉJÀ proches y vont — sinon tout le clan pèlerine à chaque lot de 3 fer déposé
    # (le fer draine à 3/upgrade, les lointains marchaient pour rien).
    if (entity.spec.can_build and entity.clan_id is not None and entity.hunger < 70
            and (entity.pick == "stone_pick" or entity.tool == "stone_axe")):
        _fu = next((f for f in _cb.get("forge", [])
                    if ((entity.pick == "stone_pick" and f.iron >= IRON_PICK_COST)
                        or (entity.tool == "stone_axe" and f.iron >= IRON_AXE_COST))
                    and _dist(entity.x, entity.y, f.x, f.y) < 24), None)
        if _fu is not None:
            if _dist(entity.x, entity.y, _fu.x, _fu.y) < 1.5:
                if _try_forge_upgrade(entity, _fu, events):
                    entity.state = State.BUILDING
                    return True
            else:
                entity.state = State.BUILDING
                entity.target_x = float(_fu.x); entity.target_y = float(_fu.y)
                _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
                return True

    # 4.1 Travailler sur un chantier en cours du clan (n'importe quel humain peut contribuer)
    # Ne pas interrompre un humain qui se rend déjà sur son propre chantier planifié
    if (_role_ok(entity.role, "build") and entity.spec.can_build and entity.clan_id is not None
            and entity.hunger < 80 and entity._build_target_type is None):
        clan_sites = [b for btl in [v for k, v in _cb.items() if k.startswith("site_")] for b in btl if b.work_done < b.work_needed]
        if clan_sites:
            near = min(clan_sites,
                       key=lambda b: _dist(entity.x, entity.y, b.x, b.y))
            entity.state = State.BUILDING
            if _dist(entity.x, entity.y, near.x, near.y) < 1.5:
                near.work_done += 2 if (_JOBS_ON and entity.role == "builder") else 1  # bâtisseur ×2 (P1)
                if (_HEROES_ON and entity.etype == EntityType.HUMAN
                        and near.work_done >= near.work_needed):   # P5 E4 : le FINISSEUR gagne le crédit
                    entity.built_count += 1
                    if entity.built_count >= HERO_BUILDS:
                        _name_hero(entity, "builds", world, events)
                entity.target_x = None
                entity.target_y = None
            else:
                entity.target_x = float(near.x)
                entity.target_y = float(near.y)
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed, world)
            return True
    # 4. Reproduction (bloquée en automne et en hiver)
    if (SEASON_REPRO_ALLOWED[season]
            and entity.hunger < spec.repro_hunger_min
            and entity.repro_cooldown_left == 0
            and entity.sex == Sex.FEMALE
            and entity.age > spec.max_age * 0.20):
        # Cap population par maisons pour les clans humains
        repro_allowed = True
        if entity.spec.can_build and entity.clan_id is not None:
            base = BUILDING_SPECS["house"].pop_bonus
            cap = sum(base * b.level for b in _cb.get("house", []))
            _clan_age = clans.get(entity.clan_id) if clans else None
            if _clan_age is not None:
                cap += AGE_POP_BONUS * _clan_age.age   # bonus d'âge (A1)
                # P5 E2 — Fête des moissons : natalité facilitée (le SEUIL de cap est ×MULT).
                # Ne touche QUE ce seuil comparé à un tirage existant ; gated _FEAST_ON → transparent.
                if _FEAST_ON and _clan_age.feast_ticks > 0:
                    cap = int(cap * FEAST_BIRTH_MULT)
            clan_pop = ctx.clan_human_pop.get(entity.clan_id, 0)   # perf P1 (mémo par-tick)
            if clan_pop >= cap:
                repro_allowed = False
        if repro_allowed:
            partner = None
            for e in all_entities:
                same_clan = (entity.clan_id is None or e.clan_id == entity.clan_id)
                if (e is not entity and e.alive
                        and e.etype == entity.etype
                        and e.sex == Sex.MALE
                        and e.hunger < spec.repro_hunger_min
                        and same_clan
                        and _dist(entity.x, entity.y, e.x, e.y) < 3):
                    partner = e
                    break
            if partner:
                entity.state = State.SEEKING_MATE
                entity.gestation_left = spec.gestation
                # I3 : rebond démographique — une espèce animale sous le plancher
                # recolonise 2× plus vite (anti-cascade). Humains exclus (démographie
                # régie par le logement/clan, pas par ce filet écologique).
                _cd = spec.repro_cooldown
                if (entity.etype is not EntityType.HUMAN
                        and (species_counts or {}).get(entity.etype.value, 0) < REBOUND_FLOOR):
                    _cd //= 2
                entity.repro_cooldown_left = _cd
                partner.repro_cooldown_left = _cd
                return True
            # E2 : aucun partenaire adjacent (<3) → viser le mâle éligible le plus
            # proche EN VISION et s'en approcher. Sans ça, la repro n'arrive que si
            # deux éligibles se croisent par hasard → goulot fatal à basse densité
            # (recolonisation, requins isolés). État SEEKING_MATE enfin utilisé pour
            # du déplacement. Même filtre clan que la recherche adjacente ci-dessus.
            suitor, best_d = None, spec.vision * spec.vision
            ex, ey = entity.x, entity.y
            for e in all_entities:
                if (e is entity or not e.alive
                        or e.etype != entity.etype
                        or e.sex != Sex.MALE
                        or e.hunger >= spec.repro_hunger_min):
                    continue
                if entity.clan_id is not None and e.clan_id != entity.clan_id:
                    continue
                dx = ex - e.x; dy = ey - e.y; d = dx*dx + dy*dy
                if d < best_d:
                    best_d = d; suitor = e
            if suitor is not None:
                entity.state = State.SEEKING_MATE
                entity.target_x = float(suitor.x); entity.target_y = float(suitor.y)
                _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
                return True
    # 4.24 Se déplacer vers le chantier planifié et le démarrer une fois sur place
    if (entity.spec.can_build
            and entity._build_target_type is not None
            and entity.clan_id is not None):
        btype_t = entity._build_target_type
        btx = entity._build_target_x
        bty = entity._build_target_y
        bx_t = int(btx)
        by_t = int(bty)
        # Pour le champ de blé (multiples permis), annuler seulement si la case est prise ou invalide
        # Pour les autres types (unique en cours), annuler si n'importe quel chantier de ce type existe
        if btype_t == "wheatfield":
            existing_site = _cb.get("site_wheatfield", [])
            should_cancel = (any(s.x == bx_t and s.y == by_t for s in existing_site)
                             or not world.is_valid(bx_t, by_t)
                             or not world.is_walkable(bx_t, by_t))
        else:
            existing_site = _cb.get(f"site_{btype_t}", [])
            should_cancel = (bool(existing_site)
                             or not world.is_valid(bx_t, by_t)
                             or not world.is_walkable(bx_t, by_t))
        if should_cancel:
            entity._build_target_type = None
            entity._build_target_x    = None
            entity._build_target_y    = None
        elif _dist(entity.x, entity.y, btx, bty) < 1.5:
            # Arrivé à l'emplacement : vérifier ressources et créer le chantier
            bspec_t    = BUILDING_SPECS[btype_t]
            clan_houses_t = _cb.get("house", [])
            can_build_t   = False
            # Éviter double déduction si un autre humain a émis le même start_site ce tick
            already_this_tick = any(
                ev.get("type") == "start_site"
                and ev.get("btype") == btype_t
                and ev.get("clan_id") == entity.clan_id
                for ev in events
            )
            if not already_this_tick:
                if btype_t == "house":
                    if not clan_houses_t:
                        if entity.wood >= bspec_t.first_cost:
                            entity.wood -= bspec_t.first_cost
                            can_build_t = True
                    else:
                        donor_t = max(clan_houses_t, key=lambda b: b.wood)
                        if donor_t.wood >= bspec_t.wood_cost:
                            donor_t.wood -= bspec_t.wood_cost
                            can_build_t = True
                elif btype_t in ("well", "mill"):
                    dw = max(clan_houses_t, key=lambda b: b.wood)  if clan_houses_t else None
                    ds = max(clan_houses_t, key=lambda b: b.stone) if clan_houses_t else None
                    if (dw and ds
                            and dw.wood  >= bspec_t.wood_cost
                            and ds.stone >= bspec_t.stone_cost):
                        dw.wood  -= bspec_t.wood_cost
                        ds.stone -= bspec_t.stone_cost
                        can_build_t = True
                elif btype_t == "church":  # bloc C1 : bois donor + pierre du POOL (comme forge)
                    dw = max(clan_houses_t, key=lambda b: b.wood) if clan_houses_t else None
                    if (dw and dw.wood >= bspec_t.wood_cost
                            and sum(h.stone for h in clan_houses_t) >= bspec_t.stone_cost):
                        if _spend_pool_stone(clan_houses_t, bspec_t.stone_cost):
                            dw.wood -= bspec_t.wood_cost
                            can_build_t = True
                elif btype_t == "monument":  # P5 E3 : 20 pierre du POOL + 4 or du TRÉSOR d'église
                    clan_churches_t = _cb.get("church", [])
                    if (sum(h.stone for h in clan_houses_t) >= bspec_t.stone_cost
                            and sum(ch.gold for ch in clan_churches_t) >= MONUMENT_GOLD):
                        if _spend_pool_stone(clan_houses_t, bspec_t.stone_cost):
                            _need_gold = MONUMENT_GOLD   # prélève l'or au trésor (comme l'offrande C2)
                            for ch in clan_churches_t:
                                _take = min(ch.gold, _need_gold)
                                ch.gold -= _take
                                _need_gold -= _take
                                if _need_gold == 0:
                                    break
                            can_build_t = True
                elif btype_t == "forge":   # bloc B : bois d'une maison + pierre du POOL clan
                    dw = max(clan_houses_t, key=lambda b: b.wood) if clan_houses_t else None
                    if (dw and dw.wood >= bspec_t.wood_cost
                            and sum(h.stone for h in clan_houses_t) >= bspec_t.stone_cost):
                        if _spend_pool_stone(clan_houses_t, bspec_t.stone_cost):
                            dw.wood -= bspec_t.wood_cost
                            can_build_t = True
                elif btype_t == "market":  # bloc D1 : bois seul, d'une maison
                    dw = max(clan_houses_t, key=lambda b: b.wood) if clan_houses_t else None
                    if dw and dw.wood >= bspec_t.wood_cost:
                        dw.wood -= bspec_t.wood_cost
                        can_build_t = True
                elif btype_t == "wheatfield":
                    can_build_t = True  # pas de coût en ressources
            if can_build_t:
                entity.building_type = btype_t
                _ss = {"type": "start_site", "btype": btype_t,
                       "x": bx_t, "y": by_t,
                       "clan_id": entity.clan_id,
                       "work_needed": bspec_t.build_time}
                if btype_t == "monument":   # P5 E3 : dédicace figée à la POSE = dernier jalon du clan
                    _cl = clans.get(entity.clan_id) if clans else None
                    _ss["dedication"] = _cl.last_deed if _cl else ""
                events.append(_ss)
            entity._build_target_type = None
            entity._build_target_x    = None
            entity._build_target_y    = None
            return True
        else:
            # En chemin vers le chantier prévu
            entity.state   = State.BUILDING
            entity.target_x = btx
            entity.target_y = bty
            _move_toward(entity, btx, bty, _eff_speed, world)
            return True
    # 4.25 Démarre une construction de maison (l'humain doit rejoindre l'emplacement)
    if (entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_type is None
            and entity._build_target_type is None
            and entity.hunger < 65
            and clans):
        clan = clans.get(entity.clan_id)
        if clan:
            bspec = BUILDING_SPECS["house"]
            clan_houses = _cb.get("house", [])
            clan_sites_h = _cb.get("site_house", [])
            # Cap atteint ou chantier déjà en cours → pas de nouvelle construction
            _total_h = len(clan_houses) + len(clan_sites_h)
            if bspec.max_per_clan == 0 or _total_h < bspec.max_per_clan:
                # Ne construire que si la population dépasse ~70% de la capacité actuelle
                _pop_cap_h = sum(bspec.pop_bonus * b.level for b in clan_houses)
                _clan_age_h = clans.get(entity.clan_id) if clans else None
                if _clan_age_h is not None:
                    _pop_cap_h += AGE_POP_BONUS * _clan_age_h.age   # bonus d'âge (A1)
                _clan_pop_h = ctx.clan_human_pop.get(entity.clan_id, 0)   # perf P1 (mémo par-tick)
                _need_house = (not clan_houses) or (_clan_pop_h >= _pop_cap_h * 0.70)
                # Vérifier si un autre humain a déjà planifié une maison ce tick
                already_planned = (
                    _need_house and (
                        any(ev.get("type") == "start_site" and ev.get("btype") == "house"
                            and ev.get("clan_id") == entity.clan_id for ev in events)
                        or any(e._build_target_type == "house" and e.clan_id == entity.clan_id
                               for e in all_entities if e.alive and e is not entity)
                    )
                )
                # Source de bois : porté (1re maison) ou stockage (maisons suivantes)
                if not clan_houses:
                    can_build_house = (_need_house and not already_planned
                                       and entity.wood >= bspec.first_cost)
                else:
                    donor = max(clan_houses, key=lambda b: b.wood)
                    can_build_house = (_need_house and not already_planned
                                       and donor.wood >= bspec.wood_cost)
                if can_build_house:
                    for _ in range(30):
                        angle = random.uniform(0, 2 * math.pi)
                        dist  = random.uniform(bspec.min_from_fire, bspec.max_from_fire)
                        bx = int(clan.cx + math.cos(angle) * dist)
                        by = int(clan.cy + math.sin(angle) * dist)
                        if not world.is_valid(bx, by) or not world.is_walkable(bx, by):
                            continue
                        if any(_dist(bx, by, b.x, b.y) < bspec.min_dist
                               for b in (buildings or [])
                               if b.btype in ("house", "site_house")):
                            continue
                        # Emplacement valide → planifier le déplacement (pas de déduction bois encore)
                        entity._build_target_type = "house"
                        entity._build_target_x    = float(bx)
                        entity._build_target_y    = float(by)
                        entity.state   = State.BUILDING
                        entity.target_x = float(bx)
                        entity.target_y = float(by)
                        _move_toward(entity, entity.target_x, entity.target_y,
                                     _eff_speed, world)
                        return True
    # 4.26 Construction d'un moulin (si le clan a ≥2 champs et pas encore de moulin)
    if (entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_type is None
            and entity._build_target_type is None
            and entity.hunger < 65
            and clans):
        bspec_m = BUILDING_SPECS["mill"]
        clan_mills   = _cb.get("mill", [])
        clan_sites_m = _cb.get("site_mill", [])
        clan_fields  = _cb.get("wheatfield", [])
        if (len(clan_fields) >= 2 and not clan_sites_m
                and (bspec_m.max_per_clan == 0 or len(clan_mills) < bspec_m.max_per_clan)):
            clan_houses = _cb.get("house", [])
            donor = max(clan_houses, key=lambda b: b.wood) if clan_houses else None
            donor_s = max(clan_houses, key=lambda b: b.stone) if clan_houses else None
            has_res = (donor and donor.wood >= bspec_m.wood_cost
                       and donor_s and donor_s.stone >= bspec_m.stone_cost)
            if has_res:
                # Placement adjacent à un champ de blé existant
                already_planned_m = any(
                    e._build_target_type == "mill" and e.clan_id == entity.clan_id
                    for e in all_entities if e.alive and e is not entity
                )
                if not already_planned_m:
                    for field in clan_fields:
                        for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
                            mx, my = field.x + ddx, field.y + ddy
                            if not world.is_valid(mx, my) or not world.is_walkable(mx, my):
                                continue
                            # C3 : ne bâtir un moulin QUE près de l'eau — sinon il ne
                            # cuit jamais (même test que la production) et le blé livré
                            # y pourrit : trou noir alimentaire permanent.
                            if not _tile_near_water(world, mx, my):
                                continue
                            if any(_dist(mx, my, b.x, b.y) < bspec_m.min_dist
                                   for b in (buildings or []) if b.btype == "mill"):
                                continue
                            # Planifier le déplacement (ressources déduites à l'arrivée)
                            entity._build_target_type = "mill"
                            entity._build_target_x    = float(mx)
                            entity._build_target_y    = float(my)
                            entity.state   = State.BUILDING
                            entity.target_x = float(mx)
                            entity.target_y = float(my)
                            _move_toward(entity, entity.target_x, entity.target_y,
                                         _eff_speed, world)
                            return True
    # 4.27 Construction d'un puit (si le clan a ≥1 maison et pas encore de puit)
    if (entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_type is None
            and entity._build_target_type is None
            and entity.hunger < 65
            and clans):
        bspec_well = BUILDING_SPECS["well"]
        clan_wells  = _cb.get("well", [])
        clan_sites_w = _cb.get("site_well", [])
        clan_houses = _cb.get("house", [])
        if (len(clan_houses) >= 2 and not clan_sites_w
                and (bspec_well.max_per_clan == 0 or len(clan_wells) < bspec_well.max_per_clan)):
            donor   = max(clan_houses, key=lambda b: b.wood)   if clan_houses else None
            donor_s = max(clan_houses, key=lambda b: b.stone)  if clan_houses else None
            has_res = (donor   and donor.wood    >= bspec_well.wood_cost
                       and donor_s and donor_s.stone >= bspec_well.stone_cost)
            if has_res:
                clan = clans.get(entity.clan_id)
                already_planned_w = (
                    any(ev.get("type") == "start_site" and ev.get("btype") == "well"
                        and ev.get("clan_id") == entity.clan_id for ev in events)
                    or any(e._build_target_type == "well" and e.clan_id == entity.clan_id
                           for e in all_entities if e.alive and e is not entity)
                )
                if clan and not already_planned_w:
                    for _ in range(30):
                        angle = random.uniform(0, 2 * math.pi)
                        dist  = random.uniform(bspec_well.min_from_fire, bspec_well.max_from_fire)
                        wx = int(clan.cx + math.cos(angle) * dist)
                        wy = int(clan.cy + math.sin(angle) * dist)
                        if not world.is_valid(wx, wy) or not world.is_walkable(wx, wy):
                            continue
                        if any(_dist(wx, wy, b.x, b.y) < bspec_well.min_dist
                               for b in (buildings or []) if b.btype == "well"):
                            continue
                        # Planifier le déplacement (la déduction des ressources se fait à l'arrivée)
                        entity._build_target_type = "well"
                        entity._build_target_x    = float(wx)
                        entity._build_target_y    = float(wy)
                        entity.state   = State.BUILDING
                        entity.target_x = float(wx)
                        entity.target_y = float(wy)
                        _move_toward(entity, entity.target_x, entity.target_y,
                                     _eff_speed, world)
                        return True
    # 4.28 Construction d'une FORGE (bloc B) : clan de l'Âge du Fer, ≥1 maison, pas
    # encore de forge. Placement autour du feu de camp (pas d'eau requise).
    if (entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_type is None
            and entity._build_target_type is None
            and entity.hunger < 65
            and clans):
        _clan_f = clans.get(entity.clan_id)
        if _clan_f is not None and _clan_f.age >= FER_AGE:
            bspec_forge  = BUILDING_SPECS["forge"]
            clan_forges  = _cb.get("forge", [])
            clan_sites_f = _cb.get("site_forge", [])
            clan_houses  = _cb.get("house", [])
            if (clan_houses and not clan_sites_f
                    and (bspec_forge.max_per_clan == 0
                         or len(clan_forges) < bspec_forge.max_per_clan)):
                donor   = max(clan_houses, key=lambda b: b.wood)
                has_res = (donor.wood >= bspec_forge.wood_cost
                           and sum(h.stone for h in clan_houses) >= bspec_forge.stone_cost)
                already_planned_f = (
                    any(ev.get("type") == "start_site" and ev.get("btype") == "forge"
                        and ev.get("clan_id") == entity.clan_id for ev in events)
                    or any(e._build_target_type == "forge" and e.clan_id == entity.clan_id
                           for e in all_entities if e.alive and e is not entity))
                if has_res and not already_planned_f:
                    for _ in range(30):
                        angle = random.uniform(0, 2 * math.pi)
                        dist  = random.uniform(bspec_forge.min_from_fire, bspec_forge.max_from_fire)
                        fx = int(_clan_f.cx + math.cos(angle) * dist)
                        fy = int(_clan_f.cy + math.sin(angle) * dist)
                        if not world.is_valid(fx, fy) or not world.is_walkable(fx, fy):
                            continue
                        if any(_dist(fx, fy, b.x, b.y) < bspec_forge.min_dist
                               for b in (buildings or []) if b.btype == "forge"):
                            continue
                        entity._build_target_type = "forge"
                        entity._build_target_x = float(fx); entity._build_target_y = float(fy)
                        entity.state = State.BUILDING
                        entity.target_x = float(fx); entity.target_y = float(fy)
                        _move_toward(entity, entity.target_x, entity.target_y,
                                     _eff_speed, world)
                        return True
    # 4.29 Construction d'un MARCHÉ (bloc D1) : clan de l'Âge de Pierre, ≥2 maisons,
    # pas encore de marché. Bois seul (un coût pierre le rendrait inconstructible
    # précisément chez l'acheteur pauvre-pierre). Placement près du feu (place du
    # village). Miroir du bloc forge 4.28.
    if (entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_type is None
            and entity._build_target_type is None
            and entity.hunger < 65
            and clans):
        _clan_mk = clans.get(entity.clan_id)
        if _clan_mk is not None and _clan_mk.age >= MARKET_AGE:
            bspec_mk  = BUILDING_SPECS["market"]
            clan_mks  = _cb.get("market", [])
            clan_sites_mk = _cb.get("site_market", [])
            clan_houses = _cb.get("house", [])
            if (len(clan_houses) >= 2 and not clan_sites_mk
                    and (bspec_mk.max_per_clan == 0
                         or len(clan_mks) < bspec_mk.max_per_clan)):
                donor = max(clan_houses, key=lambda b: b.wood)
                already_planned_mk = (
                    any(ev.get("type") == "start_site" and ev.get("btype") == "market"
                        and ev.get("clan_id") == entity.clan_id for ev in events)
                    or any(e._build_target_type == "market" and e.clan_id == entity.clan_id
                           for e in all_entities if e.alive and e is not entity))
                if donor.wood >= bspec_mk.wood_cost and not already_planned_mk:
                    for _ in range(30):
                        angle = random.uniform(0, 2 * math.pi)
                        dist  = random.uniform(bspec_mk.min_from_fire, bspec_mk.max_from_fire)
                        mx2 = int(_clan_mk.cx + math.cos(angle) * dist)
                        my2 = int(_clan_mk.cy + math.sin(angle) * dist)
                        if not world.is_valid(mx2, my2) or not world.is_walkable(mx2, my2):
                            continue
                        if any(_dist(mx2, my2, b.x, b.y) < bspec_mk.min_dist
                               for b in (buildings or []) if b.btype == "market"):
                            continue
                        entity._build_target_type = "market"
                        entity._build_target_x = float(mx2); entity._build_target_y = float(my2)
                        entity.state = State.BUILDING
                        entity.target_x = float(mx2); entity.target_y = float(my2)
                        _move_toward(entity, entity.target_x, entity.target_y,
                                     _eff_speed, world)
                        return True
    # 4.30 Construction d'une ÉGLISE (bloc C1) : clan de l'Âge d'ACIER, ≥2 maisons,
    # pas encore d'église. Bois du donor + pierre au POOL (miroir forge 4.28).
    if (entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_type is None
            and entity._build_target_type is None
            and entity.hunger < 65
            and clans):
        _clan_ch = clans.get(entity.clan_id)
        if _clan_ch is not None and _clan_ch.age >= CHURCH_AGE:
            bspec_ch  = BUILDING_SPECS["church"]
            clan_chs  = _cb.get("church", [])
            clan_sites_ch = _cb.get("site_church", [])
            clan_houses = _cb.get("house", [])
            if (len(clan_houses) >= 2 and not clan_sites_ch
                    and (bspec_ch.max_per_clan == 0
                         or len(clan_chs) < bspec_ch.max_per_clan)):
                donor = max(clan_houses, key=lambda b: b.wood)
                has_res = (donor.wood >= bspec_ch.wood_cost
                           and sum(h.stone for h in clan_houses) >= bspec_ch.stone_cost)
                already_planned_ch = (
                    any(ev.get("type") == "start_site" and ev.get("btype") == "church"
                        and ev.get("clan_id") == entity.clan_id for ev in events)
                    or any(e._build_target_type == "church" and e.clan_id == entity.clan_id
                           for e in all_entities if e.alive and e is not entity))
                if has_res and not already_planned_ch:
                    for _ in range(30):
                        angle = random.uniform(0, 2 * math.pi)
                        dist  = random.uniform(bspec_ch.min_from_fire, bspec_ch.max_from_fire)
                        cx2 = int(_clan_ch.cx + math.cos(angle) * dist)
                        cy2 = int(_clan_ch.cy + math.sin(angle) * dist)
                        if not world.is_valid(cx2, cy2) or not world.is_walkable(cx2, cy2):
                            continue
                        if any(_dist(cx2, cy2, b.x, b.y) < bspec_ch.min_dist
                               for b in (buildings or []) if b.btype == "church"):
                            continue
                        entity._build_target_type = "church"
                        entity._build_target_x = float(cx2); entity._build_target_y = float(cy2)
                        entity.state = State.BUILDING
                        entity.target_x = float(cx2); entity.target_y = float(cy2)
                        _move_toward(entity, entity.target_x, entity.target_y,
                                     _eff_speed, world)
                        return True
    # 4.30b MONUMENT (P5 E3) : clan de l'Âge ACIER avec une église (donc un trésor d'or), ≥2 maisons,
    # PAS de monument vivant. Coût 20 pierre POOL + 4 or trésor, déduits à la pose (4.24). Priorité
    # APRÈS l'église. Gated _MONUMENT_ON → MONUMENT_OFF = zéro planner/tirage → hash E2 exact.
    if (_MONUMENT_ON and entity.spec.can_build
            and entity.clan_id is not None
            and entity.building_type is None
            and entity._build_target_type is None
            and entity.hunger < 65
            and clans):
        _clan_mo = clans.get(entity.clan_id)
        if _clan_mo is not None and _clan_mo.age >= CHURCH_AGE:   # Acier (= âge des églises/or)
            bspec_mo = BUILDING_SPECS["monument"]
            clan_mos      = _cb.get("monument", [])       # monuments VIVANTS (une ruine a btype "ruin")
            clan_sites_mo = _cb.get("site_monument", [])
            clan_houses   = _cb.get("house", [])
            clan_churches = _cb.get("church", [])
            if (len(clan_houses) >= 2 and not clan_sites_mo
                    and len(clan_mos) < bspec_mo.max_per_clan
                    and sum(h.stone for h in clan_houses) >= bspec_mo.stone_cost
                    and sum(ch.gold for ch in clan_churches) >= MONUMENT_GOLD):
                already_planned_mo = (
                    any(ev.get("type") == "start_site" and ev.get("btype") == "monument"
                        and ev.get("clan_id") == entity.clan_id for ev in events)
                    or any(e._build_target_type == "monument" and e.clan_id == entity.clan_id
                           for e in all_entities if e.alive and e is not entity))
                if not already_planned_mo:
                    for _ in range(30):
                        angle = random.uniform(0, 2 * math.pi)
                        dist  = random.uniform(bspec_mo.min_from_fire, bspec_mo.max_from_fire)
                        cx2 = int(_clan_mo.cx + math.cos(angle) * dist)
                        cy2 = int(_clan_mo.cy + math.sin(angle) * dist)
                        if not world.is_valid(cx2, cy2) or not world.is_walkable(cx2, cy2):
                            continue
                        if any(_dist(cx2, cy2, b.x, b.y) < bspec_mo.min_dist
                               for b in (buildings or []) if b.btype == "monument"):
                            continue
                        entity._build_target_type = "monument"
                        entity._build_target_x = float(cx2); entity._build_target_y = float(cy2)
                        entity.state = State.BUILDING
                        entity.target_x = float(cx2); entity.target_y = float(cy2)
                        _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
                        return True
    # 4.3a Récolte : champ mûr (priorité élevée : affamé OU adjacent à un champ mûr)
    if _role_ok(entity.role, "farm") and entity.spec.can_build and entity.clan_id is not None:
        clan_fields = _cb.get("wheatfield", [])
        clan_mills  = _cb.get("mill", [])
        ripe_adj = next((b for b in clan_fields
                         if b.stage >= 4 and _dist(entity.x, entity.y, b.x, b.y) < 1.5), None)
        if ripe_adj is not None:
            entity.state = State.FARMING
            ripe_adj.stage = 1
            ripe_adj.grow_ticks = 0
            ripe_adj.watered_ticks = 0
            # C2 : le récolteur mange sa part SI il a faim (il a récolté pour ça).
            # Le moulin ne reçoit du grain que d'un fermier rassasié (surplus).
            # Avant, tout le blé partait au moulin même quand le récolteur mourait
            # de faim → l'agriculture était un trou noir alimentaire.
            mill_dst = (next((m for m in clan_mills
                              if m.wheat < (MILL_MAX_BREAD * MILL_L2_BREAD_MULT * 2
                                            if (_GRANARY_ON and m.level >= 2) else MILL_MAX_BREAD * 2)), None)
                        if entity.hunger < WHEAT_HUNGER_THRESH else None)
            if mill_dst is not None:
                mill_dst.wheat += 1
            else:
                food = WHEAT_HARVEST_FOOD + (SICKLE_HARVEST_BONUS if entity.sickle else 0.0)
                if _JOBS_ON and entity.role == "farmer":
                    food += 10.0                       # spécialiste : meilleure part (P1)
                entity.hunger = max(0.0, entity.hunger - food)
            return True
        # Chercher un champ mûr si affamé
        if entity.hunger > WHEAT_HUNGER_THRESH:
            ripe_far = next((b for b in clan_fields if b.stage >= 4), None)
            if ripe_far is not None:
                entity.state    = State.FARMING
                entity.target_x = float(ripe_far.x)
                entity.target_y = float(ripe_far.y)
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed, world)
                return True
    # 4.3b/c Entretien des champs : arrosage + plantation (quand reposé et non trop affamé)
    # DÉCISION P1 (écart spec assumé) : l'entretien reste UNIVERSEL (pas guardé farmer) — seule la
    # RÉCOLTE (4.3a) est réservée. Arroser/planter occupe les non-spécialistes sans casser la chaîne
    # alimentaire ; guarder aussi l'entretien affamerait un clan à peu de fermiers.
    if (entity.spec.can_build and entity.clan_id is not None
            and entity.hunger < WHEAT_WORK_THRESH):
        clan_fields = _cb.get("wheatfield", [])
        clan_mills  = _cb.get("mill", [])

        # 4.3b Arrosage (si possède l'arrosoir)
        if entity.watering_can is not None:
            if entity.can_filled:
                # Champ à arroser adjacent (pas encore arrosé, pas encore mûr)
                dry = next((b for b in clan_fields
                            if b.stage < 4 and b.watered_ticks == 0
                            and _dist(entity.x, entity.y, b.x, b.y) < 1.5), None)
                if dry is not None:
                    entity.state    = State.FARMING
                    entity.can_filled = False
                    dry.watered_ticks = WATERED_TICKS
                    return True
                # Se déplacer vers le champ le plus sec
                dry_far = next((b for b in clan_fields
                                if b.stage < 4 and b.watered_ticks == 0), None)
                if dry_far is not None:
                    entity.state    = State.FARMING
                    entity.target_x = float(dry_far.x)
                    entity.target_y = float(dry_far.y)
                    _move_toward(entity, entity.target_x, entity.target_y,
                                 _eff_speed, world)
                    return True
            else:
                # Arrosoir vide → chercher de l'eau : puit du clan d'abord, sinon tuile eau
                clan_wells = _cb.get("well", [])
                near_well = next((b for b in clan_wells
                                  if _dist(entity.x, entity.y, b.x, b.y) < 1.5), None)
                if near_well is not None:
                    entity.state      = State.FARMING
                    entity.can_filled = True
                    return True
                for adx in range(-2, 3):
                    for ady in range(-2, 3):
                        wx, wy = entity.ix + adx, entity.iy + ady
                        if (world.is_valid(wx, wy)
                                and world.biome_grid[wy, wx] in WATER_BIOMES):
                            entity.state    = State.FARMING
                            entity.can_filled = True
                            return True
                # Se déplacer vers le puit le plus proche du clan
                if clan_wells:
                    nearest_well = min(clan_wells,
                                       key=lambda b: _dist(entity.x, entity.y, b.x, b.y))
                    entity.state    = State.FARMING
                    entity.target_x = float(nearest_well.x)
                    entity.target_y = float(nearest_well.y)
                    _move_toward(entity, entity.target_x, entity.target_y,
                                 _eff_speed, world)
                    return True
                # Cherche une source d'eau dans le champ de vision
                r = int(entity.traits["vision"])
                best_w, best_wd = None, float("inf")
                for vdx in range(-r, r + 1):
                    for vdy in range(-r, r + 1):
                        wx, wy = entity.ix + vdx, entity.iy + vdy
                        if (world.is_valid(wx, wy)
                                and world.biome_grid[wy, wx] in WATER_BIOMES):
                            d = _dist(entity.x, entity.y, wx, wy)
                            if d < best_wd:
                                best_wd = d
                                best_w  = (wx, wy)
                if best_w:
                    entity.state    = State.FARMING
                    entity.target_x = float(best_w[0])
                    entity.target_y = float(best_w[1])
                    _move_toward(entity, entity.target_x, entity.target_y,
                                 _eff_speed, world)
                    return True
        # 4.3c Plantation : si le clan peut encore planter
        bspec_w = BUILDING_SPECS["wheatfield"]
        clan_sites_wf = _cb.get("site_wheatfield", [])
        _total_wf = len(clan_fields) + len(clan_sites_wf)
        if (entity.building_type is None
                and entity._build_target_type is None
                and (bspec_w.max_per_clan == 0 or _total_wf < bspec_w.max_per_clan)
                and clans):
            clan = clans.get(entity.clan_id)
            already_planned_wf = (
                any(ev.get("type") == "start_site" and ev.get("btype") == "wheatfield"
                    and ev.get("clan_id") == entity.clan_id for ev in events)
                or any(e._build_target_type == "wheatfield" and e.clan_id == entity.clan_id
                       for e in all_entities if e.alive and e is not entity)
            )
            if clan and not already_planned_wf:
                for _ in range(30):
                    angle = random.uniform(0, 2 * math.pi)
                    dist  = random.uniform(bspec_w.min_from_fire, bspec_w.max_from_fire)
                    fx = int(clan.cx + math.cos(angle) * dist)
                    fy = int(clan.cy + math.sin(angle) * dist)
                    if (not world.is_valid(fx, fy)
                            or not world.is_walkable(fx, fy)
                            or world.biome_grid[fy, fx] not in (int(Biome.GRASS), int(Biome.DIRT))):
                        continue
                    if any(_dist(fx, fy, b.x, b.y) < bspec_w.min_dist for b in clan_fields):
                        continue
                    # Emplacement valide → planifier le déplacement
                    entity._build_target_type = "wheatfield"
                    entity._build_target_x    = float(fx)
                    entity._build_target_y    = float(fy)
                    entity.state   = State.BUILDING
                    entity.target_x = float(fx)
                    entity.target_y = float(fy)
                    _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
                    return True
    # 4.5 Coupe du bois (l'humain doit être SUR la tuile forêt pour couper)
    # Arrête de couper si le total de bois du clan dépasse le seuil global
    _clan_houses_wood = _cb.get("house", [])
    _total_clan_wood = sum(b.wood for b in _clan_houses_wood)
    _wood_full = bool(_clan_houses_wood) and _total_clan_wood >= CLAN_WOOD_CAP
    if (_role_ok(entity.role, "chop") and entity.spec.can_chop and entity.wood < MAX_CARRY
            and entity.hunger < 70 and entity.chop_cooldown_left == 0
            and not _wood_full):
        # Coupe si debout sur une tuile d'arbre debout
        if world.is_choppable(entity.ix, entity.iy):
            entity.state = State.CHOPPING
            wood = world.chop_tree(entity.ix, entity.iy)
            # Hache fer = même rendement/coup que la pierre (un bonus serait écrêté
            # par MAX_CARRY, prouvé au gate-review) mais COUPE 2× PLUS VITE (cooldown).
            if entity.tool in ("stone_axe", "iron_axe"):
                wood += STONE_AXE_BONUS
            elif entity.tool == "axe":
                wood += AXE_BONUS
            entity.wood = min(MAX_CARRY, entity.wood + wood)
            cd = (IRON_AXE_COOLDOWN if entity.tool == "iron_axe" else CHOP_COOLDOWN)
            if _JOBS_ON and entity.role == "woodcutter":   # spécialiste : coupe plus vite (P1)
                cd = max(1, int(cd * JOB_BONUS_COOLDOWN))
            entity.chop_cooldown_left = cd
            return True
        # Arbre dans le champ de vision → se déplace vers le plus proche (numpy)
        r = int(entity.traits["vision"])
        ex_i, ey_i = entity.ix, entity.iy
        x0 = max(0, ex_i - r); x1 = min(world.width,  ex_i + r + 1)
        y0 = max(0, ey_i - r); y1 = min(world.height, ey_i + r + 1)
        chop_sub = (world._forest_mask[y0:y1, x0:x1]
                    & (world.tree_grid[y0:y1, x0:x1] >= TREE_STUMP_THRESHOLD))
        _ys, _xs = np.where(chop_sub)
        best_tree = None
        if len(_xs):
            abs_xs = _xs + x0; abs_ys = _ys + y0
            dists_sq = (abs_xs - entity.x)**2 + (abs_ys - entity.y)**2
            idx = int(np.argmin(dists_sq))
            best_tree = (int(abs_xs[idx]), int(abs_ys[idx]))
        if best_tree:
            entity.state = State.CHOPPING
            entity.target_x = float(best_tree[0])
            entity.target_y = float(best_tree[1])
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed, world)
            return True
        # Aucun arbre visible → chercher la forêt la plus proche sur toute la map
        # (important pour les clans éloignés de la forêt)
        _need_wood = (
            entity.wood < MAX_CARRY
            and (not _clan_houses_wood or entity.wood < 2)
        )
        if _need_wood:
            forest_tiles = np.argwhere(
                world._forest_mask & (world.tree_grid >= TREE_STUMP_THRESHOLD)
            )
            if len(forest_tiles):
                dists = (forest_tiles[:, 1] - entity.x)**2 + (forest_tiles[:, 0] - entity.y)**2
                best = forest_tiles[int(np.argmin(dists))]
                entity.state = State.CHOPPING
                entity.target_x = float(best[1])
                entity.target_y = float(best[0])
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed, world)
                return True
    # 4.55 Mine le FER (bloc B) : mineur d'un clan de l'Âge du Fer, avec pioche.
    # Hystérésis : l'expédition ne se déclenche que si une forge du clan passe SOUS
    # IRON_RESTOCK_THRESHOLD — sinon la vanne se rouvrirait à chaque upgrade (−3) et
    # les mineurs pendouleraient fer↔forge en permanence, affamant la chaîne pierre.
    _cobj_fer = clans.get(entity.clan_id) if (clans and entity.clan_id is not None) else None
    if (_role_ok(entity.role, "mine") and entity.spec.can_mine and entity.pick is not None
            and entity.iron < MAX_IRON_CARRY and entity.hunger < 65
            and _cobj_fer is not None and _cobj_fer.age >= FER_AGE
            and any(f.iron < IRON_RESTOCK_THRESHOLD for f in _cb.get("forge", []))):
        # Gisement de fer adjacent → mine directement (IRON_PER_MINE=3 sature la
        # portée en un coup ; pas de bonus pioche fer ici, il serait écrêté).
        for adx in range(-1, 2):
            for ady in range(-1, 2):
                if adx == 0 and ady == 0:
                    continue
                tx, ty = entity.ix + adx, entity.iy + ady
                if world.is_iron_mineable(tx, ty):
                    entity.state = State.MINING
                    got = world.mine_iron(tx, ty)
                    entity.iron = min(MAX_IRON_CARRY, entity.iron + got)
                    return True
        # Sinon → gisement de fer minable le plus proche. Double mémoïsation ctx :
        # la LISTE des tuiles (1 argwhere/tick partagé) + le NEAREST par bucket 8×8
        # (les mineurs d'un clan sont groupés → un seul dists+argmin par zone/tick,
        # pas un par mineur en transit à chaque tick — gate-review perf).
        if ctx.iron_tiles is None:
            ctx.iron_tiles = np.argwhere(
                world._iron_mask & (world.iron_grid >= IRON_STUMP_THRESHOLD))
            ctx.iron_nearest = {}
        iron_tiles = ctx.iron_tiles
        if len(iron_tiles):
            _bucket = (entity.ix >> 3, entity.iy >> 3)
            best = ctx.iron_nearest.get(_bucket)
            if best is None:
                dists = (iron_tiles[:, 1] - entity.x)**2 + (iron_tiles[:, 0] - entity.y)**2
                best = iron_tiles[int(np.argmin(dists))]
                ctx.iron_nearest[_bucket] = best
            entity.state = State.MINING
            entity.target_x = float(best[1]); entity.target_y = float(best[0])
            _move_toward(entity, entity.target_x, entity.target_y, _eff_speed * 0.8, world)
            return True
    # 4.56 Mine l'OR (bloc C2) : mineur d'un clan à ÉGLISE dont le trésor est passé
    # sous l'hystérésis (GOLD_RESTOCK). L'or reçu en offrande ferme la vanne →
    # substitution source↔circulation, anti-pompe par construction. Le fer (4.55)
    # garde la priorité : l'outil avant le trésor. Pas de bonus pioche (1 < carry 2).
    if (_role_ok(entity.role, "mine") and entity.spec.can_mine and entity.pick is not None
            and entity.gold < MAX_GOLD_CARRY and entity.hunger < 65
            and _cobj_fer is not None and _cobj_fer.age >= GOLD_AGE):
        _chs_gold = _cb.get("church", [])
        # F1 : un seul flux minier, deux destinations. Priorité au trésor d'église (règle actuelle) ;
        # SINON, sous _MONEY_ON, la DEMANDE du marché tire l'or (église requise → le puits de
        # débordement, la dorure, existe). gold_dest figé au départ. MONEY_OFF → seul le restock
        # trésor<4 subsiste → condition et flux RNG IDENTIQUES à l'actuel (imputation exacte).
        _dest = None
        if _chs_gold:
            if _chs_gold[0].gold < GOLD_RESTOCK_THRESHOLD:
                _dest = "church"
            elif _MONEY_ON:
                _mkt_g = _cb.get("market", [])
                if _mkt_g and _mkt_g[0].gold < MONEY_RESTOCK:
                    _dest = "market"
        if _dest is not None:
            entity.gold_dest = _dest
            for adx in range(-1, 2):
                for ady in range(-1, 2):
                    if adx == 0 and ady == 0:
                        continue
                    tx, ty = entity.ix + adx, entity.iy + ady
                    if world.is_gold_mineable(tx, ty):
                        entity.state = State.MINING
                        entity.gold = min(MAX_GOLD_CARRY, entity.gold + world.mine_gold(tx, ty))
                        return True
            if ctx.gold_tiles is None:
                ctx.gold_tiles = np.argwhere(
                    world._gold_mask & (world.gold_grid >= GOLD_STUMP_THRESHOLD))
                ctx.gold_nearest = {}
            gold_tiles = ctx.gold_tiles
            if len(gold_tiles):
                _bucket = (entity.ix >> 3, entity.iy >> 3)
                best = ctx.gold_nearest.get(_bucket)
                if best is None:
                    dists = (gold_tiles[:, 1] - entity.x)**2 + (gold_tiles[:, 0] - entity.y)**2
                    best = gold_tiles[int(np.argmin(dists))]
                    ctx.gold_nearest[_bucket] = best
                entity.state = State.MINING
                entity.target_x = float(best[1]); entity.target_y = float(best[0])
                _move_toward(entity, entity.target_x, entity.target_y, _eff_speed * 0.8, world)
                return True
    # 4.6 Mine la pierre (entités avec can_mine + pioche, si portée non pleine et pas trop affamées)
    # ÉCART spec P1 (assumé) : le minage n'a PAS de cooldown → le bonus « mineur cooldown ×0.85 »
    # de la spec est inapplicable ici ; seul le guard miner s'applique. Bonus mineur autrement en S2c.
    if (_role_ok(entity.role, "mine") and entity.spec.can_mine
            and entity.pick is not None
            and entity.stone < MAX_STONE_CARRY
            and entity.hunger < 65):
        # Roche adjacente → mine directement
        for adx in range(-1, 2):
            for ady in range(-1, 2):
                if adx == 0 and ady == 0:
                    continue
                tx, ty = entity.ix + adx, entity.iy + ady
                if world.is_mineable(tx, ty):
                    entity.state = State.MINING
                    stone = world.mine_stone(tx, ty)
                    if entity.pick == "iron_pick":
                        stone += IRON_PICK_BONUS
                    elif entity.pick == "stone_pick":
                        stone += STONE_PICK_BONUS
                    entity.stone = min(MAX_STONE_CARRY, entity.stone + stone)
                    return True
        # Roche dans le champ de vision → se déplace vers la plus proche (numpy)
        r = int(entity.traits["vision"])
        ex_i, ey_i = entity.ix, entity.iy
        x0 = max(0, ex_i - r); x1 = min(world.width,  ex_i + r + 1)
        y0 = max(0, ey_i - r); y1 = min(world.height, ey_i + r + 1)
        mine_sub = (world._mountain_mask[y0:y1, x0:x1]
                    & (world.stone_grid[y0:y1, x0:x1] >= STONE_STUMP_THRESHOLD))
        _ys, _xs = np.where(mine_sub)
        best_rock = None
        if len(_xs):
            abs_xs = _xs + x0; abs_ys = _ys + y0
            dists_sq = (abs_xs - entity.x)**2 + (abs_ys - entity.y)**2
            idx = int(np.argmin(dists_sq))
            best_rock = (int(abs_xs[idx]), int(abs_ys[idx]))
        if best_rock:
            entity.state = State.MINING
            entity.target_x = float(best_rock[0])
            entity.target_y = float(best_rock[1])
            _move_toward(entity, entity.target_x, entity.target_y,
                         _eff_speed * 0.8, world)
            return True
        # Aucune roche visible → chercher la montagne minable la plus proche sur
        # TOUTE la carte (symétrie exacte avec le scan bois plus haut). Sans ça, un
        # clan éloigné de toute montagne ne mine jamais → chaîne pierre cassée : pas
        # de puit/moulin/niveau 2 (C1). Gardé au stock de pierre du clan pour ne pas
        # envoyer tous les mineurs à l'autre bout de la carte en permanence.
        _clan_stone_stored = sum(b.stone for b in _cb.get("house", []))
        if _clan_stone_stored < CLAN_STONE_BOOTSTRAP:
            rock_tiles = np.argwhere(
                world._mountain_mask & (world.stone_grid >= STONE_STUMP_THRESHOLD)
            )
            if len(rock_tiles):
                dists = (rock_tiles[:, 1] - entity.x)**2 + (rock_tiles[:, 0] - entity.y)**2
                best = rock_tiles[int(np.argmin(dists))]
                entity.state = State.MINING
                entity.target_x = float(best[1])
                entity.target_y = float(best[0])
                _move_toward(entity, entity.target_x, entity.target_y,
                             _eff_speed * 0.8, world)
                return True
    # 4.65 Continuation d'une exploration longue distance déjà engagée
    if (entity.etype == EntityType.HUMAN
            and entity.clan_id is not None
            and clans
            and entity.hunger < 68
            and entity.thirst < 58
            and entity.target_x is not None):
        clan = clans.get(entity.clan_id)
        if clan and _dist(clan.cx, clan.cy, entity.target_x, entity.target_y) > 75:
            # C4 : arrivé à destination → libère la cible et laisse choisir une
            # nouvelle tâche. Sans ça, _move_toward ne bouge plus (déjà sur place)
            # mais le return True garde l'humain figé en EXPLORING des centaines de
            # ticks, jusqu'à ce que faim/soif casse la condition.
            if _dist(entity.x, entity.y, entity.target_x, entity.target_y) < 0.5:
                entity.target_x = None
                entity.target_y = None
            else:
                entity.state = State.EXPLORING
                _move_toward(entity, entity.target_x, entity.target_y, _eff_speed, world)
                return True
    # 4.7 Exploration (humain sans tâche urgente : explore loin du clan) — scout/versatile (P1).
    # _role_ok EN TÊTE (avant random()) : sous JOBS_OFF transparent (random consommé comme avant).
    if (_role_ok(entity.role, "scout")
            and entity.etype == EntityType.HUMAN
            and entity.clan_id is not None
            and entity.hunger < 65
            and entity.thirst < 55
            and clans
            and random.random() < 0.45):
        clan = clans.get(entity.clan_id)
        if clan:
            # Vérifie si la zone locale du clan est vide de ressources
            _cx_i, _cy_i = int(clan.cx), int(clan.cy)
            _r_loc = 30
            _x0l = max(0, _cx_i - _r_loc); _x1l = min(world.width,  _cx_i + _r_loc)
            _y0l = max(0, _cy_i - _r_loc); _y1l = min(world.height, _cy_i + _r_loc)
            _local_food  = float(world.food_grid[_y0l:_y1l, _x0l:_x1l].max())
            _local_trees = bool(world._forest_mask[_y0l:_y1l, _x0l:_x1l].any())
            _zone_vide   = _local_food < 5 and not _local_trees
            # Zone vide → exploration forcée; sinon 8 % de chance
            far_explore = _zone_vide or random.random() < 0.08
            dist_range  = (80, 200) if far_explore else (20, 70)
            for _ in range(12):
                angle = random.uniform(0, 2 * math.pi)
                dist  = random.uniform(*dist_range)
                tx = int(clan.cx + math.cos(angle) * dist)
                ty = int(clan.cy + math.sin(angle) * dist)
                tx = max(0, min(world.width  - 1, tx))
                ty = max(0, min(world.height - 1, ty))
                if world.is_walkable(tx, ty):
                    entity.target_x = float(tx)
                    entity.target_y = float(ty)
                    entity.state = State.EXPLORING if far_explore else State.WANDERING
                    _move_toward(entity, entity.target_x, entity.target_y,
                                 _eff_speed, world)
                    return True
    return False


def _beh_wander(entity, ctx):
    """Errance terminale : troupeau, biais de clan, répulsion, marche aléatoire."""
    world = ctx.world; all_entities = ctx.all_entities
    births = ctx.births; events = ctx.events; tick = ctx.tick
    season = ctx.season; clans = ctx.clans; buildings = ctx.buildings
    temp_c = ctx.temp_c; species_counts = ctx.species_counts
    raining = ctx.raining; heatwave = ctx.heatwave; clan_bldg = ctx.clan_bldg
    predators = ctx.predators; predator_grid = ctx.predator_grid
    entity_grid = ctx.entity_grid
    spec = entity.spec
    # 5. Déambule
    entity.state = State.WANDERING
    # Troupeau : espèces avec can_herd suivent le groupe s'il y en a un proche
    if entity.spec.can_herd:
        if _herd_move(entity, all_entities, world, entity_grid):
            return
    # Humains : biais vers le territoire du clan (rayon proportionnel à la satiété)
    if (entity.etype == EntityType.HUMAN and entity.clan_id is not None
            and clans and random.random() < 0.6
            and (entity.target_x is None
                 or _dist(entity.x, entity.y, entity.target_x, entity.target_y) < 0.5)):
        clan = clans.get(entity.clan_id)
        if clan:
            # Affamé/assoiffé → reste proche; reposé → explore loin
            wander_r = 18 if (entity.hunger > 55 or entity.thirst > 55) else 50
            # S2c — Garde en paix : un warrior de clan EN PAIX reste en garnison près du feu
            # (rayon WARBAND_GUARD_R) au lieu d'errer loin → visible comme garde, groupé et prêt
            # à partir en guerre. En guerre, la marche de _beh_survival a déjà pris le relais (on
            # n'arrive ici que si aucun ennemi n'est engagé, mais on garde tout de même l'ancre
            # de paix distincte). Gated _WARBEH_ON/_JOBS_ON/_SOCIETY_ON → transparent aux switches.
            if (_WARBEH_ON and _JOBS_ON and _SOCIETY_ON
                    and entity.role == "warrior" and clan.mode == "peace"):
                wander_r = min(wander_r, WARBAND_GUARD_R)
            # P5 E2 — Fête : pendant la fête, TOUS les membres du clan convergent vers le feu
            # (rayon FEAST_RADIUS) pour danser autour. Réduit juste le rayon d'errance (zéro
            # tirage neuf, la boucle random.uniform ci-dessous est inchangée). Gated _FEAST_ON.
            if _FEAST_ON and clan.feast_ticks > 0:
                wander_r = min(wander_r, FEAST_RADIUS)
            for _ in range(8):
                tx = clan.cx + random.uniform(-wander_r, wander_r)
                ty = clan.cy + random.uniform(-wander_r, wander_r)
                tx = max(0, min(world.width  - 1, tx))
                ty = max(0, min(world.height - 1, ty))
                if world.is_walkable(int(tx), int(ty)):
                    entity.target_x = tx
                    entity.target_y = ty
                    break
    # Répulsion légère entre humains trop proches (évite l'entassement)
    if entity.etype == EntityType.HUMAN:
        _rep_scan = (_grid_neighbors(entity_grid, entity.ix, entity.iy, reach=1)
                     if entity_grid is not None else all_entities)
        for other in _rep_scan:
            if other is entity or not other.alive or other.etype != EntityType.HUMAN:
                continue
            d = _dist(entity.x, entity.y, other.x, other.y)
            if 0.01 < d < 0.8:
                repulse_x = entity.x - other.x
                repulse_y = entity.y - other.y
                norm = math.sqrt(repulse_x**2 + repulse_y**2) or 1
                tx = max(0.0, min(world.width  - 1.0, entity.x + (repulse_x / norm) * 1.5))
                ty = max(0.0, min(world.height - 1.0, entity.y + (repulse_y / norm) * 1.5))
                if world.is_walkable(int(tx), int(ty)):
                    entity.target_x = tx
                    entity.target_y = ty
                break
    _random_walk(entity, world)


class Simulation:
    def __init__(self, world: "World"):
        from .world import World
        self.world    = world
        self.entities: list[Entity] = []
        self.tick_count = 0
        self.events_log: list[dict] = []
        self.stats_history: list[dict] = []
        # Owner-grid du territoire (T1) : int8 H×W, clan_id possédant chaque tuile ou -1
        # (sauvage/eau). Dérivé des bâtiments (pas sérialisé) ; recalculé périodiquement
        # dans step() et servi au front via /api/territory. None tant que non calculé.
        self.territory_grid = None
        # ── Chroniques du monde (bloc K) : annales persistantes des JALONS ──────
        # Dérivées des tick_events déjà émis (aucune clé ajoutée à la sortie de
        # step() → le hash du guard ne bouge pas). Servies par /api/chronicle.
        self.chronicle: list[dict] = []
        self._chronicle_seen: set = set()    # jalons "première fois" déjà actés
        self._prev_species: set = set()      # espèces vivantes au tick précédent
        self.clans: list[Clan] = []
        # Relations inter-clans (P2b) : clé = paire d'ids TRIÉE (a<b), valeur ∈ [-100,+100],
        # symétrique ; absence = neutre. _ally_state/_rival_state = états booléens dérivés
        # (hystérésis), recalculés au load — pas sérialisés (dérivés de la valeur).
        self.relations: dict = {}
        self._ally_state: set = set()
        self._rival_state: set = set()
        self._next_clan_id: int = N_CLANS   # P4 : id monotone des clans nés (scission/essaimage)
        self.cults: dict = {}               # P5 : registre {cult_id: Cult} (jamais purgé — l'histoire)
        self._next_cult_id: int = 0         # P5 : id de culte monotone
        self.money_dawn: bool = False       # P6 F1 : 1er paiement en or déjà eu lieu ? (annale unique)
        self.buildings: list[Building] = []
        self._next_building_id = 0
        self.raining: bool = False
        self.storming: bool = False   # orage (sous-type de pluie)
        self.rain_ticks_left: int = 0
        self.heatwave: bool = False
        self.heatwave_ticks_left: int = 0
        # Observabilité (I0) : durées des derniers ticks (ms), pour /api/metrics.
        # Hors sortie de step() → neutre pour le hash déterministe.
        self._step_ms: deque = deque(maxlen=512)

    def populate(self, counts: dict[EntityType, int] = None):
        """Spawn initial des entités sur des cases marchables."""
        # Animaux : placement aléatoire (les humains sont gérés séparément par clan)
        defaults = {
            EntityType.BOAR:         10,
            EntityType.CHICKEN:      30,
            EntityType.HORNED_SHEEP: 20,
            EntityType.HORSE:        16,
            EntityType.PIG:          30,
            EntityType.SHEEP:        30,
            EntityType.FISH:         55,
            EntityType.SHARK:        3,
        }
        counts = counts or defaults
        for etype, n in counts.items():
            aquatic = SPECS[etype].aquatic
            for _ in range(n):
                for attempt in range(100):
                    x = random.randint(1, self.world.width  - 2)
                    y = random.randint(1, self.world.height - 2)
                    if self.world.is_walkable(x, y, aquatic):
                        self.entities.append(spawn(etype, x + random.random(), y + random.random()))
                        break

        # ── Placement des clans : 4 emplacements bien espacés sur terrain marchable ──
        MIN_CAMPFIRE_DIST = min(self.world.width, self.world.height) // (N_CLANS + 1)
        FOREST_SEARCH_R   = 25   # rayon max pour trouver une forêt proche du campfire

        def _has_forest_nearby(x, y, r=FOREST_SEARCH_R):
            x0 = max(0, x - r); x1 = min(self.world.width,  x + r + 1)
            y0 = max(0, y - r); y1 = min(self.world.height, y + r + 1)
            return bool((self.world.biome_grid[y0:y1, x0:x1] == int(Biome.FOREST)).any())

        clan_positions: list[tuple[int, int]] = []
        for _ in range(12000):
            x = random.randint(10, self.world.width  - 10)
            y = random.randint(10, self.world.height - 10)
            if not self.world.is_walkable(x, y):
                continue
            if not _has_forest_nearby(x, y):
                continue
            if all(_dist(x, y, cx, cy) >= MIN_CAMPFIRE_DIST for cx, cy in clan_positions):
                clan_positions.append((x, y))
            if len(clan_positions) == N_CLANS:
                break
        # Fallback : compléter sans contrainte de distance mais avec forêt
        if len(clan_positions) < N_CLANS:
            for _ in range(8000):
                x = random.randint(10, self.world.width  - 10)
                y = random.randint(10, self.world.height - 10)
                if (self.world.is_walkable(x, y)
                        and _has_forest_nearby(x, y)
                        and (x, y) not in clan_positions):
                    clan_positions.append((x, y))
                if len(clan_positions) == N_CLANS:
                    break
        # Dernier recours : sans contrainte de forêt (évite le crash)
        if len(clan_positions) < N_CLANS:
            for _ in range(5000):
                x = random.randint(10, self.world.width  - 10)
                y = random.randint(10, self.world.height - 10)
                if self.world.is_walkable(x, y) and (x, y) not in clan_positions:
                    clan_positions.append((x, y))
                if len(clan_positions) == N_CLANS:
                    break

        self.clans = []
        self.relations = {}; self._ally_state = set(); self._rival_state = set()  # P2 : monde neuf = sans histoire
        self._next_clan_id = N_CLANS   # P4 : les 4 clans initiaux occupent 0..N_CLANS-1
        self.cults = {}; self._next_cult_id = 0   # P5 : monde neuf = cultes refondés ci-dessous
        for cid, (cx, cy) in enumerate(clan_positions):
            # ── Chef : mâle, placé sur le campfire ──────────────────────────
            chief = spawn(EntityType.HUMAN,
                          cx + random.uniform(-0.3, 0.3),
                          cy + random.uniform(-0.3, 0.3),
                          sex=Sex.MALE)
            chief.clan_id = cid
            chief.hunger  = 10.0
            chief.age     = SPECS[EntityType.HUMAN].max_age * 0.2   # adulte
            self.entities.append(chief)

            # ── 4 membres : 2♂ + 2♀, dans un rayon de 4 tuiles du campfire ─
            for sex in (Sex.MALE, Sex.FEMALE, Sex.MALE, Sex.FEMALE):
                for _ in range(80):
                    mx = cx + random.uniform(-4, 4)
                    my = cy + random.uniform(-4, 4)
                    mx = max(1.0, min(self.world.width  - 2.0, mx))
                    my = max(1.0, min(self.world.height - 2.0, my))
                    if self.world.is_walkable(int(mx), int(my)):
                        member = spawn(EntityType.HUMAN, mx, my, sex=sex)
                        member.clan_id = cid
                        member.hunger  = random.uniform(5, 75)
                        member.age     = SPECS[EntityType.HUMAN].max_age * random.uniform(0.1, 0.35)
                        self.entities.append(member)
                        break

            self.clans.append(Clan(id=cid, cx=float(cx), cy=float(cy),
                                   color=CLAN_COLORS[cid], chief_id=chief.id))
            if _CULTS_ON:   # P5 : chaque clan de départ fonde SON culte
                self.clans[-1].cult_id = self._found_cult(cid, 0)
            fire = Building(id=self._next_building_id,
                            clan_id=cid,
                            x=cx, y=cy,
                            btype="campfire")
            self._next_building_id += 1
            self.buildings.append(fire)

        # Territoire (T1) disponible dès le boot, avant le 1er step (pour /api/territory).
        self.territory_grid = self._compute_territory()

    def _compute_territory(self):
        """Owner-grid du territoire (T1) : chaque tuile terrestre au clan dont un bâtiment
        est le plus proche, sous TERRITORY_MAX_DIST ; -1 = sauvage/eau. PUR (aucun RNG →
        n'affecte pas le hash). Déterministe : ancres triées (clan_id, y, x), égalités
        tranchées par l'ordre de traitement (< strict), donc indépendant du shuffle
        d'entités. Vectorisé NumPy (~8 ms sur 220×160)."""
        w = self.world
        H, W = w.height, w.width
        anchors = sorted(
            ((b.clan_id, b.y, b.x) for b in self.buildings
             if b.clan_id is not None and b.clan_id >= 0),
            key=lambda t: (t[0], t[1], t[2]))
        # int16 (audit #2) : les clan_id sont un compteur MONOTONE non borné (chaque scission/
        # essaimage l'incrémente) → en int8, np.int8(cid) déborde dès le clan id 128 (OverflowError
        # NumPy≥2 → crash du step). Certain sur run long. int16 borne à 32767, largement au-delà.
        owner = np.full((H, W), -1, dtype=np.int16)
        if not anchors:
            return owner
        best = np.full((H, W), np.inf, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)[:, None]
        xs = np.arange(W, dtype=np.float32)[None, :]
        for cid, ay, ax in anchors:
            d = (xs - ax) ** 2 + (ys - ay) ** 2
            closer = d < best
            best = np.where(closer, d, best)
            owner = np.where(closer, np.int16(cid), owner)
        owner[best > TERRITORY_MAX_DIST ** 2] = -1   # au-delà du rayon d'influence
        owner[~w._walkable] = -1                       # l'eau n'est possédée par personne
        return owner

    def step(self) -> dict:
        """Avance d'un tick. Retourne les données à broadcaster."""
        _t0 = time.perf_counter()
        self.tick_count += 1
        season = get_season(self.tick_count)
        temp_c = get_temperature(self.tick_count)

        # Territoire (T1) : recalcul périodique de l'owner-grid (pur, hors hash ; le front
        # le lit via /api/territory). Le territoire évolue lentement → inutile chaque tick.
        if self.territory_grid is None or self.tick_count % TERRITORY_RECOMPUTE_PERIOD == 0:
            self.territory_grid = self._compute_territory()

        # Sentiers (F3) : la nature reprend lentement ses droits — décroissance VECTORISÉE
        # périodique (−1, plancher 0). Un chemin entretenu reste marqué, un chemin abandonné
        # s'efface en quelques saisons. Hors hash (grille non émise dans step()).
        if _TRAILS_ON and self.tick_count % TRAIL_DECAY_PERIOD == 0:
            _tg = self.world.trail_grid
            _tg -= (_tg >= 1)

        # Carte explorée (G1) : marquage périodique VECTORISÉ autour des humains (r=2), plus
        # large autour d'un éclaireur en mission (r=4 : il est là pour regarder). Cosmétique
        # pur, hors payload step(), SANS décroissance — la connaissance ne s'oublie pas.
        if _CARTO_ON and self.tick_count % EXPLORE_MARK_PERIOD == 0:
            _eg = self.world.explored_grid
            _hh, _ww = _eg.shape
            _walk: list = []; _miss: list = []
            for _e in self.entities:
                if _e.alive and _e.etype == EntityType.HUMAN:
                    (_miss if _e.expedition_phase is not None else _walk).append((_e.ix, _e.iy))
            for _pts, _r in ((_walk, EXPLORE_MARK_R), (_miss, EXPLORE_MARK_R_SCOUT)):
                if not _pts:
                    continue
                _ax = np.array([p[0] for p in _pts], dtype=np.int32)
                _ay = np.array([p[1] for p in _pts], dtype=np.int32)
                for _dy in range(-_r, _r + 1):
                    _yy = np.clip(_ay + _dy, 0, _hh - 1)
                    for _dx in range(-_r, _r + 1):
                        _eg[_yy, np.clip(_ax + _dx, 0, _ww - 1)] = 1

        births: list[Entity] = []
        tick_events: list[dict] = []

        # ── Météo : déclenchement / arrêt pluie / orage / canicule ──────────
        if self.raining:
            self.rain_ticks_left -= 1
            if self.rain_ticks_left <= 0:
                self.raining  = False
                self.storming = False
            # La pluie coupe la canicule
            if self.heatwave:
                self.heatwave = False
                self.heatwave_ticks_left = 0
                tick_events.append({"type": "heatwave_end"})
        elif self.heatwave:
            self.heatwave_ticks_left -= 1
            if self.heatwave_ticks_left <= 0:
                self.heatwave = False
                tick_events.append({"type": "heatwave_end"})
        elif season == "summer" and random.random() < HEATWAVE_PROB:
            self.heatwave = True
            self.heatwave_ticks_left = random.randint(HEATWAVE_DURATION_MIN, HEATWAVE_DURATION_MAX)
            tick_events.append({"type": "heatwave_start"})
        elif random.random() < RAIN_PROB[season]:
            self.raining  = True
            self.storming = random.random() < STORM_PROB[season]
            self.rain_ticks_left = random.randint(RAIN_DURATION_MIN, RAIN_DURATION_MAX)

        self.world.regen_food(SEASON_REGEN_MULT[season])
        # Arbres et roches qui franchissent le seuil de repousse ce tick
        tree_changes = [{"x": x, "y": y, "stump": False}
                        for x, y in self.world.regen_trees()]
        rock_changes = [{"x": x, "y": y, "depleted": False}
                        for x, y in self.world.regen_stones()]
        self.world.regen_iron()   # bloc B : régen lente des gisements de fer
        self.world.regen_gold()   # bloc C2 : régen très lente des filons d'or
        # Fertilité : DIRT → GRASS (regen lente), GRASS → DIRT collecté depuis consume_fertility
        biome_changes = self.world.regen_fertility(self.raining, season)
        # NB : les bascules GRASS→DIRT du PIÉTINEMENT (buffer _biome_changes) sont
        # drainées en FIN de tick (après la phase entités), pas ici — sinon le
        # buffer porte un état inter-ticks invisible de la sauvegarde : un événement
        # en attente au moment du save était PERDU au rechargement → le replay
        # byte-à-byte divergeait (bug trouvé par bisection à t=5081, endurance 20k).

        # Mélange pour éviter les biais d'ordre
        random.shuffle(self.entities)

        clans_dict = {c.id: c for c in self.clans}

        # Recruter les humains libres proches d'un membre de clan
        for e in self.entities:
            if e.alive and e.etype == EntityType.HUMAN and e.clan_id is None:
                for other in self.entities:
                    if (other is not e and other.alive
                            and other.etype == EntityType.HUMAN
                            and other.clan_id is not None
                            and _dist(e.x, e.y, other.x, other.y) < 4):
                        e.clan_id = other.clan_id
                        break

        # Comptage par espèce (pour cap par-espèce spec.max_pop, cf. repro)
        species_counts: dict[str, int] = {}
        for e in self.entities:
            if e.alive:
                species_counts[e.etype.value] = species_counts.get(e.etype.value, 0) + 1

        # Index bâtiments par clan pour éviter O(n_humains × n_bâtiments) dans tick_entity
        clan_bldg: dict = {}
        for _b in self.buildings:
            _cid = _b.clan_id
            if _cid not in clan_bldg:
                clan_bldg[_cid] = {}
            if _b.btype not in clan_bldg[_cid]:
                clan_bldg[_cid][_b.btype] = []
            clan_bldg[_cid][_b.btype].append(_b)

        # Dispatch des caravanes (D1) : évaluation périodique des routes de troc.
        # 100 % sans RNG (paires triées, nearest + tie-break id) → hash-neutre
        # tant qu'aucun marché n'existe.
        if self.tick_count % TRADE_CHECK_PERIOD == 0 and self.clans:
            self._dispatch_caravans(clan_bldg, tick_events)
        # Pèlerinages (C1) : APRÈS les caravanes (gardes croisées → pas de double-booking)
        if self.tick_count % PILGRIM_CHECK_PERIOD == 0 and len(self.clans) >= 2:
            self._dispatch_pilgrims(clan_bldg, tick_events)
        # Expéditions d'éclaireurs (P7 G1) : APRÈS les deux autres dispatches (mêmes gardes
        # croisées). Sort tout de suite si aucun clan n'est à l'échéance de sa période.
        if _CARTO_ON and self.clans:
            self._dispatch_expeditions(clan_bldg, tick_events)
        # Société : le chef réévalue le mode de gouvernement du clan (déterministe, déphasé).
        self._update_society(clan_bldg, tick_events)
        # Métiers (P1) : assignation des rôles selon la pop (déterministe, déphasé *41).
        self._update_jobs(clan_bldg, tick_events)
        # Fête des moissons (P5 E2) : déclenchement au 1er tick d'automne + décompte (déterministe).
        self._update_feasts(clan_bldg, tick_events)

        # Liste des prédateurs actifs (pour _find_predator_nearby, évite O(n²))
        active_predators = [e for e in self.entities if e.alive and e.spec.is_predator]

        # Grille spatiale des prédateurs (cellule 8×8 tuiles) pour accélérer la détection
        predator_grid: dict = {}
        for _p in active_predators:
            _key = (_p.ix >> 3, _p.iy >> 3)
            if _key not in predator_grid:
                predator_grid[_key] = []
            predator_grid[_key].append(_p)

        # Grille spatiale générique de TOUTES les entités vivantes (troupeau, répulsion)
        # → évite les scans O(n) par entité (poste CPU n°1 : _herd_move).
        entity_grid: dict = {}
        for _e in self.entities:
            if not _e.alive:
                continue
            _ek = (_e.ix >> 3, _e.iy >> 3)
            if _ek not in entity_grid:
                entity_grid[_ek] = []
            entity_grid[_ek].append(_e)

        ctx = _TickCtx(self.world, self.entities, births, tick_events,
                       self.tick_count, season, clans_dict, self.buildings,
                       temp_c, species_counts, self.raining, self.heatwave,
                       clan_bldg, active_predators, predator_grid, entity_grid)
        for e in self.entities:
            _tick_entity(e, ctx)

        if _MONEY_ON and not self.money_dawn:   # F1 : 1er paiement en or de la partie → annale (1×)
            for _ev in tick_events:
                if _ev.get("type") == "trade_exchange" and _ev.get("pay_good") == "gold":
                    self.money_dawn = True
                    tick_events.append({"type": "money_dawn"})
                    break

        # Traiter les constructions (dédoublonnage intra-tick inclus)
        new_this_tick: list[Building] = []
        # NB : la CRÉATION des vrais bâtiments passe uniquement par la promotion
        # des chantiers `site_*` (plus bas). Les anciens handlers `build_*` ici
        # étaient du code mort (les events `build_*` ne sont émis qu'À la promotion,
        # après cette boucle) — retirés le 2026-07-09.
        for ev in tick_events:
            if ev["type"] == "start_site":
                btype = ev["btype"]
                # Ne créer le site que s'il n'en existe pas déjà un du même type pour ce clan
                already = any(
                    b.btype == f"site_{btype}" and b.clan_id == ev["clan_id"]
                    for b in self.buildings + new_this_tick
                )
                if not already:
                    site = Building(
                        id=self._next_building_id,
                        clan_id=ev["clan_id"],
                        x=ev["x"], y=ev["y"],
                        btype=f"site_{btype}",
                        work_needed=ev["work_needed"],
                        dedication=ev.get("dedication", ""),   # P5 E3 : porté au chantier monument
                    )
                    self._next_building_id += 1
                    new_this_tick.append(site)
        self.buildings.extend(new_this_tick)

        # P7 G3 : achèvement/abandon des migrations AVANT la promotion générique — un
        # `site_campfire` promu en `campfire` ordinaire donnerait DEUX feux au clan.
        self._update_migrations(tick_events)

        # Promouvoir les chantiers terminés en bâtiments réels
        for b in self.buildings:
            if b.btype.startswith("site_") and b.work_done >= b.work_needed > 0:
                real_btype = b.btype[5:]  # "site_house" → "house"
                b.btype = real_btype
                b.work_needed = 0
                tick_events.append({"type": f"build_{real_btype}",
                                    "clan_id": b.clan_id,
                                    "x": b.x, "y": b.y})
                if _MONUMENT_ON and real_btype == "monument":   # P5 E3 : la gloire APAISE le clan (one-shot)
                    _cm = clans_dict.get(b.clan_id)
                    if _cm is not None:
                        _cm.tension = max(0, _cm.tension - MONUMENT_TENSION)
                    tick_events.append({"type": "monument_built", "clan_id": b.clan_id,
                                        "dedication": b.dedication})

        # Drain des étals de marché (D1) : les imports réintègrent les maisons
        if self.tick_count % MARKET_DRAIN_PERIOD == 0:
            self._drain_markets(clan_bldg)

        # Églises (C1) : cloche des offices + combustion des offrandes
        self._church_upkeep(clan_bldg, tick_events)

        # Production de pain dans les moulins
        for b in self.buildings:
            if b.btype != "mill":
                continue
            _cap_bread = (MILL_MAX_BREAD * MILL_L2_BREAD_MULT
                          if (_GRANARY_ON and b.level >= 2) else MILL_MAX_BREAD)
            if b.bread >= _cap_bread or b.wheat <= 0:
                b.mill_ticks = 0
                continue
            # Vérifie qu'une tuile d'eau est accessible dans un rayon de 6 tuiles
            if not _tile_near_water(self.world, b.x, b.y):
                continue
            b.mill_ticks += 1
            if b.mill_ticks >= MILL_BREAD_TICKS:
                b.wheat      -= MILL_BREAD_COST_WHEAT
                b.bread      += 1
                b.mill_ticks  = 0

        # Croissance des champs de blé (×2 si arrosé, +1 bonus si pluie)
        for b in self.buildings:
            if b.btype == "wheatfield" and b.stage < 4:
                if b.watered_ticks > 0:
                    b.grow_ticks   += 2
                    b.watered_ticks -= 1
                else:
                    b.grow_ticks += 1
                if self.raining:
                    b.grow_ticks += RAIN_WHEAT_BONUS
                if b.grow_ticks >= WHEAT_TICKS_PER_STAGE:
                    b.stage      += 1
                    b.grow_ticks  = 0

        # ── Incendie spontané (canicule) ─────────────────────────────────────
        if self.heatwave and random.random() < HEATWAVE_FIRE_PROB:
            ys, xs = np.where(
                (self.world.tree_grid >= TREE_STUMP_THRESHOLD) & self.world._forest_mask
            )
            if len(xs):
                idx = random.randrange(len(xs))
                lx, ly = int(xs[idx]), int(ys[idx])
                if self.world.ignite(lx, ly):
                    tick_events.append({"type": "heatwave_fire", "x": lx, "y": ly})

        # ── Foudre (orage) : frappe un arbre aléatoire debout ─────────────────
        if self.storming and random.random() < LIGHTNING_PROB:
            # Tire une tuile forêt avec arbre debout au hasard
            ys, xs = np.where(
                (self.world.tree_grid >= TREE_STUMP_THRESHOLD) & self.world._forest_mask
            )
            if len(xs):
                idx = random.randrange(len(xs))
                lx, ly = int(xs[idx]), int(ys[idx])
                if self.world.ignite(lx, ly):
                    tick_events.append({"type": "lightning",
                                        "x": lx, "y": ly})

        # ── Propagation et extinction des feux de forêt ───────────────────────
        _was_standing = (self.world.tree_grid >= TREE_STUMP_THRESHOLD) & self.world._forest_mask
        fire_changes = self.world.step_fire(season, self.raining)
        _now_stump = _was_standing & (self.world.tree_grid < TREE_STUMP_THRESHOLD)
        if _now_stump.any():
            _ys, _xs = np.where(_now_stump)
            for _x, _y in zip(_xs.tolist(), _ys.tolist()):
                tree_changes.append({"x": int(_x), "y": int(_y), "stump": True})

        # Auto-upgrade L1 → L2 si le clan a assez de ressources en stock
        for b in self.buildings:
            bspec = BUILDING_SPECS.get(b.btype)
            if not bspec or bspec.max_level <= 1 or b.level >= bspec.max_level:
                continue
            if b.btype == "mill" and not _GRANARY_ON:
                continue        # F4 gated : GRANARY_OFF → le moulin ne devient jamais grenier
            clan_houses = [x for x in self.buildings
                           if x.clan_id == b.clan_id and x.btype == "house"]
            # Pour upgrader une maison L2, le feu de camp doit être L2 d'abord
            if b.btype == "house":
                clan_fire = next((x for x in self.buildings
                                  if x.clan_id == b.clan_id and x.btype == "campfire"), None)
                if clan_fire is None or clan_fire.level < 2:
                    continue
            total_wood  = sum(x.wood  for x in clan_houses)
            total_stone = sum(x.stone for x in clan_houses)
            if total_wood < bspec.upgrade_wood or total_stone < bspec.upgrade_stone:
                continue
            # Prélève les ressources (d'abord dans les maisons les plus pleines)
            rem_w = bspec.upgrade_wood
            for h in sorted(clan_houses, key=lambda x: -x.wood):
                take = min(int(h.wood), rem_w); h.wood -= take; rem_w -= take
                if rem_w == 0: break
            rem_s = bspec.upgrade_stone
            for h in sorted(clan_houses, key=lambda x: -x.stone):
                take = min(h.stone, rem_s); h.stone -= take; rem_s -= take
                if rem_s == 0: break
            b.level = 2
            tick_events.append({"type": "upgrade_building", "btype": b.btype,
                                 "clan_id": b.clan_id, "x": b.x, "y": b.y})

        # Ajoute les naissances
        self.entities.extend(births)

        # Purge les morts
        if _HEROES_ON:   # P5 E4 : un héros qui meurt entre aux ANNALES (une fois, juste avant le retrait)
            for _e in self.entities:
                if not _e.alive and _e.hero_name is not None:
                    tick_events.append({"type": "hero_fallen", "entity_id": _e.id,
                                        "clan_id": _e.clan_id, "name": _e.hero_name,
                                        "age_seasons": int(_e.age // TICKS_PER_SEASON)})
        self.entities = [e for e in self.entities if e.alive]

        # Clans entièrement éteints : leurs structures durables (maison/moulin/puit)
        # deviennent des RUINES (E8) au lieu de disparaître en silence — récit
        # émergent + info visible. Le reste (feu de camp, chantiers) s'efface.
        alive_clan_ids = {e.clan_id for e in self.entities
                          if e.alive and e.etype == EntityType.HUMAN
                          and e.clan_id is not None}
        dead_clan_ids = {c.id for c in self.clans} - alive_clan_ids
        if dead_clan_ids:
            clans_by_id = {c.id: c for c in self.clans}
            ruins_per_clan = {}
            kept = []
            for b in self.buildings:
                if b.clan_id not in dead_clan_ids:
                    kept.append(b)
                elif b.btype in ("house", "mill", "well", "forge", "market", "church", "monument"):
                    ruins_per_clan[b.clan_id] = ruins_per_clan.get(b.clan_id, 0) + 1
                    _was_monument = b.btype == "monument"   # P5 E3 : un monument hante le paysage ×MULT
                    b.btype = "ruin"
                    b.clan_id = -1            # orphelin : ne matche plus aucun clan vivant
                    b.ruin_ticks = RUIN_LIFETIME * (MONUMENT_RUIN_MULT if _was_monument else 1)
                    kept.append(b)
                # feu de camp + site_* d'un clan mort → pas de ruine (effacés)
            self.buildings = kept
            for cid in dead_clan_ids:
                c = clans_by_id.get(cid)
                tick_events.append({"type": "clan_extinct", "clan_id": cid,
                                    "x": int(c.cx) if c else 0,
                                    "y": int(c.cy) if c else 0,
                                    "ruins": ruins_per_clan.get(cid, 0)})
            self.clans = [c for c in self.clans if c.id not in dead_clan_ids]
            if _WAR2_ON:   # P3 fix : purger les clés relations d'un clan éteint (bug depuis P2 :
                for cid in dead_clan_ids:   # elles restaient gelées à vie dans le save)
                    self._purge_clan_relations(cid)
            if _SOCIETY_ON:   # société (D1) : la guerre s'arrête quand la cible s'éteint
                for c in self.clans:
                    if c.war_target in dead_clan_ids:
                        c.mode = "peace"; c.war_target = -1; c.mode_ticks = 0
                        tick_events.append({"type": "clan_mode", "clan_id": c.id, "mode": "peace"})

        # Décroissance des ruines : la nature les reprend (borne la mémoire → le
        # jeu tourne à l'infini sans accumulation de bâtiments morts).
        if any(b.btype == "ruin" for b in self.buildings):
            for b in self.buildings:
                if b.btype == "ruin":
                    b.ruin_ticks -= 1
            self.buildings = [b for b in self.buildings
                              if b.btype != "ruin" or b.ruin_ticks > 0]

        # ── Science & âges technologiques (bloc A1) ──────────────────────────
        # Chaque clan vivant accumule de la science (bâtiments durables + pop) et
        # franchit ses âges. Événement `clan_age_up` à chaque passage → visible.
        if self.clans:
            _DURABLE = ("house", "mill", "well", "wheatfield", "forge", "market", "church")
            _bld_per_clan: dict = {}
            for b in self.buildings:
                if b.btype in _DURABLE:
                    _bld_per_clan[b.clan_id] = _bld_per_clan.get(b.clan_id, 0) + 1
            _pop_per_clan: dict = {}
            for e in self.entities:
                if e.alive and e.etype == EntityType.HUMAN and e.clan_id is not None:
                    _pop_per_clan[e.clan_id] = _pop_per_clan.get(e.clan_id, 0) + 1
            for c in self.clans:
                c.science += (_bld_per_clan.get(c.id, 0) * SCIENCE_PER_BUILDING
                              + _pop_per_clan.get(c.id, 0) * SCIENCE_PER_POP)
                while (c.age + 1 < len(AGE_NAMES)
                       and c.science >= AGE_SCIENCE_THRESHOLDS[c.age + 1]):
                    c.age += 1
                    tick_events.append({"type": "clan_age_up", "clan_id": c.id,
                                        "age": c.age, "age_name": AGE_NAMES[c.age]})

        # Log événements
        # Chroniques (bloc K) : distille les jalons du tick dans les annales
        self._update_relations(tick_events)   # P2b : deltas de relation issus des events du tick
        self._locate_ally_breaks(tick_events)  # x/y des ruptures — APRÈS le dernier _rel_apply du
                                               # tick (society PUIS relations), sinon on manquerait
                                               # les ruptures nées de la seconde passe
        self._update_chronicle(tick_events)

        self.events_log.extend(tick_events)
        if len(self.events_log) > 200:
            self.events_log = self.events_log[-200:]

        # Stats
        stats = self._compute_stats()
        self.stats_history.append({"tick": self.tick_count, **stats})
        if len(self.stats_history) > 300:
            self.stats_history = self.stats_history[-300:]

        # Arbres abattus et roches minées ce tick (collectés depuis world)
        # Drain du piétinement DU TICK MÊME (cf. note près de regen_fertility) :
        # aucun buffer d'événements ne survit à la frontière du tick → un save en
        # fin de tick capture TOUT l'état observable, le replay reste byte-à-byte.
        biome_changes += self.world.drain_biome_changes()
        for _cx, _cy in self.world._chop_changes:
            tree_changes.append({"x": _cx, "y": _cy, "stump": True})
        for _cx, _cy in self.world._mine_changes:
            rock_changes.append({"x": _cx, "y": _cy, "depleted": True})
        self.world._chop_changes.clear()
        self.world._mine_changes.clear()

        # Métrique de durée (hors dict retourné → hash neutre)
        self._step_ms.append((time.perf_counter() - _t0) * 1000.0)

        return {
            "tick":         self.tick_count,
            "season":       season,
            "temp_c":       temp_c,
            "raining":      self.raining,
            "storming":     self.storming,
            "heatwave":     self.heatwave,
            "fire_changes": fire_changes,
            "entities":     [e.to_dict() for e in self.entities],
            "events":       tick_events,
            "stats":        stats,
            "clans":        self._clans_wire(),
            "tree_changes":  tree_changes,
            "rock_changes":  rock_changes,
            "biome_changes": biome_changes,
            "buildings":     [b.to_dict() for b in self.buildings],
        }

    def _compute_stats(self) -> dict:
        counts = {t.value: 0 for t in EntityType}
        for e in self.entities:
            counts[e.etype.value] += 1
        return {"populations": counts, "total": len(self.entities)}

    # ── Économie / caravanes (blocs D1+D2) ───────────────────────────────────
    def _update_jobs(self, clan_bldg: dict, tick_events: list):
        """Métiers (P1) : assigne le rôle de chaque humain selon la pop du clan, périodiquement
        (déphasé par clan), 100 % déterministe (tri id + seuils, zéro RNG). Churn minimal : les
        membres gardant un quota pour leur rôle actuel le conservent (passe 1), les slots restants
        sont recrutés parmi les non-conservés dans l'ordre des id (passe 2), le reste → versatile.
        Le chef reste versatile. Event `clan_jobs` au changement. JOBS_OFF → no-op."""
        if not _JOBS_ON or not self.clans:
            return
        for c in self.clans:
            if (self.tick_count + c.id * 41) % JOB_PERIOD != 0:
                continue
            cb = clan_bldg.get(c.id, {})
            has_site = any(bt.startswith("site_") for bt in cb)
            q = _job_quotas(len([e for e in self.entities
                                 if e.alive and e.etype is EntityType.HUMAN
                                 and e.clan_id == c.id and e.id != c.chief_id]),
                            c, c.mode == "war", bool(cb.get("market")),
                            bool(cb.get("church")), has_site)
            members = sorted((e for e in self.entities
                              if e.alive and e.etype is EntityType.HUMAN
                              and e.clan_id == c.id and e.id != c.chief_id),
                             key=lambda e: e.id)
            avail = dict(q)
            assign = {}
            for e in members:                       # passe 1 : conservation
                if avail.get(e.role, 0) > 0:
                    avail[e.role] -= 1; assign[e.id] = e.role
            for e in members:                       # passe 2 : recrutement (ordre id)
                if e.id in assign:
                    continue
                r = "versatile"
                for cand in _JOB_ROLES:
                    if avail.get(cand, 0) > 0:
                        avail[cand] -= 1; r = cand; break
                assign[e.id] = r
            changed = False
            counts = {}
            for e in members:
                nr = assign.get(e.id, "versatile")
                if e.role != nr:
                    e.role = nr; changed = True
                if nr != "versatile":
                    counts[nr] = counts.get(nr, 0) + 1
            chief = next((e for e in self.entities if e.id == c.chief_id), None)
            if chief is not None and chief.role != "versatile":
                chief.role = "versatile"; changed = True
            if changed:
                tick_events.append({"type": "clan_jobs", "clan": c.id, "jobs": counts})

    def _update_society(self, clan_bldg: dict, tick_events: list):
        """Société : le chef choisit PAIX/GUERRE/FAMINE selon l'état du clan. Réévalué
        périodiquement, DÉPHASÉ par clan (un scan des entités seulement sur un tick de décision),
        100 % déterministe (thresholds, zéro RNG). mode_ticks avance chaque tick (hystérésis).

        Priorité : FAMINE (survie) > GUERRE (rival en contact + assez nombreux, hystérésis) > PAIX.
        Émet `clan_mode` au changement (consommé par les chroniques I3). SOCIETY_OFF → no-op."""
        if not _SOCIETY_ON or not self.clans:
            return
        for c in self.clans:          # hystérésis : le temps passé dans le mode courant
            c.mode_ticks += 1
        n = len(self.clans)
        phase = max(1, MODE_PERIOD // n)
        due = [c for c in self.clans if (self.tick_count + c.id * phase) % MODE_PERIOD == 0]
        if not due:
            return
        # Métriques par clan : population humaine, faim moyenne, stock de pain (un seul scan).
        # P2 succession : dans le MÊME scan, on repère (a) les clans dont le chief_id pointe
        # encore un membre vivant du clan (`chief_ok`) et (b) le candidat successeur = membre le
        # plus ÂGÉ, tie-break id min (`cand`). Zéro coût de scan supplémentaire.
        pop: dict = {}
        hsum: dict = {}
        chief_of = {c.id: c.chief_id for c in self.clans}
        chief_ok: set = set()
        cand: dict = {}                          # clan_id -> (age, id) du successeur
        for e in self.entities:
            if e.alive and e.etype == EntityType.HUMAN and e.clan_id is not None:
                pop[e.clan_id] = pop.get(e.clan_id, 0) + 1
                hsum[e.clan_id] = hsum.get(e.clan_id, 0.0) + e.hunger
                if e.id == chief_of.get(e.clan_id):
                    chief_ok.add(e.clan_id)      # chef en vie ET dans son propre clan
                cur = cand.get(e.clan_id)
                if cur is None or (e.age, -e.id) > (cur[0], -cur[1]):
                    cand[e.clan_id] = (e.age, e.id)
        # P2 succession : un clan dont le chef n'est plus un membre valide (mort ou défecté) promeut
        # son doyen. Gated _POLITICS_ON (POLITICS_OFF → aucun changement → hash S2c). Sur toute la
        # flotte de clans (pas seulement `due`) → un clan orphelin retrouve un chef dès ce scan.
        if _POLITICS_ON:
            for c in self.clans:
                if c.id not in chief_ok and c.id in cand:
                    c.chief_id = cand[c.id][1]
                    tick_events.append({"type": "clan_new_chief",
                                        "clan_id": c.id, "chief_id": c.chief_id})
        # P2b — décay + friction + voisinage (piggyback du scan `due`). Chaque paire au plus
        # UNE fois par appel (dédup `_seen`). Gated _POL_ON : sous RELATIONS_OFF/POLITICS_OFF,
        # zéro écriture → relations vides → décisions inchangées → hash empilé exact.
        if _POL_ON:
            rel, ally, rival = self.relations, self._ally_state, self._rival_state
            diplo_of = {c.id: _chief_personality(c.chief_id)[1] for c in self.clans}
            _seen = set()
            for c in due:
                for o in self.clans:
                    if o.id == c.id:
                        continue
                    k = _rel_key(c.id, o.id)
                    if k in _seen:
                        continue
                    _seen.add(k)
                    v = rel.get(k, 0)
                    _at_war = ((c.mode == "war" and c.war_target == o.id) or
                               (o.mode == "war" and o.war_target == c.id))
                    _neighbor = (not _at_war and k not in rival
                                 and _dist(c.cx, c.cy, o.cx, o.cy) < REL_NEIGHBOR_DIST)
                    # EXCLUSIFS (reco Regigigas) : un contact pacifique entretenu CONSTRUIT (+2) ;
                    # sinon les liens s'estompent (positif→0) ou les rancunes se réconcilient
                    # (négatif→0 si au moins un chef conciliant). Cumuler décay+voisinage donnait
                    # net 0 sur une paire positive → alliance inatteignable (T5 Regigigas).
                    # H1 : l'injustice OCCUPE la relation — l'érosion REMPLACE la cordialité
                    # de voisinage, elle ne s'y ajoute pas (sinon +2 −4 = −2 et le réglage
                    # deviendrait illisible). Sur l'ÉTAT d'hystérésis `ally`, pas sur le seuil brut.
                    _erode = False
                    if _ENVYALLY_ON and k in ally:
                        _wc = _clan_wealth(clan_bldg.get(c.id, {}))
                        _wo = _clan_wealth(clan_bldg.get(o.id, {}))
                        _lo, _hi = (_wc, _wo) if _wc <= _wo else (_wo, _wc)
                        _erode = (_hi >= ENVY_ALLY_MIN
                                  and _lo * ENVY_ALLY_RATIO_D <= _hi * ENVY_ALLY_RATIO_N)
                    if _erode:
                        # L'érosion SE DÉCLARE : c'est le seul site d'appel qui nomme sa cause, et
                        # c'est ce qui permet à l'annale de dire POURQUOI l'alliance s'est éteinte.
                        _rel_apply(rel, ally, rival, c.id, o.id, -ENVY_ALLY_EROSION, tick_events,
                                   cause=REL_CAUSE_ENVY)
                    elif _neighbor:
                        _rel_apply(rel, ally, rival, c.id, o.id, REL_D_NEIGHBOR, tick_events)
                    elif v > 0:
                        _rel_apply(rel, ally, rival, c.id, o.id, -1, tick_events)
                    elif v < 0 and (diplo_of[c.id] >= 0 or diplo_of[o.id] >= 0):
                        _rel_apply(rel, ally, rival, c.id, o.id, +1, tick_events)   # 2 rancuniers = jamais
        absorptions = []   # P3 : (vainqueur, perdant) conquêtes à appliquer APRÈS la boucle
        tributes = []      # P3 : (vainqueur, perdant) tributs (idem, buildings mutés hors boucle)
        rebellions = []    # P4 : ids de clans-mères qui se scindent (fondation différée)
        colonies = []      # P4.1 : ids de clans-mères qui essaiment (fondation différée)
        for c in due:
            p = pop.get(c.id, 0)
            avg_h = (hsum.get(c.id, 0.0) / p) if p else 0.0
            bread = sum(getattr(b, "bread", 0) for b in clan_bldg.get(c.id, {}).get("mill", []))
            rivals = [o for o in self.clans if o.id != c.id and pop.get(o.id, 0) >= 1]
            # ── P4 : vie interne (tension causale + soupapes), gated _UNREST_ON ─────────
            # La tension monte avec la misère (famine/guerre/surpop/temper), baisse en paix
            # prospère. À l'éval due : SCISSION ≥90 (différée), sinon COUP ≥70 (inline, change
            # le chef → nouvelle ère AVANT la décision de mode ci-dessous). Ordre spec : split>coup.
            if _UNREST_ON:
                _houses = clan_bldg.get(c.id, {}).get("house", [])
                cap = (sum(BUILDING_SPECS["house"].pop_bonus * b.level for b in _houses)
                       + AGE_POP_BONUS * c.age)
                dt = 0
                if c.mode == "famine": dt += 8
                elif c.mode == "war":  dt += 3
                if (_CULTS_ON and c.mode == "war" and c.war_target >= 0 and c.cult_id >= 0):
                    _wt = next((cc for cc in self.clans if cc.id == c.war_target), None)
                    if _wt is not None and _wt.cult_id >= 0 and _wt.cult_id != c.cult_id:
                        dt += HOLY_TENSION   # P5 : guerre SAINTE (cultes ≠) → haine religieuse
                if p > cap:            dt += min(OVERPOP_TENSION_MAX, p - cap)  # overpop SCALÉ (boucle P3↔P4)
                dt += min(OVEREXTEND_MAX, max(0, (p - OVEREXTEND_SPAN) // 10))  # surextension PERSISTANTE (span-of-control)
                _tp = _chief_personality(c.chief_id)[0]
                dt += (1 if _tp > 0 else -1 if _tp < 0 else 0)                  # temper du chef
                if c.mode == "peace" and p <= cap: dt -= 2                      # prospérité calme → apaisement
                c.tension = max(0, min(100, c.tension + dt))
                if c.tension >= TENSION_SPLIT and p >= REBEL_MIN_POP and len(self.clans) < MAX_CLANS:
                    rebellions.append(c.id)          # fondation différée (mute self.clans hors boucle)
                elif (c.tension >= TENSION_COUP and len(self.clans) > 1
                      # H2 : la branche est SAUTÉE tant que le cooldown court. Elle tombe alors sur
                      # l'essaimage, qui exige `tension < SWARM_TENSION_MAX` (30) — inatteignable à
                      # ≥70 : aucune fuite, la garde préexistante ferme le chemin. La tension
                      # continue donc de monter et la scission à 90 reprend son rôle.
                      and (not _COUPCD_ON or c.last_coup_tick < 0
                           or self.tick_count - c.last_coup_tick >= COUP_COOLDOWN)):
                    # Un HÉGÉMON (monoclan) ne fait PAS de coup : sans rival politique, sa seule issue
                    # est la FRAGMENTATION → sa tension grimpe jusqu'à la scission (le cycle des empires).
                    # En multi-clan, le coup reste la soupape (churn politique) qui préempte la scission.
                    self._coup(c, tick_events)       # inline : renverse le chef, tension −40
                elif (_SWARM_ON and c.tension < SWARM_TENSION_MAX and p >= SWARM_MIN_POP
                      and p > cap and len(self.clans) < MAX_CLANS
                      # A9, trou résiduel (finding d'audit F2) : un clan DÉJÀ en migration ne
                      # doit pas essaimer. `_swarm_split` passe self_clan_id=mother_id à
                      # `_site_reserved`, ce qui EXEMPTE sa propre réservation — l'exemption est
                      # juste quand un clan ré-évalue SA cible, elle n'a aucun cas légitime ici :
                      # la mère pouvait fonder une colonie sur sa PROPRE terre de destination et
                      # y planter un second feu.
                      and c.migrating_to < 0):
                    colonies.append(c.id)            # P4.1 essaimage pacifique (fondation différée)
            # Cible de guerre : SANS P2, rival le + peuplé (tie-break id). AVEC P2 (P2b) : jamais
            # un ALLIÉ, puis la relation la plus BASSE (le + rival, tie-break id min).
            if not _POL_ON:
                war_tgt = max(rivals, key=lambda o: (pop.get(o.id, 0), -o.id)).id if rivals else -1
            else:
                _cands = [o for o in rivals if _rel_key(c.id, o.id) not in self._ally_state]
                if _ENVY_ON:
                    # F2 ENVIE : à choix égal, on convoite le voisin RICHE et FAIBLE (richesse par
                    # tête). Réordonnancement PUR d'un choix existant — les conditions d'entrée en
                    # guerre (rivaux, pop, cooldown, temper) sont inchangées → aucune guerre créée.
                    # ENVY_OFF → clé (relation, id) actuelle EXACTE.
                    _envy = {o.id: _clan_wealth(clan_bldg.get(o.id, {})) // max(1, pop.get(o.id, 0))
                             for o in _cands}
                    war_tgt = (min(_cands, key=lambda o: (-_envy[o.id],
                                                          self.relations.get(_rel_key(c.id, o.id), 0),
                                                          o.id)).id
                               if _cands else -1)
                else:
                    war_tgt = (min(_cands, key=lambda o: (self.relations.get(_rel_key(c.id, o.id), 0), o.id)).id
                               if _cands else -1)
            temper = _chief_personality(c.chief_id)[0] if _POL_ON else 0   # P2a : belliqueux → guerre + tôt
            # P3 : riposte (être attaqué annule le cooldown) + aide d'allié (rejoindre la guerre
            # subie par un allié). Calculés une fois par clan due, gated _WAR2_ON (sinon inertes).
            _attacked = False
            aid_tgt = -1; aid_ally = -1
            if _WAR2_ON and c.mode != "war":
                for o in self.clans:
                    if o.mode != "war" or o.war_target < 0 or o.id == c.id:
                        continue
                    if o.war_target == c.id:
                        _attacked = True                       # o m'agresse → riposte
                    a_id = o.war_target                        # A = clan attaqué par o
                    if (a_id != c.id and _rel_key(c.id, a_id) in self._ally_state
                            and _rel_key(c.id, o.id) not in self._ally_state):
                        # A est mon allié, o (l'agresseur) n'est pas mon allié → candidat à l'aide
                        _r = self.relations.get(_rel_key(c.id, o.id), 0)
                        if aid_tgt < 0 or (_r, o.id) < (self.relations.get(_rel_key(c.id, aid_tgt), 0), aid_tgt):
                            aid_tgt, aid_ally = o.id, a_id
            old = c.mode
            old_wt = c.war_target
            _is_aid = False
            if avg_h >= FAMINE_HUNGER or (bread == 0 and avg_h >= 40):
                new_mode, target = "famine", -1                # survie : prime sur tout
            elif (_GRANARY_ON and c.mode == "famine"
                  and bread < FAMINE_EXIT_BREAD and avg_h >= FAMINE_HUNGER - 10):
                # F4 — SORTIE par les RÉSERVES : sans grenier garni, la crise continue même si la
                # faim moyenne redescend ponctuellement (anti-clignotement). GRANARY_OFF → branche
                # absente → mode famine EXACTEMENT l'actuel.
                new_mode, target = "famine", -1
            elif c.mode == "war":
                _wt = c.war_target
                _wtpop = pop.get(_wt, 0) if _wt >= 0 else 0
                # P3a — CONQUÊTE : la cible touche le plancher S2c → victoire totale immédiate
                # (le plancher bloque de toute façon les kills ; avant, la guerre traînait au timeout).
                if _WAR2_ON and _wt >= 0 and 0 < _wtpop <= WAR_MIN_CLAN_POP:
                    absorptions.append((c.id, _wt))
                    new_mode, target = "peace", -1
                elif c.mode_ticks >= WAR_MAX_TICKS or not rivals:
                    # TIMEOUT : victoire de X (→ tribut) si + de kills infligés que subis, sinon paix blanche.
                    if (_WAR2_ON and _wt >= 0 and _wtpop > 0
                            and c.war_kills_for > c.war_kills_against):
                        tributes.append((c.id, _wt))
                    new_mode, target = "peace", -1
                elif c.war_target < 0 or pop.get(c.war_target, 0) == 0:
                    new_mode, target = ("war", war_tgt) if war_tgt >= 0 else ("peace", -1)
                else:
                    new_mode, target = "war", c.war_target
            elif (c.mode_ticks < PEACE_MIN_TICKS - temper * WAR_TEMPER_STEP
                  and not _attacked and aid_tgt < 0):
                new_mode, target = "peace", -1                 # cooldown (sauf riposte/aide P3)
            elif _WAR2_ON and aid_tgt >= 0 and p >= WAR_MIN_POP:
                new_mode, target, _is_aid = "war", aid_tgt, True   # P3b — entrée en guerre aux côtés d'un allié
            elif p >= WAR_MIN_POP and rivals:
                new_mode, target = ("war", war_tgt) if war_tgt >= 0 else ("peace", -1)
            else:
                new_mode, target = "peace", -1
            c.war_target = target
            _holy = False
            # P2b : déclaration/retarget vers un NOUVEL ennemi → -60 sur la paire (ils deviennent rivaux).
            if _POL_ON and new_mode == "war" and target >= 0 and target != old_wt:
                _rel_apply(self.relations, self._ally_state, self._rival_state,
                           c.id, target, REL_D_WAR, tick_events)
                if _WAR2_ON:   # P3 : nouvelle déclaration → compteurs d'issue remis à zéro
                    c.war_kills_for = 0
                    c.war_kills_against = 0
                if _CULTS_ON and c.cult_id >= 0:   # P5 : guerre SAINTE (cultes ≠) → pénalité additionnelle
                    _tc = next((cc for cc in self.clans if cc.id == target), None)
                    if _tc is not None and _tc.cult_id >= 0 and _tc.cult_id != c.cult_id:
                        _holy = True
                        _rel_apply(self.relations, self._ally_state, self._rival_state,
                                   c.id, target, HOLY_REL_PENALTY, tick_events)
                if _is_aid:
                    tick_events.append({"type": "clan_war_aid", "clan_id": c.id,
                                        "ally": aid_ally, "target": target})
            if new_mode != old:
                c.mode = new_mode
                c.mode_ticks = 0
                _ev = {"type": "clan_mode", "clan_id": c.id, "mode": new_mode}
                if _holy:
                    _ev["holy"] = True   # guerre sainte → bannière teintée au front
                tick_events.append(_ev)
        # P3 — application différée (mutations de self.clans / buildings hors de la boucle de décision).
        # Absorptions d'ABORD (retirent des clans) puis tributs (skip si un clan a disparu entre-temps).
        if _WAR2_ON:
            for x_id, y_id in absorptions:
                if any(cc.id == x_id for cc in self.clans) and any(cc.id == y_id for cc in self.clans):
                    self._absorb_clan(x_id, y_id, tick_events)
            for x_id, y_id in tributes:
                if any(cc.id == x_id for cc in self.clans) and any(cc.id == y_id for cc in self.clans):
                    self._tribute(x_id, y_id, clan_bldg, tick_events)
        if _UNREST_ON:   # P4 : scissions (fondation d'un clan neuf → mute self.clans)
            for mid in rebellions:
                self._rebel_split(mid, tick_events)
            for mid in colonies:   # P4.1 : essaimages (idem, différés)
                self._swarm_split(mid, tick_events)
        if _CULTS_ON:   # P5 E1 : conversion & schisme (après les mutations de clans du tick)
            self._update_cults(due, pop, tick_events)
        # P7 G3 : la décision de migrer se prend APRÈS les mutations de clans du tick (absorptions,
        # scissions, essaimages) — un clan absorbé entre-temps n'a plus de terre à quitter, et un
        # clan tout juste fondé lit un `_reservation_ctx` à jour.
        self._decide_migrations(due, pop, tick_events)

    def _cult_guardian(self, cult_id, pop):
        """P5 E1 — gardien d'un culte, DÉRIVÉ de l'état vivant (rien à sérialiser) : le clan
        fondateur s'il est vivant et toujours de ce culte, sinon le plus peuplé (tie id min).
        None si < 2 clans partagent le culte (monoclan → rien à rompre → pas de schisme)."""
        members = [c for c in self.clans if c.cult_id == cult_id]
        if len(members) < 2:
            return None
        cult = self.cults.get(cult_id)
        if cult is not None:
            for c in members:
                if c.id == cult.founder_clan:   # fondateur vivant → gardien
                    return c.id
        return min(members, key=lambda c: (-pop.get(c.id, 0), c.id)).id   # sinon le + peuplé

    def _update_feasts(self, clan_bldg: dict, tick_events: list):
        """P5 E2 — Fête des moissons. Appelée CHAQUE tick (décompte), 100 % déterministe (zéro RNG).

        Décompte : tout clan en fête voit feast_ticks décroître ; guerre/famine l'INTERROMPT (→ 0).
        Déclenchement : PENDANT l'automne (à la 1ère fenêtre paix + moisson mûre de la saison, 1×/an et
        par clan via le verrou feast_year), chaque clan EN PAIX dont ≥ FEAST_FIELDS_MIN champs de blé
        sont MÛRS (stage 4 = la moisson sur pied) lance une fête de FEAST_TICKS ticks. On teste toute la
        saison (pas le seul tick de bascule) car les champs mûrs FLUCTUENT (moissonnés) et la paix est
        intermittente : une seule chance/an raterait presque toujours la fenêtre. La fête ne CONSOMME
        RIEN (les champs restent stage 4 — l'abondance est la porte ; la boucle des fermiers n'est pas
        perturbée). Effets (convergence + natalité×seuil) portés par _beh_wander et la repro, gated
        _FEAST_ON. FEAST_OFF → no-op → HEAD."""
        if not _FEAST_ON or not self.clans:
            return
        tick = self.tick_count
        # Décompte (chaque tick) : la fête s'éteint d'elle-même à 0, ou est brisée par la guerre/famine.
        for c in self.clans:
            if c.feast_ticks > 0:
                if c.mode != "peace":
                    c.feast_ticks = 0          # guerre/famine → fin immédiate et silencieuse
                else:
                    c.feast_ticks -= 1
        # Déclenchement : à N'IMPORTE QUEL tick d'automne (1ère fenêtre paix+moisson ; verrou feast_year).
        if get_season(tick) != "autumn":
            return
        year = tick // (4 * TICKS_PER_SEASON)
        for c in self.clans:
            if c.mode != "peace" or c.feast_year == year or c.feast_ticks > 0:
                continue
            fields = clan_bldg.get(c.id, {}).get("wheatfield", [])
            ripe = sum(1 for f in fields if f.stage >= 4)   # moisson sur pied (stage 4 = mûr)
            if ripe < FEAST_FIELDS_MIN:
                continue
            c.feast_ticks = FEAST_TICKS          # ne consomme RIEN : les champs restent mûrs
            c.feast_year = year
            tick_events.append({"type": "feast_start", "clan_id": c.id, "fields": ripe})

    def _update_cults(self, due, pop, tick_events):
        """P5 E1 — à l'éval due : CONVERSION (1× irréversible, verrou cult_converted) puis SCHISME
        (exclusif). Conversion : un clan converge vers le culte d'un ALLIÉ plus GROS (rel ≥ 60, pop
        stricte >, autre culte) — verrou anti-ping-pong (triangle à 2 gros amis). Schisme : un clan
        RIVAL du gardien de son culte fonde le sien (religion suit politique), ré-arme le verrou.
        100 % déterministe (conditions + tie-breaks, zéro RNG). Skip les clans absorbés ce tick."""
        alive = {c.id for c in self.clans}
        for c in due:
            if c.id not in alive or c.cult_id < 0:
                continue
            # CONVERSION (1× ; le schisme ré-arme le verrou) — candidat rel max, tie id min
            if not c.cult_converted:
                best = None; best_key = None
                for o in self.clans:
                    if o.id == c.id or o.cult_id < 0 or o.cult_id == c.cult_id:
                        continue
                    if pop.get(o.id, 0) <= pop.get(c.id, 0):   # pop(B) > pop(A) STRICT
                        continue
                    r = self.relations.get(_rel_key(c.id, o.id), 0)
                    if r < CULT_CONVERT_REL:
                        continue
                    key = (-r, o.id)
                    if best_key is None or key < best_key:
                        best_key = key; best = o
                if best is not None:
                    c.cult_id = best.cult_id
                    c.cult_converted = True
                    _rel_apply(self.relations, self._ally_state, self._rival_state,
                               c.id, best.id, 5, tick_events)   # la conversion rapproche
                    tick_events.append({"type": "cult_converted", "clan_id": c.id,
                                        "to_cult": c.cult_id, "name": self.cults[c.cult_id].name})
                    continue   # exclusif : pas de schisme le même tick
            # SCHISME (exclusif, après conversion) : rival du gardien de son culte
            guardian = self._cult_guardian(c.cult_id, pop)
            if (guardian is not None and guardian != c.id
                    and _rel_key(c.id, guardian) in self._rival_state):
                old_cult = c.cult_id
                new_cult = self._found_cult(c.id, self.tick_count)
                c.cult_id = new_cult
                c.cult_converted = False   # une foi neuve peut re-converger plus tard
                c.last_deed = f"la fondation de sa foi, {self.cults[new_cult].name}"  # P5 E3 jalon
                for o in self.clans:       # rancune avec chaque ancien co-cultiste
                    if o.id != c.id and o.cult_id == old_cult:
                        _rel_apply(self.relations, self._ally_state, self._rival_state,
                                   c.id, o.id, -10, tick_events)
                tick_events.append({"type": "cult_schism", "clan_id": c.id, "old_cult": old_cult,
                                    "new_cult": new_cult, "name": self.cults[new_cult].name})

    def _dispatch_caravans(self, clan_bldg: dict, tick_events: list):
        """D2 : rafraîchit les BOARDS de cours (affiché = négocié) puis évalue les
        routes et recrute AU PLUS UNE caravane par appel. Mission paramétrée :
        QUOI acheter (GOODS_ORDER), AVEC QUOI payer (bois, ou pierre en glut),
        au taux SPOT du vendeur (recalculé à l'arrivée). Aucun RNG (ordres triés,
        paliers entiers) → déterministe, hash-neutre sans marché."""
        busy = set()
        candidates: dict[int, list] = {}
        for e in self.entities:
            if not e.alive or e.etype != EntityType.HUMAN or e.clan_id is None:
                continue
            if e.trade_phase is not None:
                busy.add(e.clan_id)          # max 1 marchand par clan
            elif (e.hunger < MERCHANT_HUNGER_MAX and e.thirst < MERCHANT_THIRST_MAX
                    and e.gestation_left == 0 and e.wood == 0 and e.stone == 0
                    and e.iron == 0 and e.gold == 0
                    and e._build_target_type is None
                    and e.colonist_dest is None      # P7 G2 : ni un colon en route (sinon la
                    # colonne perd ses hommes en chemin : la garde inverse existait déjà)
                    and e.expedition_phase is None   # P7 G1 : un éclaireur en mission n'est
                    # pas recrutable marchand (sinon son slot d'expédition fuit → le clan
                    # reste marqué « expédition en vol » pour toujours)
                    and e.pilgrim_phase is None):   # garde croisée (gate-review C1 :
                    # sans elle, un pèlerin en mission était recrutable marchand →
                    # offrande écrasée, Σ violée, bénédiction gratuite, renom fantôme)
                candidates.setdefault(e.clan_id, []).append(e)

        def pool_wood(cid):  return sum(h.wood for h in clan_bldg.get(cid, {}).get("house", []))
        def pool_stone(cid): return sum(h.stone for h in clan_bldg.get(cid, {}).get("house", []))
        def pool_iron(cid):  return sum(f.iron for f in clan_bldg.get(cid, {}).get("forge", []))
        def has_forge(cid):  return bool(clan_bldg.get(cid, {}).get("forge", []))
        def market(cid):
            mks = clan_bldg.get(cid, {}).get("market", [])
            return mks[0] if mks else None

        # (a) Boards : cours + recherche, event market_price au CHANGEMENT seulement
        for cid in sorted(clan_bldg):
            mkt = market(cid)
            if mkt is None:
                continue
            rs = _stone_rate(pool_stone(cid))
            ri = _iron_rate(pool_iron(cid))
            if rs != mkt.rate_stone:
                tick_events.append({"type": "market_price", "clan_id": cid,
                                    "good": "stone", "rate": rs, "x": mkt.x, "y": mkt.y})
            if ri != mkt.rate_iron:
                tick_events.append({"type": "market_price", "clan_id": cid,
                                    "good": "iron", "rate": ri, "x": mkt.x, "y": mkt.y})
            mkt.rate_stone = rs
            mkt.rate_iron = ri
            mkt.wants_stone = 1 if pool_stone(cid) < STONE_WANT_FULL else 0
            mkt.wants_iron = 1 if (has_forge(cid) and pool_iron(cid) < IRON_WANT_FULL) else 0

        # (b) Boucle acheteurs (le commerce exige 2 clans ; les boards, non)
        if len(self.clans) < 2:
            return
        rate_of = {"stone": lambda cid: _stone_rate(pool_stone(cid)),
                   "iron":  lambda cid: _iron_rate(pool_iron(cid))}
        for a in sorted(self.clans, key=lambda c: c.id):
            mka = market(a.id)
            if mka is None or a.id in busy or a.id not in candidates:
                continue
            for good in GOODS_ORDER:
                # fenêtre d'achat
                if good == "stone":
                    if pool_stone(a.id) >= STONE_WANT_FULL:
                        continue
                else:
                    if not has_forge(a.id):
                        continue
                    pi = pool_iron(a.id)
                    if pi >= IRON_WANT_BARGAIN:
                        continue
                sellers = [c for c in sorted(self.clans, key=lambda c: c.id)
                           if c.id != a.id and market(c.id) is not None
                           and rate_of[good](c.id) > 0]
                if good == "iron":
                    pi = pool_iron(a.id)
                    if pi >= IRON_WANT_FULL:   # fenêtre bargain : rate 3 exigé
                        sellers = [c for c in sellers if rate_of["iron"](c.id) >= 3]
                if not sellers:
                    continue
                b = min(sellers, key=lambda c: (_dist(mka.x, mka.y,
                                                      market(c.id).x, market(c.id).y), c.id))
                # choix du paiement : ce que le VENDEUR valorise le plus
                pays = []
                if pool_wood(a.id) >= TRADE_WOOD_SURPLUS:
                    pays.append(("wood", _scarcity(pool_wood(b.id), WOOD_VALUE_TIERS)))
                if good == "iron" and pool_stone(a.id) >= PAY_STONE_MIN:
                    pays.append(("stone", _scarcity(pool_stone(b.id), STONE_VALUE_TIERS)))
                if _MONEY_ON and market(a.id).gold >= 1:   # F1 : l'or (valeur 4 > max scarcity 3 → préféré, zéro tie-break)
                    pays.append(("gold", MONEY_PAY_VALUE))
                if not pays:
                    continue
                pay = max(pays, key=lambda p: p[1])[0] if len(pays) > 1 else pays[0][0]
                merch = min(candidates[a.id],
                            key=lambda e: (_dist(e.x, e.y, mka.x, mka.y), e.id))
                merch.pray_ticks = 0   # pas de prière reportée (gate-review C1)
                merch.trade_phase = "load"
                merch.trade_dest_cid = b.id
                merch.trade_ticks = 0
                merch.trade_good = good
                merch.trade_pay = pay
                merch.state = State.TRADING
                ev = {"type": "trade_deal", "clan_id": a.id, "dest_clan_id": b.id,
                      "good": good, "pay": pay}
                if good == "stone" and pay == "wood":   # clés legacy D1
                    ev["wood"] = TRADE_WOOD_LOT
                    ev["stone"] = rate_of["stone"](b.id)
                tick_events.append(ev)
                return   # une seule caravane lancée par évaluation

    def _drain_markets(self, clan_bldg: dict):
        """Étal → maisons : 1 bois + 1 pierre par marché tous les MARKET_DRAIN_PERIOD
        ticks. Les imports réintègrent l'économie normale ; les piles de l'étal
        fondent à vue (spectacle). Ordre de liste sérialisé → déterministe."""
        for cid, groups in clan_bldg.items():
            for mkt in groups.get("market", []):
                houses = groups.get("house", [])
                if not houses:
                    continue
                if mkt.wood > 0:
                    h = next((h for h in houses if h.wood < MAX_WOOD_PER_HOUSE), None)
                    if h is not None:
                        mkt.wood -= 1; h.wood += 1
                if mkt.stone > 0:
                    mkt.stone -= 1; houses[0].stone += 1
                if mkt.iron > 0:   # D2 : le fer d'étal rejoint la forge (sinon reste visible)
                    f = next((f for f in groups.get("forge", [])
                              if f.iron < FORGE_MAX_IRON), None)
                    if f is not None:
                        mkt.iron -= 1; f.iron += 1

    # ── Foi / pèlerinages (bloc C1) ──────────────────────────────────────────
    def _dispatch_pilgrims(self, clan_bldg: dict, tick_events: list):
        """Recrute AU PLUS UN pèlerin par appel. A pèlerine vers B ssi
        renom(B) >= renom(A) + CHURCH_FAME_GAP (sans église : −999) → hystérésis
        monotone : chaque visite renforce l'attracteur, UNE Compostelle émerge.
        Appelé APRÈS _dispatch_caravans (gardes croisées trade/pilgrim aux deux
        dispatches → pas de double-booking aux ticks communs). Sans RNG."""
        busy = set()
        candidates: dict[int, list] = {}
        for e in self.entities:
            if not e.alive or e.etype != EntityType.HUMAN or e.clan_id is None:
                continue
            if e.pilgrim_phase is not None:
                busy.add(e.clan_id)          # max 1 pèlerin par clan
            elif (e.hunger < MERCHANT_HUNGER_MAX and e.thirst < MERCHANT_THIRST_MAX
                    and e.gestation_left == 0 and e.wood == 0 and e.stone == 0
                    and e.iron == 0 and e.gold == 0
                    and e._build_target_type is None
                    and e.colonist_dest is None      # P7 G2 : idem pour un colon en route
                    and e.expedition_phase is None   # P7 G1 : idem, pas de double-booking
                    and e.trade_phase is None):
                candidates.setdefault(e.clan_id, []).append(e)

        def pool_wood(cid):
            return sum(h.wood for h in clan_bldg.get(cid, {}).get("house", []))

        def treasury(cid):
            chs = clan_bldg.get(cid, {}).get("church", [])
            return chs[0].gold if chs else 0

        def renom(cid):
            chs = clan_bldg.get(cid, {}).get("church", [])
            return chs[0].pilgrims_served if chs else -999

        for a in sorted(self.clans, key=lambda c: c.id):
            houses_a = clan_bldg.get(a.id, {}).get("house", [])
            if a.id in busy or a.id not in candidates or not houses_a:
                continue
            # C2 : OR-D'ABORD — un trésor garni paie l'offrande en pièce (et ignore
            # le plancher bois : un clan à trésor pèlerine même pauvre en bois) ;
            # sinon la route bois C1 inchangée.
            gold_ok = treasury(a.id) >= OFFERING_GOLD
            wood_ok = pool_wood(a.id) >= PILGRIM_WOOD_MIN
            if not gold_ok and not wood_ok:
                continue
            ra = renom(a.id)
            dests = [c for c in sorted(self.clans, key=lambda c: c.id)
                     if c.id != a.id and clan_bldg.get(c.id, {}).get("church", [])
                     and renom(c.id) >= ra + CHURCH_FAME_GAP]
            if not dests:
                continue
            h0 = houses_a[0]
            b = min(dests, key=lambda c: (_dist(h0.x, h0.y,
                        clan_bldg[c.id]["church"][0].x,
                        clan_bldg[c.id]["church"][0].y), c.id))
            pil = min(candidates[a.id],
                      key=lambda e: (_dist(e.x, e.y, h0.x, h0.y), e.id))
            pil.pray_ticks = 0   # pas de prière reportée (gate-review C1)
            pil.pilgrim_pay = "gold" if gold_ok else "wood"
            pil.pilgrim_phase = "load"
            pil.pilgrim_dest_cid = b.id
            pil.pilgrim_ticks = 0
            pil.state = State.PILGRIMAGE
            tick_events.append({"type": "pilgrim_depart", "clan_id": a.id,
                                "dest_clan_id": b.id, "pay": pil.pilgrim_pay})
            return   # un seul pèlerin lancé par évaluation

    def _dispatch_expeditions(self, clan_bldg: dict, tick_events: list):
        """P7 G1 — détache UN éclaireur vers le site remarquable inconnu le plus proche.
        Cadence : période SCOUT_PERIOD déphasée par clan (pattern caravanes) → les clans
        ne partent pas au même tick. UNE expédition en vol par clan. Éligible = rôle scout
        s'il en existe un, SINON un versatile : c'est la définition que le moteur donne
        DÉJÀ de l'exploration (_ROLE_SECTIONS["scout"] = ("scout", "versatile")) — sans ça
        le chantier serait resté inerte sous 25 habitants, palier d'apparition du rôle.
        AUCUN tirage : clans triés, sites triés, tie-break id → hash-neutre sans carto."""
        due = [c for c in sorted(self.clans, key=lambda c: c.id)
               if (self.tick_count + c.id * SCOUT_PHASE) % SCOUT_PERIOD == 0]
        if not due:
            return
        sites = self.world.site_catalogue()
        if not sites:
            return
        busy = set()
        pop: dict = {}
        candidates: dict[int, list] = {}
        for e in self.entities:
            if not e.alive or e.etype != EntityType.HUMAN or e.clan_id is None:
                continue
            pop[e.clan_id] = pop.get(e.clan_id, 0) + 1
            if e.expedition_phase is not None:
                if self.tick_count - e.expedition_t0 > EXPEDITION_TIMEOUT:
                    # mission périmée sur un éclaireur que la cascade ne visite plus (famine,
                    # combat permanent) → on la clôt ICI, sinon le clan ne repartirait jamais.
                    # Ce balayage ne passe qu'aux ticks où un clan est à l'échéance (retour
                    # précoce plus haut) : la mission périmée peut donc survivre au plus une
                    # période avant d'être ramassée — latence acceptée, pas une fuite.
                    tick_events.append({"type": "expedition_lost", "clan_id": e.clan_id,
                                        "site": e.expedition_site})
                    _clear_expedition(e)
                else:
                    busy.add(e.clan_id)      # max 1 éclaireur en mission par clan
            elif (e.hunger < SCOUT_HUNGER_MAX and e.thirst < SCOUT_THIRST_MAX
                    and e.gestation_left == 0
                    and e._build_target_type is None
                    and e.trade_phase is None and e.pilgrim_phase is None
                    and e.colonist_dest is None       # P7 G2 : un colon marche vers sa terre
                    and _role_ok(e.role, "scout")):   # gardes croisées : pas de double-booking
                candidates.setdefault(e.clan_id, []).append(e)
        for c in due:
            if (c.id in busy or c.age < SITE_MIN_AGE
                    or pop.get(c.id, 0) < SITE_MIN_POP
                    or c.id not in candidates):
                continue
            scout = min(candidates[c.id],
                        key=lambda e: (0 if e.role == "scout" else 1, e.id))
            # La cible doit être sur SA masse terrestre : sans navigation, un site au-delà
            # d'une mer n'est pas une découverte, c'est une mission perdue d'avance (mesuré :
            # 1 site sur 24 est insulaire sur les mondes servis). Référence = la tuile de
            # l'éclaireur, c'est lui qui marche.
            comp = self.world.land_component(int(scout.x), int(scout.y))
            unknown = [s for s in sites
                       if s[0] not in c.known_sites
                       and s[0] not in c.failed_sites      # terre à laquelle on a renoncé
                       and (comp < 0 or self.world.land_component(s[1], s[2]) == comp)]
            if not unknown:
                continue      # ce clan a fait le tour du monde qui lui est accessible
            sid, sx, sy, _score = min(
                unknown, key=lambda s: ((s[1] - c.cx) ** 2 + (s[2] - c.cy) ** 2, s[0]))
            scout.expedition_site = sid
            scout.expedition_phase = "out"
            scout.expedition_t0 = self.tick_count
            scout.state = State.EXPLORING
            tick_events.append({"type": "expedition_depart", "clan_id": c.id,
                                "site": sid, "x": sx, "y": sy})

    def _church_upkeep(self, clan_bldg: dict, tick_events: list):
        """Cloche (1 event par office, déphasé par clan) + cierges : l'offrande de
        l'autel se CONSUME (1 bois / ALTAR_BURN_PERIOD ticks) — le puits économique.
        L'étal du marché se drainait vers l'économie ; l'autel BRÛLE."""
        for cid in sorted(clan_bldg):
            for ch in clan_bldg[cid].get("church", []):
                if (self.tick_count + cid * 37) % CHURCH_SERVICE_PERIOD == 0:
                    tick_events.append({"type": "church_bell", "clan_id": cid,
                                        "x": ch.x, "y": ch.y})
                if self.tick_count % ALTAR_BURN_PERIOD == 0 and ch.wood > 0:
                    ch.wood -= 1

    # ── Chroniques du monde (bloc K) ─────────────────────────────────────────
    _SPECIES_FR = {"boar": "sangliers", "chicken": "poules", "horned_sheep": "mouflons",
                   "horse": "chevaux", "human": "humains", "pig": "cochons",
                   "sheep": "moutons", "fish": "poissons", "shark": "requins"}
    _FIRST_BUILD_FR = {"house": "sa première maison", "mill": "son premier moulin",
                       "well": "son premier puit", "forge": "sa forge",
                       "market": "son premier marché",
                       "church": "son premier sanctuaire"}
    CHRONICLE_MAX = 600   # cap mémoire : le monde tourne à l'infini, pas ses annales

    def _update_relations(self, tick_events: list[dict]):
        """P2b — applique les deltas de relation issus des ÉVÉNEMENTS de CE tick (scan unique,
        style _update_chronicle). La déclaration de guerre (-60), le décay et le voisinage sont
        gérés dans _update_society. Émet clan_allies/clan_rivals (dans tick_events, avant les
        chroniques). RELATIONS_OFF / POLITICS_OFF → no-op (hash inchangé)."""
        if not _POL_ON or not self.clans:
            return
        rel, ally, rival = self.relations, self._ally_state, self._rival_state
        by_id = {c.id: c for c in self.clans} if _WAR2_ON else None   # P3 : compteurs de kills
        for ev in list(tick_events):          # snapshot : _rel_apply append dans tick_events
            t = ev.get("type")
            if t == "clan_fight":
                a, v = ev["attacker_clan"], ev["victim_clan"]
                _rel_apply(rel, ally, rival, a, v, REL_D_FIGHT, tick_events)
                if _WAR2_ON:   # compteurs portés par le DÉCLARANT (issue de guerre P3a)
                    ca = by_id.get(a); cv = by_id.get(v)
                    if ca is not None and ca.war_target == v:
                        ca.war_kills_for += 1        # X frappe sa cible
                    if cv is not None and cv.war_target == a:
                        cv.war_kills_against += 1     # X encaisse de sa cible
            elif t == "clan_defect":
                _rel_apply(rel, ally, rival, ev["from_clan"], ev["to_clan"], REL_D_DEFECT, tick_events)
                if _UNREST_ON:   # P4 : assimiler un défecteur ennemi déstabilise l'assimilateur
                    ct = by_id.get(ev["to_clan"])
                    if ct is not None: ct.tension = min(100, ct.tension + DEFECT_TENSION)
            elif t == "trade_exchange":
                _rel_apply(rel, ally, rival, ev["clan_id"], ev["dest_clan_id"], REL_D_TRADE, tick_events)
            elif t == "clan_tribute" and _UNREST_ON:   # P4 §1 : l'humiliation du tribut payé
                cf = by_id.get(ev["from_clan"])
                if cf is not None: cf.tension = min(100, cf.tension + TRIBUTE_TENSION)
            elif t == "clan_absorbed" and _UNREST_ON:  # P4 fix : l'annexion sur-étend le vainqueur
                cb = by_id.get(ev["by"])
                if cb is not None: cb.tension = min(100, cb.tension + CONQUEST_TENSION)
        # P3c mariages — un franchissement d'alliance (event clan_allies) est SCELLÉ par un mariage :
        # le clan le + peuplé de la paire donne son + jeune adulte non-chef à l'autre, +10 sur la paire.
        # Zéro état nouveau (déclenché par l'event P2), spam-proof par l'hystérésis de _rel_apply.
        if _WAR2_ON:
            weddings = [ev for ev in list(tick_events) if ev.get("type") == "clan_allies"]
            if weddings:
                mpop = {}
                for e in self.entities:
                    if e.alive and e.etype == EntityType.HUMAN and e.clan_id is not None:
                        mpop[e.clan_id] = mpop.get(e.clan_id, 0) + 1
                chiefs = {c.chief_id for c in self.clans}
                clans_by_id = {c.id: c for c in self.clans}
                adult = SPECS[EntityType.HUMAN].max_age * 0.20
                for ev in weddings:
                    a, b = ev["a"], ev["b"]
                    if a not in clans_by_id or b not in clans_by_id:
                        continue
                    donor, recv = (a, b) if (mpop.get(a, 0), -a) >= (mpop.get(b, 0), -b) else (b, a)
                    if mpop.get(donor, 0) <= WAR_MIN_POP:
                        continue                       # donneur trop petit → alliance sans mariage
                    cand = None
                    for e in self.entities:            # + jeune adulte non-chef (tie-break id min)
                        if (e.alive and e.etype == EntityType.HUMAN and e.clan_id == donor
                                and e.id not in chiefs and e.age > adult):
                            if cand is None or (e.age, e.id) < (cand.age, cand.id):
                                cand = e
                    if cand is None:
                        continue
                    cand.clan_id = recv                # le/la marié(e) rejoint son nouveau feu
                    _rel_apply(rel, ally, rival, a, b, REL_D_MARRIAGE, tick_events)
                    tick_events.append({"type": "clan_marriage", "from_clan": donor,
                                        "to_clan": recv, "entity_id": cand.id})

    def _locate_ally_breaks(self, tick_events):
        """Pose x/y sur les ruptures d'alliance du tick — le point MÉDIAN des deux feux, parce
        qu'une rupture n'a pas de lieu propre : elle appartient aux deux clans à la fois.
        `_rel_apply` est une fonction de module et ne connaît pas les clans ; plutôt que de lui
        faire porter des coordonnées à travers ses 14 sites d'appel, on enrichit ICI, à l'unique
        point où tous les events du tick ont convergé.
        x/y sont OPTIONNELS et volontairement absents si un clan a disparu dans le même tick
        (conquête, extinction) : la frise centrera ce qui a un lieu et laissera le reste inerte —
        même contrat que l'enrichissement d'annales du lot UI, aucun événement fabriqué."""
        breaks = [ev for ev in tick_events if ev.get("type") == "clan_ally_break"]
        if not breaks:
            return
        pos = {c.id: (c.cx, c.cy) for c in self.clans}
        for ev in breaks:
            pa, pb = pos.get(ev["a"]), pos.get(ev["b"])
            if pa and pb:
                ev["x"], ev["y"] = int((pa[0] + pb[0]) / 2), int((pa[1] + pb[1]) / 2)

    def _purge_clan_relations(self, cid):
        """P3 : retire toute trace du clan `cid` des relations + états dérivés (conquête §3 OU
        extinction naturelle E8 — même bug de clés gelées depuis P2). Gated _WAR2_ON aux appels."""
        self.relations   = {k: v for k, v in self.relations.items() if cid not in k}
        self._ally_state  = {k for k in self._ally_state  if cid not in k}
        self._rival_state = {k for k in self._rival_state if cid not in k}

    def _absorb_clan(self, x_id, y_id, tick_events):
        """P3a conquête — le clan Y (tombé au plancher) est ABSORBÉ par X (victoire totale).
        Membres (chef inclus) + bâtiments durables → X (butin intégral, colonie émergente) ;
        feu de camp & chantiers de Y effacés (rien n'est mort → pas de ruine E8) ; Y retiré ;
        relations purgées ; guerre passive des tiers sur Y remise à paix (chemin extinction)."""
        n = 0
        for e in self.entities:
            if e.alive and e.etype == EntityType.HUMAN and e.clan_id == y_id:
                e.clan_id = x_id; n += 1          # le prochain _update_jobs les re-répartit
        kept = []
        for b in self.buildings:
            if b.clan_id != y_id:
                kept.append(b); continue
            if b.btype == "campfire" or b.btype.startswith("site_"):
                continue                          # un clan = un feu ; chantiers inachevés perdus
            b.clan_id = x_id; kept.append(b)      # maisons/moulins/forges/marchés → X
        self.buildings = kept
        self.clans = [c for c in self.clans if c.id != y_id]
        _winner = next((c for c in self.clans if c.id == x_id), None)
        if _winner is not None:   # P5 E3 : la conquête = jalon commémoré (dédicace d'un futur monument)
            _winner.last_deed = f"sa victoire sur le clan {y_id + 1}"
        for c in self.clans:                      # tiers en guerre sur Y → paix (comme extinction D1)
            if c.war_target == y_id:
                c.mode = "peace"; c.war_target = -1; c.mode_ticks = 0
                tick_events.append({"type": "clan_mode", "clan_id": c.id, "mode": "peace"})
        self._purge_clan_relations(y_id)
        tick_events.append({"type": "clan_absorbed", "clan_id": y_id, "by": x_id, "members": n})

    def _tribute(self, x_id, y_id, clan_bldg, tick_events):
        """P3 tribut — le perdant Y verse TRIBUTE_PCT % de chaque ressource stockée au vainqueur X.
        Pool = somme par ressource ; prélèvement/crédit triés par id ; caps respectés ; surplus
        perdu (pillage gaspillé) ; division ENTIÈRE (zéro float). Event si transfert > 0."""
        yb = clan_bldg.get(y_id, {}); xb = clan_bldg.get(x_id, {})
        # (ressource, sources Y [(btype, attr)], cibles X [(btype, attr, cap|None)])
        RES = [
            ("wood",  [("house", "wood"), ("market", "wood")],
                      [("house", "wood", MAX_WOOD_PER_HOUSE), ("market", "wood", MARKET_MAX_STOCK)]),
            ("stone", [("house", "stone"), ("market", "stone")],
                      [("house", "stone", None), ("market", "stone", MARKET_MAX_STOCK)]),
            ("iron",  [("forge", "iron"), ("market", "iron")],
                      [("forge", "iron", FORGE_MAX_IRON), ("market", "iron", MARKET_MAX_STOCK)]),
            ("bread", [("mill", "bread")], [("mill", "bread", MILL_MAX_BREAD)]),
        ]
        moved = {}
        for res, sources, targets in RES:
            src = [(b, attr) for bt, attr in sources for b in yb.get(bt, [])]
            pool = sum(getattr(b, attr) for b, attr in src)
            want = pool * TRIBUTE_PCT // 100
            got = 0
            for b, attr in sorted(src, key=lambda ba: ba[0].id):
                t = min(getattr(b, attr), want - got)
                setattr(b, attr, getattr(b, attr) - t); got += t
                if got >= want:
                    break
            rem = got                                     # crédit chez X (caps ; surplus perdu)
            tgt = [(b, attr, cap) for bt, attr, cap in targets for b in xb.get(bt, [])]
            for b, attr, cap in sorted(tgt, key=lambda t: t[0].id):
                room = rem if cap is None else min(rem, cap - getattr(b, attr))
                if room <= 0:
                    continue
                setattr(b, attr, getattr(b, attr) + room); rem -= room
                if rem <= 0:
                    break
            if got > 0:
                moved[res] = got
        if moved:
            tick_events.append({"type": "clan_tribute", "from_clan": y_id, "to_clan": x_id,
                                "wood": moved.get("wood", 0), "stone": moved.get("stone", 0),
                                "iron": moved.get("iron", 0), "bread": moved.get("bread", 0)})

    def _coup(self, clan, tick_events):
        """P4 §2.2 — coup d'État : le plus JEUNE adulte non-chef renverse le chef (contraste avec
        la succession-doyen P2.1 : la légitimité promeut l'ancien, la colère porte un jeune loup).
        Nouvelle ère politique (personnalité P2a du nouveau chef). tension −40. ≥1 challenger requis."""
        adult = SPECS[EntityType.HUMAN].max_age * 0.20
        challengers = [e for e in self.entities
                       if e.alive and e.etype == EntityType.HUMAN and e.clan_id == clan.id
                       and e.id != clan.chief_id and e.age > adult]
        if not challengers:
            return
        young = min(challengers, key=lambda e: (e.age, e.id))
        clan.chief_id = young.id
        clan.tension = max(0, clan.tension - 40)
        clan.last_coup_tick = self.tick_count      # H2 : arme le cooldown (hors switch — c'est un
                                                   # fait d'état civil du clan, comme founded_tick ;
                                                   # le gater rendrait un save incohérent d'un
                                                   # lancement à l'autre. Seule la GARDE est gatée.)
        tick_events.append({"type": "clan_coup", "clan_id": clan.id, "chief_id": young.id})

    def _found_cult(self, founder_clan, tick):
        """P5 — enregistre un culte neuf (fondation initiale ou schisme). Nom déterministe."""
        cid = self._next_cult_id
        self._next_cult_id += 1
        self.cults[cid] = Cult(id=cid, name=_cult_name(self.world.seed, cid),
                               founder_clan=founder_clan, founded_tick=tick)
        return cid

    def _found_clan(self, leader, members, cx, cy, cult_id=-1, tick_events=None,
                    known_sites=None, failed_sites=None):
        """P4 §3 — fonde un clan neuf à (cx,cy) : `leader` = chef, `members` (entités) le rejoignent.
        Compteur d'id MONOTONE, couleur id%4, campfire posé. AUCUN RNG (position fournie) → le flux
        `random` global reste byte-identique. Retourne le Clan neuf (âge Bois, tension 0).
        P5 : `cult_id` = culte HÉRITÉ de la mère (scission/colonie gardent la foi ; rupture religieuse
        = le schisme, distinct de la rupture politique)."""
        nid = self._next_clan_id
        self._next_clan_id += 1
        nc = Clan(id=nid, cx=float(cx), cy=float(cy),
                  color=CLAN_COLORS[nid % len(CLAN_COLORS)], chief_id=leader.id)
        if _CULTS_ON:
            nc.cult_id = cult_id
        if _CARTO_ON and failed_sites:
            nc.failed_sites = list(failed_sites)
        if _CARTO_ON and known_sites:
            # P7 G1 : le savoir part avec les hommes — colons et rebelles emportent la carte
            # de la mère (même pattern que l'héritage de culte E1). Copie, pas partage : les
            # deux clans exploreront ensuite chacun de leur côté.
            nc.known_sites = list(known_sites)
        # A10 — un clan neuf doit VIVRE quelque part avant de conclure qu'il s'est trompé.
        # MESURÉ sans cette garde (sonde g3-live seed 7) : c5, fondé à t=10800, levait le camp à
        # t=11160 — 360 ticks sur place, une demi-saison ; 6 des 7 départs du run étaient de cette
        # nature. La cause est structurelle : le clan hérite des `known_sites` de sa mère, donc il
        # connaît d'emblée les belles terres et n'a aucun passé de migration à purger.
        # Le champ est posé HORS de `_MIGRATE_ON` : c'est un fait d'état civil du clan, pas une
        # décision de migration — le gater rendrait la valeur dépendante d'un switch et le
        # round-trip d'un save incohérent d'un lancement à l'autre.
        nc.founded_tick = self.tick_count
        self.clans.append(nc)
        for e in members:
            e.clan_id = nid                       # gardent leur rôle jusqu'au prochain _update_jobs
        self.buildings.append(Building(id=self._next_building_id, clan_id=nid,
                                       x=int(cx), y=int(cy), btype="campfire"))
        self._next_building_id += 1
        if _HEROES_ON and tick_events is not None:   # P5 E4 : le chef FONDATEUR entre dans la légende
            _name_hero(leader, "founder", self.world, tick_events)
        return nc

    def _rebel_split(self, mother_id, tick_events):
        """P4 §2.1 — scission : les K=pop//3 membres les PLUS ÉLOIGNÉS du feu de la mère font
        sécession → clan neuf mené par le plus éloigné. Rancune fondatrice (rel mère/née = −50,
        rival). Re-vérifie les conditions (l'état a pu changer depuis la décision différée)."""
        mother = next((c for c in self.clans if c.id == mother_id), None)
        if mother is None or len(self.clans) >= MAX_CLANS:
            return
        members = [e for e in self.entities if e.alive and e.etype == EntityType.HUMAN
                   and e.clan_id == mother_id]
        if len(members) < REBEL_MIN_POP:
            return
        K = len(members) // 3
        if K < 1:
            return
        # tri stable : les plus éloignés du feu d'abord (−dist², tie-break id) → rebelles = tête
        members.sort(key=lambda e: (-((e.x - mother.cx) ** 2 + (e.y - mother.cy) ** 2), e.id))
        rebels = members[:K]
        leader = rebels[0]
        nc = self._found_clan(leader, rebels, leader.x, leader.y, cult_id=mother.cult_id,
                              tick_events=tick_events, known_sites=mother.known_sites,
                              failed_sites=mother.failed_sites)
        mother.tension = max(0, mother.tension - 60)   # la mère respire par le départ
        _rel_apply(self.relations, self._ally_state, self._rival_state,
                   mother_id, nc.id, REL_D_REBELLION, tick_events)   # née rivale d'entrée
        tick_events.append({"type": "clan_rebellion", "clan_id": mother_id,
                            "new_clan": nc.id, "members": K, "chief_id": leader.id})

    def _decide_migrations(self, due, pop, tick_events):
        """P7 G3 — le chef regarde sa terre et décide, à la cadence des modes (AUCUNE période
        neuve). Deux exodes qui racontent deux histoires : le PULL est l'appel d'une terre bien
        meilleure, le PUSH la fuite d'une terre épuisée.
        A9 (a) : on itère en ORDRE D'ID CROISSANT et la réservation est écrite AU MOMENT de la
        décision — c'est cet ordre, déterministe et rejouable, qui départage deux clans que la
        même terre appelle le même tick (mesuré : c14 et c17 visaient tous deux s8)."""
        if not _MIGRATE_ON or not due:
            return
        cat = self.world.site_catalogue()
        if not cat:
            return
        fresh = None                       # conv FRAÎCHE : ~19 ms, calculée au plus une fois/tick
        frozen = self.world.conv_grid()
        for c in sorted(due, key=lambda c: c.id):
            if c.migrating_to >= 0:        # déjà en route : on ne redécide pas sous ses pieds
                continue
            fx, fy = int(c.cx), int(c.cy)
            if not (0 <= fx < self.world.width and 0 <= fy < self.world.height):
                continue
            if fresh is None:
                fresh = self.world.conv_fresh()
            local = int(fresh[fy, fx])
            # L'hystérésis compte l'état du TERROIR, pas la permission de partir : elle s'accumule
            # même quand la guerre ou la tension ferment la porte (sinon un clan en crise longue
            # devrait re-souffrir 2 évals après chaque accalmie).
            c.push_evals = (c.push_evals + 1) if _terroir_epuise(local, int(frozen[fy, fx])) else 0
            if c.mode != "peace" or c.tension >= 30 or pop.get(c.id, 0) < MIGRATE_MIN_POP:
                continue
            # A10 — DEUX délais aux sémantiques distinctes, jamais confondus :
            # SÉDENTARISATION depuis la fondation (court : la fenêtre d'opportunité d'un clan est
            # PRÉCOCE et périssable — mesuré, guerre et tension ferment 54 à 67 % des évals
            # ensuite, donc un délai long ne reporte pas un départ, il le SUPPRIME) ...
            if self.tick_count - c.founded_tick < MIGRATION_SETTLE:
                continue
            # ... et COOLDOWN depuis la dernière migration (long : on ne déménage pas en boucle).
            # La sentinelle -1 « n'a JAMAIS migré » reste intacte et opposable.
            if c.last_migration_tick >= 0 and \
                    self.tick_count - c.last_migration_tick < MIGRATION_COOLDOWN:
                continue
            comp = self.world.land_component(fx, fy)
            ctx = self._reservation_ctx()      # relu par clan : la réservation du précédent COMPTE
            best = None
            for sid, sx, sy, _sc in cat:
                if sid not in c.known_sites or sid in c.failed_sites:
                    continue                   # A5 : jamais une terre où nos hommes ont renoncé
                if (sx - fx) ** 2 + (sy - fy) ** 2 < MIGRATE_MIN_DIST ** 2:
                    continue                   # trop près : le village bougerait pour rien
                if comp >= 0 and self.world.land_component(sx, sy) != comp:
                    continue                   # A3 : on ne migre pas au-delà d'une mer
                if not self.world.is_walkable(sx, sy):
                    continue                   # ceinture : un feu ne tombe JAMAIS dans l'eau
                if self._site_reserved(sid, sx, sy, ctx, self_clan_id=c.id):
                    continue                   # A9 : habitée, promise, ou déjà en chemin
                gain = int(frozen[sy, sx]) - local
                cand = (gain, -((sx - fx) ** 2 + (sy - fy) ** 2), -sid, sid, sx, sy)
                if best is None or cand > best:
                    best = cand                # meilleur gain, puis la plus PROCHE, puis id min
            if best is None:
                continue
            gain = best[0]
            pousse = c.push_evals >= MIGRATE_PUSH_EVALS and gain >= 1
            tire = gain >= MIGRATE_MIN_GAIN
            if not (tire or pousse):
                continue
            self._start_migration(c, best[3], best[4], best[5],
                                  "pull" if tire else "push", gain, tick_events)

    def _start_migration(self, clan, sid, sx, sy, cause, gain, tick_events):
        """Pose le CHANTIER du nouveau feu et RÉSERVE la terre dans le même souffle (A9 a) : entre
        la décision et la réservation il ne doit pas s'écouler un seul clan de la boucle."""
        site = Building(id=self._next_building_id, clan_id=clan.id, x=int(sx), y=int(sy),
                        btype="site_campfire", work_needed=MIGRATION_WORK, work_done=0)
        self._next_building_id += 1
        self.buildings.append(site)
        clan.migrating_to = sid
        clan.migration_t0 = self.tick_count
        clan.push_evals = 0
        tick_events.append({"type": "clan_migration_start", "clan_id": clan.id, "site": sid,
                            "cause": cause, "gain": gain, "x": int(sx), "y": int(sy)})

    def _update_migrations(self, tick_events):
        """P7 G3 — achèvement et abandon, CHAQUE TICK (le chantier se termine quand les bâtisseurs
        finissent, pas quand le chef réfléchit). Placé AVANT la promotion générique des chantiers :
        sans ça le `site_campfire` deviendrait un `campfire` ordinaire et le clan porterait DEUX
        feux jusqu'à sa prochaine éval — territoire, `_cb` et `_site_reserved` verraient double."""
        if not _MIGRATE_ON:
            return
        for clan in sorted([c for c in self.clans if c.migrating_to >= 0], key=lambda c: c.id):
            site = next((b for b in self.buildings if b.btype == "site_campfire"
                         and b.clan_id == clan.id), None)
            if site is None:                      # chantier disparu (ruiné, conquis) → on renonce
                self._abandon_migration(clan, None, "lost", tick_events)
                continue
            if site.work_done >= site.work_needed > 0:
                self._complete_migration(clan, site, tick_events)
            elif self.tick_count - clan.migration_t0 >= MIGRATION_TIMEOUT:
                self._abandon_migration(clan, site, "timeout", tick_events)

    def _abandon_migration(self, clan, site, reason, tick_events):
        """Renoncement — et le site rejoint `failed_sites` (A5) : une terre que le village entier
        n'a pas su atteindre ne doit plus JAMAIS être proposée à ce clan, ni en migration, ni en
        colonie, ni en expédition. Sans quoi il la reviserait à chaque éval, à vie."""
        sid = clan.migrating_to
        if site is not None and site in self.buildings:
            self.buildings.remove(site)           # le chantier avorté s'efface (rien n'a été bâti)
        if sid >= 0 and sid not in clan.failed_sites:
            clan.failed_sites.append(sid)
            clan.failed_sites.sort()
        clan.migrating_to = -1
        clan.migration_t0 = -1
        clan.push_evals = 0                       # A9 (c) : la réservation tombe AVEC l'abandon
        tick_events.append({"type": "clan_migration_failed", "clan_id": clan.id,
                            "site": sid, "reason": reason})

    def _complete_migration(self, clan, site, tick_events):
        """BASCULE ATOMIQUE — le village enjambe : pas de mode neuf, pas de phase sans-abri.
        Dans le MÊME tick : le nouveau feu s'allume, tout l'ancien clan tombe en ruines, le centre
        suit. A9 (c) : la sentinelle n'est remise QU'APRÈS la pose du feu — il n'existe jamais un
        tick où la terre n'est ni réservée ni occupée, sinon un voisin s'y glisserait."""
        from_x, from_y = int(clan.cx), int(clan.cy)
        sid = clan.migrating_to
        # RIEN NE SUIT — ni les matériaux, ni les vivres : la friction de l'exode est TOTALE.
        # L'arbitrage d'origine (ffafb37e) faisait suivre min(pain, 5) ; RETIRÉ après mesure,
        # le mécanisme était mort trois fois. (1) Illisible : tous les lecteurs de `bread` sont
        # mill-only (repas, verrou de famine F4, tribut, wire), or on versait sur le FEU.
        # (2) Toujours vide : ceux qui migrent sont les clans PAUVRES, sans moulin — mesuré 0
        # pain sur 12 bascules ; les clans riches (30 et 120 pains au monde live) ne migrent
        # jamais. (3) Sans objet : 0 tick de famine sur ces 12 bascules à friction RÉELLE, faim
        # moyenne à l'arrivée 12-22 pour un seuil à 55, soit plus du double de marge.
        for b in list(self.buildings):
            if b.clan_id != clan.id or b is site:
                continue
            _was_monument = b.btype == "monument"  # un monument laisse un vestige durable (E3)
            b.btype = "ruin"                      # y compris l'ancien feu et les chantiers avortés
            # ORPHELINE, comme sur le chemin d'extinction (finding d'audit F3) : sans ça la ruine
            # reste attachée au clan et (a) ANCRE SON TERRITOIRE 15000 ticks depuis une terre qu'il
            # a quittée — `_compute_territory` prend pour ancre tout bâtiment à clan_id >= 0 —,
            # (b) ses stocks comptent encore dans `_clan_wealth`, ce qui biaise l'envie F2, et
            # (c) à une extinction ultérieure elle serait SUPPRIMÉE au lieu de vieillir.
            b.clan_id = -1
            b.ruin_ticks = RUIN_LIFETIME * (MONUMENT_RUIN_MULT if _was_monument else 1)
            b.work_needed = 0
            b.work_done = 0
        site.btype = "campfire"                   # le feu s'allume : la terre reste occupée sans
        site.work_needed = 0                      # discontinuité (A9 c)
        site.work_done = 0
        clan.cx, clan.cy = float(site.x), float(site.y)
        clan.migrating_to = -1
        clan.migration_t0 = -1
        clan.push_evals = 0
        clan.last_migration_tick = self.tick_count
        tick_events.append({"type": "clan_migration", "clan_id": clan.id, "site": sid,
                            "from_x": from_x, "from_y": from_y,
                            "to_x": site.x, "to_y": site.y})

    def _reservation_ctx(self):
        """A9 — l'état des engagements sur les terres, calculé UNE fois par décision (le prédicat
        est appelé pour chaque site du catalogue : sans ce pré-calcul on rebalaierait entités et
        bâtiments 24 fois). Rien n'est stocké : tout est DÉRIVÉ de l'état courant."""
        fires = [(b.x, b.y) for b in self.buildings if b.btype == "campfire"]
        migrating = {c.migrating_to: c.id for c in self.clans if c.migrating_to >= 0}
        colons = [e.colonist_dest for e in self.entities
                  if e.alive and e.colonist_dest is not None]
        return fires, migrating, colons

    def _site_reserved(self, sid, sx, sy, ctx, self_clan_id=None):
        """A9 — une terre est PRISE dès qu'on s'y engage, pas seulement quand le feu y brûle.
        Le télescopage est SYMÉTRIQUE, d'où un prédicat unique lu par les DEUX blocs : sans le
        terme `migrating_to`, une colonie se fonderait sous les pieds d'un village en route ;
        sans le terme `colonist_dest`, un village migrerait vers la terre où marche une colonie.
        `self_clan_id` : un clan ne compte JAMAIS sa propre réservation quand il ré-évalue sa
        cible, sinon il se déclarerait lui-même intrus et renoncerait à son propre chantier."""
        fires, migrating, colons = ctx
        r2 = SITE_OCCUPIED_R ** 2
        if any((fx - sx) ** 2 + (fy - sy) ** 2 <= r2 for fx, fy in fires):
            return True
        holder = migrating.get(sid)
        if holder is not None and holder != self_clan_id:
            return True
        # Le colon marche vers l'ANCRE du site : la comparaison est exacte en pratique, mais on la
        # pose en distance — un colon dérouté reste un engagement sur cette terre.
        return any((cx - sx) ** 2 + (cy - sy) ** 2 <= r2 for cx, cy in colons)

    def _swarm_split(self, mother_id, tick_events):
        """P4.1 essaimage — un clan prospère et à l'étroit envoie K=pop//4 colons (les + éloignés du
        feu, leader = le + éloigné) fonder une COLONIE, de préférence SUR la ruine la plus proche du
        feu-mère (consommée → recolonisation E8), sinon à la position du leader. Colonie ALLIÉE de la
        mère (+50 POSÉ direct, sans event clan_allies → pas de mariage auto). Re-vérifie les conditions."""
        mother = next((c for c in self.clans if c.id == mother_id), None)
        if mother is None or len(self.clans) >= MAX_CLANS:
            return
        members = [e for e in self.entities if e.alive and e.etype == EntityType.HUMAN
                   and e.clan_id == mother_id]
        if len(members) < SWARM_MIN_POP:
            return
        K = len(members) // 4
        if K < 1:
            return
        members.sort(key=lambda e: (-((e.x - mother.cx) ** 2 + (e.y - mother.cy) ** 2), e.id))
        colonists = members[:K]
        leader = colonists[0]
        # G2 — FONDATION DIRIGÉE : si la mère connaît une belle terre au loin, la colonie s'y fonde
        # plutôt que sur la ruine d'à côté. Réordonnancement PUR d'une décision existante (pattern
        # de l'envie F2) : les conditions d'entrée de l'essaimage ne bougent pas, seul (fx, fy) change.
        site_pick = None
        if _COLONY_ON and mother.known_sites:
            _cat = self.world.site_catalogue()
            _comp = self.world.land_component(int(mother.cx), int(mother.cy))
            _min_d = min(self.world.width, self.world.height) // 4   # A6 : relatif au monde
            _ctx = self._reservation_ctx()
            _cands = []
            for _sid, _sx, _sy, _sc in _cat:
                if _sid not in mother.known_sites or _sid in mother.failed_sites:
                    continue                     # A5 : jamais une terre où nos hommes ont renoncé
                if _sc < COLONY_MIN_SCORE:
                    continue
                if (_sx - mother.cx) ** 2 + (_sy - mother.cy) ** 2 < _min_d ** 2:
                    continue                     # trop près : c'est un essaimage local, pas une colonie
                if _comp >= 0 and self.world.land_component(_sx, _sy) != _comp:
                    continue                     # A3 : pas de colonie au-delà d'une mer
                if not self.world.is_walkable(_sx, _sy):
                    continue                     # ceinture : un feu ne tombe JAMAIS dans l'eau
                if self._site_reserved(_sid, _sx, _sy, _ctx, self_clan_id=mother_id):
                    continue                     # A9 (d) : habité, OU promis à une migration en
                                                 # cours, OU déjà visé par une colonie en vol —
                                                 # occupation DÉRIVÉE, aucun état stocké de plus
                _cands.append((_sid, _sx, _sy, _sc))
            if _cands:                           # la meilleure terre, la plus proche à score égal
                site_pick = max(_cands, key=lambda c: (c[3],
                                                       -((c[1] - mother.cx) ** 2 + (c[2] - mother.cy) ** 2),
                                                       -c[0]))
        # Recolonisation : la ruine la plus proche du feu-mère (tie-break id bâtiment) si elle existe.
        ruins = [b for b in self.buildings if b.btype == "ruin"]
        ruin = (min(ruins, key=lambda b: ((b.x - mother.cx) ** 2 + (b.y - mother.cy) ** 2, b.id))
                if ruins else None)
        on_ruin = False
        if site_pick is not None and ruin is not None:
            # A7 — l'exil ne se décide pas contre une ruine qui vaut le coup. On compare les deux
            # terres avec le MÊME juge et à la MÊME époque : la conv FIGÉE du monde vierge, des
            # deux côtés (le score du catalogue EST cette conv à l'ancre). Lookup O(1), aucun
            # calcul frais — la conv fraîche reste réservée au PUSH de la migration (G3).
            if site_pick[3] < int(self.world.conv_grid()[ruin.y, ruin.x]) + SWARM_DIRECT_MARGIN:
                site_pick = None          # la ruine tient la comparaison → on repeuple l'ancien lieu
        if site_pick is not None:
            # La ruine n'est PAS consommée : elle reste disponible pour le prochain essaimage local.
            fx, fy = float(site_pick[1]), float(site_pick[2])
        elif ruin is not None:
            fx, fy = ruin.x, ruin.y
            self.buildings.remove(ruin)   # consommée → le feu de la colonie prend sa place
            on_ruin = True
        else:
            fx, fy = leader.x, leader.y
        nc = self._found_clan(leader, colonists, fx, fy, cult_id=mother.cult_id,
                              tick_events=tick_events, known_sites=mother.known_sites,
                              failed_sites=mother.failed_sites)
        # Alliance d'entrée POSÉE directement (pas de _rel_apply → pas d'event clan_allies → pas de
        # mariage auto ; le mariage P3 scelle les FRANCHISSEMENTS, ici la fondation POSE l'état).
        k = _rel_key(mother_id, nc.id)
        self.relations[k] = min(REL_MAX, self.relations.get(k, 0) + REL_D_COLONY)
        self._ally_state.add(k)
        _ev = {"type": "clan_swarm", "clan_id": mother_id, "new_clan": nc.id,
               "members": K, "x": int(fx), "y": int(fy), "on_ruin": on_ruin}
        if site_pick is not None:      # discipline golden : clé ABSENTE hors fondation dirigée
            _ev["site"] = site_pick[0]   # (site et on_ruin sont mutuellement exclusifs)
            for _e in colonists:         # la colonne se met en marche vers sa terre promise
                _e.colonist_dest = (int(site_pick[1]), int(site_pick[2]))
                _e.colonist_t0 = self.tick_count
        tick_events.append(_ev)

    def _clans_wire(self):
        """Sérialise les clans + wire politique P2 (gated _POL_ON, DISCIPLINE GOLDEN : une clé
        n'apparaît que si non vide → RELATIONS_OFF/étages-off = payload pré-P2 exact).
        chief_trait dérivé du chief_id (temper) ; allies/rivals dérivés des états d'hystérésis.
        P6 F2 : `wealth` recalculée ICI à chaque tick (pure fonction de l'état, 0 accepté) — la
        mémoriser à l'éval due la figerait après un load → replay divergent."""
        out = []
        _wcb: dict = {}
        if _ECON_ON:                       # pools par clan (une passe, comme clan_bldg dans step)
            for _b in self.buildings:
                _wcb.setdefault(_b.clan_id, []).append(_b)
        for c in self.clans:
            d = c.to_dict()
            if _ECON_ON:                   # F2 : richesse du clan (la pauvreté est une info → 0 émis)
                d["wealth"] = _clan_wealth({"_": _wcb.get(c.id, [])})
            if _POL_ON:
                temper, _ = _chief_personality(c.chief_id)
                if temper != 0:
                    d["chief_trait"] = "warlike" if temper > 0 else "dovish"
                allies = sorted(p[0] if p[1] == c.id else p[1]
                                for p in self._ally_state if c.id in p)
                rivals = sorted(p[0] if p[1] == c.id else p[1]
                                for p in self._rival_state if c.id in p)
                if allies:
                    d["allies"] = allies
                if rivals:
                    d["rivals"] = rivals
            if _UNREST_ON:   # P4 : jauge de tension (discipline golden : clé présente seulement sous ON)
                d["tension"] = c.tension
            if _CULTS_ON and c.cult_id >= 0 and c.cult_id in self.cults:   # P5 : nom du culte du clan
                d["cult"] = self.cults[c.cult_id].name
            if _FEAST_ON and c.feast_ticks > 0:   # P5 E2 : fête en cours (fanions + danse au feu)
                d["feast"] = True
            out.append(d)
        return out

    def _update_chronicle(self, tick_events: list[dict]):
        """Distille les tick_events du tick en JALONS d'annales (première maison/
        forge d'un clan, âges franchis, clans éteints, espèces disparues…).
        N'influence AUCUN comportement (pur enregistreur) et n'ajoute rien à la
        sortie de step() → déterminisme et hash du guard intacts."""
        t = self.tick_count
        add = self.chronicle.append
        for ev in tick_events:
            et = ev.get("type")
            if et == "clan_age_up":
                add({"t": t, "kind": "age", "msg":
                     f"Le clan {ev['clan_id'] + 1} entre dans l'Âge de {ev['age_name']}"})
            elif et == "clan_extinct":
                r = ev.get("ruins", 0)
                add({"t": t, "kind": "extinct", "msg":
                     f"Le clan {ev['clan_id'] + 1} s'éteint"
                     + (f" — {r} ruine{'s' if r > 1 else ''} demeure{'nt' if r > 1 else ''}" if r else "")})
            elif et == "clan_allies":   # P2b : franchissement du seuil d'alliance (hystérésis anti-spam)
                add({"t": t, "kind": "ally", "msg":
                     f"Le clan {ev['a'] + 1} et le clan {ev['b'] + 1} scellent une alliance"})
            elif et == "clan_ally_break":   # H1 : la sortie d'alliance, symétrique de l'entrée
                # TOUTE sortie est annoncée, pas seulement celle causée par l'envie : n'annoncer
                # que l'injustice donnerait au spectateur une vision biaisée de son monde, où les
                # alliances ne mourraient que de jalousie. Une alliance qui s'éteint doucement est
                # aussi de l'Histoire — elle a juste droit à un texte sobre.
                _lieu = {"x": ev["x"], "y": ev["y"]} if "x" in ev else {}
                if ev.get("cause") == REL_CAUSE_ENVY:
                    add({"t": t, "kind": "ally", "cat": "annals", **_lieu, "msg":
                         f"L'alliance du clan {ev['a'] + 1} et du clan {ev['b'] + 1} se brise : "
                         f"trop de richesse d'un côté, trop peu de l'autre"})
                else:
                    add({"t": t, "kind": "ally", "cat": "annals", **_lieu, "msg":
                         f"L'alliance entre le clan {ev['a'] + 1} et le clan {ev['b'] + 1} s'éteint"})
            elif et == "clan_rivals":
                add({"t": t, "kind": "rival", "msg":
                     f"Le clan {ev['a'] + 1} et le clan {ev['b'] + 1} deviennent rivaux"})
            elif et == "clan_absorbed":   # P3a — guerre gagnée = annales
                add({"t": t, "kind": "war", "cat": "annals", "msg":
                     f"Le clan {ev['by'] + 1} conquiert le clan {ev['clan_id'] + 1} et absorbe ses {ev['members']} survivants"})
            elif et == "clan_migration":   # P7 G3 — l'exode ABOUTI est un événement d'annales.
                # Le départ et l'échec restent SILENCIEUX (même parti pris qu'en G1 : seule la
                # découverte se racontait, pas le départ de l'éclaireur) — on ne raconte pas une
                # intention, on raconte ce qui a eu lieu.
                # Formulation neutre quant à la CAUSE : l'event d'arrivée ne la porte pas, et
                # l'exode a deux histoires (appel d'une terre meilleure / fuite d'un terroir
                # épuisé). Écrire « terres épuisées » mentirait sur les PULL, majoritaires.
                # G4 : sous TOPO le lieu se NOMME ; sans lui la phrase reste vraie, juste anonyme.
                _lieu = (site_name(self.world.seed, ev["site"])
                         if _TOPO_ON and ev.get("site") is not None else None)
                add({"t": t, "kind": "explo", "cat": "annals", "msg":
                     (f"Le clan {ev['clan_id'] + 1} quitte ses terres pour {_lieu}" if _lieu else
                      f"Le clan {ev['clan_id'] + 1} lève le camp et rallume son feu sur une terre "
                      f"lointaine")})
            elif et == "site_discovered" and _TOPO_ON and ev.get("dist", 0) >= TOPO_ANNAL_DIST:
                # G4 — seule la découverte LOINTAINE fait date. Un site à deux pas du feu n'est pas
                # une expédition, c'est une promenade : sans ce filtre les annales se noieraient
                # sous les découvertes de proximité (mesuré en G1 : médiane 22 tuiles).
                _k = ("site_annal", ev["site"])
                if _k not in self._chronicle_seen:      # une terre ne se découvre qu'UNE fois
                    self._chronicle_seen.add(_k)
                    add({"t": t, "kind": "explo", "cat": "annals", "msg":
                         f"Les éclaireurs du clan {ev['clan_id'] + 1} atteignent "
                         f"{site_name(self.world.seed, ev['site'])}"})
            elif et == "clan_swarm" and _TOPO_ON and ev.get("site") is not None:
                # G4 — la fondation DIRIGÉE se raconte (la clé `site` est absente d'un essaimage
                # local, donc ce test suffit à les distinguer sans drapeau supplémentaire).
                add({"t": t, "kind": "explo", "cat": "annals", "msg":
                     f"Le clan {ev['clan_id'] + 1} fonde une colonie "
                     f"{a_lieu(site_name(self.world.seed, ev['site']))}"})
            elif et == "clan_tribute":    # P3
                add({"t": t, "kind": "war", "msg":
                     f"Vaincu, le clan {ev['from_clan'] + 1} paie tribut au clan {ev['to_clan'] + 1}"})
            elif et == "clan_marriage":   # P3c
                add({"t": t, "kind": "ally", "msg":
                     f"Un mariage unit le clan {ev['from_clan'] + 1} et le clan {ev['to_clan'] + 1}"})
            elif et == "clan_war_aid":    # P3b
                add({"t": t, "kind": "war", "msg":
                     f"Le clan {ev['clan_id'] + 1} entre en guerre aux côtés de son allié le clan {ev['ally'] + 1}"})
            elif et == "clan_rebellion":  # P4 scission — fondation d'un clan = annales
                add({"t": t, "kind": "rebellion", "cat": "annals", "msg":
                     f"Le clan {ev['clan_id'] + 1} éclate : {ev['members']} rebelles fondent le clan {ev['new_clan'] + 1}"})
            elif et == "clan_coup":       # P4 coup d'État
                add({"t": t, "kind": "coup", "msg":
                     f"Un coup d'État renverse le chef du clan {ev['clan_id'] + 1}"})
            elif et == "cult_schism":     # P5 E1 : rupture religieuse — fondation d'une foi = annales
                add({"t": t, "kind": "cult", "cat": "annals", "msg":
                     f"Le clan {ev['clan_id'] + 1} rompt avec sa foi et fonde {ev['name']}"})
            elif et == "cult_converted":  # P5 E1 : conversion
                add({"t": t, "kind": "cult", "msg":
                     f"Le clan {ev['clan_id'] + 1} se convertit à {ev['name']}"})
            elif et == "feast_start":     # P5 E2 : fête des moissons
                add({"t": t, "kind": "feast", "msg":
                     f"Le clan {ev['clan_id'] + 1} célèbre la fête des moissons "
                     f"({ev.get('fields', 0)} champs mûrs)"})
            elif et == "monument_built":  # P5 E3 : monument de prestige
                _ded = ev.get("dedication") or "sa gloire"
                add({"t": t, "kind": "monument", "cat": "annals", "msg":
                     f"Le clan {ev['clan_id'] + 1} érige un monument à {_ded}"})
            elif et == "hero_named":      # P5 E4 : une figure entre dans la légende
                _via = {"kills": "au combat", "builds": "par ses œuvres",
                        "founder": "en fondant son clan"}.get(ev.get("via"), "")
                add({"t": t, "kind": "hero", "cat": "annals", "msg":
                     f"{ev['name']} se distingue {_via} (clan {ev['clan_id'] + 1})"})
            elif et == "hero_fallen":     # P5 E4 : mort d'un héros → annales
                _s = ev.get("age_seasons", 0)
                _life = f"après {_s} saison{'s' if _s > 1 else ''}" if _s >= 1 else "au terme d'une vie brève"
                add({"t": t, "kind": "hero", "cat": "annals", "msg":
                     f"{ev['name']}, héros du clan {ev['clan_id'] + 1}, tombe {_life}"})
            elif et == "money_dawn":      # P6 F1 : l'or devient monnaie (1× dans la partie) → annales
                add({"t": t, "kind": "econ", "cat": "annals", "msg":
                     "L'or devient monnaie : les marchés acceptent désormais la pièce en paiement"})
            elif et == "clan_swarm":      # P4.1 essaimage — fondation d'une colonie = annales
                _w = "sur d'anciennes ruines" if ev.get("on_ruin") else "en terre vierge"
                add({"t": t, "kind": "swarm", "cat": "annals", "msg":
                     f"Le clan {ev['clan_id'] + 1} essaime : une colonie s'installe {_w} (clan {ev['new_clan'] + 1})"})
            elif et in ("build_house", "build_mill", "build_well", "build_forge", "build_market", "build_church"):
                btype = et[6:]
                key = ("first_build", ev.get("clan_id"), btype)
                if key not in self._chronicle_seen:
                    self._chronicle_seen.add(key)
                    add({"t": t, "kind": "build", "msg":
                         f"Le clan {ev['clan_id'] + 1} érige {self._FIRST_BUILD_FR[btype]}"})
            elif et in ("craft_iron_pick", "craft_iron_axe"):
                key = ("first_iron", ev.get("clan_id"))
                if key not in self._chronicle_seen:
                    self._chronicle_seen.add(key)
                    add({"t": t, "kind": "iron", "msg":
                         f"Le clan {ev['clan_id'] + 1} forge son premier outil en fer"})
            elif et == "upgrade_building" and ev.get("btype") == "campfire":
                key = ("fire_l2", ev.get("clan_id"))
                if key not in self._chronicle_seen:
                    self._chronicle_seen.add(key)
                    add({"t": t, "kind": "build", "msg":
                         f"Le grand feu du clan {ev['clan_id'] + 1} rayonne (niveau 2)"})
            elif et == "trade_complete":
                a, b = ev.get("clan_id"), ev.get("dest_clan_id")
                _delivered = ev.get("stone", 0) > 0 or ev.get("iron", 0) > 0
                if _delivered:   # un retour bredouille (refus) n'est pas une route
                    key = ("first_trade", min(a, b), max(a, b))
                    if key not in self._chronicle_seen:
                        self._chronicle_seen.add(key)
                        add({"t": t, "kind": "trade", "msg":
                             f"Une route commerciale s'ouvre entre le clan {a + 1} et le clan {b + 1}"})
                if ev.get("good") == "iron" and ev.get("iron", 0) > 0:
                    key = ("first_iron_trade", min(a, b), max(a, b))
                    if key not in self._chronicle_seen:
                        self._chronicle_seen.add(key)
                        add({"t": t, "kind": "trade", "msg":
                             f"Le métal voyage : le clan {b + 1} fournit le fer au clan {a + 1}"})
            elif et == "trade_exchange" and ev.get("pay_good") == "stone":
                key = ("first_stone_pay", ev.get("clan_id"))
                if key not in self._chronicle_seen:
                    self._chronicle_seen.add(key)
                    add({"t": t, "kind": "trade", "msg":
                         f"Le clan {ev['clan_id'] + 1} paie en pierre — première monnaie minérale"})
            elif et == "trade_refused":
                key = ("first_refusal", ev.get("dest_clan_id"))
                if key not in self._chronicle_seen:
                    self._chronicle_seen.add(key)
                    add({"t": t, "kind": "trade", "msg":
                         f"Le clan {ev['dest_clan_id'] + 1} renvoie une caravane : son bien s'est fait rare"})
            elif et == "market_price" and ev.get("good") == "stone" and ev.get("rate") == 12:
                key = ("stone_parity", ev.get("clan_id"))
                if key not in self._chronicle_seen:
                    self._chronicle_seen.add(key)
                    add({"t": t, "kind": "trade", "msg":
                         f"Au marché du clan {ev['clan_id'] + 1}, la pierre s'échange à parité contre le bois"})
            elif et == "church_bell":
                key = ("first_procession", ev.get("clan_id"))
                if key not in self._chronicle_seen:
                    self._chronicle_seen.add(key)
                    add({"t": t, "kind": "faith", "msg":
                         f"La cloche du clan {ev['clan_id'] + 1} sonne pour la première fois — le village se rassemble"})
            elif et == "pilgrim_blessed":   # SUR bénédiction effective (leçon MAJOR gate D2)
                a, b = ev.get("clan_id"), ev.get("dest_clan_id")
                key = ("first_pilgrim", min(a, b), max(a, b))
                if key not in self._chronicle_seen:
                    self._chronicle_seen.add(key)
                    add({"t": t, "kind": "faith", "msg":
                         f"Un pèlerin du clan {a + 1} vient prier au sanctuaire du clan {b + 1}"})
                if ev.get("pay") == "gold":
                    key = ("first_gold_offering", min(a, b), max(a, b))
                    if key not in self._chronicle_seen:
                        self._chronicle_seen.add(key)
                        add({"t": t, "kind": "faith", "msg":
                             f"Une pièce d'or voyage en offrande : le clan {a + 1} dore le sanctuaire du clan {b + 1}"})
                if ev.get("served") == CHURCH_FAME_MILESTONE:
                    key = ("church_fame", b)
                    if key not in self._chronicle_seen:
                        self._chronicle_seen.add(key)
                        add({"t": t, "kind": "faith", "msg":
                             f"Le sanctuaire du clan {b + 1} rayonne au-delà des frontières"})
            elif et == "gold_deposit":
                key = ("first_gold", ev.get("clan_id"))
                if key not in self._chronicle_seen:
                    self._chronicle_seen.add(key)
                    add({"t": t, "kind": "faith", "msg":
                         f"Le clan {ev['clan_id'] + 1} tire l'or de la montagne — son sanctuaire a désormais un trésor"})
            elif et == "church_gilt" and ev.get("gilt", 0) >= GILT_MILESTONE:
                key = ("church_gilt", ev.get("clan_id"))
                if key not in self._chronicle_seen:
                    self._chronicle_seen.add(key)
                    add({"t": t, "kind": "faith", "msg":
                         f"Couvert d'or, le sanctuaire du clan {ev['clan_id'] + 1} resplendit au soleil"})
            elif et == "heatwave_start":
                add({"t": t, "kind": "weather", "msg": "Une canicule s'abat sur le monde"})
        # Espèces disparues / revenues (comparaison avec le tick précédent).
        # Recompute depuis self.entities (post-purge, post-naissances) : exact,
        # contrairement au species_counts du tick (les morts d'âge n'y sont pas).
        alive = {e.etype.value for e in self.entities if e.alive}
        if self._prev_species:
            for sp in sorted(self._prev_species - alive):
                add({"t": t, "kind": "species", "msg":
                     f"Les {self._SPECIES_FR.get(sp, sp)} ont disparu du monde"})
            for sp in sorted(alive - self._prev_species):
                add({"t": t, "kind": "species", "msg":
                     f"Les {self._SPECIES_FR.get(sp, sp)} réapparaissent"})
        self._prev_species = alive
        if len(self.chronicle) > self.CHRONICLE_MAX:
            self.chronicle = self.chronicle[-self.CHRONICLE_MAX:]

    def full_state(self) -> dict:
        """État complet pour un nouveau client qui se connecte."""
        return {
            "world":     self.world.to_dict(),
            "tick":      self.tick_count,
            # Le front en dérive son calendrier : il ne doit JAMAIS figer la valeur,
            # sinon un changement de TIME_SCALE lui fait afficher des dates fausses
            # en silence. Envoyé une seule fois, à la connexion → coût nul par tick.
            "ticks_per_season": TICKS_PER_SEASON,
            "season":    get_season(self.tick_count),
            "temp_c":    get_temperature(self.tick_count),
            "raining":   self.raining,
            "storming":  self.storming,
            "heatwave":  self.heatwave,
            "entities":  [e.to_dict() for e in self.entities],
            "stats":     self._compute_stats(),
            "history":   self.stats_history[-100:],
            "events":    self.events_log[-50:],
            "clans":     self._clans_wire(),
            "buildings": [b.to_dict() for b in self.buildings],
        }

    # ── Sauvegarde / reprise ─────────────────────────────────────────────────
    def save_state(self) -> dict:
        """Snapshot complet et sans perte de l'état de la simulation. Inclut
        l'état des RNG (random + numpy) pour une reprise EXACTE : un sim rechargé
        continue le même flux aléatoire que l'original."""
        np_s = np.random.get_state()
        return {
            "version": 1,
            "tick_count": self.tick_count,
            "raining": self.raining, "storming": self.storming,
            "rain_ticks_left": self.rain_ticks_left,
            "heatwave": self.heatwave, "heatwave_ticks_left": self.heatwave_ticks_left,
            "_next_building_id": self._next_building_id,
            "entity_id_counter": get_id_counter(),
            "py_random_state": list(random.getstate()),
            "np_random_state": [np_s[0], np_s[1].tolist(), int(np_s[2]),
                                int(np_s[3]), float(np_s[4])],
            "world": self.world.to_state(),
            "entities": [e.to_state() for e in self.entities],
            "clans": [asdict(c) for c in self.clans],
            "buildings": [asdict(b) for b in self.buildings],
            # Annales (bloc K) : l'histoire du monde survit aux reboots. COPIE (I5,
            # audit #89/#101) : sans ça save_state aliase la liste vivante self.chronicle
            # → sérialisée hors state_lock, un step() concurrent l'allongerait pendant le
            # dump JSON (snapshot déchiré, viole le replay byte-à-byte).
            "chronicle": [dict(c) for c in self.chronicle],
            "chronicle_seen": [list(k) for k in self._chronicle_seen],
            # Relations inter-clans (P2b) : liste [a,b,v] triée → save déterministe. Les états
            # allié/rival dérivés ne sont PAS sauvés (recalculés au load depuis la valeur).
            "relations": [[a, b, v] for (a, b), v in sorted(self.relations.items())],
            # États allié/rival (P2b) : SÉRIALISÉS (audit #1). L'hystérésis a un seuil d'ENTRÉE
            # (±40) ≠ seuil de SORTIE (±35) → une paire dans [35,40) est encore alliée en run
            # continu ; les recalculer au seuil dur au load la déclasserait → divergence.
            "ally_state":  sorted([a, b] for (a, b) in self._ally_state),
            "rival_state": sorted([a, b] for (a, b) in self._rival_state),
            "next_clan_id": self._next_clan_id,   # P4 : compteur monotone (asdict n'atteint pas les attrs Simulation)
            # Cultes (P5) : registre trié + compteur (jamais purgé). Clan.cult_id/cult_converted via asdict.
            "cults": [[c.id, c.name, c.founder_clan, c.founded_tick]
                      for c in sorted(self.cults.values(), key=lambda x: x.id)],
            "next_cult_id": self._next_cult_id,
            "money_dawn": self.money_dawn,   # P6 F1
        }

    def load_state(self, d: dict):
        """Restaure un snapshot save_state(). Remplace intégralement l'état.

        TRANSACTIONNEL (I1 durcissement) : tout le nouvel état est construit dans des
        variables LOCALES d'abord. Une clé manquante, un from_state qui lève ou un RNG
        malformé fait partir l'exception AVANT toute mutation de `self` → la simulation
        vivante reste intacte (avant, un load à moitié appliqué laissait un world neuf
        collé à de vieux clans + RNG non restauré = état hybride corrompu, replay cassé)."""
        from .world import World
        # World.from_state re-seed random/np GLOBALEMENT dans __init__. Si la
        # construction échoue APRÈS, la sim vivante (intacte) repartirait sur un autre
        # flux aléatoire → reprise non exacte. On capture les RNG globaux et on les
        # restaure en cas d'échec → transactionnalité complète (objet ET flux RNG).
        _py_save = random.getstate()
        _np_save = np.random.get_state()
        try:
            # ── Phase 1 : construire (peut lever) — self n'est PAS touché ──────
            world     = World.from_state(d["world"])
            entities  = [Entity.from_state(e) for e in d["entities"]]
            _cles_ignorees: set = set()
            clans     = [_depuis_dict(Clan, c, _cles_ignorees) for c in d["clans"]]
            buildings = [_depuis_dict(Building, b, _cles_ignorees) for b in d["buildings"]]
            tick_count          = d["tick_count"]
            raining             = d["raining"]; storming = d["storming"]
            rain_ticks_left     = d["rain_ticks_left"]
            heatwave            = d["heatwave"]
            heatwave_ticks_left = d["heatwave_ticks_left"]
            next_building_id    = d["_next_building_id"]
            entity_id_counter   = d["entity_id_counter"]
            if _cles_ignorees:
                # Visibilité SANS échec (le chargement doit aboutir) : un champ retiré est
                # attendu, un champ RENOMMÉ ne l'est pas et se réinitialiserait aux défauts
                # sans un mot. Une ligne au journal serveur, une seule fois par chargement.
                print(f"[load] clés de save inconnues, ignorées : {sorted(_cles_ignorees)} "
                      f"— normal après un retrait de champ ; SUSPECT après un renommage "
                      f"(le champ repartirait à sa valeur par défaut).", flush=True)
            # Types des compteurs validés en phase 1 (gate F3) : un compteur non-int (save
            # trafiqué) passerait le commit puis lèverait TypeError au 1er spawn/bâtiment —
            # crash DIFFÉRÉ qu'aucune sonde bornée ne voit → vecteur crash-loop résiduel.
            if not isinstance(entity_id_counter, int) or not isinstance(next_building_id, int):
                raise ValueError("compteurs d'id de save invalides (non entiers)")
            chronicle       = [dict(c) for c in d.get("chronicle", [])]   # copie (symétrie I5)
            chronicle_seen  = {tuple(k) for k in d.get("chronicle_seen", [])}
            prev_species    = {e.etype.value for e in entities if e.alive}
            # Relations P2b : absent d'un vieux save → dict vide (tout neutre, pas de migration).
            relations       = {(int(a), int(b)): int(v) for a, b, v in d.get("relations", [])}
            # États allié/rival (audit #1) : restaurés depuis le save (préserve l'hystérésis
            # 40/35). Vieux save (absent) → fallback recalcul au seuil dur (best-effort, une
            # paire dans la fenêtre d'hystérésis peut y perdre son statut, mais rien de mieux).
            if "ally_state" in d:
                ally_state  = {(int(a), int(b)) for a, b in d["ally_state"]}
                rival_state = {(int(a), int(b)) for a, b in d.get("rival_state", [])}
            else:
                ally_state  = {k for k, v in relations.items() if v >= REL_ALLY}
                rival_state = {k for k, v in relations.items() if v <= REL_RIVAL}
            # P4 : compteur d'id de clans. Vieux save (absent) → max(ids)+1 pour rester monotone.
            next_clan_id    = d.get("next_clan_id",
                                    (max((c.id for c in clans), default=N_CLANS - 1) + 1))
            # Cultes (P5) : registre restauré, sinon vieux save → refondation À FROID (1 culte/clan,
            # ids = ordre des clans triés, noms régénérés depuis le seed → identiques à un run frais).
            if "cults" in d:
                cults = {cid: Cult(id=cid, name=name, founder_clan=fc, founded_tick=ft)
                         for cid, name, fc, ft in d["cults"]}
                next_cult_id = d.get("next_cult_id", (max(cults, default=-1) + 1))
            else:
                cults = {}
                for _i, _c in enumerate(sorted(clans, key=lambda x: x.id)):
                    cults[_i] = Cult(id=_i, name=_cult_name(world.seed, _i),
                                     founder_clan=_c.id, founded_tick=0)
                    _c.cult_id = _i; _c.cult_converted = False
                next_cult_id = len(clans)
            # États RNG : construire PUIS les valider par un DRY-RUN sur des générateurs
            # JETABLES (gate F2). setstate/set_state rejettent une longueur/version invalide que
            # la simple construction du tuple ne détecte pas ; sans ce dry-run, l'échec surviendrait
            # en phase 2 (commit), APRÈS le remplacement de self → RNG hybride (le bug même que I1
            # doit éliminer, ex. POST /api/load sur la sim vivante). Les jetables ne touchent aucun
            # flux global → la phase 2 réappliquera les MÊMES états sans risque de lever.
            prs = d["py_random_state"]
            py_rng_state = (prs[0], tuple(prs[1]), prs[2])
            nrs = d["np_random_state"]
            np_rng_state = (nrs[0], np.array(nrs[1], dtype=np.uint32), nrs[2], nrs[3], nrs[4])
            random.Random().setstate(py_rng_state)              # lève ici (phase 1) si invalide
            np.random.RandomState().set_state(np_rng_state)     # idem, sans muter le global
        except Exception:
            random.setstate(_py_save)
            np.random.set_state(_np_save)
            raise
        # ── Phase 2 : commit (aucune opération faillible ici) ─────────────────
        self.world = world
        self.tick_count = tick_count
        self.raining = raining; self.storming = storming
        self.rain_ticks_left = rain_ticks_left
        self.heatwave = heatwave
        self.heatwave_ticks_left = heatwave_ticks_left
        self._next_building_id = next_building_id
        set_id_counter(entity_id_counter)
        self.entities = entities
        self.clans = clans
        self.buildings = buildings
        self.events_log = []
        self.stats_history = []
        self.chronicle = chronicle
        self._chronicle_seen = chronicle_seen
        self._prev_species = prev_species
        self.relations = relations
        self._ally_state = ally_state
        self._rival_state = rival_state
        self._next_clan_id = next_clan_id
        self.cults = cults
        self._next_cult_id = next_cult_id
        self.money_dawn = d.get("money_dawn", False)   # P6 F1 (vieux save → False)
        # World.from_state a re-seedé random/np dans __init__ → on restaure les RNG
        # en DERNIER, sinon la reprise repartirait sur le flux de la génération initiale.
        random.setstate(py_rng_state)
        np.random.set_state(np_rng_state)

    def save(self, path: str):
        """Écrit un snapshot JSON de façon atomique (tmp + os.replace)."""
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.save_state(), f, separators=(",", ":"))
        os.replace(tmp, path)

    def load(self, path: str):
        # Durcissement I2 (audit #64) : json.load accepte NaN/Infinity par défaut, qui
        # passeraient dans l'état puis feraient diverger/crasher step() plus tard (crash
        # différé → crash-loop avec LOAD_ON_START). Un save légitime n'a que des valeurs
        # finies → on rejette les non-finies au parsing.
        def _reject_nonfinite(tok):
            raise ValueError(f"valeur non-finie interdite dans le save: {tok!r}")
        with open(path) as f:
            self.load_state(json.load(f, parse_constant=_reject_nonfinite))

    # ── Observabilité (I0) ───────────────────────────────────────────────────
    def metrics(self) -> dict:
        """Métriques d'observabilité (endpoint /api/metrics). Baseline du
        pathfinding : taux de blocage (_stuck_resets) + distribution du temps de
        step. Purement dérivé de l'état → ne modifie rien."""
        ms = list(self._step_ms)
        ms_sorted = sorted(ms)

        def _pct(p: float) -> float:
            if not ms_sorted:
                return 0.0
            k = min(len(ms_sorted) - 1, int(p / 100.0 * len(ms_sorted)))
            return round(ms_sorted[k], 2)

        return {
            "tick": self.tick_count,
            "entities": len(self.entities),
            "stuck_resets_total": int(self.world._stuck_resets),
            "step_ms": {
                "avg": round(sum(ms) / len(ms), 2) if ms else 0.0,
                "p50": _pct(50), "p95": _pct(95),
                "max": round(max(ms), 2) if ms else 0.0,
                "n": len(ms),
            },
        }
