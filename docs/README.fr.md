# BioSim — Simulateur de vie

Simulateur de vie style WorldBox, observable en temps réel via navigateur. Aucune interaction utilisateur — la simulation tourne seule et évolue de manière autonome.

**Stack :** Python · FastAPI + uvicorn · NumPy · HTML Canvas 2D + WebSocket

---

## Fonctionnalités

- Monde procédural (fBm + domain warping), 7 types de terrain : eau, rivière, prairie, forêt, désert, montagne, + terre battue (*dirt* = herbe surpâturée, état dérivé)
- 9 espèces animales avec traits héréditaires et mutations
- **Civilisation humaine profonde** (empilée bloc par bloc, cf. section *Civilisation*) : clans, **métiers**, **âges technologiques** (Bois → Pierre → Fer → Acier), fer/forge, **économie** (marchés, caravanes, loi de l'offre), **religion** (sanctuaires, pèlerinages, or), **société** (gouvernements, guerres périodiques), **politique** (personnalité des chefs, alliances/rivalités, mariages), **guerre 2.0** (conquête, tribut, absorption), **vie interne** (tension, coups d'État, scissions, essaimage) — un **cycle des empires** émergent
- Saisons, météo dynamique (pluie, orages, canicules, incendies de forêt)
- **Chroniques** (annales narratives) + calendrier (années, saisons)
- Interface web temps réel : zoom/pan, tooltip, graphes de population, log d'événements, panneau clan
- Persistance (save/load sans perte, reprise exacte du flux aléatoire), monitoring système (CPU, RAM, température)

---

## Installation

```bash
git clone https://github.com/<votre-utilisateur>/biosim.git
cd biosim
pip install -r requirements.txt
python server.py
```

Ouvre ensuite **http://localhost:8080** dans ton navigateur (ou `http://<ip-machine>:8080` depuis le réseau local).

> **Note :** Sur certaines distributions Linux sans environnement virtuel, ajouter `--break-system-packages` à la commande pip.

---

## Lancement automatique au démarrage (systemd)

```bash
sudo nano /etc/systemd/system/biosim.service
```

Contenu :

```ini
[Unit]
Description=BioSim Life Simulator
After=network.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
User=<your_user>
WorkingDirectory=/home/<your_user>/biosim
# Écoute sur 0.0.0.0:8080 par défaut. Pour restreindre au loopback (derrière un
# reverse-proxy / tunnel), décommenter :
# Environment=BIOSIM_HOST=127.0.0.1
# Environment=BIOSIM_PORT=8080
ExecStart=/usr/bin/python3 server.py
Restart=always
RestartSec=5
# Durcissement (defense-in-depth : pas de RCE connue, mais le service tourne sous
# ton compte → on limite ce qu'il peut atteindre si une vuln émergeait) :
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=true
RestrictAddressFamilies=AF_INET AF_INET6

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable biosim
sudo systemctl start biosim
sudo systemctl status biosim
```

---

## Architecture

```
biosim/
├── server.py              # FastAPI, WebSocket broadcast, routes API
├── engine/
│   ├── world.py           # Génération procédurale du monde, biomes, nourriture
│   ├── entities.py        # Espèces (EntitySpec), états, traits héréditaires
│   └── simulation.py      # Boucle IA, comportements, mort, reproduction, clans
└── static/
    └── index.html         # Canvas, zoom/pan, tooltip, graphes, log événements
```

---

## Monde

- Taille : **220×160 tuiles**
- 7 types de terrain : eau, rivière, prairie, forêt, désert, montagne, + terre battue (dirt, état dérivé du surpâturage)
- Génération par bruit fBm avec domain warping
- Fertilité du sol (`fertility_grid`) : dégradation par pâturage, régénération lente
- Arbres (`tree_grid`) : abattage et repousse (~625 ticks)

---

## Espèces

| Espèce       | Couleur    | Rôle              | Compétences        |
|--------------|------------|-------------------|--------------------|
| Humain       | Jaune      | Prédateur/Bâtisseur | Couper, miner, construire |
| Sanglier     | Marron     | Prédateur         | —                  |
| Mouton       | Blanc      | Proie             | Troupeau           |
| Mouton cornu | Beige      | Proie             | Troupeau           |
| Cheval       | Marron clair | Neutre          | Troupeau           |
| Cochon       | Rose       | Proie             | —                  |
| Poulet       | Orange     | Proie             | —                  |
| Poisson      | Bleu clair | Proie (aquatique) | —                  |
| Requin       | Bleu foncé | Prédateur (aquatique) | —              |

**Traits héréditaires :** `speed`, `vision`, `hunger_rate` — initialisés à ±5% de la valeur de l'espèce, transmis avec mutation ±5% à chaque naissance.

---

## IA & comportements

États (par priorité) : `flee → hunt → eat → seek_food → reproduce → wander / resting / dead`

- **Troupeaux** : moutons, moutons cornés, chevaux se déplacent vers le centroïde des congénères proches (rayon 12 tuiles)
- **Fuite** : proies fuient les prédateurs en portée
- **Chasse** : prédateurs poursuivent les proies
- **Soif** : toutes les entités cherchent de l'eau (biomes WATER/RIVER) quand soif > seuil

---

## Civilisation

Les clans humains (4 au départ : rouge, bleu, vert, violet, chacun centré sur un feu de camp) portent une simulation de civilisation approfondie, empilée **bloc par bloc**. Chaque bloc est déterministe et gardé par un kill-switch d'imputation (voir *Déterminisme & tests*).

### Bâtiments

| Bâtiment    | Coût (bois/pierre)  | Effet                         |
|-------------|---------------------|-------------------------------|
| Feu de camp | — (initial)         | Centre du territoire, chaleur |
| Maison      | 15 bois             | +3 pop max par maison (×2 au niveau 2) |
| Champ de blé | —                  | Production alimentaire        |
| Moulin      | 12 bois + 8 pierre  | Transforme blé → pain (bâti près de l'eau) |
| Puit        | 8 bois + 6 pierre   | Source d'eau sans se déplacer |
| Forge       | 14 bois + 12 pierre | Âge du Fer : forge les outils en fer (1/clan) |
| Marché      | bois                | Échange inter-clan (caravanes, troc puis loi de l'offre) |
| Sanctuaire  | bois                | Religion : pèlerinages, bénédictions, trésor d'or |

### Âges technologiques

Chaque clan accumule de la **science** (bâtiments durables + population) et franchit 4 âges : **Bois → Pierre → Fer → Acier** (seuils 0 / 700 / 2500 / 6000). Un âge avancé = plus grande cité (cap de pop) et débloque des systèmes (fer à l'Âge du Fer, etc.).

### Métiers

À une certaine taille, un clan répartit sa population en **métiers** (déterministe, par quotas) : `farmer`, `woodcutter`, `miner`, `builder`, `warrior`, `merchant`, `priest`, `scout` (le chef reste polyvalent). Chaque métier biaise le comportement (le bûcheron coupe plus vite, le guerrier est le seul agresseur en guerre, etc.). Visible au survol d'un feu de camp (compteurs).

### Économie & religion

- **Marchés & caravanes** : un clan riche en bois mais pauvre en pierre envoie une caravane troquer chez un clan thésauriseur. Puis **loi de l'offre** (D2) : les cours (pierre, fer) bougent par paliers.
- **Sanctuaires** : les fidèles font des **pèlerinages**, offrent du bois/or, reçoivent une bénédiction ; l'or circule et dore le sanctuaire.

### Société, politique & guerre

- **Gouvernements** : le chef choisit un mode PAIX / GUERRE / FAMINE ; les guerres sont **périodiques** (un événement, pas un état permanent).
- **Politique (mémoire géopolitique)** : chaque chef a une **personnalité** dérivée de son identité (belliqueux ↔ pacifique, change à chaque succession → *ères* politiques) ; les clans gardent une **relation** −100..+100 (la guerre creuse la rancune, le commerce/voisinage rapproche) → **alliances** et **rivalités**, scellées par des **mariages** inter-clans.
- **Guerre 2.0** : une guerre a des **issues** — **conquête** (absorption : membres et bâtiments du perdant passent au vainqueur), **tribut** (20 % des ressources) ou paix blanche ; **riposte**, **aide d'un allié**. Un chef mort ou défecté est remplacé par succession.

### Vie interne — le cycle des empires

Chaque clan porte une **tension** interne (0-100) nourrie par la misère (famine, guerre qui traîne, surpopulation, humiliation du tribut, surextension d'un empire trop grand) et apaisée par la paix prospère. Trois soupapes :

- **Coup d'État** (tension ≥ 70) : le plus jeune adulte renverse le chef.
- **Scission** (tension ≥ 90) : un groupe fait sécession et **fonde un clan indépendant**. Un empire hégémonique (dernier clan debout) finit toujours par éclater → le monde se **re-diversifie** tout seul.
- **Essaimage** (clan prospère mais à l'étroit) : une colonie part s'installer, de préférence **sur les ruines** d'un clan éteint (recolonisation) ; colonie alliée de la mère.

Résultat : conquête → domination → révolte → fragmentation → nouvelles guerres, **à l'infini**. Les sécessionnistes étant les plus éloignés du feu (= les peuples fraîchement conquis), les empires se re-fragmentent le long des anciennes frontières.

*Pour l'historique détaillé des blocs (S2c, P1, D1/D2, C1/C2, A1, B, K, P2, P3, P4/P4.1) : `NEXT-STEPS.md` côté agent.*

### Outils

| Outil        | Coût           | Effet                              |
|--------------|----------------|------------------------------------|
| Hache bois   | 5 bois stockés | +3 bois/coup (total 5/coup)        |
| Hache pierre | 5 pierre       | +5 bois/coup (total 7/coup)        |
| Pioche bois  | 8 bois         | Permet de miner la pierre          |
| Pioche pierre | 5 pierre      | +1 pierre/coup (total 2/coup)      |
| Hache/pioche fer | forge + fer | Âge du Fer : outils plus rapides (upgrade des outils pierre) |
| Faucille     | 5 pierre       | +25 nourriture à la récolte        |
| Arrosoir     | 6 bois         | Accélère ×2 la croissance du blé   |
| Canne à pêche | 4 bois        | Permet de pêcher                   |

---

## Saisons & météo

**Saisons** (300 ticks chacune, 1200 ticks = 1 an) :

| Saison    | Regen nourriture | Reproduction | Notes               |
|-----------|-----------------|--------------|---------------------|
| Printemps | ×1.6            | Oui          | Pluie fréquente     |
| Été       | ×1.1            | Oui          | Canicules possibles |
| Automne   | ×0.6            | Non          | Orages              |
| Hiver     | ×0.10           | Non          | Faim ×1.5, feu essentiel |

**Événements météo :**
- **Pluie / Orage** : réduit la soif, accélère la croissance du blé, foudre possible sur les arbres
- **Canicule** (été uniquement) : soif ×2.2, risque d'incendie spontané

---

## Interface web

| Fonctionnalité | Description |
|----------------|-------------|
| Carte          | Zoom molette, pan drag |
| Tooltip        | Survole une entité : état, faim, soif, âge, sexe, traits, inventaire |
| Vitesse        | Slider 1× à 20× (ticks/seconde) |
| Pause          | Bouton ⏸ |
| Populations    | Barres temps réel + graphe d'évolution |
| Événements     | Log naissances, morts, chasses, constructions |
| Saison / Météo | Badge saison + overlay météo |
| Monitoring     | CPU%, RAM%, température (polling 3 s via `/api/sysinfo`) |

**Indicateurs visuels :**
- Contour rouge = en chasse
- Contour bleu = en fuite
- Sol mort (DeadGrass) = fertilité épuisée par le pâturage

---

## API

| Méthode | Route              | Description                        |
|---------|--------------------|------------------------------------|
| GET     | `/`                | Interface web                      |
| GET     | `/api/world`       | État complet initial (JSON)        |
| WS      | `/ws`              | Stream des ticks en temps réel     |
| POST    | `/api/speed?tps=N` | Change la vitesse (1–20 tps)       |
| POST    | `/api/pause`       | Toggle pause                       |
| GET     | `/api/sysinfo`     | CPU, RAM, température              |

---

## Performance

- Coût d'un tick sur Raspberry Pi 5 (mesuré) : ~15-20 ms à ~200 entités, montant avec la population (grille spatiale pour garder les voisinages ~linéaires).
- Cap de population : `MAX_PER_SPECIES = 200` par espèce (9 espèces → ~1800 entités max théorique). **Pas de cap global** à 5000.
- Monde 220×160, NumPy vectorisé (regen nourriture/fertilité/arbres/roches/feu).
- `psutil` est **optionnel** (monitoring CPU/RAM de `/api/sysinfo`) ; sans lui, ces champs valent `null`.

---

## Déterminisme & tests

La simulation est **100 % déterministe** (même graine → même déroulé, byte-à-byte), invariant essentiel puisque le monde doit tourner à l'infini sans crash ni dérive.

- **Filet de tests** (sans pytest) : `python3 tests/test_smoke.py` — tests unitaires de comportement (conquête, tribut, scission, coup, essaimage, économie, religion…), endurance, save/load, et les **goldens de déterminisme**.
- **Guard de déterminisme** : `tests/determinism_guard.py` rejoue un run seedé et compare le hash SHA256 de toute la séquence à un *golden* versionné, sur 3 gabarits (BASE Âge Bois, CIV Âge Acier, PROD 220×160). Un refactor « sans changer le comportement » doit laisser le hash intact ; un changement volontaire le **regolde**.
- **Kill-switches d'imputation** : chaque bloc de civilisation s'éteint par une variable d'environnement (`SOCIETY_OFF`, `JOBS_OFF`, `WARBEH_OFF`, `POLITICS_OFF`, `RELATIONS_OFF`, `WAR2_OFF`, `UNREST_OFF`, `SWARM_OFF`…) qui restaure **exactement** le hash d'avant le bloc — preuve que chaque bloc est isolé et n'a pas de fuite.

---

## Ajouter une espèce

1. `engine/entities.py` → ajouter dans `EntityType` et `SPECS`
2. `server.py` → ajouter dans `sim.populate()`
3. `static/index.html` → ajouter dans `SPECIES`

Exemple (cerf) :

```python
# entities.py
EntityType.DEER: EntitySpec(
    name="deer", color="#c0a060",
    is_predator=False, is_prey=True, prey_types=[],
    max_age=2000, hunger_rate=0.07, max_hunger=100,
    eat_amount=20, eat_meat=0, speed=1.1, vision=7,
    repro_hunger_min=45, repro_cooldown=130, gestation=60, litter_size=(1, 2),
    flee_distance=6,
),
```

```js
// index.html → SPECIES
deer: { label: "Cerfs", color: "#c0a060" },
```

---

## Crédits

- Sprites 2D : **MiniWorldSprites** par [Shade & octoshrimpy](https://merchant-shade.itch.io/16x16-mini-world-sprites) (Itch.io, CC0 1.0)
- Planches par métier : dérivées de **Ninja Adventure** par [Pixel-boy](https://pixel-boy.itch.io/ninja-adventure-asset-pack) (CC0 1.0)

---

## Licence

MIT
