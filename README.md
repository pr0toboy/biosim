# BioSim — Simulateur de vie

Simulateur de vie style WorldBox, observable en temps réel via navigateur. Aucune interaction utilisateur — la simulation tourne seule et évolue de manière autonome.

**Stack :** Python · FastAPI + uvicorn · NumPy · HTML Canvas 2D + WebSocket

---

## Fonctionnalités

- Monde procédural (fBm + domain warping) avec 6 biomes : eau, rivière, prairie, forêt, désert, montagne
- 9 espèces animales avec traits héréditaires et mutations
- 4 clans humains avec bâtiments, outils, agriculture et pêche
- Saisons, météo dynamique (pluie, orages, canicules, incendies de forêt)
- Interface web temps réel : zoom/pan, tooltip, graphes de population, log d'événements
- Monitoring système (CPU, RAM, température)

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

[Service]
User=<your_user>
WorkingDirectory=/home/<your_user>/biosim
ExecStart=/usr/bin/python3 server.py
Restart=always
RestartSec=5

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
- 6 biomes : eau, rivière, prairie, forêt, désert, montagne
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

## Clans humains

4 clans (rouge, bleu, vert, violet), chacun centré sur un feu de camp.

### Bâtiments disponibles

| Bâtiment    | Coût (bois/pierre) | Effet                         |
|-------------|--------------------|-------------------------------|
| Feu de camp | — (initial)        | Centre du territoire, chaleur |
| Maison      | 15 bois            | +3 pop max par maison (max 12/clan) |
| Champ de blé | —                 | Production alimentaire        |
| Moulin      | 12 bois + 8 pierre | Transforme blé → pain (65 nourriture) |
| Entrepôt    | 10 bois + 4 pierre | Stockage ressources           |
| Puit        | 8 bois + 6 pierre  | Source d'eau sans se déplacer |

### Outils

| Outil        | Coût           | Effet                              |
|--------------|----------------|------------------------------------|
| Hache bois   | 5 bois stockés | +3 bois/coup (total 5/coup)        |
| Hache pierre | 5 pierre       | +5 bois/coup (total 7/coup)        |
| Pioche bois  | 8 bois         | Permet de miner la pierre          |
| Pioche pierre | 5 pierre      | +1 pierre/coup (total 2/coup)      |
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

- ~6–7 ms/tick sur matériel ARM (Raspberry Pi 5 ou équivalent)
- Cap entités : 5 000
- Monde 220×160 avec NumPy vectorisé (regen, fertilité, arbres)

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

- Sprites 2D : **MiniWorldSprites** par [Cainos](https://cainos.itch.io/pixel-art-top-down-basic) (Itch.io)

---

## Licence

MIT
