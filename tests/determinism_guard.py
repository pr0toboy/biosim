"""Empreinte déterministe d'un run seedé : hash de toute la séquence step().
Lancé en processus frais → _next_id repart de 0, random/np.random seedés par World."""
import sys, json, hashlib
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.world import World
from engine.simulation import Simulation

SEED = 424242
TICKS = 500

world = World(width=140, height=100, seed=SEED)
sim = Simulation(world)
sim.populate()

h = hashlib.sha256()
for _ in range(TICKS):
    data = sim.step()
    # sérialisation stable (tri des clés) de tout ce que step produit
    h.update(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())

print(h.hexdigest(), f"(pop finale {len(sim.entities)})")
