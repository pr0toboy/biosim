"""Baseline d'observabilité (I0) — à capturer AVANT tout changement de
comportement du mouvement. Sert de référence aux gates du pathfinding (I1-I4) :
taux de blocage (_stuck_resets), distribution du temps de step, stabilité de pop.

Usage :
  python3 tests/nav_baseline.py            # baseline complète (spec du plan)
  python3 tests/nav_baseline.py --quick    # version rapide (validation du harnais)
Écrit tests/baselines/baseline_I0.json (--quick : baseline_I0_quick.json).
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.world import World
from engine.simulation import Simulation


def run(width, height, seed, ticks, sample_every=1000):
    world = World(width=width, height=height, seed=seed)
    sim = Simulation(world)
    sim.populate()
    pops = []
    resets_at = []
    for t in range(1, ticks + 1):
        sim.step()
        if t % sample_every == 0:
            pops.append(len(sim.entities))
            resets_at.append(int(world._stuck_resets))
    m = sim.metrics()
    entity_ticks = sum(pops) * sample_every if pops else max(1, ticks) * len(sim.entities)
    total_resets = int(world._stuck_resets)
    return {
        "world": [width, height], "seed": seed, "ticks": ticks,
        "pop_final": len(sim.entities),
        "pop_min": min(pops) if pops else len(sim.entities),
        "pop_max": max(pops) if pops else len(sim.entities),
        "pop_samples": pops,
        "stuck_resets_total": total_resets,
        "stuck_resets_per_1k_ticks": round(total_resets / (ticks / 1000.0), 1),
        "stuck_resets_per_Mentity_tick": round(total_resets / (entity_ticks / 1e6), 1) if entity_ticks else 0.0,
        "step_ms": m["step_ms"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        runs = [run(140, 100, s, 300) for s in (11, 22)]
        runs.append(run(220, 160, 999, 400))
        out = "tests/baselines/baseline_I0_quick.json"
    else:
        # Spec du plan : 5 seeds × 2000 ticks (140×100) + 1×10000 ticks (220×160)
        runs = [run(140, 100, s, 2000) for s in (101, 202, 303, 404, 505)]
        runs.append(run(220, 160, 700, 10000))
        out = "tests/baselines/baseline_I0.json"

    payload = {"increment": "I0", "runs": runs}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"écrit {out}")
    for r in runs:
        print(f"  {r['world']} seed={r['seed']} {r['ticks']}t : "
              f"pop {r['pop_min']}-{r['pop_max']} (fin {r['pop_final']}), "
              f"stuck/1k={r['stuck_resets_per_1k_ticks']}, "
              f"step avg={r['step_ms']['avg']}ms p95={r['step_ms']['p95']}ms max={r['step_ms']['max']}ms")


if __name__ == "__main__":
    main()
