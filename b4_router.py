"""
b4_router.py — the router inside the grower, measured. Run: python3 b4_router.py

THE TASK (context conflict): 4 hidden frequencies, 4 classes defined as 2-subsets.
In context A, class c means subset[c]. In context B, the SAME subsets are relabeled
by a derangement (class c means subset[c+1 mod 4]). So identical spectral content
carries different labels depending on context — a context-blind linear readout is
structurally doomed (classes 0 and 2 even share identical mixed-context feature
means). Context is announced only by a pilot tone inside the signal; no label or
context bit ever reaches the physical inference path.

THREE SYSTEMS, equal ear budget, identical growth rule:
  routed     : cue -> master steering -> slave entrainment -> cluster gating
               (router.py; the B2 mechanism doing real work)
  monolithic : all ears always hear ('open'); the ablation that measures what
               routing adds
  oracle     : gates driven by the true context bit; the ceiling that measures
               what the physical router loses to its own dynamics

Also reported: per-trial routing accuracy, and the grown ear tables — routing-aware
growth should plant the SAME frequency twice, once per meaning.
"""

import json
import numpy as np
from core import make_signal
from router import RoutedGrower, F_CUE

SUBSETS = [(0, 1), (2, 3), (0, 2), (1, 3)]


def conflict_task(rng, f_lo=9.0, f_hi=38.0):
    grid = np.linspace(f_lo, f_hi, 4)
    jit = 0.3 * (grid[1] - grid[0])
    return sorted(float(f + rng.uniform(-jit, jit)) for f in grid)


def conflict_batch(freqs, rng, batch=24, T=1200, fs=400.0, noise=0.35):
    S = np.zeros((batch, T))
    y = rng.integers(0, 4, batch)
    ctx = rng.integers(0, 2, batch)
    for i in range(batch):
        sub = SUBSETS[y[i]] if ctx[i] == 0 else SUBSETS[(y[i] + 1) % 4]
        comps = [F_CUE[ctx[i]]] + [freqs[j] for j in sub]
        S[i] = make_signal(comps, [1.0] * len(comps), T, fs, noise=noise, rng=rng)
    return S, y, ctx


def run_system(route, seed, episodes=90, budget=14):
    rng = np.random.default_rng(seed)
    freqs = conflict_task(rng)
    g = RoutedGrower(4, seed=seed, route=route, budget=budget)
    for _ in range(episodes):
        S, y, ctx = conflict_batch(freqs, rng)
        g.step(S, y, true_ctx=ctx)
    S, y, ctx = conflict_batch(freqs, rng, batch=200)
    acc, r = g.evaluate(S, y, true_ctx=ctx)
    route_acc = float(np.mean(r["routed"] == ctx)) if route == "physical" else None
    ears = sorted(zip(np.round(g.net.f, 1), ["AB"[c] for c in g.net.cluster]))
    return dict(acc=acc, n=g.net.n, route_acc=route_acc, ears=ears,
                freqs=np.round(freqs, 1).tolist())


if __name__ == "__main__":
    print("== B4  The router inside the grower (context-conflict task) ==")
    print("   chance = 0.25; context-blind linear readout is structurally capped\n")
    out = {}
    for route, name in (("physical", "ROUTED (cue->master->slaves->gates)"),
                        ("open", "monolithic (no routing, same budget)"),
                        ("oracle", "oracle routing (true context bit)")):
        accs, ns, r_accs = [], [], []
        ears_example = None
        for s in range(4):
            r = run_system(route, seed=s)
            accs.append(r["acc"])
            ns.append(r["n"])
            if r["route_acc"] is not None:
                r_accs.append(r["route_acc"])
            if s == 0:
                ears_example = r
        line = f"  {name:<38} acc {np.mean(accs):.3f} +/- {np.std(accs):.3f} " \
               f"({np.mean(ns):.1f} ears)"
        if r_accs:
            line += f" | routing correct {np.mean(r_accs):.3f}"
        print(line)
        out[route] = dict(acc=float(np.mean(accs)), std=float(np.std(accs)),
                          ears=float(np.mean(ns)),
                          route_acc=float(np.mean(r_accs)) if r_accs else None)
        if route == "physical" and ears_example:
            print(f"     seed-0 hidden freqs {ears_example['freqs']}")
            print(f"     seed-0 grown ears   {ears_example['ears']}")
    with open("b4_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nb4_results.json written.")
