"""
benchmark.py — the honest measurement. Run: python3 benchmark.py
Every number in the README comes from here. ~3-5 minutes, numpy only.
"""

import json
import numpy as np
from grow import Grower
from core import Net
from tasks import (spectrum_task, spectrum_batch, timbre_batch, TIMBRES,
                   tongue_map, ring_memory)

RESULTS = {}


def fixed_bank_eval(freqs, task, rng, episodes=30):
    """Baseline: a FIXED bank (no growth) trained with the same online protocol."""
    g = Grower(4, seed=0)
    g.net = Net(g.fs)
    for f in freqs:
        g.net.add_node(f, mu=-0.5, b=1.0)
    g.gate.trip = 1e9                        # growth disabled
    for _ in range(episodes):
        S, y = spectrum_batch(task, rng)
        g.step(S, y)
    St, yt = spectrum_batch(task, rng, batch=200)
    return g.evaluate(St, yt)


def bench_spectrum(seeds=range(8)):
    print("\n== T1  Hidden-spectrum classification (4 classes, 6 secret freqs) ==")
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        task = spectrum_task(rng)
        g = Grower(4, seed=s)
        for _ in range(80):
            S, y = spectrum_batch(task, rng)
            g.step(S, y)
        St, yt = spectrum_batch(task, rng, batch=200)
        acc = g.evaluate(St, yt)
        n = g.net.n
        acc_rand = fixed_bank_eval(np.random.default_rng(1000 + s)
                                   .uniform(3, 45, n), task, rng)
        acc_oracle = fixed_bank_eval(task["freqs"], task, rng)
        rows.append((acc, n, acc_rand, acc_oracle))
        print(f"  seed {s}: grown {acc:.2f} ({n} nodes) | random-{n} {acc_rand:.2f} "
              f"| oracle-6 {acc_oracle:.2f}")
    a = np.array(rows)
    print(f"  MEAN : grown {a[:,0].mean():.3f} ({a[:,1].mean():.1f} nodes) | "
          f"same-size random {a[:,2].mean():.3f} | oracle {a[:,3].mean():.3f}")
    RESULTS["T1"] = dict(grown=a[:, 0].mean(), nodes=a[:, 1].mean(),
                         random=a[:, 2].mean(), oracle=a[:, 3].mean())


def bench_timbre(seeds=range(6)):
    print("\n== T2  Harmonic timbre (same fundamental, 4 harmonic profiles) ==")
    print("     flat growth vs fractal comb growth (grow f -> also 2f, 3f)")
    accs = {"flat": [], "comb": []}
    trips = {"flat": [], "comb": []}
    for s in seeds:
        for mode in ("flat", "comb"):
            rng = np.random.default_rng(s)
            g = Grower(len(TIMBRES), seed=s, mode=mode, budget=10,
                       f_range=(3.0, 24.0))
            for _ in range(60):
                S, y = timbre_batch(rng)
                g.step(S, y)
            St, yt = timbre_batch(rng, batch=200)
            accs[mode].append(g.evaluate(St, yt))
            trips[mode].append(sum(1 for h in g.history if h["grew"]))
    for mode in ("flat", "comb"):
        print(f"  {mode:>4}: acc {np.mean(accs[mode]):.3f} +/- {np.std(accs[mode]):.3f} "
              f"| growth events {np.mean(trips[mode]):.1f}")
    RESULTS["T2"] = {m: dict(acc=float(np.mean(accs[m])),
                             trips=float(np.mean(trips[m]))) for m in accs}


def bench_tongue():
    print("\n== T3  Enslavement map (Arnold tongue), measured not asserted ==")
    dets = np.linspace(0.1, 2.5, 9)
    cps = np.linspace(1, 16, 9)
    m = tongue_map(dets, cps)
    for row in m[::-1]:
        print("   " + "".join("#" if v else "." for v in row))
    print("   detuning 0.1 -> 2.5 Hz across; coupling 16 -> 1 downward. "
          "# = child locked to master.")
    RESULTS["T3"] = dict(locked_fraction=float(m.mean()))


def bench_ring():
    print("\n== T4  Loop memory: does a ring remember a direction it was shown? ==")
    for closed, name in ((True, "closed ring"), (False, "open chain")):
        ok = 0
        for s in range(6):
            for d in (+1, -1):
                r = ring_memory(d, closed=closed, seed=s)
                if abs(r["winding"][-1] - d) < 0.2:
                    ok += 1
        print(f"  {name}: direction retained after release {ok}/12")
        RESULTS.setdefault("T4", {})[name] = f"{ok}/12"


if __name__ == "__main__":
    bench_spectrum()
    bench_timbre()
    bench_tongue()
    bench_ring()
    with open("results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults.json written. These are the only numbers the README may cite.")
