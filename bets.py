"""
bets.py — the three bets from the README, turned into measurements.
Run: python3 bets.py   (~3-4 min, numpy only). Writes bets_results.json.

B1 SCALING : does surprise-gated growth keep its edge over same-size random
             placement as class count and dictionary size scale up?
B2 ROUTING : can a master oscillator SELECT which slave computes — entrain one
             and not the other — and does the Arnold tongue measured in T3
             predict where that routing resolution breaks down?
B3 PHASE   : can class information carried ONLY in relative phase (identical
             amplitude spectra) be read out — and do amplitude features fail
             at chance, as they must?
"""

import json
import numpy as np
from core import Net, TWO_PI
from grow import Grower
from tasks import spectrum_task, spectrum_batch

RESULTS = {}


# ====================================================================== B1
def fixed_bank_eval(freqs, task, rng, n_classes, episodes=40):
    g = Grower(n_classes, seed=0)
    g.net = Net(g.fs)
    for f in freqs:
        g.net.add_node(f, mu=-0.5, b=1.0)
    g.gate.trip = 1e9
    for _ in range(episodes):
        S, y = spectrum_batch(task, rng)
        g.step(S, y)
    St, yt = spectrum_batch(task, rng, batch=200)
    return g.evaluate(St, yt)


def bet_scaling(seeds=range(3)):
    print("== B1  Scaling: classes x dictionary up, per-class complexity fixed ==")
    configs = [(4, 6), (6, 10), (8, 14), (10, 18)]
    out = []
    for n_cls, dict_size in configs:
        accs, rands, oracles, ns = [], [], [], []
        for s in seeds:
            rng = np.random.default_rng(s)
            task = spectrum_task(rng, n_classes=n_cls, dict_size=dict_size,
                                 per_class=3, min_sep=2.0)
            g = Grower(n_cls, seed=s, budget=dict_size + 4)
            for _ in range(120):
                S, y = spectrum_batch(task, rng)
                g.step(S, y)
            St, yt = spectrum_batch(task, rng, batch=200)
            accs.append(g.evaluate(St, yt))
            ns.append(g.net.n)
            rands.append(fixed_bank_eval(
                np.random.default_rng(1000 + s).uniform(3, 45, g.net.n),
                task, rng, n_cls))
            oracles.append(fixed_bank_eval(task["freqs"], task, rng, n_cls))
        row = dict(classes=n_cls, dict=dict_size,
                   grown=float(np.mean(accs)), nodes=float(np.mean(ns)),
                   random=float(np.mean(rands)), oracle=float(np.mean(oracles)))
        out.append(row)
        print(f"  {n_cls:>2} classes / {dict_size:>2} freqs: "
              f"grown {row['grown']:.2f} ({row['nodes']:.1f} nodes) | "
              f"random-same-size {row['random']:.2f} | oracle {row['oracle']:.2f}")
    RESULTS["B1"] = out


# ====================================================================== B2
def _route_trial(master_f, slave_fs, w=6.0, fs=400.0, steps=3200, tail=1000,
                 seed=0, switch_to=None):
    """Master + k slaves, equal coupling. Returns per-slave locked flags
    (and, if switch_to is given, re-lock latency in master cycles)."""
    rng = np.random.default_rng(seed)
    net = Net(fs)
    m = net.add_node(master_f, mu=1.0, b=0.0)
    for f in slave_fs:
        net.add_node(f, mu=1.0, b=0.0, parent=m, w_down=w)
    n = net.n
    z = (0.1 * rng.standard_normal((1, n)) +
         0.1j * rng.standard_normal((1, n)) + 1e-3)
    dt = 1.0 / fs
    half = steps // 2
    ph = np.zeros((steps, n))
    for t in range(steps):
        if switch_to is not None and t == half:
            net.f[0] = switch_to
        iw = 1j * TWO_PI * net.f
        lin = net.mu + iw - np.abs(z) ** 2
        z = z * np.exp(lin * dt) + dt * (z @ net.W.T)
        ph[t] = np.angle(z[0])

    def locked(a, b, t0, t1):
        d = np.unwrap(ph[t0:t1, a] - ph[t0:t1, b])
        f_rel = abs(d[-1] - d[0]) / ((t1 - t0) / fs) / TWO_PI
        return f_rel < 0.1

    if switch_to is None:
        return [locked(0, k + 1, steps - tail, steps) for k in range(len(slave_fs))]
    # switching latency: first window after the switch where the new target locks
    win = int(0.4 * fs)
    for t0 in range(half, steps - win, win // 2):
        if locked(0, 2, t0, t0 + win) and not locked(0, 1, t0, t0 + win):
            return (t0 - half) / fs * switch_to     # latency in master cycles
    return np.nan


def bet_routing():
    print("\n== B2  Routing by entrainment: master selects which slave locks ==")
    # basic routing: slaves at 12 and 18 Hz, master targets one of them
    ok = 0
    trials = 20
    for s in range(trials):
        target = s % 2
        master = (12.0, 18.0)[target]
        lock = _route_trial(master, [12.0, 18.0], seed=s)
        if lock[target] and not lock[1 - target]:
            ok += 1
    print(f"  slaves 6 Hz apart, coupling 6: correct exclusive routing {ok}/{trials}")
    # resolution limit: shrink slave separation; T3's tongue at w=6 (~1 Hz) should
    # predict the breakdown — when both slaves sit inside the master's tongue,
    # both lock and routing turns ambiguous.
    res = {}
    for sep in (6.0, 3.0, 1.5, 0.75):
        okk = 0
        for s in range(10):
            target = s % 2
            fa, fb = 15.0 - sep / 2, 15.0 + sep / 2
            master = (fa, fb)[target]
            lock = _route_trial(master, [fa, fb], seed=100 + s)
            if lock[target] and not lock[1 - target]:
                okk += 1
        res[sep] = okk
        print(f"  separation {sep:>4.2f} Hz: exclusive routing {okk}/10")
    # switching latency
    lats = [_route_trial(12.0, [12.0, 18.0], seed=200 + s, switch_to=18.0)
            for s in range(8)]
    lats = [l for l in lats if np.isfinite(l)]
    print(f"  switch 12->18 Hz: re-routed in {np.mean(lats):.1f} master cycles "
          f"(n={len(lats)}/8)")
    RESULTS["B2"] = dict(basic=f"{ok}/{trials}", resolution=res,
                         switch_cycles=float(np.mean(lats)) if lats else None)


# ====================================================================== B3
def phase_batch(rng, batch=32, T=1200, fs=400.0, f0=8.0, noise=0.3):
    """Two classes with IDENTICAL amplitude spectra: fundamental + 2nd harmonic,
    harmonic phase tied to the fundamental, offset 0 (class 0) or pi (class 1).
    A random global time-shift per signal removes every absolute-phase cue."""
    y = rng.integers(0, 2, batch)
    t = np.arange(T) / fs
    S = np.zeros((batch, T))
    for i in range(batch):
        psi = rng.uniform(0, TWO_PI)
        d = np.pi * y[i]
        S[i] = (np.sin(TWO_PI * f0 * t + psi)
                + 0.8 * np.sin(2 * (TWO_PI * f0 * t + psi) + d)
                + rng.normal(0, noise, T))
    return S, y


def phase_features(net, S, i_f, i_2f):
    """Amplitude feats + the quadratic phase-coupling invariant of the (f, 2f)
    pair: C = <z_2f * conj(z_f)^2> — its angle is (theta_2f - 2*theta_f), which
    is exactly the class phase offset, immune to the global time-shift."""
    batch, T = S.shape
    dt = 1.0 / net.fs
    n = net.n
    z = np.zeros((batch, n), complex) + 1e-3
    iw = 1j * TWO_PI * net.f
    acc_amp = np.zeros((batch, n))
    acc_C = np.zeros(batch, complex)
    half = T // 2
    for t in range(T):
        lin = net.mu + iw - np.abs(z) ** 2
        z = z * np.exp(lin * dt) + dt * (z @ net.W.T + S[:, t:t + 1] * net.B)
        if t >= half:
            acc_amp += np.abs(z)
            acc_C += z[:, i_2f] * np.conj(z[:, i_f]) ** 2
    amp = acc_amp / (T - half)
    C = acc_C / (T - half)
    C = C / (np.abs(C) + 1e-12)
    return amp, np.stack([C.real, C.imag], axis=1)


def _nearest_mean_acc(Xtr, ytr, Xte, yte):
    m = np.stack([Xtr[ytr == c].mean(axis=0) for c in (0, 1)])
    sc = Xtr.std(axis=0) + 1e-9
    d = (((Xte[:, None, :] - m[None]) / sc) ** 2).sum(axis=2)
    return float(np.mean(np.argmin(d, axis=1) == yte))


def bet_phase(seeds=range(5)):
    print("\n== B3  Phase-borne information (identical amplitude spectra) ==")
    a_amp, a_ph = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        net = Net(400.0)
        i1 = net.add_node(8.0, mu=-0.5, b=1.0)
        i2 = net.add_node(16.0, mu=-0.5, b=1.0)
        Str, ytr = phase_batch(rng, batch=64)
        Ste, yte = phase_batch(rng, batch=200)
        Atr, Ptr = phase_features(net, Str, i1, i2)
        Ate, Pte = phase_features(net, Ste, i1, i2)
        a_amp.append(_nearest_mean_acc(Atr, ytr, Ate, yte))
        a_ph.append(_nearest_mean_acc(np.hstack([Atr, Ptr]), ytr,
                                      np.hstack([Ate, Pte]), yte))
    print(f"  amplitude features only : {np.mean(a_amp):.3f} +/- {np.std(a_amp):.3f} "
          f"(must be ~chance 0.5, or the task leaks)")
    print(f"  + phase coupling C(f,2f): {np.mean(a_ph):.3f} +/- {np.std(a_ph):.3f}")
    RESULTS["B3"] = dict(amplitude=float(np.mean(a_amp)),
                         with_phase=float(np.mean(a_ph)))


if __name__ == "__main__":
    bet_scaling()
    bet_routing()
    bet_phase()
    with open("bets_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nbets_results.json written. README bet ledger updates from these only.")
