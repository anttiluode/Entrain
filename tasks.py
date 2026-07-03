"""
tasks.py — the worlds the network must wire itself to.

T1 SPECTRUM   : classes are secret subsets of a hidden frequency dictionary.
                The network must discover which ears to grow.
T2 TIMBRE     : classes share the SAME fundamental and differ only in harmonic
                profile (saw-ish / square-ish / pure / formant-ish). The world
                where the fractal comb prior should earn its keep — tested, not
                assumed.
T3 TONGUE     : the enslavement map — detuning x coupling -> does the child lock
                to the master? Measured Arnold tongue.
T4 RING MEMORY: a loop of self-sustaining nodes is imprinted with a traveling
                wave (clockwise or counter). After input stops, does the loop
                remember the direction? Amplitude memory is trivial for a limit
                cycle; DIRECTION lives only in the loop's relative phases —
                that's what loop topology buys.
"""

import numpy as np
from core import Net, make_signal, TWO_PI


# ---------------- T1: hidden spectrum ----------------
def spectrum_task(rng, n_classes=4, dict_size=6, per_class=3,
                  f_lo=4.0, f_hi=40.0, min_sep=2.5):
    # jittered grid: guaranteed to terminate at any density (a sequential
    # rejection sampler deadlocks near capacity — measured, see README kill list).
    # As dict_size grows the spacing shrinks: density IS the difficulty knob.
    grid = np.linspace(f_lo, f_hi, dict_size)
    spacing = grid[1] - grid[0] if dict_size > 1 else (f_hi - f_lo)
    jit = min(0.35 * spacing, 0.5 * max(spacing - min_sep, 0) + 0.15 * spacing)
    freqs = sorted(float(f + rng.uniform(-jit, jit)) for f in grid)
    classes = []
    while len(classes) < n_classes:
        sub = tuple(sorted(rng.choice(dict_size, per_class, replace=False)))
        if sub not in classes:
            classes.append(sub)
    return dict(freqs=freqs, classes=classes)


def spectrum_batch(task, rng, batch=32, T=800, fs=400.0, noise=0.4):
    S = np.zeros((batch, T))
    y = rng.integers(0, len(task["classes"]), batch)
    for i in range(batch):
        idx = task["classes"][y[i]]
        fl = [task["freqs"][j] for j in idx]
        S[i] = make_signal(fl, [1.0] * len(fl), T, fs, noise=noise, rng=rng)
    return S, y


# ---------------- T2: harmonic timbre ----------------
TIMBRES = {  # amplitude of harmonics 1..4 of a shared fundamental
    "saw-ish":     [1.0, 0.55, 0.30, 0.18],
    "square-ish":  [1.0, 0.00, 0.45, 0.00],
    "pure":        [1.0, 0.00, 0.00, 0.00],
    "formant-ish": [0.35, 1.0, 0.35, 0.00],
}


def timbre_batch(rng, f0=5.0, batch=32, T=800, fs=400.0, noise=0.35):
    names = list(TIMBRES)
    y = rng.integers(0, len(names), batch)
    S = np.zeros((batch, T))
    for i in range(batch):
        amps = TIMBRES[names[y[i]]]
        fl = [f0 * (h + 1) for h in range(4)]
        S[i] = make_signal(fl, amps, T, fs, noise=noise, rng=rng)
    return S, y


# ---------------- T3: Arnold tongue ----------------
def tongue_point(detune, w, fs=400.0, f0=12.0, steps=4000, tail=1200):
    net = Net(fs)
    m = net.add_node(f0, mu=1.0, b=0.0)
    net.add_node(f0 + detune, mu=1.0, b=0.0, parent=m, w_down=w)
    r = net.run(np.zeros((1, steps)), record=True)
    f = net.instantaneous_freq(r["traj"], tail=tail)
    return bool(abs(f[0] - f[1]) < 0.1)


def tongue_map(detunes, couplings, fs=400.0):
    return np.array([[tongue_point(d, w, fs) for d in detunes] for w in couplings])


# ---------------- T4: ring memory (traveling-wave direction) ----------------
def ring_memory(direction, n_ring=8, w=3.0, f0=8.0, fs=400.0,
                imprint_steps=1600, hold_steps=4000, closed=True, seed=0):
    """Imprint a traveling wave (direction=+1 cw / -1 ccw) on a loop of
    self-sustaining nodes via per-node phase-delayed drive; then release and
    measure the winding sign over time. Returns dict with winding trace."""
    rng = np.random.default_rng(seed)
    net = Net(fs)
    for k in range(n_ring):
        net.add_node(f0, mu=0.6, b=0.0)
    for k in range(n_ring):
        j = (k + 1) % n_ring
        if closed or j != 0:
            net.W[j, k] = w
            net.W[k, j] = w
    dt = 1.0 / fs
    z = np.full((1, n_ring), 1e-3, complex)
    iw = 1j * TWO_PI * net.f
    t = 0.0
    winding = []

    def wind(zrow):
        ph = np.angle(zrow)
        d = np.angle(np.exp(1j * (np.roll(ph, -1) - ph)))
        return float(np.sum(d) / TWO_PI)

    for step in range(imprint_steps + hold_steps):
        drive = np.zeros(n_ring)
        if step < imprint_steps:
            for k in range(n_ring):
                drive[k] = 1.2 * np.sin(TWO_PI * f0 * t + direction * TWO_PI * k / n_ring)
        lin = net.mu + iw - np.abs(z) ** 2
        z = z * np.exp(lin * dt) + dt * (z @ net.W.T + drive[None, :])
        t += dt
        if step % 40 == 0:
            winding.append(wind(z[0]))
    return dict(winding=np.array(winding), imprint_end=imprint_steps // 40)
