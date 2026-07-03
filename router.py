"""
router.py — the router moves INSIDE the grower. B2 made this an engineering step:
lock is reliable at 6 Hz slave separation, switching takes ~6 master cycles, so a
trial a few seconds long gives the physics room to route before features accumulate.

The machine, end to end, with no label ever touching the inference path:

  input s(t) --> CUE resonators (damped, at the two pilot tones)
                    |  amplitudes a_A, a_B                (physical state)
                    v
             MASTER oscillator, frequency steered:        (control law, documented)
                    f_m(t) = 12*wA + 18*wB,  wA = a_A^2/(a_A^2+a_B^2)
                    |
                    v   entrainment (the B2 mechanism, coupling 6)
             GATE SLAVES at 12 Hz (cluster A) and 18 Hz (cluster B)
                    |  lock coherence c_k(t) = |EMA exp(i(th_m - th_k))|
                    v
             DETECTOR clusters: input coupling of cluster k's ears
             is multiplied by c_k(t)^2 — the unenslaved cluster is DEAF.

Growth (surprise-gated, as before) now also decides WHICH CLUSTER to grow into:
misclassified signals vote by their cue-resonator amplitudes, and the new ear is
planted in the winning context's cluster, covered-frequency check done PER CLUSTER —
so the same frequency may honestly be grown twice, once per meaning.

The claim to measure (b4_router.py): on a context-conflict task the routed grower
beats a context-blind monolithic grower at equal budget, and approaches an oracle
that routes by the true label. If it doesn't, that goes in the README too.
"""

import numpy as np
from clutch import MagnitudeGate
from core import residual_peak, TWO_PI

F_SLAVE = (12.0, 18.0)       # cluster gate frequencies (B2-proven separation)
F_CUE = (3.5, 44.0)          # pilot tones announcing context A / B
W_MASTER = 6.0               # master->slave coupling (B2-proven)


class RoutedNet:
    """Fixed plumbing (cues, master, slaves) + growable, cluster-tagged ears."""

    def __init__(self, fs=400.0, gate_sharpness=2.0):
        self.fs = fs
        self.p = gate_sharpness
        # growable detector arrays
        self.f = np.zeros(0)
        self.cluster = np.zeros(0, int)      # 0 = context A ears, 1 = context B ears
        # fixed plumbing state initialized per run

    def add_ear(self, f, cluster):
        self.f = np.append(self.f, float(f))
        self.cluster = np.append(self.cluster, int(cluster))

    @property
    def n(self):
        return len(self.f)

    def run(self, S, route="physical", true_ctx=None):
        """Integrate. route: 'physical' (cue->master->slaves->gates),
        'open' (all ears always hear; the monolithic ablation),
        'oracle' (gates driven by the true context bit; the ceiling).
        Returns feats (batch, n_ears) and routing diagnostics."""
        batch, T = S.shape
        dt = 1.0 / self.fs
        mu_det, mu_osc = -0.5, 1.0
        # state: cues (2), master (1), slaves (2), ears (n)
        zc = np.zeros((batch, 2), complex) + 1e-3
        zm = np.zeros((batch, 1), complex) + 1e-3 + 0.1
        zs = np.zeros((batch, 2), complex) + 1e-3 + 0.1
        ze = np.zeros((batch, self.n), complex) + 1e-3
        coher = np.zeros((batch, 2))          # EMA phasor magnitude per slave
        ph_ema = np.zeros((batch, 2), complex)
        acc = np.zeros((batch, self.n))
        acc_c = np.zeros((batch, 2))
        half = 2 * T // 3
        a_ema = np.zeros((batch, 2))
        iwc = 1j * TWO_PI * np.array(F_CUE)
        iws = 1j * TWO_PI * np.array(F_SLAVE)
        iwe = 1j * TWO_PI * self.f
        alpha = dt / 0.15                     # ~150 ms EMA for cue amps & coherence
        for t in range(T):
            s = S[:, t:t + 1]
            # cues
            zc = zc * np.exp((mu_det + iwc - np.abs(zc) ** 2) * dt) + dt * s
            a_ema = (1 - alpha) * a_ema + alpha * np.abs(zc)
            # master frequency steered by cue amplitudes
            a2 = a_ema ** 2
            wA = a2[:, 0:1] / (a2.sum(axis=1, keepdims=True) + 1e-9)
            fm = F_SLAVE[0] * wA + F_SLAVE[1] * (1 - wA)
            zm = zm * np.exp((mu_osc + 1j * TWO_PI * fm - np.abs(zm) ** 2) * dt)
            # slaves entrained by master
            zs = zs * np.exp((mu_osc + iws - np.abs(zs) ** 2) * dt) \
                + dt * W_MASTER * zm
            # lock coherence per slave
            rel = np.exp(1j * (np.angle(zm) - np.angle(zs)))
            ph_ema = (1 - alpha) * ph_ema + alpha * rel
            coher = np.abs(ph_ema)
            # gates
            if route == "physical":
                g = coher ** self.p
            elif route == "open":
                g = np.ones((batch, 2))
            else:                             # oracle
                g = np.zeros((batch, 2))
                g[np.arange(batch), true_ctx] = 1.0
            gain = g[:, self.cluster]         # (batch, n_ears)
            # ears: input coupling multiplied by their cluster's gate
            ze = ze * np.exp((mu_det + iwe - np.abs(ze) ** 2) * dt) \
                + dt * gain * s
            if t >= half:
                acc += np.abs(ze)
                acc_c += coher
        feats = acc / (T - half)
        return dict(feats=feats, coher=acc_c / (T - half),
                    routed=np.argmax(acc_c, axis=1))


class RoutedGrower:
    """The Grower with the router inside. Same Clutch gate, same residual-peak
    placement — but growth is planted into the cluster whose context the failing
    signals belong to (voted by cue-resonator amplitude, never by the label)."""

    def __init__(self, n_classes, fs=400.0, budget=16, seed=0, route="physical",
                 f_range=(6.0, 40.0)):
        self.rng = np.random.default_rng(seed)
        self.net = RoutedNet(fs)
        self.net.add_ear(self.rng.uniform(*f_range), 0)
        self.net.add_ear(self.rng.uniform(*f_range), 1)
        self.route = route
        self.budget = budget
        self.f_range = f_range
        self.gate = MagnitudeGate(gain=3.0, leak=0.35, trip=2.5)
        self.n_classes = n_classes
        self.sum = None
        self.cnt = np.zeros(n_classes)
        self.history = []

    def _resize(self):
        if self.sum is None or self.sum.shape[1] != self.net.n:
            self.sum = np.zeros((self.n_classes, self.net.n))
            self.cnt = np.zeros(self.n_classes)

    def predict(self, feats):
        m = self.sum / np.maximum(self.cnt[:, None], 1)
        sc = feats.std(axis=0) + 1e-9
        d = (((feats[:, None, :] - m[None]) / sc) ** 2).sum(axis=2)
        return np.argmin(d, axis=1)

    def step(self, S, labels, true_ctx=None):
        self._resize()
        r = self.net.run(S, route=self.route, true_ctx=true_ctx)
        feats = r["feats"]
        pred = self.predict(feats) if self.cnt.min() > 0 else \
            self.rng.integers(0, self.n_classes, len(labels))
        err = float(np.mean(pred != labels))
        for c in range(self.n_classes):
            mask = labels == c
            if mask.any():
                self.sum[c] = 0.9 * self.sum[c] + feats[mask].sum(axis=0)
                self.cnt[c] = 0.9 * self.cnt[c] + mask.sum()
        grew = None
        if self.gate.update(err) and self.net.n < self.budget:
            grew = self._grow(S, labels, pred, r)
            self.gate.clear()
            self.sum = None
        self.history.append(dict(err=err, n=self.net.n, grew=grew))
        return err

    def _grow(self, S, labels, pred, r):
        wrong = pred != labels
        pool = S[wrong] if wrong.sum() >= 4 else S
        pool_route = r["routed"][wrong] if wrong.sum() >= 4 else r["routed"]
        # context vote by physical routing state of the failing signals
        cl = int(np.round(pool_route.mean())) if self.route != "open" else \
            int(self.rng.integers(0, 2))
        covered = list(self.net.f[self.net.cluster == cl]) + list(F_CUE)
        if self.route == "open":              # monolithic: one shared pool of ears
            covered = list(self.net.f) + list(F_CUE)
        f_new = residual_peak(pool, self.net.fs, covered=covered,
                              min_f=self.f_range[0], max_f=self.f_range[1],
                              guard=1.2)
        if f_new is None:
            return None
        self.net.add_ear(f_new, cl)
        return (round(f_new, 1), "AB"[cl])

    def evaluate(self, S, labels, true_ctx=None):
        r = self.net.run(S, route=self.route, true_ctx=true_ctx)
        return float(np.mean(self.predict(r["feats"]) == labels)), r
