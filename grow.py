"""
grow.py — the self-wiring rule. This is the repo's actual claim.

A Grower starts with ONE randomly-tuned ear and a task it is bad at. Then:

  1. Each episode it hears a labelled batch, classifies by nearest class-mean over
     its resonance features, and computes its error rate.
  2. The error feeds a Clutch MagnitudeGate (the same ~40-line leaky integrator that
     gates compute in the companion Spaces). While the network keeps failing,
     surprise accumulates; when the gate trips, the network GROWS.
  3. Growth is targeted, not random: the new node is placed at the strongest spectral
     peak of the *misclassified* signals that no existing node covers
     (core.residual_peak). It literally grows an ear for what it failed to hear.
     In "comb" mode it sprouts a fractal harmonic branch (f, 2f, 3f) instead of a
     single node — a structural prior for harmonic-rich worlds.
  4. Nodes that never help separate classes (low F-ratio between/within class
     variance) are pruned. Wiring is earned, not kept.

What this is NOT: gradient descent, and not magic. It is structure search gated by
surprise. benchmark.py measures where that beats a same-size random bank and where
it merely matches an oracle.
"""

import numpy as np
from clutch import MagnitudeGate
from core import Net, residual_peak


class Grower:
    def __init__(self, n_classes, fs=400.0, budget=14, mode="flat", seed=0,
                 f_range=(3.0, 45.0), gate=None):
        self.rng = np.random.default_rng(seed)
        self.fs = fs
        self.budget = budget
        self.mode = mode                     # "flat" | "comb"
        self.f_range = f_range
        self.net = Net(fs)
        self.net.add_node(self.rng.uniform(*f_range), mu=-0.5, b=1.0)
        self.gate = gate or MagnitudeGate(gain=3.0, leak=0.35, trip=2.5)
        self.n_classes = n_classes
        self.sum = None                      # per-class feature sums
        self.cnt = np.zeros(n_classes)
        self.history = []                    # per-episode dict: err, n, grew, pruned
        self.feat_log = []                   # (feats, labels) ring for F-ratio
        self.episode = 0

    # ---------------- classifier ----------------
    def _resize(self):
        n = self.net.n
        if self.sum is None:
            self.sum = np.zeros((self.n_classes, n))
        elif self.sum.shape[1] != n:
            s = np.zeros((self.n_classes, n))
            k = min(n, self.sum.shape[1])
            s[:, :k] = self.sum[:, :k]
            self.sum = s

    def _means(self):
        return self.sum / np.maximum(self.cnt[:, None], 1)

    def predict(self, feats):
        m = self._means()
        scale = feats.std(axis=0) + 1e-9
        d = (((feats[:, None, :] - m[None]) / scale) ** 2).sum(axis=2)
        return np.argmin(d, axis=1)

    # ---------------- one training episode ----------------
    def step(self, signals, labels):
        self.episode += 1
        self._resize()
        feats = self.net.run(signals)["feats"]
        pred = self.predict(feats) if self.cnt.min() > 0 else \
            self.rng.integers(0, self.n_classes, len(labels))
        err = float(np.mean(pred != labels))
        # update running class means (after predicting: honest online protocol)
        for c in range(self.n_classes):
            mask = labels == c
            if mask.any():
                self.sum[c, :] = 0.9 * self.sum[c, :] + feats[mask].sum(axis=0)
                self.cnt[c] = 0.9 * self.cnt[c] + mask.sum()
        self.feat_log.append((feats, labels))
        if len(self.feat_log) > 6:
            self.feat_log.pop(0)

        grew, pruned = None, None
        if self.gate.update(err) and self.net.n < self.budget:
            grew = self._grow(signals, labels, pred)
            self.gate.clear()
            self.sum = None                  # feature space changed: relearn means
            self.cnt = np.zeros(self.n_classes)
            self.feat_log = []
        elif self.episode > 12 and self.net.n > 3:
            pruned = self._prune()
        self.history.append(dict(err=err, n=self.net.n, grew=grew, pruned=pruned))
        return err

    def _grow(self, signals, labels, pred):
        wrong = pred != labels
        pool = signals[wrong] if wrong.sum() >= 4 else signals
        f_new = residual_peak(pool, self.fs, covered=list(self.net.f),
                              min_f=self.f_range[0], max_f=self.f_range[1], guard=1.2)
        if f_new is None:
            return None
        added = [f_new]
        parent = int(np.argmin(np.abs(self.net.f - f_new)))
        self.net.add_node(f_new, mu=-0.5, b=1.0, parent=parent)
        if self.mode == "comb":              # fractal harmonic branch
            base = len(self.net.f) - 1
            for h in (2.0, 3.0):
                fh = f_new * h
                if fh < self.f_range[1] and np.abs(self.net.f - fh).min() > 1.2 \
                        and self.net.n < self.budget:
                    self.net.add_node(fh, mu=-0.5, b=1.0, parent=base)
                    added.append(fh)
        return added

    def _prune(self):
        """Drop the node with the weakest class-separating power, if truly weak."""
        if len(self.feat_log) < 4:
            return None
        F = self._f_ratio()
        i = int(np.argmin(F))
        if F[i] < 0.05:
            f_dead = float(self.net.f[i])
            self.net.remove_node(i)
            self.sum = None
            self.cnt = np.zeros(self.n_classes)
            self.feat_log = []
            return f_dead
        return None

    def _f_ratio(self):
        X = np.concatenate([f for f, _ in self.feat_log], axis=0)
        y = np.concatenate([l for _, l in self.feat_log], axis=0)
        n = self.net.n
        X = X[:, :n] if X.shape[1] >= n else \
            np.pad(X, ((0, 0), (0, n - X.shape[1])))
        gm = X.mean(axis=0)
        between = np.zeros(n)
        within = np.zeros(n) + 1e-9
        for c in np.unique(y):
            xc = X[y == c]
            between += len(xc) * (xc.mean(axis=0) - gm) ** 2
            within += ((xc - xc.mean(axis=0)) ** 2).sum(axis=0)
        return between / within

    def evaluate(self, signals, labels):
        feats = self.net.run(signals)["feats"]
        return float(np.mean(self.predict(feats) == labels))
