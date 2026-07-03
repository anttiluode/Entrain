"""
core.py — the oscillator substrate.

Each node is a Stuart-Landau oscillator (the normal form of anything near a Hopf
bifurcation — the honest 'generic oscillator'):

    dz_i/dt = (mu_i + i*2*pi*f_i - |z_i|^2) * z_i  +  sum_j W_ij * z_j  +  B_i * s(t)

  mu_i < 0 : damped resonator — |z_i| rises only while the input (or a parent)
             contains energy near f_i. This is a detector / 'sensor' node.
  mu_i > 0 : self-sustaining limit cycle — keeps ringing after input stops.
             A ring of these is a memory loop.

'Enslavement' is entrainment: couple a child to a master and, inside the Arnold
tongue (small detuning, enough coupling), the child abandons its own frequency and
locks to the master's. benchmark.py measures that tongue instead of asserting it.

Plain-words honesty: a bank of damped nodes is a resonator filterbank (cochlea-style).
The claimed novelty of this repo is NOT the resonator — it is the growth rule in
grow.py that decides, from task failure alone, which resonators to grow and where
to attach them. See README for what that buys and what it doesn't.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


class Net:
    """A growable network of Stuart-Landau nodes with sparse complex state.

    State z has shape (batch, n) so many input signals integrate in parallel.
    Wiring is a dense (n, n) real matrix W (n stays small: tens of nodes).
    B (n,) couples the scalar input s(t) into each node ('sensor gain').
    """

    def __init__(self, fs=400.0):
        self.fs = fs
        self.f = np.zeros(0)          # natural frequencies (Hz)
        self.mu = np.zeros(0)         # <0 damped detector, >0 self-sustaining
        self.B = np.zeros(0)          # input coupling
        self.W = np.zeros((0, 0))     # node-to-node coupling (real weights)
        self.parent = []              # tree bookkeeping (-1 = root/free node)

    # ---------------- construction ----------------
    def add_node(self, f, mu=-0.5, b=1.0, parent=-1, w_down=0.0, w_up=0.0):
        """Add one node; optionally wire it under `parent` (down = parent->child)."""
        i = len(self.f)
        self.f = np.append(self.f, float(f))
        self.mu = np.append(self.mu, float(mu))
        self.B = np.append(self.B, float(b))
        W = np.zeros((i + 1, i + 1))
        W[:i, :i] = self.W
        self.W = W
        self.parent.append(int(parent))
        if parent >= 0:
            self.W[i, parent] = w_down     # child receives the master
            self.W[parent, i] = w_up       # optional feedback loop
        return i

    def remove_node(self, i):
        keep = [k for k in range(len(self.f)) if k != i]
        self.f, self.mu, self.B = self.f[keep], self.mu[keep], self.B[keep]
        self.W = self.W[np.ix_(keep, keep)]
        remap = {old: new for new, old in enumerate(keep)}
        self.parent = [(-1 if p == i else remap.get(p, -1))
                       for k, p in enumerate(self.parent) if k != i]

    @property
    def n(self):
        return len(self.f)

    # ---------------- dynamics ----------------
    def run(self, S, record=False, z0=None):
        """Integrate the whole batch. S: (batch, T) real input signals.
        Returns dict with feats (batch, n) mean |z| over the second half —
        the resonance features — and optionally the full trajectory."""
        batch, T = S.shape
        dt = 1.0 / self.fs
        n = self.n
        z = (np.zeros((batch, n), complex) + 1e-3) if z0 is None else z0.copy()
        iw = 1j * TWO_PI * self.f                       # (n,)
        traj = np.zeros((T, n), complex) if record else None
        acc = np.zeros((batch, n))
        half = T // 2
        for t in range(T):
            # exponential integrator: the stiff rotation/decay part is integrated
            # exactly (explicit Euler on oscillators is unstable and the error
            # grows with frequency — measured, not guessed; see README kill list)
            lin = self.mu + iw - np.abs(z) ** 2          # (batch, n)
            forcing = z @ self.W.T + S[:, t:t + 1] * self.B
            z = z * np.exp(lin * dt) + dt * forcing
            if t >= half:
                acc += np.abs(z)
            if record:
                traj[t] = z[0]
        return dict(feats=acc / (T - half), z=z, traj=traj)

    def instantaneous_freq(self, traj, tail=200):
        """Mean d(angle)/dt over the last `tail` steps, per node, in Hz."""
        ph = np.unwrap(np.angle(traj[-tail:]), axis=0)
        return (ph[-1] - ph[0]) / (tail / self.fs) / TWO_PI


# ---------------- signals ----------------
def make_signal(freqs, amps, T, fs, noise=0.3, rng=None):
    rng = rng or np.random.default_rng()
    t = np.arange(T) / fs
    s = np.zeros(T)
    for f, a in zip(freqs, amps):
        s += a * np.sin(TWO_PI * f * t + rng.uniform(0, TWO_PI))
    return s + rng.normal(0, noise, T)


def residual_peak(signals, fs, covered, min_f=1.0, max_f=None, guard=1.0):
    """Frequency of the strongest average spectral peak NOT within `guard` Hz of
    any already-covered frequency. This is where growth places the next ear."""
    T = signals.shape[1]
    spec = np.abs(np.fft.rfft(signals, axis=1)).mean(axis=0)
    fgrid = np.fft.rfftfreq(T, 1.0 / fs)
    max_f = max_f or fs / 2 - 2
    mask = (fgrid >= min_f) & (fgrid <= max_f)
    for fc in covered:
        mask &= np.abs(fgrid - fc) > guard
    if not mask.any():
        return None
    return float(fgrid[mask][np.argmax(spec[mask])])
