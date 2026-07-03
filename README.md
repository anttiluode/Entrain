# entrain — a self-wiring network of oscillator fractals that computes by frequency

Index.html is at: https://anttiluode.github.io/Entrain/ 

Lineage: Geometric AI Core V5 (fractal growth + carrier), the Resonator phasor gates,
and the Clutch (surprise-gated compute). This repo distills the "fractals enslaving
sub-fractals by frequency" intuition into machinery that is **simulated, measured, and
benchmarked against baselines**, negative and modest results stated.
*Do not hype. Do not lie. Just show.*

## The idea, in physics words

- A node is a **Stuart–Landau oscillator** — the normal form of anything near a Hopf
  bifurcation, i.e. the honest generic oscillator. Damped nodes (`mu<0`) are
  **resonators**: they light up only when the input carries energy near their
  frequency. Self-sustaining nodes (`mu>0`) are **limit cycles**: they keep ringing.
- **"Enslavement" is entrainment.** Couple a child to a master and, inside the Arnold
  tongue, the child abandons its own frequency and runs at the master's. We *measure*
  the tongue (T3) instead of asserting it.
- **"Loops as memory"** is a ring of self-sustaining nodes. Amplitude memory is
  trivial for a limit cycle; what the *loop topology* buys is **relative phase** —
  a ring imprinted with a traveling wave remembers its direction after the input
  stops (T4). One bit, stored in geometry.
- **The actual claim of this repo is the growth rule** (`grow.py`): the network starts
  as ONE randomly tuned ear, feeds its task error into the Clutch's leaky-integrator
  gate, and when accumulated surprise trips, it **grows a new resonator at the
  strongest spectral peak of the signals it misclassified** — attaching it to the
  nearest-frequency existing node (fractal attachment), optionally sprouting a
  harmonic comb branch (f, 2f, 3f). Useless nodes are pruned by class-separation
  F-ratio. It grows ears for what it fails to hear, and only for that.

## Verified [V] / Killed [K] / Bet [B] — all numbers from `python3 benchmark.py`

**[V] T1 — hidden-spectrum classification** (4 classes = secret subsets of 6 hidden
frequencies; 8 seeds, 80 online episodes):

| bank | test accuracy | nodes |
|---|---:|---:|
| **grown (surprise-gated)** | **0.974** | **3.6** |
| random frequencies, same size | 0.873 | 3.6 |
| oracle (all 6 true freqs) | 1.000 | 6 |

The grown net nearly matches the oracle with **~half the nodes**, beating same-size
random placement by ~10 points at this easiest scale (the gap widens as the task
scales up — see B1 below). Note the honest reading of "half the nodes": measured
dictionary coverage is only 33–50% — the network does **not** reconstruct the world's
spectrum, it grows the *minimal discriminative* subset. Parsimony, not omniscience.

**[V] T2 — harmonic timbre** (4 classes share one fundamental, differ only in harmonic
profile; 6 seeds): fractal comb growth **1.000 ± 0.000** acc with **1.2** growth
events vs flat growth **0.990 ± 0.008** with **2.0** events. The comb prior wins, but
*modestly* — this is a small, real advantage in a world built to favor it, not a
breakthrough. Stated as such.

**[V] T3 — the enslavement map.** Master at 12 Hz, child detuned 0.1–2.5 Hz, coupling
1–16: the measured lock region is a clean Arnold tongue (triangular: lock range grows
with coupling; threshold scales with detuning as phase-reduction theory predicts).

**[V] T4 — loop memory.** Closed ring of 8 self-sustaining nodes imprinted with a
traveling wave: direction retained after input release **12/12**; the same nodes as an
open chain: **0/12**. The bit lives in the loop.

**[K] Explicit Euler on oscillators.** The first integrator produced "resonance" that
was actually numerical instability growing with frequency (higher-f nodes always won,
regardless of input). Killed; replaced with an exponential integrator (rotation
integrated exactly). If you reimplement this, you will hit the same trap.

**[K] Rejection sampling near capacity.** The first task generator drew dictionary
frequencies by sequential rejection; at 14+ frequencies with 2 Hz separation it
deadlocks (a bad prefix leaves no room and the loop retries forever). Replaced by a
jittered grid, which also makes density an explicit difficulty knob. T1 numbers above
were re-measured after this change.

**[K] Lazy gate.** With the Clutch gate at (gain 3, leak 1.0, trip 4) the network grew
once and plateaued at 75% accuracy — 20% error was subcritical, so surprise never
accumulated again. Retuned to (3, 0.35, 2.5): growth continues until error ≲ 10%.
The gate's calibration IS the stopping criterion for structure search.

**Bets → measured.** The three bets from the first version of this README were run
(`python3 bets.py`, ~3 min). All three resolved; one came with a limit the tongue map
had already predicted.

**[V] B1 — scaling.** Class count and dictionary size up, per-class signal complexity
held fixed (3 seeds each):

| classes / freqs | grown | nodes | same-size random | oracle |
|---|---:|---:|---:|---:|
| 4 / 6 | 0.96 | 3.7 | 0.83 | 1.00 |
| 6 / 10 | 1.00 | 3.7 | 0.75 | 1.00 |
| 8 / 14 | 1.00 | 4.7 | 0.80 | 1.00 |
| 10 / 18 | 0.99 | 5.3 | 0.87 | 1.00 |

The edge over random placement *persists* as the world scales, and parsimony gets more
striking: at 10 classes over an 18-frequency dictionary the network still solves the
task with ~5 ears. Caveat honestly stated: per-class complexity was held at 3
components — this scales the *search space*, not the per-signal difficulty.

**[V] B2 — routing by entrainment, with a measured resolution limit.** A master and
two slaves with equal coupling: setting the master's frequency to one slave's locks
that slave and not the other — **20/20** correct exclusive routing at 6 Hz separation,
re-routing after a mid-run master switch in **6.3 master cycles**. And the limit: at
coupling 6 the T3 tongue half-width is ~1 Hz, predicting that routing must fail when
slaves sit closer than that. Measured: 10/10 at 3 Hz separation, **3/10 at 1.5 Hz,
0/10 at 0.75 Hz**. The tongue map predicts the router's resolution. That is the kind
of cross-check that separates a mechanism from a demo.

**[V] B3 — phase carries class information.** Two classes with *identical* amplitude
spectra (fundamental + 2nd harmonic; only the harmonic's relative phase differs, and a
random global time-shift removes all absolute-phase cues). Amplitude features:
**0.489 ± 0.022** — chance, as they must be; the control holds. Adding the quadratic
phase-coupling invariant of the (f, 2f) node pair, angle(<z_2f·conj(z_f)²>):
**1.000 ± 0.000**. The information was never in the amplitudes; the substrate can read
it out of relative phase with two nodes and one product.

**[V] B4 — the router inside the grower** (`python3 b4_router.py`). The B2
mechanism now does real work: cue resonators read a pilot tone out of the signal,
their amplitudes steer the master's frequency, the master enslaves the matching
cluster's gate slave, and that slave's lock coherence multiplies the input coupling
of its cluster's ears — the unenslaved cluster is physically deaf (gating contrast
~20x, measured). No label or context bit touches the inference path.

Task: context conflict — the same four frequencies carry *different* class labels in
context A vs B (labels deranged), so a context-blind linear readout is structurally
capped near 0.5 (two classes even share identical mixed-context feature means).
Four seeds, equal budget, identical growth rule:

| system | accuracy | routing correct |
|---|---:|---:|
| **routed (cue→master→slaves→gates)** | **0.941 ± 0.102** | **1.000** |
| monolithic (no routing) | 0.525 ± 0.023 | — |
| oracle (true context bit) | 0.924 ± 0.132 | — |

The physical router matches the oracle. And routing-aware growth produced its
predicted signature: the network planted **the same frequency twice, once per
cluster** — one ear per *meaning* (e.g. seed 0 grew 9.7/17.3/25.7/35.3 into both
clusters). Same spectral content, different context, different ear.

The measured failure mode, stated: on 1 of 4 seeds accuracy fell to 0.77 with routing
still perfect — an ear planted 0.6 Hz off its true frequency *blocked* (via the
coverage guard) a corrected ear at the right spot, and budget filled with
near-duplicates. The router held; the placement refinement is the weak joint. That is
a growth-rule limitation already latent in T1, made visible here, and the obvious next
fix (allow re-growth inside the guard when the blocking ear has low F-ratio) is left
as an open bet rather than quietly patched after seeing the test data.

**[B] Open bets, still unverified:** placement refinement (re-growth inside the
coverage guard when the blocking ear is useless — the B4 failure mode); growth that
decides *where in the tree* to attach by function rather than frequency proximity;
more than two contexts (the tongue map says slave frequencies need >1.5 Hz separation
at coupling 6, so the routing capacity of one master is finite and measurable);
per-signal complexity scaling. Not claimed.

## What this is not

Not gradient descent, not a transformer competitor, not "AI". It is **structure search
gated by surprise** over a resonator substrate — a growth rule with measured wins over
random placement and measured limits against an oracle. A fixed FFT plus logistic
regression would also solve T1; the point is that this system *finds its own features
by failing*, with no spectral analysis given to it in advance beyond where to place
the next ear.

## Files

- `core.py` — Stuart–Landau nodes, exponential integrator, entrainment measures
- `grow.py` — the surprise-gated growth/pruning rule (imports the Clutch)
- `clutch.py` — the gate, unchanged from the Clutch Space
- `tasks.py` — the four worlds (spectrum, timbre, tongue, ring)
- `benchmark.py` — T1-T4 numbers; run it yourself (~3 min, numpy only)
- `bets.py` — B1-B3 numbers (scaling, routing, phase); ~3 min
- `router.py` / `b4_router.py` — the router inside the grower + its measurement (~4 min)
- `index.html` — self-contained live demo: inject frequencies, watch it grow ears,
  toggle the carrier and watch enslavement (no webcam, no server, no dependencies)

```bash
pip install numpy
python3 benchmark.py
```

Built by Antti Luode (PerceptionLab) with Claude as implementation collaborator.
