# Neighbourhood model (Approximation 3, n = 2) — SPEC

Reference: Keeling, House, Cooper, Pellis (2016), *Systematic
Approximations to SIS Dynamics on Networks*, PLoS Comp Biol 12(12):
e1005296.  See §3.3 "Neighbourhood model, n = 2", Eqs 9–10.  The
closure is identical to the one-step "approximate master equation" of
Lindquist, Ma, van den Driessche & Willeboordse (2011) and
Marceau et al. (2010) for SIS on a *k*-regular network.

This file describes Phase C of the approximation hierarchy locked in
plan.md.  Scope is **n = 2 only, SIS only, k-regular only**.

## Variable layout

For SIS on a k-regular network the state vector is

    u = [ S_0, S_1, …, S_k,   I_0, I_1, …, I_k ]      length 2(k+1)

with `S_y` (resp. `I_y`) the expected number of susceptible (resp.
infectious) nodes whose y of k neighbours are infectious.  For the
canonical k = 3 example the eight variables are

    u = [ S_0, S_1, S_2, S_3,  I_0, I_1, I_2, I_3 ]

Internally we order variables so that `S_*` come before `I_*` and `y`
runs from 0 upward — see `_build_neighbourhood_index` in
`neighbourhood_based.jl`.

## Dynamics (Eq 9 of paper)

For each `y ∈ {0,…,k}`:

    dS_y/dt =  γ · I_y                               # ego recovers
             - β · y · S_y                           # ego is infected
             + γ · [(y+1) S_{y+1} − y S_y]            # an inf nbr recovers
             + ω_S · [(k−y+1) S_{y−1} − (k−y) S_y]    # a sus nbr is infected

    dI_y/dt = -γ · I_y
             + β · y · S_y
             + γ · [(y+1) I_{y+1} − y I_y]
             + ω_I · [(k−y+1) I_{y−1} − (k−y) I_y]

with `S_{-1} ≡ S_{k+1} ≡ 0` and likewise for `I`.  Sign convention: `β`
is the per-edge transmission rate (the paper's `τ`), `γ` is the
per-node recovery rate.  `ω_S` (resp. `ω_I`) is the per-S-neighbour
infection rate of a susceptible neighbour of an S-ego (resp. an
I-ego).

## Closure (Eq 10 — consistent-overlap, n = 2)

Let V be the susceptible neighbour under consideration and let `y'` be
its own count of infectious neighbours (out of k).  V's k connections
include the ego A; the remaining (k−1) carry V's other infection
risk.  Rate that V is infected = β · (# infectious neighbours of V) =
β · y'.

The closure assumes that the joint distribution of V's neighbourhood
factorises through the shared overlap (V), conditional on V being a
susceptible neighbour of an ego of given state.  We sample V uniformly
over directed edges (ego → V) consistent with the ego's class:

* Pick V uniformly over directed S→S edges where V is the S-end.  The
  weight of an `S_{y'}` candidate is `(k − y')`  (its number of S-edges).
* Pick V uniformly over directed I→S edges where V is the S-end.  The
  weight of an `S_{y'}` candidate is `y'`        (its number of I-edges).

Hence

    ω_S = β · ⟨y'⟩_{S-end of SS}  =  β · Σ_{y'} y'(k−y') [S_{y'}]
                                          / Σ_{y'} (k−y') [S_{y'}]

    ω_I = β · ⟨y'⟩_{S-end of IS}  =  β · Σ_{y'} y'² [S_{y'}]
                                          / Σ_{y'} y'  [S_{y'}]

Note ω_I correctly includes the unit contribution of the ego itself:
the S-end of an IS edge has by construction at least one infectious
neighbour (the ego), so `⟨y'⟩` is automatically ≥ 1.  We do **not**
add a separate `+β` for the ego.

`safe_ratio(num, den; tol=1e-12)` from `motif_based.jl` is used for
both ratios (returns 0 when denominator < tol).  Both ratios are
finite at the disease-free equilibrium: the SS-denominator equals
`k·N > 0` at DFE, the IS-denominator vanishes but its numerator does
too — `safe_ratio` returns 0 cleanly.

### Worked example (k = 3, simple state)

Take `S_0 = 80, S_1 = 10, S_2 = 5, S_3 = 0, I_y = 0` (a near-DFE
state), `β = 1, γ = 1`.

    Σ y(k−y) S_y = 0·3·80 + 1·2·10 + 2·1·5 + 3·0·0 = 30
    Σ (k−y) S_y = 3·80 + 2·10 + 1·5 + 0·0           = 265
    ω_S         = 1 · 30 / 265                       ≈ 0.1132

    Σ y² S_y    = 0 + 10 + 20 + 0 = 30
    Σ y S_y     = 0 + 10 + 10 + 0 = 20
    ω_I         = 1 · 30 / 20      = 1.5

These two numbers feed directly into the (k−y±1) shift terms above.

## Conservation laws

1.  Population:  Σ_y S_y + Σ_y I_y = N      (constant in time).
2.  Edge balance:  Σ_y y · S_y = Σ_y (k−y) · I_y  (= number of
    undirected SI edges; preserved because each SI edge appears once
    on each side).
3.  Total directed-edge count:  Σ_y (k S_y + k I_y) = k N (trivial
    consequence of (1)).

Tests assert (1) and (2) at the IC and along the trajectory.

## Initial conditions

Random-mixing IC with infected fraction ε on N nodes:

    S_y = N (1−ε) · Binomial(k, y; ε)        # ε = P(specific nbr is I)
    I_y = N  ε    · Binomial(k, y; ε)

Trivially satisfies conservation (1) and (2) (both sides equal
N·k·ε).

## Tests (subset of plan.md test list)

* Conservation (1), (2) at IC and at t = T_end.
* Symbolic-validator agreement: numeric and symbolic RHS agree to
  ~1e-12 at IC, 5 random states, and ε → 0 (DFE limit).
* Reduces toward mean-field SIS at large k (qualitative — covered by
  the high-β sanity check that prevalence → 1 − γ/(βk)).
* `generate_neighbourhood(model, k, n)` throws on `n ∉ {2}` and on
  non-SIS models.
* Gillespie comparison on N = 500 random 3-regular at one canonical
  (β, γ): mean prevalence at t = t_end matches within 0.05.

## Closure transcription notes

The paper's Eq 10 is written in the ratio form
`τ · [SSX…]/([SSX…] + [ISX…])` etc.  For the SIS, n=2 case those
brackets reduce exactly to the sums above:

    [SSX_2…X_{n-1}] with n=2 ⇒ count of SS-edges with the given S-end
                              = Σ_{y'} (k−y') [S_{y'}]
    [ISX_2…X_{n-1}] with n=2 ⇒ count of IS-edges with the given S-end
                              = Σ_{y'} y' [S_{y'}]
    Σ y' (k−y') [S_{y'}]       counts (S-S edge, with the OTHER nbr of
                              the S-end being I)  =  pre-rate of
                              infection through the S-S link.
    Σ y'² [S_{y'}] counts (I-S edge, with another I nbr of S-end);
                  the S-end has an I-edge to ego AND y' total I-nbrs,
                  contributing y'·y' weight.

So the "ω_S, ω_I" form above is algebraically identical to Eq 10.
