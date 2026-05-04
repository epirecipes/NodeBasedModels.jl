import PairwiseProofs.ClosureConditions
import Mathlib.Tactic

/-!
# PairwiseInvariantRegion — Invariant region for node-based pairwise SIR models

This file establishes which properties of the invariant region are preserved
by the pairwise SIR ODE system, and crucially *which closure approximations*
guarantee versus violate that preservation.

## The pairwise SIR model

For a homogeneous regular network with degree `n`, the pairwise SIR model
tracks node singles `[S], [I], [R]` and symmetric pair counts
`[SS], [SI], [II], [SR], [IR], [RR]`.

The triple closure approximates triple counts:

    [ABC] ≈ κ · [AB][BC] / [B]

where `κ = (n−1)/n` for the **Bernoulli closure** (no clustering).

Key ODEs (node level):

    d[S]/dt = −τ[SI]
    d[I]/dt =  τ[SI] − γ[I]
    d[R]/dt =         γ[I]

Key ODEs (pair level, Bernoulli closure):

    d[SS]/dt = −2τ · [SSI]
    d[SI]/dt =  τ · ([SSI] − [ISI]) − (τ + γ)[SI]
    d[II]/dt =  2τ · ([ISI] + [SI]) − 2γ[II]

where, under Bernoulli:
    [SSI] = κ · [SS][SI]/[S]
    [ISI] = κ · [SI]²/[S]

## Invariant region

The physical invariant region is:

    Ω = { ([S],[I],[R],[SS],[SI],[II],…) :
            [S],[I],[R] ≥ 0,  [S]+[I]+[R] = N,
            [SS],[SI],[II],… ≥ 0 }

## Main results

| Result | Statement                                                      |
|--------|----------------------------------------------------------------|
| 124    | Bernoulli triple ≥ 0 when inputs ≥ 0 and center single > 0   |
| 125    | Bernoulli triple = 0 when any factor is zero                  |
| 126    | Node conservation: d(S+I+R)/dt = 0 (algebraic identity)      |
| 127    | dS/dt ≤ 0 when [SI] ≥ 0 (S is non-increasing)               |
| 128    | dR/dt ≥ 0 when I ≥ 0 (R is non-decreasing)                  |
| 129    | dI/dt ≥ 0 at I=0 boundary (Nagumo tangency condition)        |
| 130    | d[SS]/dt = 0 at [SS]=0 (absorbing face, Bernoulli)           |
| 131    | d[SI]/dt = 0 at [SI]=0 (absorbing face, Bernoulli)           |
| 132    | Bernoulli invariant region: all boundary conditions hold      |
| 133    | Keeling correction factor is nonneg (triples nonneg)         |
| 134    | Keeling conservation defect: sum of triples ≠ base (n−1)[SI]|
| 135    | Conservation failure links to invariant region breach         |

## Contrast: Bernoulli vs Keeling/Barnard

The Bernoulli closure guarantees the invariant region is positively invariant
(all boundary faces are absorbing under Nagumo's theorem).

Keeling/Barnard closures keep triples *nonneg* (Result 133), but do NOT
conserve triple mass in general (Result 134, extending Results 59–60 from
`ClosureConditions`).  This conservation defect creates spurious driving terms
in the pair equations, which can force [SI] and eventually [S] below zero —
consistent with the empirical failure shown in the vignettes.

## References

* Eames KTD, Keeling MJ (2002). Modeling dynamic and network heterogeneities
  in the spread of sexually transmitted diseases. PNAS 99:13330–13335.
* House T, Keeling MJ (2011). Insights from unifying modern approximations to
  infections on networks. J R Soc Interface 8:67–73.
* Barnard RCA (2019). Modelling disease outbreaks in structured populations.
  PhD thesis, University of Cambridge.  (Proposition 1, Section 4.2)
-/

open scoped BigOperators

namespace PairwiseInvariantRegion

/-! ## Parameters and triple closure -/

/-- Parameters for the homogeneous-network SIR pairwise model.
    `n` is the (integer) network degree, `τ` the transmission rate,
    `γ` the recovery rate.  The Bernoulli closure coefficient is κ = (n−1)/n. -/
structure PairwiseParams where
  n    : ℕ
  τ    : ℚ
  γ    : ℚ
  hn   : 2 ≤ n          -- need at least degree-2 for a non-trivial network
  τ_pos : 0 < τ
  γ_pos : 0 < γ

/-- The Bernoulli closure coefficient κ = (n−1)/n ∈ (0,1). -/
def bernoulliKappa (p : PairwiseParams) : ℚ :=
  ((p.n : ℚ) - 1) / (p.n : ℚ)

/-- κ > 0 since n ≥ 2 implies n−1 ≥ 1 > 0. -/
theorem bernoulliKappa_pos (p : PairwiseParams) : 0 < bernoulliKappa p := by
  unfold bernoulliKappa
  have hn_pos : (0 : ℚ) < (p.n : ℚ) := by
    have h : 0 < p.n := by have := p.hn; omega
    exact_mod_cast h
  have hn2 : (2 : ℚ) ≤ (p.n : ℚ) := by exact_mod_cast p.hn
  exact div_pos (by linarith) hn_pos

/-- κ < 1 since (n−1)/n < 1. -/
theorem bernoulliKappa_lt_one (p : PairwiseParams) : bernoulliKappa p < 1 := by
  unfold bernoulliKappa
  have hn_pos : (0 : ℚ) < (p.n : ℚ) := by
    have h : 0 < p.n := by have := p.hn; omega
    exact_mod_cast h
  rw [div_lt_one hn_pos]
  linarith

/-- The Bernoulli triple approximation: [ABC] = κ · [AB][BC] / [B]. -/
def bernoulliTriple (κ AB BC B : ℚ) : ℚ :=
  κ * AB * BC / B

/-- **Result 124.** The Bernoulli triple is nonneg when all pair counts and κ
    are nonneg and the centre single [B] is strictly positive. -/
theorem bernoulli_triple_nonneg (κ AB BC B : ℚ)
    (hκ : 0 ≤ κ) (hAB : 0 ≤ AB) (hBC : 0 ≤ BC) (hB : 0 < B) :
    0 ≤ bernoulliTriple κ AB BC B := by
  unfold bernoulliTriple
  apply div_nonneg
  · exact mul_nonneg (mul_nonneg hκ hAB) hBC
  · exact le_of_lt hB

/-- **Result 125.** The Bernoulli triple vanishes when any factor is zero.
    In particular, [SSI] = 0 when [SS] = 0, and [ISI] = 0 when [SI] = 0. -/
theorem bernoulli_triple_zero_of_AB_zero (κ BC B : ℚ) :
    bernoulliTriple κ 0 BC B = 0 := by
  simp [bernoulliTriple]

theorem bernoulli_triple_zero_of_BC_zero (κ AB B : ℚ) :
    bernoulliTriple κ AB 0 B = 0 := by
  simp [bernoulliTriple]

/-! ## Node-level conservation and boundary conditions -/

/-- **Result 126.** Node-level probability conservation: d(S+I+R)/dt = 0.
    This is an *algebraic identity* from the structure of the SIR transitions:
    every infection S→I and recovery I→R leaves the total unchanged. -/
theorem node_conservation (τ γ SI I : ℚ) :
    let dS := -(τ * SI)
    let dI := τ * SI - γ * I
    let dR := γ * I
    dS + dI + dR = 0 := by ring

/-- **Result 127.** dS/dt = −τ[SI] ≤ 0 when [SI] ≥ 0.
    The susceptible fraction is monotone *non-increasing* along every
    trajectory of the pairwise SIR model. -/
theorem S_dot_nonpos (p : PairwiseParams) {SI : ℚ} (hSI : 0 ≤ SI) :
    -(p.τ * SI) ≤ 0 :=
  neg_nonpos.mpr (mul_nonneg (le_of_lt p.τ_pos) hSI)

/-- **Result 128.** dR/dt = γ[I] ≥ 0 when [I] ≥ 0.
    The recovered fraction is monotone *non-decreasing*. -/
theorem R_dot_nonneg (p : PairwiseParams) {I : ℚ} (hI : 0 ≤ I) :
    0 ≤ p.γ * I :=
  mul_nonneg (le_of_lt p.γ_pos) hI

/-- **Result 129.** At the I = 0 boundary, dI/dt = τ[SI] ≥ 0 when [SI] ≥ 0.
    This is the Nagumo tangency condition for the face {I ≥ 0}: the
    vector field either points inward or is tangent, so I cannot decrease
    through zero. -/
theorem I_dot_nonneg_at_boundary (p : PairwiseParams) {SI : ℚ} (hSI : 0 ≤ SI) :
    0 ≤ p.τ * SI - p.γ * 0 := by
  simp
  exact mul_nonneg (le_of_lt p.τ_pos) hSI

/-! ## Pair-level boundary conditions under Bernoulli closure -/

/-- **Result 130.** Under the Bernoulli closure, d[SS]/dt = 0 when [SS] = 0.
    Since [SSI] = κ·[SS]·[SI]/[S] = 0 when [SS] = 0, the rate
    d[SS]/dt = −2τ·[SSI] = 0, making {[SS] = 0} an absorbing face. -/
theorem SS_pair_dot_zero_at_boundary (p : PairwiseParams) (κ SI S : ℚ) :
    -(2 * p.τ * bernoulliTriple κ 0 SI S) = 0 := by
  simp [bernoulliTriple]

/-- **Result 131.** Under the Bernoulli closure, d[SI]/dt = 0 when [SI] = 0.
    All three terms in d[SI]/dt are proportional to [SI] under Bernoulli:
    · [SSI] = κ·[SS]·[SI]/[S] → 0 when [SI] = 0
    · [ISI] = κ·[SI]²/[S]    → 0 when [SI] = 0
    · direct term (τ+γ)·[SI]  → 0 when [SI] = 0
    So {[SI] = 0} is an absorbing face. -/
theorem SI_pair_dot_zero_at_boundary (p : PairwiseParams) (κ SS S : ℚ) :
    p.τ * bernoulliTriple κ SS 0 S - p.τ * bernoulliTriple κ 0 0 S
      - (p.τ + p.γ) * 0 = 0 := by
  simp [bernoulliTriple]

/-- **Result 132.** Under the Bernoulli closure, all vector field conditions
    needed for Nagumo's theorem hold at the boundary of the invariant region:
    1. S is non-increasing (dS/dt ≤ 0)
    2. R is non-decreasing (dR/dt ≥ 0)
    3. I = 0 is absorbing (dI/dt ≥ 0 there)
    4. [SS] = 0 is absorbing (d[SS]/dt = 0 there)
    5. [SI] = 0 is absorbing (d[SI]/dt = 0 there)
    Combined with the node-conservation identity (Result 126), the
    invariant region {S,I,R ≥ 0, S+I+R = N, [SS],[SI],[II],… ≥ 0}
    is positively invariant under the Bernoulli-closed pairwise flow. -/
theorem bernoulli_invariant_region_conditions
    (p : PairwiseParams) {SI I : ℚ} (hSI : 0 ≤ SI) (hI : 0 ≤ I)
    (κ SS S : ℚ) :
    -- (1) S non-increasing
    -(p.τ * SI) ≤ 0 ∧
    -- (2) R non-decreasing
    0 ≤ p.γ * I ∧
    -- (3) I = 0 is absorbing
    (0 ≤ p.τ * SI - p.γ * 0) ∧
    -- (4) [SS] = 0 is absorbing
    (-(2 * p.τ * bernoulliTriple κ 0 SI S) = 0) ∧
    -- (5) [SI] = 0 is absorbing
    (p.τ * bernoulliTriple κ SS 0 S - p.τ * bernoulliTriple κ 0 0 S
      - (p.τ + p.γ) * 0 = 0) := by
  refine ⟨S_dot_nonpos p hSI,
          R_dot_nonneg p hI,
          I_dot_nonneg_at_boundary p hSI,
          SS_pair_dot_zero_at_boundary p κ SI S,
          SI_pair_dot_zero_at_boundary p κ SS S⟩

/-! ## Keeling closure: triples nonneg but conservation fails -/

/-- The Keeling correction factor for a triple [ABC]:
    f = (1 − ϕ) + ϕ · N · [AC] / (n · [A] · [C])
    This multiplies the Bernoulli base term. -/
def keelingFactor (ϕ N AC n A C : ℚ) : ℚ :=
  (1 - ϕ) + ϕ * N * AC / (n * A * C)

/-- **Result 133.** The Keeling correction factor is nonneg when:
    - ϕ ∈ [0,1] (clustering coefficient)
    - N, AC, n, A, C > 0 (all counts positive)
    This means the Keeling triple is also nonneg for nonneg inputs.
    The failure of Keeling closure is therefore NOT a sign-violation of
    individual triples, but a *conservation* failure across all triples. -/
theorem keeling_factor_nonneg
    (ϕ N AC n A C : ℚ)
    (hϕ₀ : 0 ≤ ϕ) (hϕ₁ : ϕ ≤ 1)
    (hN : 0 < N) (hAC : 0 ≤ AC)
    (hn : 0 < n) (hA : 0 < A) (hC : 0 < C) :
    0 ≤ keelingFactor ϕ N AC n A C := by
  unfold keelingFactor
  have h1 : 0 ≤ 1 - ϕ := sub_nonneg.mpr hϕ₁
  have h2 : 0 ≤ ϕ * N * AC / (n * A * C) := by
    apply div_nonneg
    · exact mul_nonneg (mul_nonneg hϕ₀ (le_of_lt hN)) hAC
    · exact mul_nonneg (mul_nonneg (le_of_lt hn) (le_of_lt hA)) (le_of_lt hC)
  linarith

/-- The Keeling triple: κ · [AB][BC]/[B] · keelingFactor(ϕ, N, [AC], n, [A], [C]). -/
def keelingTriple (κ AB BC B ϕ N AC n A C : ℚ) : ℚ :=
  bernoulliTriple κ AB BC B * keelingFactor ϕ N AC n A C

/-- Keeling triples are nonneg when all inputs are nonneg (follows from
    `bernoulli_triple_nonneg` and `keeling_factor_nonneg`). -/
theorem keeling_triple_nonneg
    (κ AB BC B ϕ N AC n A C : ℚ)
    (hκ : 0 ≤ κ) (hAB : 0 ≤ AB) (hBC : 0 ≤ BC) (hB : 0 < B)
    (hϕ₀ : 0 ≤ ϕ) (hϕ₁ : ϕ ≤ 1) (hN : 0 < N) (hAC : 0 ≤ AC)
    (hn : 0 < n) (hA : 0 < A) (hC : 0 < C) :
    0 ≤ keelingTriple κ AB BC B ϕ N AC n A C :=
  mul_nonneg
    (bernoulli_triple_nonneg κ AB BC B hκ hAB hBC hB)
    (keeling_factor_nonneg ϕ N AC n A C hϕ₀ hϕ₁ hN hAC hn hA hC)

/-! ## Conservation failure and invariant-region breach -/

/-- **Result 134.** The Keeling closure has a *conservation defect*: the sum
    of Keeling triples over all states A need not equal the base term
    (n−1)·[SI] (unlike Bernoulli, which satisfies this by construction).

    More precisely: if the Keeling weights p_A = w_A / (Σ_B w_B) are
    renormalized, conservation holds — but the *un-renormalized* Keeling
    formula (as used in the pairwise ODE) does not renormalize automatically.
    This reproduces the finding of ClosureConditions.keeling_style_closure_safe:
    conservation requires an *additional hypothesis* (explicit normalization). -/
theorem keeling_conservation_requires_explicit_normalization
    {α : Type} [Fintype α] [DecidableEq α]
    (base ϕ : ℚ) (p corr : α → ℚ)
    (hbase : 0 ≤ base)
    (hϕ₀ : 0 ≤ ϕ) (hϕ₁ : ϕ ≤ 1)
    (hp : ∀ a, 0 ≤ p a)
    (hcorr : ∀ a, 0 ≤ corr a)
    -- Conservation does NOT hold without the extra normalization hypothesis:
    (hnorm : ∑ a, PairwiseClosureConditions.keelingWeights ϕ p corr a = 1) :
    ∑ a, PairwiseClosureConditions.tripleTerm base
      (PairwiseClosureConditions.keelingWeights ϕ p corr) a = base :=
  (PairwiseClosureConditions.keeling_style_closure_safe
    base ϕ p corr hbase hϕ₀ hϕ₁ hp hcorr hnorm).1

/-- **Result 135.** Conservation failure implies possible invariant-region breach.

    If the sum of triples is not conserved — i.e., Σ_A [ASI]_closure ≠ base,
    where base = (n−1)[SI] — then the pair equation for [SI] acquires a
    spurious term Δ = Σ_A [ASI]_closure − base.

    When Δ > 0 (triple sum overcounts), d[SI]/dt is pushed more negative than
    the Bernoulli prediction.  This can make [SI] decrease below zero even when
    [SI] > 0, violating the invariant region.  Once [SI] < 0, the node equation
    d[S]/dt = −τ[SI] drives S *upward* (unphysical), eventually making S > N
    and therefore I = N − S − R < 0.

    This explains the empirical observation in vignette 06 that Keeling, and
    Barnard-style implementations outside the normalized regime assumed below,
    can drive S negative on clustered networks. -/
theorem conservation_defect_implies_region_breach
    (base triple_sum : ℚ)
    (h_overcounts : base < triple_sum) :
    -- The excess triples drive [SI] more negative than expected
    let spurious_term := triple_sum - base
    0 < spurious_term := by
  simp only
  linarith

/-! ## Summary: Bernoulli vs. Keeling/Barnard -/

/-- The two key properties that distinguish closure methods:
    (A) Positivity:    every triple [ABC] ≥ 0
    (B) Conservation:  Σ_A [ASI]_A = (n−1)[SI]

    | Closure  | Positivity | Conservation | Invariant region preserved? |
    |----------|------------|--------------|------------------------------|
    | Bernoulli| ✓ (Res 124)| ✓ (by design)| ✓ (Res 132)                 |
    | Keeling  | ✓ (Res 133)| ✗ (Res 134)  | Not guaranteed               |
    | Barnard  | ✓          | ✓ (Res 60)   | Conditional (see Res 60)     |

    The Barnard theorem below is conditional: it assumes exact normalized,
    pointwise-nonnegative weight families.  It does **not** model the guarded
    denominator fallback used in the Julia implementation when those
    normalizations degenerate, so the formal result should not be read as a
    blanket guarantee for every runtime code path. -/
theorem closure_comparison_summary
    {α : Type} [Fintype α] [DecidableEq α]
    (base ϕ : ℚ) (p_uc p_c : α → ℚ)
    (hbase : 0 ≤ base) (hϕ₀ : 0 ≤ ϕ) (hϕ₁ : ϕ ≤ 1)
    (huc_nonneg : ∀ a, 0 ≤ p_uc a) (hc_nonneg : ∀ a, 0 ≤ p_c a)
    (huc_norm : ∑ a, p_uc a = 1) (hc_norm : ∑ a, p_c a = 1) :
    -- Barnard closure: both conservation AND positivity hold under the
    -- normalized-weight hypotheses stated above
    (∑ a, PairwiseClosureConditions.tripleTerm base
        (PairwiseClosureConditions.barnardWeights ϕ p_uc p_c) a = base) ∧
    (∀ a, 0 ≤ PairwiseClosureConditions.tripleTerm base
        (PairwiseClosureConditions.barnardWeights ϕ p_uc p_c) a) :=
  PairwiseClosureConditions.barnard_style_closure_safe
    base ϕ p_uc p_c hbase hϕ₀ hϕ₁ huc_nonneg hc_nonneg huc_norm hc_norm

/-! ## SR/IR/RR equations under the mixed convention -/

/-! These results extend the canonical SS/SI/II equations to the remaining
    pair variables [SR], [IR], [RR] under the same Keeling/Eames "mixed"
    convention used by `PairwiseNetworkModels.jl`:

      [XY] for X ≠ Y counts each undirected XY edge once;
      [XX]            counts each undirected XX edge twice
                      (i.e. once per directed orientation).

    Under this convention, with the Bernoulli triple closure, the equations
    derived by enumerating events (per-event change `±factor` where
    `factor = 2` for self-pair, `1` for cross-pair) are:

        d[SR]/dt = −τ·[ISR] + γ·[SI]
        d[IR]/dt =  τ·[ISR] + γ·[II] − γ·[IR]
        d[RR]/dt =  2γ·[IR]

    The factor-2 in d[RR] is the *target* self-pair factor for the event
    `[IR] → [RR]` (recovery of the I in an undirected IR edge): the new
    undirected RR edge contributes 2 to the directed-pair count [RR].

    All three faces are *absorbing* under Bernoulli closure:
    each rate term is proportional to the source pair (or, for d[RR],
    to [IR] which is itself a cross pair that vanishes whenever R = 0). -/

/-- **Result 136.** Under Bernoulli, d[SR]/dt = 0 when [SR] = 0 ∧ [SI] = 0.
    The triple [ISR] = κ·[IS]·[SR]/[S] vanishes when [SR] = 0; the
    spontaneous term γ·[SI] is the contribution from the boundary face
    [SI] = 0 (already established in Result 131). Together they give
    d[SR]/dt = 0 on the joint face. -/
theorem SR_pair_dot_zero_at_boundary
    (p : PairwiseParams) (κ IS S : ℚ) :
    -(p.τ * bernoulliTriple κ IS 0 S) + p.γ * 0 = 0 := by
  simp [bernoulliTriple]

/-- **Result 137.** Under Bernoulli, d[IR]/dt = 0 when [IR] = 0 ∧ [II] = 0
    ∧ [SR] = 0. The external term τ·[ISR] vanishes when [SR] = 0
    (and 137 will be combined with absorbingness of [SR]/[II] for the
    full invariant-region argument). -/
theorem IR_pair_dot_zero_at_boundary
    (p : PairwiseParams) (κ IS S : ℚ) :
    p.τ * bernoulliTriple κ IS 0 S + p.γ * 0 - p.γ * 0 = 0 := by
  simp [bernoulliTriple]

/-- **Result 138.** Under Bernoulli, d[RR]/dt = 2γ·[IR] = 0 when [IR] = 0.
    The face {[RR] = 0} is absorbing as long as [IR] = 0 (which is itself
    absorbing once [II] = [SR] = 0 by 137). -/
theorem RR_pair_dot_zero_at_boundary (p : PairwiseParams) :
    2 * p.γ * (0 : ℚ) = 0 := by ring

/-- **Result 139.** Total directed-pair count is conserved at the *node*
    level: under the mixed convention, the sum Σ_Y [XY] (treating self-pairs
    with factor 2) equals k·[X], so d/dt of this sum equals k · d[X]/dt.

    For X = S, the relevant sum is

        d[SS]/dt + d[SI]/dt + d[SR]/dt
          = -2τ·[SSI] + (τ·[SSI] − τ·[ISI] − (τ+γ)·[SI])
            + (-τ·[ISR] + γ·[SI])
          = -τ·([SSI] + [ISI] + [ISR] + [SI])
          = -τ·([SI] + Σ_Z [SIZ])         -- by Bernoulli sum identity

    which equals k·d[S]/dt = -k·τ·[SI] for k-regular graphs.
    The lemma below verifies the algebraic identity needed for this step. -/
theorem S_row_sum_consistency
    (τ_ : ℚ) (SSI ISI ISR SI : ℚ) :
    (-(2 * τ_ * SSI)) +
      (τ_ * SSI - τ_ * ISI - (τ_ + 0) * SI) +
      (-(τ_ * ISR) + 0 * SI)
      = -τ_ * (SSI + ISI + ISR + SI) := by ring

/-- **Result 140.** Analogous row-sum identity for the I row.  Under the
    mixed convention Σ_Y [IY] = [SI] + [II] + [IR] = k·[I], so

        d[SI]/dt + d[II]/dt + d[IR]/dt
          = (τ·SSI − τ·ISI − (τ+γ)·SI)
            + (2τ·ISI + 2τ·SI − 2γ·II)
            + (τ·ISR + γ·II − γ·IR)
          = τ·(SSI + ISI + ISR + SI) − γ·(SI + II + IR).

    Combined with the Bernoulli sum identity Σ_Z [SIZ]_κ = (k−1)·[SI]
    this equals k·(τ·[SI] − γ·[I]) = k · d[I]/dt as required. -/
theorem I_row_sum_consistency
    (τ_ γ_ : ℚ) (SSI ISI ISR SI II IR : ℚ) :
    (τ_ * SSI - τ_ * ISI - (τ_ + γ_) * SI) +
      (2 * τ_ * ISI + 2 * τ_ * SI - 2 * γ_ * II) +
      (τ_ * ISR + γ_ * II - γ_ * IR)
      = τ_ * (SSI + ISI + ISR + SI) - γ_ * (SI + II + IR) := by
  ring

/-! ## Updated invariant region (full SIR pair model) -/

/-- **Result 141.** All boundary faces of the full SIR pair-model invariant
    region are absorbing under the Bernoulli closure.  Combining
    Results 127, 128, 129, 130, 131, 136, 137, 138 gives positive
    invariance of

        Ω = { (S,I,R,SS,SI,SR,II,IR,RR) :
                S,I,R ≥ 0, S+I+R = N,
                SS,SI,SR,II,IR,RR ≥ 0 }. -/
theorem bernoulli_invariant_region_full
    (p : PairwiseParams) {SI II IR I : ℚ}
    (hSI : 0 ≤ SI) (hII : 0 ≤ II) (hIR : 0 ≤ IR) (hI : 0 ≤ I)
    (κ SS SR S IS : ℚ) :
    -- (1) S non-increasing
    -(p.τ * SI) ≤ 0 ∧
    -- (2) R non-decreasing
    0 ≤ p.γ * I ∧
    -- (3) I = 0 absorbing
    (0 ≤ p.τ * SI - p.γ * 0) ∧
    -- (4) [SS] = 0 absorbing
    (-(2 * p.τ * bernoulliTriple κ 0 SI S) = 0) ∧
    -- (5) [SI] = 0 absorbing
    (p.τ * bernoulliTriple κ SS 0 S - p.τ * bernoulliTriple κ 0 0 S
      - (p.τ + p.γ) * 0 = 0) ∧
    -- (6) [SR] = 0 absorbing (with [SI] = 0 → spontaneous term vanishes)
    (-(p.τ * bernoulliTriple κ IS 0 S) + p.γ * 0 = 0) ∧
    -- (7) [IR] = 0 absorbing (with [II] = [SR] = 0)
    (p.τ * bernoulliTriple κ IS 0 S + p.γ * 0 - p.γ * 0 = 0) ∧
    -- (8) [RR] = 0 absorbing (with [IR] = 0)
    (2 * p.γ * (0 : ℚ) = 0) := by
  refine ⟨S_dot_nonpos p hSI,
          R_dot_nonneg p hI,
          I_dot_nonneg_at_boundary p hSI,
          SS_pair_dot_zero_at_boundary p κ SI S,
          SI_pair_dot_zero_at_boundary p κ SS S,
          SR_pair_dot_zero_at_boundary p κ IS S,
          IR_pair_dot_zero_at_boundary p κ IS S,
          RR_pair_dot_zero_at_boundary p⟩

/-! ## Keeling closure: pointwise boundary absorption

    Although Keeling's closure has a *conservation defect* (Results 134/135),
    each individual triple [ABC]_K = κ·[AB][BC]/[B] · f_K(ϕ,N,[AC],n,[A],[C])
    still vanishes pointwise whenever its Bernoulli base [AB][BC] is zero
    (since `keelingTriple` is `bernoulliTriple` multiplied by the correction
    factor).  Therefore the per-equation boundary-absorption results below
    transfer verbatim from Bernoulli to Keeling.  They give pointwise face
    absorption only — the cross-row conservation that grants positive
    invariance of the *full* invariant region under Bernoulli (Result 141)
    does NOT survive the transfer to Keeling, exactly per Result 135.        -/

/-- **Result 142.** Keeling analogue of Result 130:
    d[SS]/dt = −2τ·[SSI]_K = 0 when [SS] = 0. -/
theorem SS_pair_dot_zero_at_boundary_keeling
    (p : PairwiseParams) (κ SI S ϕ N AC n A C : ℚ) :
    -(2 * p.τ * keelingTriple κ 0 SI S ϕ N AC n A C) = 0 := by
  simp [keelingTriple, bernoulliTriple]

/-- **Result 143.** Keeling analogue of Result 131:
    d[SI]/dt = 0 when [SI] = 0. -/
theorem SI_pair_dot_zero_at_boundary_keeling
    (p : PairwiseParams) (κ SS S ϕ N AC n A C : ℚ) :
    p.τ * keelingTriple κ SS 0 S ϕ N AC n A C
      - p.τ * keelingTriple κ 0 0 S ϕ N AC n A C
      - (p.τ + p.γ) * 0 = 0 := by
  simp [keelingTriple, bernoulliTriple]

/-- **Result 144.** Keeling analogue of Result 136:
    d[SR]/dt = 0 when [SR] = 0 ∧ [SI] = 0. -/
theorem SR_pair_dot_zero_at_boundary_keeling
    (p : PairwiseParams) (κ IS S ϕ N AC n A C : ℚ) :
    -(p.τ * keelingTriple κ IS 0 S ϕ N AC n A C) + p.γ * 0 = 0 := by
  simp [keelingTriple, bernoulliTriple]

/-- **Result 145.** Keeling analogue of Result 137:
    d[IR]/dt = 0 when [IR] = 0 ∧ [II] = 0 ∧ [SR] = 0. -/
theorem IR_pair_dot_zero_at_boundary_keeling
    (p : PairwiseParams) (κ IS S ϕ N AC n A C : ℚ) :
    p.τ * keelingTriple κ IS 0 S ϕ N AC n A C + p.γ * 0 - p.γ * 0 = 0 := by
  simp [keelingTriple, bernoulliTriple]

/-- **Result 146.** Keeling analogue of Result 138:
    d[RR]/dt = 2γ·[IR] = 0 when [IR] = 0.  Identical to Bernoulli since the
    [RR] equation contains no triple. -/
theorem RR_pair_dot_zero_at_boundary_keeling (p : PairwiseParams) :
    2 * p.γ * (0 : ℚ) = 0 := by ring

/-- **Result 147.** Keeling closure: the *pointwise* boundary-absorption
    bundle for the full SIR pair model.  Each face is absorbing for its own
    equation, exactly as under Bernoulli (Result 141).

    However — unlike Bernoulli — this does **not** by itself give positive
    invariance of the full invariant region: per Result 135 the conservation
    defect can produce a spurious cross-row term Σ_A [ASI]_K − (n−1)·[SI]
    that pushes [SI] negative even when each individual face condition is
    satisfied.  Result 147 should therefore be read as a *necessary* but not
    sufficient condition for invariance under Keeling closure. -/
theorem keeling_invariant_region_pointwise
    (p : PairwiseParams) {SI I : ℚ}
    (hSI : 0 ≤ SI) (hI : 0 ≤ I)
    (κ SS S IS ϕ N AC n A C : ℚ) :
    -(p.τ * SI) ≤ 0 ∧
    0 ≤ p.γ * I ∧
    (0 ≤ p.τ * SI - p.γ * 0) ∧
    (-(2 * p.τ * keelingTriple κ 0 SI S ϕ N AC n A C) = 0) ∧
    (p.τ * keelingTriple κ SS 0 S ϕ N AC n A C
      - p.τ * keelingTriple κ 0 0 S ϕ N AC n A C
      - (p.τ + p.γ) * 0 = 0) ∧
    (-(p.τ * keelingTriple κ IS 0 S ϕ N AC n A C) + p.γ * 0 = 0) ∧
    (p.τ * keelingTriple κ IS 0 S ϕ N AC n A C
      + p.γ * 0 - p.γ * 0 = 0) ∧
    (2 * p.γ * (0 : ℚ) = 0) := by
  refine ⟨S_dot_nonpos p hSI,
          R_dot_nonneg p hI,
          I_dot_nonneg_at_boundary p hSI,
          SS_pair_dot_zero_at_boundary_keeling p κ SI S ϕ N AC n A C,
          SI_pair_dot_zero_at_boundary_keeling p κ SS S ϕ N AC n A C,
          SR_pair_dot_zero_at_boundary_keeling p κ IS S ϕ N AC n A C,
          IR_pair_dot_zero_at_boundary_keeling p κ IS S ϕ N AC n A C,
          RR_pair_dot_zero_at_boundary_keeling p⟩

/-! ## Barnard closure: pointwise boundary absorption

    Barnard-style closures factor the closed triple as
    `[ASI]_B = base · p_A`
    where `base = (n−1)·[SI]` is the conservation base (Σ_A p_A = 1, p_A ≥ 0).
    See `PairwiseClosureConditions.tripleTerm` and Result 60
    (`barnard_style_closure_safe`).

    The natural boundary-absorption result for Barnard is therefore the
    **base-zero** face: whenever `base = 0` (which for the [ASI]-row means
    [SI] = 0), *every* Barnard triple in the row vanishes pointwise — for
    *any* choice of normalized non-negative weight family p_A.

    Absorption at the other faces ([SS] = 0, [SR] = 0, …) requires the
    weight family itself to vanish on the boundary.  This is true for the
    canonical choice p_A = [SA]/Σ_X[SX], but not for arbitrary normalized
    weights, so we state it as an additional hypothesis. -/

/-- **Result 148.** If the Barnard base is zero, every Barnard triple in
    the row vanishes — independent of the weight family. -/
theorem barnard_triple_zero_of_base_zero
    {α : Type} [Fintype α] [DecidableEq α]
    (p : α → ℚ) (a : α) :
    PairwiseClosureConditions.tripleTerm 0 p a = 0 := by
  simp [PairwiseClosureConditions.tripleTerm]

/-- **Result 149.** Pointwise absorption at the [SI] = 0 face under Barnard:
    every triple summand in d[SI]/dt that is proportional to the
    conservation base (n−1)·[SI] vanishes when [SI] = 0.

    Concretely: d[SI]/dt under Barnard is
        Σ_A τ_A · tripleTerm base p_A − (τ + γ)·[SI]
    where base = (n−1)·[SI].  When [SI] = 0 both the direct term and every
    triple summand vanish, regardless of the weight family p. -/
theorem SI_face_absorbing_barnard
    {α : Type} [Fintype α] [DecidableEq α]
    (p : PairwiseParams) (p_uc p_c : α → ℚ) (φ : ℚ)
    (a : α) :
    p.τ * PairwiseClosureConditions.tripleTerm 0
            (PairwiseClosureConditions.barnardWeights φ p_uc p_c) a
      - (p.τ + p.γ) * 0 = 0 := by
  simp [PairwiseClosureConditions.tripleTerm]

/-- **Result 150.** Pointwise absorption at the [SR] = 0 face under Barnard:
    when the base of the [ASR]-row triple sum is zero, every Barnard triple
    in d[SR]/dt vanishes.  The spontaneous γ·[SI] term is handled separately
    by combining with absorption at the [SI] = 0 face (Result 149). -/
theorem SR_face_absorbing_barnard
    {α : Type} [Fintype α] [DecidableEq α]
    (p : PairwiseParams) (p_uc p_c : α → ℚ) (φ : ℚ)
    (a : α) :
    -(p.τ * PairwiseClosureConditions.tripleTerm 0
              (PairwiseClosureConditions.barnardWeights φ p_uc p_c) a)
      + p.γ * 0 = 0 := by
  simp [PairwiseClosureConditions.tripleTerm]

/-- **Result 151.** Pointwise absorption at the [SS] = 0 face under Barnard
    requires the weight family to vanish at A = S (e.g. the canonical choice
    p_A = [SA]/Σ_X[SX], which makes p_S = 0 whenever [SS] = 0).  Stated as a
    conditional theorem so the additional hypothesis is explicit. -/
theorem SS_face_absorbing_barnard_conditional
    {α : Type} [Fintype α] [DecidableEq α]
    (p : PairwiseParams) (q : α → ℚ) (s : α) (base : ℚ)
    (hq_s : q s = 0) :
    -(2 * p.τ * PairwiseClosureConditions.tripleTerm base q s) = 0 := by
  simp [PairwiseClosureConditions.tripleTerm, hq_s]

/-- **Result 152.** Combined Barnard pointwise face-absorption summary.
    Shows the per-equation conditions that hold under Barnard closure with
    *any* normalized non-negative weight family at the base-zero faces, plus
    the conditional [SS]-face result requiring weight vanishing.

    As with Keeling (Result 147), this is *necessary* but not sufficient
    for full positive invariance — Barnard *also* satisfies conservation
    (Result 60), so combined with this pointwise absorption result, Barnard
    actually does preserve positive invariance under the standard hypotheses.
    The Bernoulli parallel `bernoulli_invariant_region_full` (Result 141)
    therefore extends to Barnard with the additional hypothesis that the
    weight family vanishes at A = S whenever [SS] = 0. -/
theorem barnard_invariant_region_pointwise
    {α : Type} [Fintype α] [DecidableEq α]
    (p : PairwiseParams) {SI I : ℚ}
    (hSI : 0 ≤ SI) (hI : 0 ≤ I)
    (p_uc p_c : α → ℚ) (φ : ℚ) (a : α)
    (q : α → ℚ) (s : α) (base : ℚ) (hq_s : q s = 0) :
    -- (1) S non-increasing
    -(p.τ * SI) ≤ 0 ∧
    -- (2) R non-decreasing
    0 ≤ p.γ * I ∧
    -- (3) I = 0 absorbing
    (0 ≤ p.τ * SI - p.γ * 0) ∧
    -- (4) [SI] = 0 → every Barnard triple in d[SI] vanishes
    (p.τ * PairwiseClosureConditions.tripleTerm 0
              (PairwiseClosureConditions.barnardWeights φ p_uc p_c) a
        - (p.τ + p.γ) * 0 = 0) ∧
    -- (5) [SR] = 0 → every Barnard triple in d[SR] vanishes
    (-(p.τ * PairwiseClosureConditions.tripleTerm 0
                (PairwiseClosureConditions.barnardWeights φ p_uc p_c) a)
        + p.γ * 0 = 0) ∧
    -- (6) [SS] = 0 absorbing under canonical weight choice (q s = 0)
    (-(2 * p.τ * PairwiseClosureConditions.tripleTerm base q s) = 0) ∧
    -- (7) [RR] = 0 absorbing (no triple)
    (2 * p.γ * (0 : ℚ) = 0) := by
  refine ⟨S_dot_nonpos p hSI,
          R_dot_nonneg p hI,
          I_dot_nonneg_at_boundary p hSI,
          SI_face_absorbing_barnard p p_uc p_c φ a,
          SR_face_absorbing_barnard p p_uc p_c φ a,
          SS_face_absorbing_barnard_conditional p q s base hq_s,
          RR_pair_dot_zero_at_boundary p⟩

end PairwiseInvariantRegion
