# NodeBasedModels golden files

Goldens freeze the numbers NodeBasedModels 0.1 produces, **known defects included**,
so that the 0.2 refactor (NetworkEpiCore adoption, `via`, `node_based`, closures) can
show that it changes nothing it does not mean to change. They were created in Phase 0
(WP2 of `DESIGN_NetworkEpiCore.md`).

## Layout

```
test/golden/
  GoldenIO.jl        read/write (TOML + CSV, stdlib only) and comparison
  common.jl          metadata helpers and the fixed small graphs
  generate.jl        regenerates goldens
  selftest.jl        tests of GoldenIO itself (the "golden_io" suite)
  <area>/cases.jl    case definitions: the single source of truth for an area
  <area>/<case>.toml metadata, scalars, name sets, table index, known defects
  <area>/<case>.<table>.csv   numeric tables (trajectory, rhs_probe, totals)
```

| area | what is frozen | suite that checks it | owner (design §G.2) |
|---|---|---|---|
| `pairwise` | `generate_pairwise` + `solve_pairwise`: SIR, SIS, SEIR, SIRS, two-stage, vaccination and reinfection models; Bernoulli, Keeling, Barnard, Power closures; homogeneous and heterogeneous networks | `00_legacy_pairwise.jl` | WP15 |
| `graph_level` | `generate_individual_based` (IB) and `generate_pair_based` (PB) on fixed small graphs | `00_legacy_graph_level.jl` | WP24 |
| `analysis` | `basic_reproduction_number`, `epidemic_threshold`, `early_growth_rate`, `disease_free_equilibrium` | `00_legacy_analysis.jl` | WP24 |
| `motif` | `motif_based_sis` for m = 2, 3 (k = 2, 3), the m = 4 chain and a clustered host | `00_legacy_motif.jl` | WP37 |
| `neighbourhood` | `generate_neighbourhood` n = 2 for SIS, k = 2, 3, 4 | `00_legacy_neighbourhood.jl` | – |
| `reinfection` | pairwise systems of `with_reinfection_counting` lifts and `reinfection_totals` | `00_legacy_reinfection.jl` | WP15 |

Each ODE case stores:

- `trajectory` — every state variable (and aggregate) on `t = 0:1:T`, solved with the
  default OrdinaryDiffEq algorithm at `reltol = 1e-10`, `abstol = 1e-12`. IB and PB calls
  use loose default tolerances, so their golden re-solves the `ODEProblem` the call built
  (`result.sol.prob`, i.e. the package's own initial condition and right-hand side);
- `rhs_probe` — the right-hand side at three deterministic probe states (`x:<name>` is
  the state, `f:<name>` the derivative). Probe states come from an integer hash, not an
  RNG, so they are identical on every Julia version;
- `structure` — name sets (unknowns, parameters, layouts), compared exactly.

The comparison is elementwise, `|a − b| ≤ atol + rtol·max(|a|, |b|)`, with `rtol = 1e-8`
and a per-case `atol` (1e-10 × the population scale for trajectories). Columns are matched
by name, so a reimplementation may reorder its state vector.

## Known defects frozen here, and their fixes

| ID (`VERIFIED_ISSUES.md`) | goldens | status |
|---|---|---|
| B01 | `analysis/*` | **fixed in WP24**: SIR and SIS analysis are separate (`dynamics = :SIR / :SIS`, required on heterogeneous networks), Keeling/Barnard use the fast-variable quasi-equilibria. `het_bernoulli_B01`, `clustered_B01` and `symbolic_B01` were replaced by `het_bernoulli`, `clustered` and `symbolic` (justified by `test/suites/analysis_nbm.jl`); `het_bernoulli` reproduces the 0.1 numbers as the SIS R₀/threshold and the SIR growth rate. `hom_bernoulli` (SIR) is unchanged; `hom_bernoulli_sis` is new. The Barnard entries moved from `clustered` to `clustered_barnard`, which follows Barnard's closure with the infector last (thesis eq. 4.23; justified against a reference ODE in `analysis_nbm.jl`); `symbolic` gives the Bernoulli formula for Keeling/Barnard when ϕ = 0 |
| B02 | `graph_level/ib_seir_karate_node1*`, `ib_sirs_karate_node1*` | **fixed in WP24**: every transition runs at its own rate (`p`). The `*_B02` goldens were replaced by `ib_seir_karate_node1` (σ ≠ γ, E seeded) and `ib_sirs_karate_node1`; `ib_seir_karate_node1_sigma_eq_gamma_seedI` and `ib_sirs_karate_node1_eps_eq_gamma` reproduce the 0.1 tables (justified by `test/suites/graph_level.jl`). `ib_staged_karate_via` freezes per-infector infections (B04) |
| B03 | `pairwise/*keeling*` | **fixed in WP24**: KeelingClosure takes N = Σ[X] from the state. Every Keeling `rhs_probe` table was regenerated (the probes lie off Σ[X] = N); the consistent-N trajectories are unchanged (≤ 3e-11 relative); `sir_hom6_keeling_phi02_Nkw1000_B03` and `_Nnet1000_B03` became `sir_hom6_keeling_phi02_Nkw1000` and `_Nnet1000`, which now equal the consistent cases (justified by `test/suites/00_legacy_closures.jl`) |
| Barnard (fixed in S-NBM, WP24 follow-up) | `pairwise/sir_hom6_barnard_phi02`, `analysis/clustered_barnard` (subcritical SIR entries) | **fixed**: `generate_pairwise` evaluated Barnard's triple with the infector first, which is not Barnard's closure; it now uses `_infection_triple` ([A S I], thesis eq. 4.23). The pairwise golden was regenerated; `analysis_nbm.jl` checks its trajectory against an independent transcription of eq. 4.23 and the right-hand side at probe states. Below the threshold Barnard's SIR growth rate (and R = 1 + r/γ) is now the asymptotic decay from the product-state seed instead of the continued supercritical branch: the 18 subcritical SIR scalars of `clustered_barnard` changed (thresholds, SIS and supercritical entries are unchanged); `analysis_nbm.jl` ("SIR below the threshold: the seeded decay rate") checks them against the reference ODE |
| B05 follow-up (fixed in WP37) | `motif/sis_k2_m3` | **fixed in WP37**: the specialised k = 2, m = 3 builder's single-vertex anchor lacked the slot factor n_ext/k = 1/2 at each end of a P₃, doubling external infection (dE_SSS/dt at the IC was −48.7 against the exact +2.71 at β = 0.6, γ = 0.4, N = 1000, ε = 0.05). The golden was regenerated; only `sis_k2_m3.rhs_probe.csv` and `.trajectory.csv` changed, the other five motif goldens are unchanged. Justified in `test/suites/motif.jl` by the master-equation check "k = 2, m = 3 single-vertex anchor carries the slot factor 1/2" (exact IC derivative; equal to the independent generic-chain builder), the tree-Markov exactness test, and the NetworkOutbreaks SSA ring testset (ring N = 1000: D∞(m = 3) 0.318 → 0.153, now below m = 2) |
| seeding (fixed in WP15) | `pairwise/seir_*_seedI` | 0.1 seeded the first infectious compartment (I), not the entry state (E). The `seir_*` goldens were replaced in WP15 (0.2 seeds E, DESIGN §E.2; justified by `test/suites/adopt.jl`). The `seir_*_seedI` cases pin the 0.1 numbers through `seed_state = :first_infectious`, and their tables are byte-identical to the replaced goldens |

## Regenerating

```
julia --project=. test/golden/generate.jl                                  # all areas
julia --project=. test/golden/generate.jl graph_level                      # one area
julia --project=. test/golden/generate.jl pairwise:seir_hom4_bernoulli     # one case
```

Run from `NodeBasedModels.jl/`. A golden is replaced only by the work package that owns its
area, in the bug-fix change that justifies the new numbers with a literature or simulation
test (design §G.1). Rename a case whose meaning changes rather than silently rewriting it,
and delete the old files: the check fails on stored goldens that no case defines.
