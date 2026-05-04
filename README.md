# NodeBasedModels.jl

Node-centric companion to [EdgeBasedModels.jl](../EdgeBasedModels.jl) for
infectious-disease dynamics on networks. Where EBCMs track edge-level
probabilities and a degree-distribution PGF, NodeBasedModels works at the level
of individual nodes (and pairs of nodes), supporting:

- **Population-level pairwise approximations** — mean-field `[X], [XY], [XYZ]`
  ODEs on `HomogeneousNetwork` / `HeterogeneousNetwork` with a configurable
  triple closure (`closures.jl`).
- **Reinfection-counting pairwise lifts** — SIS/SIRS-style infection-history
  towers `X₀, …, X_L` via `with_reinfection_counting`, closed by the existing
  pairwise machinery.
- **Motif closures** — connected induced-subgraph counts by shape and state
  class via `motif_based_sis`, including the tested `m = 3` random-regular
  benchmark.
- **Neighbourhood / effective-degree closures** — ego-neighbourhood SIS states
  `[S_y], [I_y]` for `n = 2` via `generate_neighbourhood`.
- **Graph-instance approximations** — order-1 individual-based and order-2
  pair-based ODE systems on a specific `Graphs.jl` graph (`individual_based.jl`,
  `pair_based.jl`).
- **Stochastic Gillespie SIR/SIS** simulators on arbitrary graphs
  (`gillespie.jl`) for benchmarking the deterministic approximations.

## Quick start

```julia
using NodeBasedModels

model   = sir_model()                       # canonical S → I → R, τ infection rate
network = regular_network(50; n_nodes = 5_000)
closure = KeelingClosure()

psys = generate_pairwise(model, network, closure;
                         tspan = (0.0, 80.0))
sol  = solve_pairwise(psys, Dict(:τ => 0.05, :γ => 0.1))
```

## Disease specification

| Constructor      | Compartments    | Transitions                    |
|------------------|-----------------|--------------------------------|
| `sir_model()`    | S, I, R         | infection (τ), recovery (γ)    |
| `sis_model()`    | S, I            | infection, recovery back to S  |
| `seir_model()`   | S, E, I, R      | infection, progression, recovery |
| `sirs_model()`   | S, I, R         | + waning (ε)                   |
| `model_from_catalyst(rxn)` | inferred from `Catalyst.ReactionSystem` | bimolecular `S + I → 2I` is recognised |

Construct your own with `CompartmentalModel`, `Compartment`, and `Transition`
(see `vignettes/05_custom_models/`).

## Network types

| Type | Use                                            |
|------|------------------------------------------------|
| `HomogeneousNetwork(n; ϕ = 0.0)` | regular degree, optional triangle fraction |
| `HeterogeneousNetwork(p_k)`      | degree distribution `p_k` |
| `GraphNetwork(g; transmission_matrix = nothing)` | specific `Graphs.jl` graph; optional per-edge τ |

Helpers: `regular_network`, `erdos_renyi_network`, `degree_distribution_network`,
`mean_degree`, `excess_degree`, `clustering`.

## Closures (population pairwise)

| Closure          | `HomogeneousNetwork` | `HeterogeneousNetwork` |
|------------------|:--------------------:|:----------------------:|
| `BernoulliClosure` | ✓ | ✓ |
| `KeelingClosure`   | ✓ | ✓ |
| `BarnardClosure`   | ✓ | ✗ throws `ArgumentError` |
| `PowerClosure`     | ✓ | ✗ throws `ArgumentError` |
| `EamesClosure`     | placeholder — always throws `ArgumentError` | placeholder |

`KirkwoodClosure` is used internally by `generate_pair_based`.

## Solvers

- `solve_pairwise(psys, params)` for population pairwise systems, including
  reinfection-counting lifts.
- `solve_motif(sys)` for motif-closure systems from `motif_based_sis`.
- `solve_neighbourhood(sys)` for neighbourhood systems from
  `generate_neighbourhood`.
- `generate_individual_based(model, net; …)` returns an `IndividualBasedResult`
  bundling the `ODEProblem` solution and helper accessors (`compartment`,
  `aggregate`, `node_state`, `population_fraction`).
- `generate_pair_based(model, net; …)` returns a `PairBasedResult` with
  `pair_prob` accessors. **Currently restricted to canonical SIR on undirected
  graphs.**
- `gillespie_sir(net, τ, γ; …)` and `gillespie_sis(net; …)` for stochastic
  ground-truth.

The moment-closure solvers now default to tighter tolerances
`reltol = 1e-8`, `abstol = 1e-10`; pass these keywords explicitly to override.

## Analysis

`basic_reproduction_number`, `epidemic_threshold`, `early_growth_rate`, and
`disease_free_equilibrium` provide both symbolic and numeric variants — see
`vignettes/03_analysis/`.

## Vignettes

The `vignettes/` directory contains end-to-end Quarto worksheets covering
basics, network types, analysis, stochastic validation, custom models, the
population-vs-graph approximation comparison, and population-pairwise
worksheets (basics, R₀/threshold, clustering effects) inherited from the
former PairwiseNetworkModels.jl.

Recent closure-family vignettes:

- [`vignettes/10_reinfection_counting/`](vignettes/10_reinfection_counting/) —
  reinfection-counting pairwise lifts.
- [`vignettes/11_motif_closures/`](vignettes/11_motif_closures/) — motif
  closures and the marginalisation obstruction.
- [`vignettes/12_neighbourhood_model/`](vignettes/12_neighbourhood_model/) —
  neighbourhood / effective-degree closure.
- [`vignettes/13_combined_comparison/`](vignettes/13_combined_comparison/) —
   standard pairwise, reinfection, motif, neighbourhood, and Gillespie on one
   SIS benchmark.
- [`vignettes/14_motif_catalogue/`](vignettes/14_motif_catalogue/) —
  supported motif shapes, automorphism orbits, and state-class counts.

## Release highlights

This release consolidates the node-based side of the ecosystem:

- Reinfection-counting pairwise lifts via `with_reinfection_counting`.
- Motif/subgraph SIS closures for `k = 2, 2 ≤ m ≤ 6` and
  `k = 3, m ∈ {2,3,4}`, with a standalone motif-shape catalogue.
- Neighbourhood/effective-degree SIS closure for `n = 2`.
- Cross-package comparison vignettes and stochastic validation against
  Gillespie ensembles.
- Lean-backed documentation of the Kirkwood marginalisation obstruction:
  `k = 3, m = 4` is implemented but not a guaranteed monotone refinement of
  `m = 3`.
- Runtime dependencies now exclude the stochastic companion package; the
  `NetworkOutbreaks` adapter is used through test/vignette environments.

## Formal proofs

The `proofs/` directory contains a Lean 4 / Mathlib formalization of the
pair-equation closures and their invariant regions
(`PairwiseProofs/PairwiseInvariantRegion.lean`,
`PairwiseProofs/ClosureConditions.lean`). Build with `lake build` from
`proofs/`.

## Theoretical guarantees and known obstructions

The companion proof directory
[`EdgeBasedModels.jl/proofs/EBCMCategory/`](../EdgeBasedModels.jl/proofs/EBCMCategory/)
contains the category-theoretic results used by the newer closure families:

- `ClosureTheorem.lean` — positive closure-consistency results.
- `MarginalisationFunctor.lean` — T1, equivariance lifts algebraic closure
  diagrams to dynamical marginalisation diagrams.
- `Obstructions.lean` and `MarginalisationCharacterization.lean` — T2/T3
  obstruction results, including the Kirkwood-form marginalisation obstruction
  that blocks treating motif `m = 4` as a guaranteed monotone refinement of
  `m = 3`. This is why the Phase B B(c) random 3-regular `m = 4` expectations
  are documented as `@test_broken` in `test/runtests.jl`.

## Consolidation note

Population-level pairwise functionality, formerly distributed as a separate
`PairwiseNetworkModels.jl` package, has been folded into NodeBasedModels.jl.
The mixed Keeling/Eames pair-counting convention used here is
`[XX] = 2·(undirected XX edges)` and `[XY] = (undirected XY edges)` for
`X≠Y`, with conservation `2·∑_{X≠Y}[XY] + ∑_X [XX] = k·N`.

## Companion package

For an edge-centric/configuration-model formulation that scales naturally to
heterogeneous degree distributions and clustered/multiplex networks, see
[EdgeBasedModels.jl](../EdgeBasedModels.jl).

## License

NodeBasedModels.jl is licensed under the MIT License.
