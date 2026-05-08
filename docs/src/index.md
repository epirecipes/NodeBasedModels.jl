# NodeBasedModels.jl

Node-centric companion to
[EdgeBasedModels.jl](https://epirecip.es/EdgeBasedModels.jl/) for
infectious-disease dynamics on networks. Where EBCMs track edge-level
probabilities and a degree-distribution PGF, NodeBasedModels works at the level
of individual nodes (and pairs of nodes), supporting:

- **Population-level pairwise approximations** — mean-field `[X], [XY], [XYZ]`
  ODEs on `HomogeneousNetwork` / `HeterogeneousNetwork` with a configurable
  triple closure.
- **Reinfection-counting pairwise lifts** — SIS/SIRS-style infection-history
  towers `X₀, …, X_L` via `with_reinfection_counting`.
- **Motif closures** — connected induced-subgraph counts by shape and state
  class via `motif_based_sis`.
- **Neighbourhood / effective-degree closures** — ego-neighbourhood SIS states
  `[S_y], [I_y]` for `n = 2` via `generate_neighbourhood`.
- **Graph-instance approximations** — order-1 individual-based and order-2
  pair-based ODE systems on a specific `Graphs.jl` graph.
- **Stochastic Gillespie SIR/SIS** simulators on arbitrary graphs.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/epirecipes/NodeBasedModels.jl")
```

## Example

```julia
using NodeBasedModels

model   = sir_model()
network = regular_network(50; n_nodes = 5_000)
closure = KeelingClosure()

psys = generate_pairwise(model, network, closure; tspan = (0.0, 80.0))
sol  = solve_pairwise(psys, Dict(:τ => 0.05, :γ => 0.1))
```

## Documentation contents

- [Vignettes](vignettes.md) — worked examples covering pairwise closures,
  motif and neighbourhood approximations, and stochastic comparisons.
- [API reference](api.md) — exported types and functions.

## Companion packages

- [EdgeBasedModels.jl](https://epirecip.es/EdgeBasedModels.jl/) — edge-based
  compartmental models on configuration-model networks.
- [NetworkOutbreaks.jl](https://epirecip.es/NetworkOutbreaks.jl/) — stochastic
  simulation algorithms.

## License

NodeBasedModels.jl is licensed under the MIT License.
