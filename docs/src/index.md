# NodeBasedModels.jl

NodeBasedModels provides node-level deterministic models of epidemics on networks, each built
as a ModelingToolkit system:

- population pairwise models with a choice of triple closure;
- the exact PGF closure of the S-anchored pairwise model;
- mean-field (mass-action) models;
- individual- and pair-based models on an explicit graph;
- reinfection-counting lifts;
- the SIS motif and neighbourhood closures.

Version 0.2 is built on NetworkEpiCore (NEC). A model is a NEC `ContactModel` and a network is
a NEC `NetworkDescriptor`. The verb is [`node_based`](@ref)`(model, net; closure, level)`. The
same objects go to EdgeBasedModels (`edge_based`) and NetworkOutbreaks (`simulate`). Unlike the
edge-based lift, `node_based` is total on T_net, so SIS and SIRS are accepted. The changes from
0.1 are listed in [Migrating from 0.1](migration.md).

## The low-level API

```julia
using NetworkEpiCore, NodeBasedModels, Catalyst

seir = @reaction_network seir begin
    @parameters τ σ γ
    τ, S + I --> E + I     # contact: per-contact (per-edge) rate τ
    σ, E --> I
    γ, I --> R
end
model = contact_model(seir)
net   = ConfigurationNetwork(PoissonDegree(5))
p     = Dict(:τ => 1/6, :σ => 1/5, :γ => 1/4)

sys = node_based(model, net)                    # PairwiseSystem, constant closure K = 1 (exact: Poisson is PT)
sol = solve_epidemic(sys; p, initial = SeedFraction(:E => 0.01), tspan = (0.0, 150.0))
mc  = model_curves(sys, sol; t = 0:1:150)
mc.values[:cumulative][end]                     # 0.8002
```

By default the seed goes into the model's unique entry state, which is E for SEIR. The initial
pairs are the image of the edge-based initial condition.

## Levels and closures

| call | system |
|---|---|
| `node_based(m, ConfigurationNetwork(d))` | population pairwise with `BernoulliClosure` (constant K = ⟨k(k−1)⟩/⟨k⟩²) |
| `closure = KeelingClosure()`, `BarnardClosure()`, `PowerClosure()` | clustering closures (`ClusteredNetwork` defaults to Keeling) |
| `closure = PGFClosure()` | S-anchored pairwise with K_ψ = ψψ''/ψ'², exact for every PGF on T_EB models |
| `level = :s_anchored` | the S-anchored subsystem (`SAnchoredSystem`) |
| `node_based(m, WellMixed(κ))` | `MeanFieldSystem`, equal to `mass_action(m; κ)` |
| `node_based(m, MultitypeNetwork(…))` | `MultitypePairwiseSystem` |
| `node_based(m, ExplicitGraph(g); level = :individual \| :pair, p)` | individual-based (NIMFA) and pair-based (Kirkwood) models |
| `level = :motif, closure = MotifClosure(k, m)`; `level = :neighbourhood, n = 2` | SIS motif and neighbourhood approximations on regular networks |
| `with_reinfection_counting(sis_model(), L)` then `node_based` | reinfection-counting lift |

## Factories

The 0.1 entry points keep their signatures, and they also accept a `ContactModel` and a
descriptor. With `cumulative = true`, `generate_pairwise(model, network, closure)` gives the same
vector field as `node_based`. The other entry points are `generate_individual_based`,
`generate_pair_based`, `motif_based_sis` and `generate_neighbourhood`. `EamesClosure` and
`mass_action(::PairwiseSystem)` raise migration errors. `gillespie_sir` and `gillespie_sis` are
deprecated; use NetworkOutbreaks' `simulate` instead.

## Companion packages

- [NetworkEpiCore.jl](https://github.com/epirecipes/NetworkEpiCore.jl): the shared model,
  network, scenario and morphism objects.
- [EdgeBasedModels.jl](https://epirecip.es/EdgeBasedModels.jl/): edge-based models, and
  `pairwise_image`, the exact map onto the S-anchored pairwise model.
- [NetworkOutbreaks.jl](https://epirecip.es/NetworkOutbreaks.jl/): exact stochastic simulation
  and the reference ensembles.
