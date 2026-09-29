"""
    NodeBasedModels

Node-centric epidemic models on networks, on the NetworkEpiCore model representation.

The verb is [`node_based`](@ref)`(model, net; closure, level)`: `model` is a
`NetworkEpiCore.ContactModel` (`sir_model()`, …), a Catalyst or ModelingToolkit model (through
`contact_model`) or a 0.1 [`CompartmentalModel`](@ref); `net` is a NetworkEpiCore network
descriptor (`ConfigurationNetwork`, `ClusteredNetwork`, `ExplicitGraph`, …). The levels
([`NODE_BASED_LEVELS`](@ref)) are the population pairwise model with a triple closure
(Bernoulli, Keeling, Barnard, Power; a [`PairwiseSystem`](@ref)), the individual- and
pair-based models on an explicit graph, and the SIS motif and neighbourhood approximations on
regular networks. Systems are solved with `solve_epidemic` and read with `model_curves`,
`compartment` and `population_fraction`, the NetworkEpiCore generics shared with
EdgeBasedModels and NetworkOutbreaks.

```julia
using NodeBasedModels
sys = node_based(seir_model(), ConfigurationNetwork(PoissonDegree(5)))
sol = solve_epidemic(sys; p = Dict(:τ => 1/6, :σ => 1/5, :γ => 1/4), tspan = (0.0, 150.0))
model_curves(sys, sol; t = 0:1:150)
```

MIGRATION.md lists the changes from NodeBasedModels 0.1.
"""
module NodeBasedModels

#=
NodeBasedModels.jl — Node-centric companion to EdgeBasedModels

A unified package for node-centric epidemic models on networks:

  Population-level pairwise models (population_pairwise.jl, lift.jl)
    → `node_based(model, net; closure, level)`: disease × network × closure → ODESystem
    → Supports homogeneous and heterogeneous degree distributions and clustering

  Individual-based models (individual_based.jl)
    → Order-1 moment closure on a specific graph (Sharkey 2008, 2011)
    → Per-node ODEs: ⟨S_i⟩, ⟨I_i⟩ with pairwise independence assumption

  Pair-based models (pair_based.jl)
    → Order-2 moment closure for SIR on an undirected specific graph (Sharkey 2008, 2011)
    → Per-node + per-edge ODEs with Kirkwood triple closure
    → Exact on tree graphs
    → Currently restricted to the canonical S → I → R model

Architecture (0.2, on NetworkEpiCore):
  Models
    → NetworkEpiCore.ContactModel (sir_model(), Catalyst/MTK via contact_model): the input
    → CompartmentalModel (compartments.jl): the lowering target; converters in compat.jl

  Networks
    → NetworkEpiCore descriptors (ConfigurationNetwork, ClusteredNetwork, ExplicitGraph, …)
    → NetworkStructure (networks.jl): HomogeneousNetwork, HeterogeneousNetwork, GraphNetwork,
      built from descriptors by network_structure (compat.jl)

  Closure approximations (closures.jl)
    → BernoulliClosure, KeelingClosure: support both HomogeneousNetwork and HeterogeneousNetwork
    → BarnardClosure, PowerClosure: HomogeneousNetwork only
    → EamesClosure: exported placeholder; removed in 0.2 (always throws ArgumentError when used)
    → KirkwoodClosure (for pair-based models on graphs)

  Analysis (analysis.jl)
    → R₀, epidemic threshold, early growth rate, DFE
=#

using LinearAlgebra
using Symbolics
using ModelingToolkit
using Graphs
using OrdinaryDiffEqDefault
using Statistics
using Random
using NetworkEpiCore

# The shared generics of NetworkEpiCore that NodeBasedModels adds methods to (DESIGN §A.2):
# NodeBasedModels defines no function of its own under these names, and re-exports the NEC
# bindings below, so `using NetworkEpiCore, NodeBasedModels` is unambiguous.
import NetworkEpiCore: basic_reproduction_number, epidemic_threshold, early_growth_rate,
                       disease_free_equilibrium, default_initial_conditions, solve_epidemic,
                       model_curves, symbolic_ode, compartment, compartments,
                       population_fraction, mean_degree, excess_degree,
                       clustering_coefficient, contact_model,
                       mass_action, with_reinfection_counting, reinfection_totals,
                       reinfection_histogram

# Include order: each file may use the types of the files before it. compat.jl, lift.jl and the
# files of later work packages come after every builder; deprecated.jl is last. Later work
# packages never edit this file: they put their `export` lines (and any
# `import NetworkEpiCore: f` they need) in their own files.

# Core types (WP15)
include("compartments.jl")
include("networks.jl")
include("closures.jl")                 # WP24
include("population_pairwise.jl")      # WP15
include("individual_based.jl")         # WP24
include("pair_based.jl")               # WP24
include("gillespie.jl")                # WP24
include("analysis.jl")                 # WP24

# The 0.1 reinfection-counting lift on CompartmentalModel. reinfection_counting.jl also defines
# the legacy name parsers base_compartment_of(::Symbol) and infection_count_of(::Symbol), which
# NetworkEpiCore now owns (identical methods): defining them in NodeBasedModels would overwrite
# NetworkEpiCore's methods. The file is therefore included into a private submodule, and
# deprecated.jl / compat.jl expose its lift and totals as methods of the NetworkEpiCore generics.
module _LegacyReinfection
using ..NodeBasedModels: CompartmentalModel, Compartment, Transition, PairwiseSystem,
                         node_variables
include("reinfection_counting.jl")
end

include("motif_based.jl")              # WP37
include("motif_symbolic.jl")
include("neighbourhood_based.jl")
include("neighbourhood_symbolic.jl")

# NetworkEpiCore adoption (WP15)
include("compat.jl")
include("lift.jl")

# Later work packages (created empty by WP15)
include("s_anchored.jl")               # WP23
include("closures_pgf.jl")             # WP23
include("wellmixed.jl")                # WP23
include("multitype_pairwise.jl")       # WP36e

# Deprecations and removals (WP15)
include("deprecated.jl")

# ─── NetworkEpiCore bindings re-exported (the same objects, DESIGN §A.7, §A.8) ───
# The model representation and the canned models
export ContactModel, Contact, NodeTransition, contact_model
export sir_model, sis_model, seir_model, sirs_model, seair_model, twostrain_model, sirv_model
export with_reinfection_counting, reinfection_totals, base_compartment_of, infection_count_of
# Network descriptors and degree distributions that node_based accepts
export NetworkDescriptor, WellMixed, ConfigurationNetwork, ClusteredNetwork, ExplicitGraph
export DegreeDistribution, RegularDegree, PoissonDegree, BinomialDegree, NegBinDegree,
       GeometricDegree, PowerLawDegree, EmpiricalDegree, MixtureDegree
# Seeding
export SeedSpec, SeedFraction, SeedCount
# Shared generics with NodeBasedModels methods
export mean_degree, excess_degree, clustering_coefficient
export basic_reproduction_number, epidemic_threshold, early_growth_rate
export disease_free_equilibrium
export default_initial_conditions, solve_epidemic, model_curves, symbolic_ode, mass_action
export compartment, compartments, population_fraction
export reinfection_histogram

# ─── NodeBasedModels ─────────────────────────────────────────────────────────

# The node-based verb (lift.jl)
export node_based, default_closure, NODE_BASED_LEVELS, network_structure

# Compartmental model types
export CompartmentalModel, Compartment, Transition
export model_from_catalyst

# Network types
export NetworkStructure, HomogeneousNetwork, HeterogeneousNetwork, GraphNetwork
export regular_network, erdos_renyi_network, degree_distribution_network
export clustering

# Closure types
export ClosureMethod
export BernoulliClosure, KeelingClosure, BarnardClosure, EamesClosure
export PowerClosure

# Pairwise system generation (mean-field)
export PairwiseSystem
export generate_pairwise
export node_variables, pair_variables, triple_closure

# Individual-based model (order 1)
export IndividualBasedResult
export generate_individual_based
export node_state, aggregate

# Pair-based model (order 2)
export PairBasedResult
export generate_pair_based, pair_prob
export KirkwoodClosure

# Gillespie stochastic simulation
export GillespieResult, GillespieSISResult
export gillespie_sir, gillespie_sir_average
export gillespie_sis, gillespie_sis_average
export sis_state, infection_count
export reinfection_histogram_series

# Convenience
export solve_pairwise

# Motif-closure framework (Phase B(a1))
export MotifClosure, MotifShape, MotifVariable, MotifSystem
export motif_based_sis, solve_motif
export build_motif_symbolic_rhs
export induced_subgraph_counts_4vertex

# Neighbourhood model (Phase C; Keeling et al. 2016, Approximation 3, n = 2)
export NeighbourhoodSystem
export generate_neighbourhood, solve_neighbourhood, neighbourhood_compartment
export build_neighbourhood_symbolic_rhs

# --- Bidirectional API parity aliases (additive, non-breaking) ---
# Mirror EdgeBasedModels' `build_*` naming so users can call either spelling.

"""
    build_pairwise(args...; kwargs...)

Alias for [`generate_pairwise`](@ref). Provided for naming parity with
EdgeBasedModels.jl's `build_edge_system` / `build_sir` family.
"""
const build_pairwise = generate_pairwise

"""
    build_individual_based(args...; kwargs...)

Alias for [`generate_individual_based`](@ref).
"""
const build_individual_based = generate_individual_based

"""
    build_pair_based(args...; kwargs...)

Alias for [`generate_pair_based`](@ref).
"""
const build_pair_based = generate_pair_based

export build_pairwise, build_individual_based, build_pair_based

# Deprecated 0.1 constructors of CompartmentalModels (deprecated.jl)
export node_sir_model, node_sis_model, node_seir_model, node_sirs_model

end # module
