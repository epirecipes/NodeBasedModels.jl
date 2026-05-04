module NodeBasedModels

#=
NodeBasedModels.jl — Node-centric companion to EdgeBasedModels

A unified package for node-centric epidemic models on networks:

  Population-level pairwise models (population_pairwise.jl)
    → Mean-field pairwise approximation: disease × network × closure → ODESystem
    → Supports homogeneous and heterogeneous degree distributions

  Individual-based models (individual_based.jl)
    → Order-1 moment closure on a specific graph (Sharkey 2008, 2011)
    → Per-node ODEs: ⟨S_i⟩, ⟨I_i⟩ with pairwise independence assumption

  Pair-based models (pair_based.jl)
    → Order-2 moment closure for SIR on an undirected specific graph (Sharkey 2008, 2011)
    → Per-node + per-edge ODEs with Kirkwood triple closure
    → Exact on tree graphs
    → Currently restricted to the canonical S → I → R model

Architecture:
  Disease structure (compartments.jl)
    → CompartmentalModel: S, I, R states + transitions
    → Also accepts Catalyst.ReactionSystem input

  Network structure (networks.jl)
    → HomogeneousNetwork: regular degree n, optional clustering ϕ
    → HeterogeneousNetwork: degree distribution p_k
    → GraphNetwork: wraps a Graphs.jl AbstractGraph (edge list); an optional
      transmission_matrix supplies heterogeneous per-edge rates

  Closure approximations (closures.jl)
    → BernoulliClosure, KeelingClosure: support both HomogeneousNetwork and HeterogeneousNetwork
    → BarnardClosure, PowerClosure: HomogeneousNetwork only
    → EamesClosure: exported placeholder; always throws ArgumentError when used
    → KirkwoodClosure (for pair-based models on graphs)

  Analysis (analysis.jl)
    → R₀, epidemic threshold, early growth rate, DFE
=#

using LinearAlgebra
using Symbolics
using ModelingToolkit
using Graphs
using OrdinaryDiffEqDefault
using JumpProcesses
using Statistics
using Random
import Catalyst

# Core types
include("compartments.jl")
include("networks.jl")
include("closures.jl")
include("population_pairwise.jl")
include("individual_based.jl")
include("pair_based.jl")
include("gillespie.jl")
include("analysis.jl")
include("reinfection_counting.jl")
include("motif_based.jl")
include("motif_symbolic.jl")
include("neighbourhood_based.jl")
include("neighbourhood_symbolic.jl")

# Compartmental model types
export CompartmentalModel, Compartment, Transition
export sir_model, sis_model, seir_model, sirs_model
export model_from_catalyst

# Network types
export NetworkStructure, HomogeneousNetwork, HeterogeneousNetwork, GraphNetwork
export regular_network, erdos_renyi_network, degree_distribution_network
export mean_degree, excess_degree, clustering

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
export node_state, aggregate, compartment, compartments, population_fraction

# Pair-based model (order 2)
export PairBasedResult
export generate_pair_based, pair_prob
export KirkwoodClosure

# Gillespie stochastic simulation
export GillespieResult, GillespieSISResult
export gillespie_sir, gillespie_sir_average
export gillespie_sis, gillespie_sis_average
export sis_state, infection_count
export reinfection_histogram, reinfection_histogram_series

# Analysis
export basic_reproduction_number, epidemic_threshold
export early_growth_rate
export disease_free_equilibrium

# Convenience
export default_initial_conditions, solve_pairwise, solve_epidemic

# Motif-closure framework (Phase B(a1))
export MotifClosure, MotifShape, MotifVariable, MotifSystem
export motif_based_sis, solve_motif
export build_motif_symbolic_rhs
export induced_subgraph_counts_4vertex

# Reinfection counting (Keeling, House, Cooper & Pellis 2016)
export with_reinfection_counting, reinfection_totals
export base_compartment_of, infection_count_of

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
const build_individual_based = generate_individual_based
const build_pair_based = generate_pair_based

export build_pairwise, build_individual_based, build_pair_based

# --- Disambiguating aliases for the cross-package `sir_model` collision ---
# Both EdgeBasedModels and NodeBasedModels export `sir_model` with different
# semantics (DiseaseProgression vs CompartmentalModel). Users who mix both
# packages in one session can use these qualified names to avoid ambiguity.

"""
    node_sir_model(args...; kwargs...)

Disambiguating alias for [`sir_model`](@ref) when used alongside
EdgeBasedModels.jl. Returns a `CompartmentalModel`.
"""
const node_sir_model  = sir_model
const node_sis_model  = sis_model
const node_seir_model = seir_model
const node_sirs_model = sirs_model

export node_sir_model, node_sis_model, node_seir_model, node_sirs_model

end # module
