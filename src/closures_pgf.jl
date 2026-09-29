# closures_pgf.jl — `PGFClosure`, the S-anchored closure K_ψ(θ) = ψ(θ)ψ''(θ)/ψ'(θ)² with an
# auxiliary θ, exact against the edge-based model for every degree PGF.
#
# Owner: WP23 (DESIGN_NetworkEpiCore.md §A.4, §D.5 M6/M8; work package in §G.2).
#
# The builder is `_closed_pairwise` (s_anchored.jl, whose header gives the equations); this file
# adds the closure's hooks and its `node_based` methods at the population and S-anchored levels.

export PGFClosure

"""
    PGFClosure()

The exact closure of the S-centred triples of a T_EB model on a configuration network with degree
PGF ψ (the dynamic survival closure of Kiss, Kenah & Rempała 2023; DESIGN §A.4, §D.5 M6):

    [Y s J] = K_ψ(θ) [Ys][sJ]/[s],        K_ψ(x) = ψ(x)ψ''(x)/ψ'(x)²,

with the auxiliary state θ of the edge-based model (the probability that an edge has not
transmitted to a test node), θ(0) = 1 and

    θ̇ = −ψ(θ)/([s]ψ'(θ)) · Σ_r τ_r [s J_r]

(the edge-based θ̇ = −Σ_r τ_r φ_{J_r} on the image [s J] = qξψ'(θ)φ_J, [s] = qξψ(θ)). The map π^PW
of M6 carries the edge-based model to this pairwise model for every degree distribution and every
T_EB model (branching at infection, several infectors, exits S → V with the survival factor ξ,
removals), so the node observables equal the edge-based ones. The constant closure
K = K_ψ(1) = `closure_constant(d)` (`BernoulliClosure`, the default of `ConfigurationNetwork`) is
exact only when K_ψ is constant, i.e. when ψ is Poisson type, ψ' = αψ^κ (M8,
`is_poisson_type`): Poisson, binomial, regular and negative binomial degrees, but not the
bimodal `:sir_bim` or the power law `:sir_pl`.

`node_based(model, ConfigurationNetwork(d); closure = PGFClosure())` builds the population
pairwise model with every pair [XY] (representation `:pgf_closure`), and `level = :s_anchored`
the S-anchored subsystem PW^S with the pairs [sX] and [ss] only (representation `:s_anchored`);
both return an [`SAnchoredSystem`](@ref). The model must be T_EB (SIS and SIRS raise an
`AdmissibilityError`: an arrow back into S breaks the edge-based construction) and the degree
parameters numeric. Other descriptors are an `ArgumentError`: `WellMixed` takes
[`MeanFieldClosure`](@ref), a `ClusteredNetwork` Keeling's closure.

```julia
sc  = scenario(:sir_bim)                                  # degrees {2, 10}, K_ψ(1) = 3/2
sys = node_based(sc; closure = PGFClosure())              # exact; the default (constant K) is not
sol = solve_epidemic(sys, sc)
model_curves(sys, sol; t = sc.tgrid)[:cumulative][end]    # 0.4956, the edge-based final size
```
"""
struct PGFClosure <: ClosureMethod end

_closure_theta(::PGFClosure) = true

"""
    NodeBasedModels._pgf_closure_factor(d::DegreeDistribution, x)

K_ψ(x) = ψ(x)ψ''(x)/ψ'(x)² for the PGF ψ of `d` (generic in `x`: numbers or symbolic θ). It is
constant, equal to `closure_constant(d)`, iff `d` is Poisson type (DESIGN §D.5 M8).
"""
_pgf_closure_factor(d::DegreeDistribution, x) =
    pgf(d, x) * pgf_derivative(d, x, 2) / pgf_derivative(d, x, 1)^2

_closure_factor(::PGFClosure, d::DegreeDistribution, θ) = _pgf_closure_factor(d, θ)

# θ̇ = −ψ(θ)/([s]ψ'(θ)) Σ_r τ_r [s J_r], where `hazard` = Σ_r τ_r [s J_r] and S = [s].
_theta_rhs(::PGFClosure, d::DegreeDistribution, θ, S, hazard) =
    -(pgf(d, θ) / (S * pgf_derivative(d, θ, 1))) * hazard

_closure_label(::PGFClosure, ::DegreeDistribution) = "PGF closure"

# ─── node_based methods ──────────────────────────────────────────────────────

function _node_based(::Val{:s_anchored}, closure::PGFClosure, cm::ContactModel,
                     net::ConfigurationNetwork; kw...)
    return _closed_pairwise(cm, net, closure; level = :s_anchored, kw...)
end

function _node_based(::Val{:population}, closure::PGFClosure, cm::ContactModel,
                     net::ConfigurationNetwork; kw...)
    return _closed_pairwise(cm, net, closure; level = :population, kw...)
end

const _PGF_NEEDS_CONFIGURATION =
    "PGFClosure closes the S-centred triples with the degree PGF of a ConfigurationNetwork " *
    "descriptor"

function _node_based(::Val{:population}, ::PGFClosure, cm::ContactModel, net::ClusteredNetwork;
                     kw...)
    throw(ArgumentError("node_based: $(_PGF_NEEDS_CONFIGURATION); on a ClusteredNetwork the " *
                        "partners of a susceptible node are correlated through triangles, so use " *
                        "KeelingClosure() (or EdgeBasedModels' clustered lift)"))
end

function _node_based(::Val{:population}, ::PGFClosure, cm::ContactModel, net::NetworkStructure;
                     kw...)
    throw(ArgumentError("node_based: $(_PGF_NEEDS_CONFIGURATION); a $(nameof(typeof(net))) " *
                        "records only moments of the degrees. Use " *
                        "node_based(model, ConfigurationNetwork(d); closure = PGFClosure())"))
end

# The 0.1 builder (`generate_pairwise` on a NetworkStructure) cannot use PGFClosure: it needs θ.
function triple_closure(::Symbol, ::Symbol, ::Symbol, ::Dict, ::Dict, net::NetworkStructure,
                        ::PGFClosure)
    throw(ArgumentError("$(_PGF_NEEDS_CONFIGURATION) and the auxiliary state θ, which the " *
                        "population pairwise builder on a $(nameof(typeof(net))) does not have. " *
                        "Use node_based(model, ConfigurationNetwork(d); closure = PGFClosure())"))
end
