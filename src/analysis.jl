# analysis.jl — Epidemic threshold, R₀, and final size for pairwise models
#
# Analytical results derived from linearization around the disease-free
# equilibrium (DFE), following Keeling (1999), Kiss et al (2017),
# and Barnard et al (2019).

"""
    basic_reproduction_number(model, network, closure)

Compute R₀ for a pairwise model. The method depends on the closure and
network type.

For homogeneous SIR with Bernoulli closure:
    R₀ = τ(n-2)/γ

For homogeneous SIR with Keeling closure (clustering ϕ):
    R₀ depends on the fast-variable analysis at DFE

For heterogeneous networks (compact pairwise):
    R₀_CPW = (a + √((2γ-a)² + 8γ(a+τ))) / (2γ)
    where a = τ(⟨k²⟩ - 2⟨k⟩)/⟨k⟩ - γ
"""
function basic_reproduction_number(model::CompartmentalModel,
                                    network::NetworkStructure,
                                    closure::ClosureMethod = BernoulliClosure())
    # Extract transmission and recovery rates as symbols
    inf_trans = infection_transitions(model)
    spon_trans = spontaneous_transitions(model)

    length(inf_trans) >= 1 || throw(ArgumentError("Model has no infection transitions"))
    length(spon_trans) >= 1 || throw(ArgumentError("Model has no spontaneous transitions"))

    inf_rates = unique(t.rate for t in inf_trans)
    spon_rates = unique(t.rate for t in spon_trans)
    length(inf_rates) == 1 || throw(ArgumentError(
        "basic_reproduction_number(model, ...) requires a single infection-rate symbol."))
    length(spon_rates) == 1 || throw(ArgumentError(
        "basic_reproduction_number(model, ...) requires a single spontaneous-rate symbol."))

    # For simple SIR/SIS-like models with one τ and one γ
    τ_sym, γ_sym = _symbolic_rate_variables(only(inf_rates), only(spon_rates))

    return _compute_R0(network, closure, τ_sym, γ_sym)
end

"""
    basic_reproduction_number(network, closure, τ, γ)

Compute R₀ directly from network parameters and rates.
"""
function basic_reproduction_number(network::NetworkStructure,
                                    closure::ClosureMethod,
                                    τ::Real, γ::Real)
    _compute_R0_numeric(network, closure, τ, γ)
end

function _compute_R0_numeric(net::HomogeneousNetwork, ::BernoulliClosure,
                              τ::Real, γ::Real)
    n = net.n
    n > 2 || return 0.0
    return τ * (n - 2) / γ
end

function _compute_R0_numeric(net::HomogeneousNetwork, ::KeelingClosure,
                              τ::Real, γ::Real)
    n = net.n
    ϕ = net.ϕ
    n > 2 || return 0.0
    # Keeling (1999) R₀ with clustering correction
    # For the unclustered part: τ(n-2)/γ
    # Clustering reduces the effective transmission through triangles
    R0_unclustered = τ * (n - 2) / γ
    # First-order correction: R₀ ≈ τ(n-2)/γ · (1 - ϕ·correction)
    # The full expression requires solving the fast-variable cubic
    # Use the leading-order approximation:
    correction = ϕ * 2 / (n - 1)
    return R0_unclustered * (1 - correction)
end

function _compute_R0_numeric(net::HomogeneousNetwork, ::BarnardClosure,
                              τ::Real, γ::Real)
    n = net.n
    ϕ = net.ϕ
    n > 2 || return 0.0
    # Barnard et al (2019) improved threshold
    # At DFE with improved closure, the fast variable α satisfies a cubic
    # Leading order: R₀ = τ(n-2)/γ · (1 - ϕ²·O(1))
    R0_base = τ * (n - 2) / γ
    correction = ϕ^2 * 2 / ((n - 1) * (n - 2))
    return R0_base * (1 - correction)
end

function _compute_R0_numeric(net::HeterogeneousNetwork, ::BernoulliClosure,
                              τ::Real, γ::Real)
    # Compact pairwise R₀ (Kiss et al 2017, eq 54)
    mk = net.mean_degree
    mk2 = net.second_moment
    mk > 0 || return 0.0

    a = τ * (mk2 - 2mk) / mk - γ
    discriminant = (2γ - a)^2 + 8γ * (a + τ)
    discriminant >= 0 || return 0.0

    return (a + sqrt(discriminant)) / (2γ)
end

function _compute_R0_numeric(net::HeterogeneousNetwork, ::KeelingClosure,
                              τ::Real, γ::Real)
    # Heterogeneous with clustering
    mk = net.mean_degree
    mk2 = net.second_moment
    ϕ = net.ϕ
    mk > 0 || return 0.0

    # Start from unclustered heterogeneous R₀
    R0_uc = _compute_R0_numeric(
        HeterogeneousNetwork(net.degree_probs; ϕ=0.0, N=net.N),
        BernoulliClosure(), τ, γ)

    # Apply clustering correction (first-order)
    excess = net.excess_degree
    correction = ϕ * 2 / (excess > 0 ? excess : 1.0)
    return R0_uc * max(0.0, 1 - correction)
end

function _compute_R0_numeric(::NetworkStructure, closure::ClosureMethod, ::Real, ::Real)
    throw(ArgumentError(
        "basic_reproduction_number is not implemented for closure $(typeof(closure)) on this network type."))
end

function _symbolic_rate_variables(τ_sym::Symbol, γ_sym::Symbol)
    return Symbolics.variable(τ_sym), Symbolics.variable(γ_sym)
end

# Symbolic R₀ (returns a Symbolics expression)
function _compute_R0(net::HomogeneousNetwork, ::BernoulliClosure, τ_sym, γ_sym)
    n = net.n
    return n > 2 ? τ_sym * (n - 2) / γ_sym : 0 * τ_sym
end

function _compute_R0(net::HomogeneousNetwork, ::KeelingClosure, τ_sym, γ_sym)
    n = net.n
    ϕ = net.ϕ
    n > 2 || return 0 * τ_sym
    correction = ϕ * 2 / (n - 1)
    return τ_sym * (n - 2) / γ_sym * (1 - correction)
end

function _compute_R0(net::HomogeneousNetwork, ::BarnardClosure, τ_sym, γ_sym)
    n = net.n
    ϕ = net.ϕ
    n > 2 || return 0 * τ_sym
    correction = ϕ^2 * 2 / ((n - 1) * (n - 2))
    return τ_sym * (n - 2) / γ_sym * (1 - correction)
end

function _compute_R0(net::HeterogeneousNetwork, ::BernoulliClosure, τ_sym, γ_sym)
    mk = net.mean_degree
    mk2 = net.second_moment
    mk > 0 || return 0 * τ_sym
    a = τ_sym * (mk2 - 2mk) / mk - γ_sym
    discriminant = (2γ_sym - a)^2 + 8γ_sym * (a + τ_sym)
    return (a + sqrt(discriminant)) / (2γ_sym)
end

function _compute_R0(net::HeterogeneousNetwork, ::KeelingClosure, τ_sym, γ_sym)
    mk = net.mean_degree
    mk > 0 || return 0 * τ_sym
    excess = net.excess_degree
    correction = net.ϕ * 2 / (excess > 0 ? excess : 1.0)
    return _compute_R0(
        HeterogeneousNetwork(net.degree_probs; ϕ=0.0, N=net.N),
        BernoulliClosure(), τ_sym, γ_sym) * max(0.0, 1 - correction)
end

function _compute_R0(::NetworkStructure, closure::ClosureMethod, τ_sym, γ_sym)
    throw(ArgumentError(
        "basic_reproduction_number(model, ...) is not implemented for closure $(typeof(closure)) on this network type."))
end

"""
    epidemic_threshold(model, network, closure)

Compute the critical transmission rate τ_c above which an epidemic occurs.

For homogeneous Bernoulli: τ_c = γ/(n-2)
For heterogeneous Bernoulli: τ_c = γ⟨k⟩/(⟨k²⟩ - ⟨k⟩)
"""
function epidemic_threshold(network::NetworkStructure,
                             closure::ClosureMethod,
                             γ::Real)
    _compute_threshold(network, closure, γ)
end

function _compute_threshold(net::HomogeneousNetwork, ::BernoulliClosure, γ::Real)
    n = net.n
    n > 2 || return Inf
    return γ / (n - 2)
end

function _compute_threshold(net::HomogeneousNetwork, ::KeelingClosure, γ::Real)
    n = net.n
    ϕ = net.ϕ
    n > 2 || return Inf
    τ_c_uc = γ / (n - 2)
    # Clustering raises the threshold
    correction = ϕ * 2 / (n - 1)
    return τ_c_uc / max(1e-10, 1 - correction)
end

function _compute_threshold(net::HeterogeneousNetwork, ::BernoulliClosure, γ::Real)
    # HMF threshold: τ_c = γ⟨k⟩/⟨k²⟩
    # CPW threshold: τ_c = γ⟨k⟩/(⟨k²⟩ - ⟨k⟩)
    mk = net.mean_degree
    mk2 = net.second_moment
    denom = mk2 - mk
    denom > 0 || return Inf
    return γ * mk / denom
end

# Catch-all: produce a consistent error rather than a MethodError when an unsupported
# (network, closure) combination reaches the threshold computation.
function _compute_threshold(net::NetworkStructure, closure::ClosureMethod, ::Real)
    throw(ArgumentError(
        "epidemic_threshold is not implemented for $(typeof(net)) with $(typeof(closure)); " *
        "use BernoulliClosure or KeelingClosure with a supported network type"))
end

"""
    early_growth_rate(network, closure, τ, γ)

Compute the early exponential growth rate r₀ of the epidemic.

For homogeneous Bernoulli: r₀ = τ(n-2) - γ
"""
function early_growth_rate(network::HomogeneousNetwork,
                            closure::BernoulliClosure,
                            τ::Real, γ::Real)
    n = network.n
    return τ * (n - 2) - γ
end

function early_growth_rate(network::HeterogeneousNetwork,
                            closure::BernoulliClosure,
                            τ::Real, γ::Real)
    mk = network.mean_degree
    mk > 0 || return -γ
    return τ * (network.second_moment - 2mk) / mk - γ
end

"""
    disease_free_equilibrium(model, network; N=1.0)

Compute the disease-free equilibrium (DFE) for a pairwise system.
At DFE: all nodes are susceptible, [S]=N, [SS]=nN, all other
variables are zero.
"""
function disease_free_equilibrium(model::CompartmentalModel,
                                   network::NetworkStructure;
                                   N::Real = 1.0)
    names = model.compartment_names
    n = mean_degree(network)

    dfe = Dict{String,Float64}()

    # Singles
    first_susc = model.susceptible_compartments[1]
    for name in names
        dfe["[$name]"] = name == first_susc ? Float64(N) : 0.0
    end

    # Pairs
    for i in eachindex(names), j in i:length(names)
        a, b = names[i], names[j]
        if a == first_susc && b == first_susc
            dfe["[$a$b]"] = n * Float64(N)
        else
            dfe["[$a$b]"] = 0.0
        end
    end

    return dfe
end
