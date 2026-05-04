# networks.jl — Network structure definitions
#
# Describes the contact network topology independently of the disease model.
# Two main types: homogeneous (all nodes have degree n) and heterogeneous
# (nodes have a degree distribution p_k).

"""
    NetworkStructure

Abstract type for network topologies used in pairwise models.
"""
abstract type NetworkStructure end

"""
    HomogeneousNetwork(n; ϕ=0.0, N=1.0)

A homogeneous (regular) network where every node has degree `n`.

- `n` — degree of each node
- `ϕ` — Keeling's clustering coefficient (ratio of triangles to triples)
- `N` — total population size (used for normalization)
"""
struct HomogeneousNetwork <: NetworkStructure
    n::Int           # node degree
    ϕ::Float64       # clustering coefficient [0,1]
    N::Float64       # population size
end

HomogeneousNetwork(n::Int; ϕ::Real = 0.0, N::Real = 1.0) =
    HomogeneousNetwork(n, Float64(ϕ), Float64(N))

"""
    HeterogeneousNetwork(degree_probs; ϕ=0.0, N=1.0)

A heterogeneous network with degree distribution `p_k` for k = 0, 1, ..., K.

- `degree_probs` — vector where `degree_probs[k+1]` = P(degree = k)
- `ϕ` — global clustering coefficient
- `N` — total population size
"""
struct HeterogeneousNetwork <: NetworkStructure
    degree_probs::Vector{Float64}  # p_k for k = 0, ..., K
    max_degree::Int
    ϕ::Float64
    N::Float64
    # Cached moments
    mean_degree::Float64           # ⟨k⟩
    second_moment::Float64         # ⟨k²⟩
    third_moment::Float64          # ⟨k³⟩
    excess_degree::Float64         # ⟨k(k-1)⟩/⟨k⟩ = (⟨k²⟩ - ⟨k⟩)/⟨k⟩
end

function HeterogeneousNetwork(degree_probs::AbstractVector;
                               ϕ::Real = 0.0, N::Real = 1.0)
    p = collect(Float64, degree_probs)
    isapprox(sum(p), 1.0; atol=1e-8) || throw(ArgumentError(
        "degree_probs must sum to 1, got $(sum(p))"))
    K = length(p) - 1
    mk = sum((k) * p[k+1] for k in 0:K)
    mk2 = sum(k^2 * p[k+1] for k in 0:K)
    mk3 = sum(k^3 * p[k+1] for k in 0:K)
    excess = mk > 0 ? (mk2 - mk) / mk : 0.0
    HeterogeneousNetwork(p, K, Float64(ϕ), Float64(N), mk, mk2, mk3, excess)
end

# ─── Convenience constructors ─────────────────────────────────────────────────

"""
    regular_network(n; ϕ=0.0, N=1.0)

Shorthand for `HomogeneousNetwork(n; ϕ=ϕ, N=N)`.
"""
regular_network(n::Int; kwargs...) = HomogeneousNetwork(n; kwargs...)

"""
    erdos_renyi_network(mean_degree; max_k=nothing, N=1.0)

Erdős–Rényi network with Poisson(mean_degree) degree distribution.
Truncated at `max_k` (default: mean_degree + 5√mean_degree).
"""
function erdos_renyi_network(κ::Real; max_k::Union{Int,Nothing} = nothing,
                              N::Real = 1.0)
    K = isnothing(max_k) ? round(Int, κ + 5 * sqrt(κ)) : max_k
    p = [exp(-κ) * κ^k / factorial(big(k)) for k in 0:K]
    p = Float64.(p ./ sum(p))
    HeterogeneousNetwork(p; N=N)
end

"""
    degree_distribution_network(probs; ϕ=0.0, N=1.0)

Create a HeterogeneousNetwork from an explicit degree distribution.
"""
degree_distribution_network(probs::AbstractVector; kwargs...) =
    HeterogeneousNetwork(probs; kwargs...)

# ─── Accessors ─────────────────────────────────────────────────────────────────

mean_degree(net::HomogeneousNetwork) = Float64(net.n)
mean_degree(net::HeterogeneousNetwork) = net.mean_degree

second_moment(net::HomogeneousNetwork) = Float64(net.n^2)
second_moment(net::HeterogeneousNetwork) = net.second_moment

excess_degree(net::HomogeneousNetwork) = Float64(net.n - 1)
excess_degree(net::HeterogeneousNetwork) = net.excess_degree

clustering(net::NetworkStructure) = net.ϕ

population_size(net::NetworkStructure) = net.N

# ─── Graph-instance network ───────────────────────────────────────────────────

"""
    GraphNetwork <: NetworkStructure

Network defined by a specific graph instance.
Used for individual-level, pair-based, and stochastic graph models.

# Fields
- `graph` — Graphs.jl AbstractGraph
- `transmission_matrix` — matrix with `T[i,j]` equal to the per-edge infection
  rate from node `j` to node `i`, or `nothing` for a uniform infection rate
  supplied by the solver keyword arguments
"""
struct GraphNetwork <: NetworkStructure
    graph::Any  # AbstractGraph from Graphs.jl
    transmission_matrix::Union{Nothing, Matrix{Float64}}
end

function GraphNetwork(g; transmission_rate::Float64=1.0)
    N = Graphs.nv(g)
    if transmission_rate == 1.0
        return GraphNetwork(g, nothing)
    end
    T = zeros(N, N)
    for e in Graphs.edges(g)
        s, d = Graphs.src(e), Graphs.dst(e)
        T[d, s] = transmission_rate  # T_ij = rate from j to i
        if !Graphs.is_directed(g)
            T[s, d] = transmission_rate
        end
    end
    return GraphNetwork(g, T)
end

# Implement NetworkStructure interface
function mean_degree(net::GraphNetwork)
    if Graphs.is_directed(net.graph)
        return Graphs.ne(net.graph) / Graphs.nv(net.graph)
    end
    return 2.0 * Graphs.ne(net.graph) / Graphs.nv(net.graph)
end

function Base.show(io::IO, net::GraphNetwork)
    N = Graphs.nv(net.graph)
    E = Graphs.ne(net.graph)
    k = round(mean_degree(net); digits=2)
    print(io, "GraphNetwork(N=$N, E=$E, ⟨k⟩=$k)")
end

function _effective_transmission_matrix(net::GraphNetwork, infection_rate::Real)
    if !isnothing(net.transmission_matrix)
        return net.transmission_matrix
    end

    g = net.graph
    N = Graphs.nv(g)
    T = zeros(Float64, N, N)
    rate = Float64(infection_rate)
    for e in Graphs.edges(g)
        s, d = Graphs.src(e), Graphs.dst(e)
        T[d, s] = rate
        if !Graphs.is_directed(g)
            T[s, d] = rate
        end
    end
    return T
end

_infection_sources(g, i::Int) =
    Graphs.is_directed(g) ? Graphs.inneighbors(g, i) : Graphs.neighbors(g, i)
