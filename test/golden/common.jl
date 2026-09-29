# common.jl — helpers shared by the NodeBasedModels golden case files.
#
# Included (not loaded as a package) by every `test/golden/<area>/cases.jl`.
# Only metadata builders and small utilities live here; they never affect the
# frozen numbers, so an area owner may extend this file without touching
# another area's goldens.

using NodeBasedModels
using Graphs

"""Solver settings of every frozen trajectory (design §G.2: ODE rtol 1e-10)."""
const GOLDEN_RELTOL = 1e-10
const GOLDEN_ABSTOL = 1e-12

"""Human-readable solver description stored in the golden metadata."""
const GOLDEN_SOLVER = "OrdinaryDiffEqDefault default algorithm, reltol = 1e-10, abstol = 1e-12, saveat = 0:1:T"

describe_model(m) = Dict{String,Any}(
    "name" => String(m.name),
    "compartments" => [String(c.name) for c in m.compartments],
    "infectious" => [String(c) for c in m.infectious_compartments],
    "susceptible" => [String(c) for c in m.susceptible_compartments],
    "transitions" => ["$(t.from) -> $(t.to) [$(t.rate), $(t.type)]" for t in m.transitions],
)

# NodeBasedModels 0.2: `sir_model()` and friends return a NetworkEpiCore ContactModel; describe
# it through the CompartmentalModel the pairwise builder lowers it to (metadata only, WP15).
describe_model(m::ContactModel) = describe_model(CompartmentalModel(m))

describe_network(net::HomogeneousNetwork) = Dict{String,Any}(
    "type" => "HomogeneousNetwork", "n" => net.n, "phi" => net.ϕ, "N" => net.N)

describe_network(net::HeterogeneousNetwork) = Dict{String,Any}(
    "type" => "HeterogeneousNetwork", "degree_probs" => net.degree_probs,
    "phi" => net.ϕ, "N" => net.N, "mean_degree" => net.mean_degree,
    "second_moment" => net.second_moment)

function describe_network(net::GraphNetwork)
    g = net.graph
    d = Dict{String,Any}(
        "type" => "GraphNetwork", "nv" => nv(g), "ne" => ne(g),
        "directed" => is_directed(g),
        "edges" => ["$(src(e))-$(dst(e))" for e in edges(g)],
        "transmission_matrix" => isnothing(net.transmission_matrix) ? "nothing (uniform infection_rate)" :
                                 "explicit (see case builder)")
    return d
end

describe_closure(c) = string(c)

"""Stringify a `Symbol => value` parameter dictionary for TOML metadata."""
describe_params(p::AbstractDict) = Dict{String,Any}(string(k) => v for (k, v) in p)

"""
    graph_from_edges(n, edges; directed = false)

Build a `SimpleGraph` / `SimpleDiGraph` on `n` vertices from an explicit edge
list, so golden graphs never depend on a random-graph generator.
"""
function graph_from_edges(n::Integer, edges; directed::Bool = false)
    g = directed ? SimpleDiGraph(n) : SimpleGraph(n)
    for (a, b) in edges
        add_edge!(g, a, b) || error("duplicate or invalid edge $a-$b")
    end
    return g
end

"""
    golden_graph(name) -> AbstractGraph

The fixed small graphs used by the graph-level goldens:

- `:karate` — Zachary's karate club (34 nodes, 78 edges, 45 triangles);
- `:kite` — Krackhardt's kite (10 nodes, 18 edges, heterogeneous degree, triangles);
- `:tree` — the complete binary tree of depth 4 (15 nodes, a tree);
- `:digraph` — an 8-node directed graph: a directed ring plus four chords.
"""
function golden_graph(name::Symbol)
    name === :karate && return smallgraph(:karate)
    name === :kite && return smallgraph(:krackhardtkite)
    name === :tree && return binary_tree(4)
    name === :digraph && return graph_from_edges(8,
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 1),
         (1, 5), (3, 7), (6, 2), (8, 4)]; directed = true)
    throw(ArgumentError("unknown golden graph $name"))
end

"""
    heterogeneous_transmission(g) -> Matrix{Float64}

A deterministic heterogeneous per-edge rate matrix on `g`: `T[i, j]` (rate from
`j` to `i`) is `0.15 + 0.05·((i + 2j) mod 5)` for each edge, so rates differ by
direction and by edge.
"""
function heterogeneous_transmission(g)
    n = nv(g)
    T = zeros(n, n)
    for e in edges(g)
        s, d = src(e), dst(e)
        T[d, s] = 0.15 + 0.05 * ((d + 2s) % 5)
        is_directed(g) || (T[s, d] = 0.15 + 0.05 * ((s + 2d) % 5))
    end
    return T
end
