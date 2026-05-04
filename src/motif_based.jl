# motif_based.jl — Motif / subgraph closure framework (Phase B(a1))
#
# Reference: Keeling, House, Cooper & Pellis (2016), "Systematic
# Approximations to Susceptible-Infectious-Susceptible Dynamics on
# Networks", Approximation 2 (motif/subgraph closure). See also the
# Kirkwood superposition approximation.
#
# Counting convention (LOCKED — do not deviate):
#
#   * Motifs are connected INDUCED subgraphs of the host network class.
#   * Each variable counts UNORDERED EMBEDDINGS of an m-vertex induced
#     subgraph: each subgraph counted exactly once regardless of vertex
#     labelling, and partitioned by canonical state class under the
#     shape's automorphism group.
#   * Multiplicity bookkeeping: when relating to "labeled" or "rooted"
#     quantities (e.g. directed pair counts in the standard Keeling
#     pairwise convention) we multiply by the orbit size explicitly.
#     For the P₂ shape with automorphism {id, swap}:
#       - state class [:S,:S] (orbit size 1):  E_SS = #undirected SS edges
#       - state class [:S,:I] (orbit size 2):  E_SI = #undirected SI edges
#       - state class [:I,:I] (orbit size 1):  E_II = #undirected II edges
#     Conversion to the mixed Keeling pairwise convention used in
#     `population_pairwise.jl`:
#       [SS]_pairwise = 2·E_SS, [SI]_pairwise = E_SI, [II]_pairwise = 2·E_II.
#
# This file implements the trivial m=2, k=2 case which is provably
# equivalent to `generate_pairwise(sis_model(), regular_network(2),
# KeelingClosure())`. Higher m or k will be added in later phases.

"""
    MotifClosure(k, m)

Motif/subgraph closure of order `m` on a `k`-regular host network class.
Subtype of [`ClosureMethod`](@ref). Currently only `(k, m) = (2, 2)` is
implemented.
"""
struct MotifClosure <: ClosureMethod
    k::Int
    m::Int
end

MotifClosure(k::Integer, m::Integer) = MotifClosure(Int(k), Int(m))

"""
    MotifShape(n_nodes, edges, automorphisms, name)

A connected induced-subgraph "shape": its vertex set, edge set, the
automorphism group (each automorphism is a permutation of `1:n_nodes`)
and a symbolic `name` (e.g. `:P2`, `:singleton`).
"""
struct MotifShape
    n_nodes::Int
    edges::Vector{Tuple{Int,Int}}
    automorphisms::Vector{Vector{Int}}
    name::Symbol
end

"""
    MotifVariable(shape, state, orbit_size)

A single dynamical variable: the count of induced subgraphs isomorphic to
`shape` whose vertex states (in canonical order) are `state`. `orbit_size`
is the size of the automorphism orbit of `state` (used for bookkeeping
when converting to/from labeled quantities).
"""
struct MotifVariable
    shape::MotifShape
    state::Vector{Symbol}
    orbit_size::Int
end

"""
    MotifSystem

Container for a motif-closure ODE system.

Fields:
- `shapes::Vector{MotifShape}` — shape table (singleton + m-vertex shape(s))
- `variables::Vector{MotifVariable}` — ordered list of dynamical variables
- `index::Dict{Tuple{Symbol,Vector{Symbol}},Int}` — `(shape.name, state) → index in variables`
- `rhs!::Function` — closed-over `(du, u, p, t) -> nothing`
- `u0::Vector{Float64}` — initial condition (in `variables` order)
- `tspan::Tuple{Float64,Float64}`
- `params::NamedTuple` — model parameters (e.g. `(β=…, γ=…, k=…, N=…)`)
- `model`, `network`, `closure` — references to the underlying objects
"""
struct MotifSystem
    shapes::Vector{MotifShape}
    variables::Vector{MotifVariable}
    index::Dict{Tuple{Symbol,Vector{Symbol}},Int}
    rhs!::Function
    u0::Vector{Float64}
    tspan::Tuple{Float64,Float64}
    params::NamedTuple
    model::Any
    network::Any
    closure::MotifClosure
end

# ─── Helpers ────────────────────────────────────────────────────────────────

"""
    safe_ratio(num, den; tol=1e-12) -> Float64

Numerically safe ratio: returns `0.0` when `den < tol`, else `num/den`.
The denominator is NOT clamped — the value is returned as-is when small,
the ratio is simply set to zero. Used for closure terms of the form
`[AB][BC]/[B]` that vanish when `[B]` does.
"""
@inline function safe_ratio(num::Real, den::Real; tol::Real = 1e-12)
    return den < tol ? 0.0 : num / den
end

# ─── Shape enumeration ─────────────────────────────────────────────────────
#
# For (k=2, m=2) the only connected induced 2-vertex subgraph of any
# k-regular graph (k≥1) is the path/edge P₂. We also carry a singleton
# "shape" for the node-level variables ⟨S⟩, ⟨I⟩ that appear in closure
# denominators.

const _SINGLETON_SHAPE = MotifShape(1, Tuple{Int,Int}[], [[1]], :singleton)
const _P2_SHAPE        = MotifShape(2, [(1,2)], [[1,2], [2,1]], :P2)
# C₃ = triangle on 3 vertices, edges {(1,2),(2,3),(1,3)}. Automorphism
# group is the full symmetric group S₃ (all 6 permutations of {1,2,3}).
# Canonical state classes: [SSS] (orbit 1), [ISS] (orbit 3), [IIS]
# (orbit 3), [III] (orbit 1).
const _C3_SHAPE        = MotifShape(3,
    [(1,2),(2,3),(1,3)],
    [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]],
    :C3)
# P₃ = path on 3 vertices, edges {(1,2),(2,3)}. Automorphism group is
# {identity, end-swap}. For a 2-regular host of length ≥ 4, P₃ is the
# unique connected 3-vertex INDUCED subgraph (the triangle C₃ does not
# occur as an induced subgraph). Host rings of length 3 (host = C₃
# itself) are not handled by this closure — assume |V(host)| ≥ 4.
const _P3_SHAPE        = MotifShape(3, [(1,2),(2,3)], [[1,2,3],[3,2,1]], :P3)

# ─── Brute-force automorphism computation (used for B(c) 4-vertex shapes) ─
"""
    _compute_automorphisms(n, edges) -> Vector{Vector{Int}}

Brute-force automorphism group of the simple graph with vertex set `1:n`
and edge list `edges`. Iterates over all `n!` permutations and keeps those
that preserve the (unordered) edge set. Used to build the 4-vertex shape
constants below — for `n = 4` this enumerates 24 permutations, which is
trivial.
"""
function _compute_automorphisms(n::Int, edges::Vector{Tuple{Int,Int}})
    edge_set = Set{Tuple{Int,Int}}()
    for (a, b) in edges
        push!(edge_set, (min(a, b), max(a, b)))
    end
    autos = Vector{Int}[]
    perm = collect(1:n)
    function visit(k_)
        if k_ > n
            ok = true
            for (a, b) in edges
                ee = (min(perm[a], perm[b]), max(perm[a], perm[b]))
                if !(ee in edge_set); ok = false; break; end
            end
            ok && push!(autos, copy(perm))
            return
        end
        for j in k_:n
            perm[k_], perm[j] = perm[j], perm[k_]
            visit(k_ + 1)
            perm[k_], perm[j] = perm[j], perm[k_]
        end
    end
    visit(1)
    return autos
end

# ─── Connected 4-vertex shapes on a 3-regular host (Phase B(c)) ────────────
# Six distinct connected induced 4-vertex subgraphs can occur in a 3-regular
# host (the constraint is deg_inside(v) ≤ 3 for every vertex of the motif):
#
#   :P4    path 1-2-3-4                        (10 canonical SIS states)
#   :K13   claw / star K_{1,3}                 ( 8 canonical SIS states)
#   :paw   triangle (1,2,3) + pendant 4 at 1   (12 canonical SIS states)
#   :C4    4-cycle 1-2-3-4-1                   ( 6 canonical SIS states)
#   :K4me  K_4 − e (diamond): K_4 minus (2,4)  ( 9 canonical SIS states)
#   :K4    complete graph K_4                  ( 5 canonical SIS states)
#
# For each shape the automorphism group is computed by brute force; the
# canonical-state counts above are verified at construction time via
# `enumerate_state_classes`.
const _P4_EDGES   = [(1,2),(2,3),(3,4)]
const _K13_EDGES  = [(1,2),(1,3),(1,4)]
const _PAW_EDGES  = [(1,2),(1,3),(2,3),(1,4)]
const _C4_EDGES   = [(1,2),(2,3),(3,4),(4,1)]
const _K4ME_EDGES = [(1,2),(2,3),(3,4),(4,1),(1,3)]
const _K4_EDGES   = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

const _P4_SHAPE   = MotifShape(4, _P4_EDGES,
                               _compute_automorphisms(4, _P4_EDGES),   :P4)
const _K13_SHAPE  = MotifShape(4, _K13_EDGES,
                               _compute_automorphisms(4, _K13_EDGES),  :K13)
const _PAW_SHAPE  = MotifShape(4, _PAW_EDGES,
                               _compute_automorphisms(4, _PAW_EDGES),  :paw)
const _C4_SHAPE   = MotifShape(4, _C4_EDGES,
                               _compute_automorphisms(4, _C4_EDGES),   :C4)
const _K4ME_SHAPE = MotifShape(4, _K4ME_EDGES,
                               _compute_automorphisms(4, _K4ME_EDGES), :K4me)
const _K4_SHAPE   = MotifShape(4, _K4_EDGES,
                               _compute_automorphisms(4, _K4_EDGES),   :K4)

const _SHAPES_4V_K3 = (_P4_SHAPE, _K13_SHAPE, _PAW_SHAPE,
                       _C4_SHAPE, _K4ME_SHAPE, _K4_SHAPE)

"""
    _path_shape(m::Int) -> MotifShape

Return the path-graph shape on `m` vertices (`P_m`): vertices `1..m`,
edges `{(i,i+1) : i=1..m-1}`, automorphism group `{identity, reflection}`.
For `m ∈ {1, 2, 3}` returns the corresponding pre-defined constant.
"""
function _path_shape(m::Int)
    m == 1 && return _SINGLETON_SHAPE
    m == 2 && return _P2_SHAPE
    m == 3 && return _P3_SHAPE
    edges = [(i, i+1) for i in 1:m-1]
    autos = [collect(1:m), collect(m:-1:1)]
    return MotifShape(m, edges, autos, Symbol("P", m))
end

"""
    enumerate_shapes(closure::MotifClosure) -> Vector{MotifShape}

Return the shape table for the requested `(k, m)`. Phase B(a3) supports
`k = 2` and `2 ≤ m ≤ 6`. The generic chain builder tracks only the
singleton, the pair `P_2`, and the highest-order path `P_m` (intermediate
`P_j` for `3 ≤ j < m` are not stored — they are obtained on-the-fly as
analytical marginals of `P_m` whenever needed).
"""
function enumerate_shapes(closure::MotifClosure)
    if closure.k == 2 && closure.m == 2
        return [_SINGLETON_SHAPE, _P2_SHAPE]
    elseif closure.k == 2 && 3 <= closure.m <= 6
        return [_SINGLETON_SHAPE, _P2_SHAPE, _path_shape(closure.m)]
    elseif closure.k == 3 && closure.m == 2
        return [_SINGLETON_SHAPE, _P2_SHAPE]
    elseif closure.k == 3 && closure.m == 3
        # Phase B(b): two distinct 3-vertex induced shapes coexist on a
        # 3-regular host — open path P₃ and triangle C₃.
        return [_SINGLETON_SHAPE, _P2_SHAPE, _P3_SHAPE, _C3_SHAPE]
    elseif closure.k == 3 && closure.m == 4
        # Phase B(c): six distinct 4-vertex induced shapes can occur in a
        # 3-regular host. We track all of them on top of the B(b) singleton,
        # P₂, P₃, C₃ layout.
        return [_SINGLETON_SHAPE, _P2_SHAPE, _P3_SHAPE, _C3_SHAPE,
                _P4_SHAPE, _K13_SHAPE, _PAW_SHAPE,
                _C4_SHAPE, _K4ME_SHAPE, _K4_SHAPE]
    end
    throw(ArgumentError(
        "Motif shape enumeration for (k=$(closure.k), m=$(closure.m)) is not yet implemented; supported: k=2 with 2 ≤ m ≤ 6, k=3 with m ∈ {2,3,4}."))
end

# ─── State enumeration / canonicalisation ──────────────────────────────────

"""
    canonical_state(shape, state) -> (canonical::Vector{Symbol}, orbit_size::Int)

Return the lexicographically-smallest representative of `state` under the
shape's automorphism group, plus the orbit size.
"""
function canonical_state(shape::MotifShape, state::AbstractVector{Symbol})
    length(state) == shape.n_nodes ||
        throw(ArgumentError("state length $(length(state)) ≠ shape.n_nodes $(shape.n_nodes)"))
    orbit = Set{Vector{Symbol}}()
    for σ in shape.automorphisms
        push!(orbit, [state[σ[i]] for i in 1:shape.n_nodes])
    end
    canon = minimum(orbit)
    return canon, length(orbit)
end

"""
    enumerate_state_classes(shape, base_states) -> Vector{Tuple{Vector{Symbol},Int}}

Enumerate all canonical state classes of `shape` over `base_states`,
returned as `(canonical_state, orbit_size)` pairs in lex order.
"""
function enumerate_state_classes(shape::MotifShape, base_states::AbstractVector{Symbol})
    seen = Dict{Vector{Symbol},Int}()
    n = shape.n_nodes
    # iterate over base_states^n
    function iter(prefix)
        if length(prefix) == n
            canon, osz = canonical_state(shape, prefix)
            seen[canon] = osz
            return
        end
        for s in base_states
            iter(vcat(prefix, [s]))
        end
    end
    iter(Symbol[])
    keys_sorted = sort(collect(keys(seen)))
    return [(k, seen[k]) for k in keys_sorted]
end

# ─── Variable layout ───────────────────────────────────────────────────────

function _build_variables(closure::MotifClosure, base_states::Vector{Symbol})
    shapes = enumerate_shapes(closure)
    vars = MotifVariable[]
    idx  = Dict{Tuple{Symbol,Vector{Symbol}},Int}()
    for sh in shapes
        for (st, osz) in enumerate_state_classes(sh, base_states)
            push!(vars, MotifVariable(sh, st, osz))
            idx[(sh.name, st)] = length(vars)
        end
    end
    return shapes, vars, idx
end

# ─── RHS for SIS, (k=2, m=2) ───────────────────────────────────────────────
#
# Variables (in the order produced by `_build_variables`):
#
#   singletons:  ⟨I⟩, ⟨S⟩          (lex order on Symbol: :I < :S)
#   P₂ pairs  :  E_II, E_IS, E_SS  (lex on canonical state)
#
# With κ ≡ (k-1)/k (the standard Keeling factor), the equations in motif
# (unordered-edge) variables are derived from the canonical Keeling
# pairwise system by substituting [SS]_pw = 2·E_SS, [II]_pw = 2·E_II,
# [SI]_pw = E_SI:
#
#   d⟨S⟩/dt =  γ·⟨I⟩ - β·E_SI
#   d⟨I⟩/dt = -γ·⟨I⟩ + β·E_SI
#   dE_SS/dt = -β·κ·(2·E_SS·E_SI)/⟨S⟩ + γ·E_SI
#   dE_SI/dt =  β·κ·(2·E_SS·E_SI - E_SI²)/⟨S⟩ - β·E_SI - γ·E_SI + 2γ·E_II
#   dE_II/dt =  β·κ·E_SI²/⟨S⟩ + β·E_SI - 2γ·E_II
#
# Closure terms expressed via `safe_ratio`:
#   triple_ISI ≈ safe_ratio(E_SI·E_SI, ⟨S⟩) · κ
#   triple_ISS ≈ safe_ratio(E_SI·(2·E_SS), ⟨S⟩) · κ        (S in middle)

function _build_sis_k2_m2_rhs(idx::Dict{Tuple{Symbol,Vector{Symbol}},Int})
    iI  = idx[(:singleton, [:I])]
    iS  = idx[(:singleton, [:S])]
    iII = idx[(:P2, [:I, :I])]
    iIS = idx[(:P2, [:I, :S])]   # canonical (lex) form of {S,I}
    iSS = idx[(:P2, [:S, :S])]

    function rhs!(du, u, p, t)
        β = p.β; γ = p.γ; k = p.k
        κ = (k - 1) / k

        S  = u[iS];  I  = u[iI]
        SS = u[iSS]; SI = u[iIS]; II = u[iII]

        triple_ISS = safe_ratio(SI * (2 * SS), S) * κ
        triple_ISI = safe_ratio(SI * SI,        S) * κ

        du[iS]  =  γ * I - β * SI
        du[iI]  = -γ * I + β * SI
        du[iSS] = -β * triple_ISS + γ * SI
        du[iIS] =  β * triple_ISS - β * triple_ISI - β * SI - γ * SI + 2γ * II
        du[iII] =  β * triple_ISI + β * SI - 2γ * II
        return nothing
    end
    return rhs!
end

# ─── Initial conditions ────────────────────────────────────────────────────
#
# Random-mixing IC consistent with the singleton / pair-count convention
# above. For a k-regular network on N nodes the total number of
# (undirected, induced) edges is N·k/2; partition by independent vertex
# states with infected fraction ε.

function _build_sis_k2_m2_ic(idx, N::Float64, ε::Float64, k::Int)
    nvars = maximum(values(idx))
    u0 = zeros(Float64, nvars)
    Etot = N * k / 2
    u0[idx[(:singleton, [:S])]] = N * (1 - ε)
    u0[idx[(:singleton, [:I])]] = N * ε
    u0[idx[(:P2, [:S, :S])]]    = Etot * (1 - ε)^2
    u0[idx[(:P2, [:I, :S])]]    = Etot * 2 * ε * (1 - ε)
    u0[idx[(:P2, [:I, :I])]]    = Etot * ε^2
    return u0
end

# ─── Public entry point ────────────────────────────────────────────────────

"""
    motif_based_sis(; β, γ, k, m,
                     tspan=(0.0,100.0), N=1.0, ε=1e-3, kwargs...) -> MotifSystem

Build a motif-closure SIS system on a `k`-regular host with motif order
`m`. Currently `k = 2` with `2 ≤ m ≤ 6` is implemented:

  * `m = 2` (Phase B(a1)) and `m = 3` (Phase B(a2)) use hand-derived
    specialised RHS builders.
  * `m ∈ {4, 5, 6}` (Phase B(a3)) uses a generic chain builder that
    tracks only the singleton, the pair `P_2`, and the highest-order
    path `P_m`. Closure enters at the (m+1)-vertex level via the
    order-m Kirkwood/path factorisation:

        L_{(s_0, s_1, …, s_m)}_{P_{m+1}}
            ≈ L_{(s_0, …, s_{m-1})}_{P_m} · L_{(s_1, …, s_m)}_{P_m}
                 / L_{(s_1, …, s_{m-1})}_{P_{m-1}}

    where `L_{(σ)}_{P_{m-1}}` is computed on-the-fly as a marginal of
    the tracked `P_m` variables.

The hidden kwarg `_use_generic_chain_builder=false` lets you A/B test
the generic builder against the specialised m=3 builder. This is for
diagnostic use only — the generic chain builder uses a strictly
higher-order closure than the original m=3 builder, so the two RHSs
agree exactly only at the random-mixing IC, not under perturbation.

For `k = 3, m = 4`, the system is implemented but should not be interpreted
as a certified monotone refinement of `m = 3`: the Lean marginalisation
theorems in `EdgeBasedModels.jl/proofs/EBCMCategory/` show that the
Kirkwood-form order-4 RHS need not project to a better order-3 RHS.

Other `(k, m)` combinations throw `ArgumentError`.
"""
function motif_based_sis(; β::Real, γ::Real, k::Integer, m::Integer,
                          tspan = (0.0, 100.0),
                          N::Real = 1.0,
                          ε::Real = 1e-3,
                          n_p3::Real = -1.0,
                          n_c3::Real = 0.0,
                          n_p4::Real   = -1.0,
                          n_k13::Real  = -1.0,
                          n_paw::Real  = 0.0,
                          n_c4::Real   = 0.0,
                          n_k4me::Real = 0.0,
                          n_k4::Real   = 0.0,
                          _use_generic_chain_builder::Bool = false,
                          kwargs...)
    k = Int(k); m = Int(m)
    supported = (k == 2 && 2 <= m <= 6) || (k == 3 && 2 <= m <= 4)
    if !supported
        throw(ArgumentError(
            "motif_based_sis(k=$k, m=$m) is not yet implemented; supported: k=2 with 2 ≤ m ≤ 6, k=3 with m ∈ {2,3,4}."))
    end
    if k == 3 && m == 4
        @warn "motif_based_sis(k=3, m=4) is implemented, but Lean T3b/T7 certify that this Kirkwood refinement need not marginalise monotonically to m=3." maxlog=1
    end

    closure = MotifClosure(k, m)
    base_states = [:S, :I]
    shapes, vars, idx = _build_variables(closure, base_states)
    if k == 2 && m == 2 && !_use_generic_chain_builder
        rhs! = _build_sis_k2_m2_rhs(idx)
        u0   = _build_sis_k2_m2_ic(idx, Float64(N), Float64(ε), k)
    elseif k == 2 && m == 3 && !_use_generic_chain_builder
        rhs! = _build_sis_k2_m3_rhs(idx)
        u0   = _build_sis_k2_m3_ic(idx, Float64(N), Float64(ε), k)
    elseif k == 3 && m == 2
        # The (k=2, m=2) builder is fully parametric in p.k (closure factor
        # κ = (k-1)/k computed from the parameter NamedTuple at runtime),
        # and the IC builder uses Etot = N·k/2. Reuse them for k=3.
        rhs! = _build_sis_k2_m2_rhs(idx)
        u0   = _build_sis_k2_m2_ic(idx, Float64(N), Float64(ε), k)
    elseif k == 3 && m == 3
        # Default n_p3 for a random 3-regular host: each vertex is centre
        # of C(3,2)=3 induced P₃'s minus the triangles incident to it,
        # giving Σ ≈ 3N - 3·n_c3 induced P₃'s. The user is encouraged to
        # supply graph-derived n_p3, n_c3 for quantitative comparisons.
        np3 = n_p3 < 0 ? max(0.0, 3.0 * Float64(N) - 3.0 * Float64(n_c3)) :
                         Float64(n_p3)
        nc3 = Float64(n_c3)
        rhs! = _build_sis_k3_m3_rhs(idx)
        u0   = _build_sis_k3_m3_ic(idx, Float64(N), Float64(ε), np3, nc3)
    elseif k == 3 && m == 4
        # Phase B(c): six 4-vertex shapes added on top of B(b) layout.
        # Default counts for an asymptotic random 3-regular host: every
        # vertex is the centre of one K_{1,3}, and each induced P_3
        # extends ≈ 2 ways into a P_4 (per endpoint, 2 choices for the
        # next neighbour minus the rare loop-back). Other 4-vertex shapes
        # (paw, C_4, K_4-e, K_4) have asymptotic density 0; users with
        # quantitative graphs should call `induced_subgraph_counts_4vertex`
        # and pass the precise counts.
        nc3   = Float64(n_c3)
        np3   = n_p3 < 0  ? max(0.0, 3.0 * Float64(N) - 3.0 * nc3) :
                            Float64(n_p3)
        np4   = n_p4 < 0  ? max(0.0, 6.0 * Float64(N))             :
                            Float64(n_p4)
        nk13  = n_k13 < 0 ? Float64(N)                              :
                            Float64(n_k13)
        npaw  = Float64(n_paw)
        nc4   = Float64(n_c4)
        nk4me = Float64(n_k4me)
        nk4   = Float64(n_k4)
        rhs! = _build_sis_k3_m4_rhs(idx)
        u0   = _build_sis_k3_m4_ic(idx, Float64(N), Float64(ε),
                                    np3, nc3, np4, nk13, npaw, nc4, nk4me, nk4)
    else
        # k=2 and m ∈ {3, 4, 5, 6} (with m=2 only via the diagnostic flag).
        m >= 3 || throw(ArgumentError(
            "_use_generic_chain_builder requires m ≥ 3 (got m=$m)."))
        rhs! = _build_sis_k2_chain_rhs(m, idx)
        u0   = _build_sis_k2_chain_ic(m, idx, Float64(N), Float64(ε), k)
    end

    model   = sis_model()
    network = regular_network(k)

    params = (β = Float64(β), γ = Float64(γ), k = k, N = Float64(N), ε = Float64(ε))
    return MotifSystem(shapes, vars, idx, rhs!, u0,
                       (Float64(tspan[1]), Float64(tspan[2])),
                       params, model, network, closure)
end

# ─── RHS for SIS, (k=2, m=3) ───────────────────────────────────────────────
#
# Variables (in the order produced by `_build_variables`):
#
#   singletons:  ⟨I⟩, ⟨S⟩
#   P₂ pairs  :  E_II, E_IS, E_SS
#   P₃ triples:  E_{(I,I,I)}, E_{(I,I,S)}, E_{(I,S,I)},
#                E_{(I,S,S)}, E_{(S,I,S)}, E_{(S,S,S)}
#
# Total = 11 ODE variables.
#
# Counting conventions (LOCKED):
#   * E_c (canonical) counts unordered induced subgraphs in canonical
#     state class c.
#   * Labelled count L_σ (ordered embedding (1=σ₁, 2=σ₂, …)):
#       L_σ = E_{canon(σ)} · |stab(canon(σ))|
#     where for P₂  (|G|=2):  L_{XX} = 2·E_{XX}, L_{XY}=L_{YX} = E_{XY}.
#     For P₃ (|G|=2): palindromes have L = 2·E, non-palindromes L = E.
#
# Closure is now lifted to the 4-node level (open path P₄ on
# (e,1,2,3)). For an external slot at position 1 (an end of the P₃),
# we use the (m+1)-th order Kirkwood factorisation that conditions on
# the existing triple and extends by one vertex:
#
#   L_{(X,σ₁,σ₂,σ₃)}_P₄ ≈ L_{(σ₁,σ₂,σ₃)}_P₃ · L_{(X,σ₁)}_P₂ / ⟨σ₁⟩
#
# (Same form for the right-side extension, with σ₃ playing σ₁'s role.)
# This factorisation
#   (a) reuses the triple variables we already track,
#   (b) has a single-vertex denominator (more robust as ⟨S⟩ → 0),
#   (c) is exact when (X,σ₁,σ₂,σ₃) factorises as
#       P(X|σ₁,σ₂,σ₃) ≈ P(X|σ₁), i.e. the next vertex depends only on
#       its immediate neighbour given the rest of the path.
#
# Pair derivatives are NOT closed at the pair level — they are exact
# marginals of triple transitions:
#
#   dE_{SS}/dt = -β·E_{(I,S,S)} + γ·E_{IS}
#   dE_{IS}/dt = -γ·E_{IS} - β·E_{IS} - 2β·E_{(I,S,I)} + β·E_{(I,S,S)} + 2γ·E_{II}
#   dE_{II}/dt = -2γ·E_{II} + β·E_{IS} + 2β·E_{(I,S,I)}
#
# Singleton derivatives are exact marginals of pair transitions:
#
#   d⟨S⟩/dt =  γ·⟨I⟩ - β·E_{IS}
#   d⟨I⟩/dt = -γ·⟨I⟩ + β·E_{IS}
#
# Triple derivatives are accumulated by enumerating, for each labelled
# triple state σ, the per-vertex transition rates (recovery, internal
# infection, external infection through a closed P₄), and then
# converting flow-of-L_σ into flow-of-E_c via dE_c = dL_{rep}/|stab(c)|.

# Pre-tabulated triple state classes for (k=2, m=3): (canonical state,
# orbit_size). Lex order matches `enumerate_state_classes`.
const _P3_TRIPLE_STATES = [
    ([:I, :I, :I], 1),
    ([:I, :I, :S], 2),
    ([:I, :S, :I], 1),
    ([:I, :S, :S], 2),
    ([:S, :I, :S], 1),
    ([:S, :S, :S], 1),
]

# All 8 labelled triple states (Vector{Symbol}, encoded as 0..7).
const _P3_LABELLED = [
    [:S, :S, :S], [:S, :S, :I], [:S, :I, :S], [:S, :I, :I],
    [:I, :S, :S], [:I, :S, :I], [:I, :I, :S], [:I, :I, :I],
]

@inline _bit(s::Symbol) = (s === :I) ? 1 : 0
@inline _enc3(σ) = (_bit(σ[1]) << 2) | (_bit(σ[2]) << 1) | _bit(σ[3])

function _build_sis_k2_m3_rhs(idx::Dict{Tuple{Symbol,Vector{Symbol}},Int})
    # Singleton + pair indices
    iI  = idx[(:singleton, [:I])]
    iS  = idx[(:singleton, [:S])]
    iII = idx[(:P2, [:I, :I])]
    iIS = idx[(:P2, [:I, :S])]
    iSS = idx[(:P2, [:S, :S])]

    # Triple indices, keyed by canonical state
    iIII = idx[(:P3, [:I, :I, :I])]
    iIIS = idx[(:P3, [:I, :I, :S])]
    iISI = idx[(:P3, [:I, :S, :I])]
    iISS = idx[(:P3, [:I, :S, :S])]
    iSIS = idx[(:P3, [:S, :I, :S])]
    iSSS = idx[(:P3, [:S, :S, :S])]

    # Map from labelled-state encoding (0..7) to (canonical_index, stab).
    # stab = |G| / |orbit| = 2/orbit_size; multiplies E_c to give L_σ.
    # Also store the canonical-state index for assembly.
    canon_idx = Vector{Int}(undef, 8)
    stab_of   = Vector{Int}(undef, 8)
    osz_of    = Vector{Int}(undef, 8)  # orbit size of canon
    for σ in _P3_LABELLED
        rev = reverse(σ)
        canon = σ <= rev ? σ : rev
        osz   = (σ == rev) ? 1 : 2
        stab  = 2 ÷ osz
        e = _enc3(σ)
        canon_idx[e + 1] =
            canon == [:I,:I,:I] ? iIII :
            canon == [:I,:I,:S] ? iIIS :
            canon == [:I,:S,:I] ? iISI :
            canon == [:I,:S,:S] ? iISS :
            canon == [:S,:I,:S] ? iSIS :
                                  iSSS
        stab_of[e + 1] = stab
        osz_of[e + 1]  = osz
    end

    # Precompute, per labelled state σ, the list of (target_enc, kind, vertex)
    # transitions. kind ∈ (:rec, :int, :ext_left, :ext_right). For each we
    # know the transition multiplicity needed to compute the rate.
    # We'll just inline the per-σ logic at runtime — only 8 states, cheap.

    function rhs!(du, u, p, t)
        β = p.β; γ = p.γ
        S  = u[iS];  Ival  = u[iI]
        SS = u[iSS]; SI = u[iIS]; II = u[iII]

        # Labelled pair counts L_{(X,Y)}: L_SS = 2·E_SS, L_II = 2·E_II,
        # L_SI = L_IS = E_SI.
        @inline function Lpair(X::Symbol, Y::Symbol)
            if X === :S && Y === :S
                return 2.0 * SS
            elseif X === :I && Y === :I
                return 2.0 * II
            else
                return SI
            end
        end
        @inline single(X::Symbol) = (X === :I) ? Ival : S

        # Labelled-flow accumulator, indexed by encoded labelled state.
        dL = zeros(Float64, 8)

        @inbounds for σ in _P3_LABELLED
            eσ = _enc3(σ)
            Lσ = u[canon_idx[eσ + 1]] * stab_of[eσ + 1]

            for i in 1:3
                if σ[i] === :I
                    # Recovery i: I → S
                    σp = (i == 1) ? [:S, σ[2], σ[3]] :
                         (i == 2) ? [σ[1], :S, σ[3]] :
                                    [σ[1], σ[2], :S]
                    flow = γ * Lσ
                    dL[eσ + 1]        -= flow
                    dL[_enc3(σp) + 1] += flow
                else
                    # σ[i] === :S, infection i: S → I
                    # Internal contribution: count internal I-neighbours
                    n_int = 0
                    if i == 1 && σ[2] === :I
                        n_int += 1
                    elseif i == 2
                        if σ[1] === :I; n_int += 1; end
                        if σ[3] === :I; n_int += 1; end
                    elseif i == 3 && σ[2] === :I
                        n_int += 1
                    end
                    flow_int = β * n_int * Lσ

                    # External contribution: only for endpoints (i=1 or 3).
                    # (m+1)-Kirkwood: L_{(I,σ)}·L_{(σ_end,I)}_P₂ / ⟨σ_end⟩
                    # = Lσ · Lpair(σ[end], :I) / ⟨σ[end]⟩
                    flow_ext = 0.0
                    if i == 1
                        # External e attaches to vertex 1 (state :S),
                        # closure L_(I,σ₁,σ₂,σ₃) ≈ Lσ · L_(I,S) / ⟨S⟩
                        flow_ext = β * safe_ratio(Lσ * Lpair(:I, :S),
                                                  single(:S))
                    elseif i == 3
                        # External e attaches to vertex 3 (state :S),
                        # closure L_(σ₁,σ₂,σ₃,I) ≈ Lσ · L_(S,I) / ⟨S⟩
                        flow_ext = β * safe_ratio(Lσ * Lpair(:S, :I),
                                                  single(:S))
                    end

                    σp = (i == 1) ? [:I, σ[2], σ[3]] :
                         (i == 2) ? [σ[1], :I, σ[3]] :
                                    [σ[1], σ[2], :I]
                    flow = flow_int + flow_ext
                    dL[eσ + 1]        -= flow
                    dL[_enc3(σp) + 1] += flow
                end
            end
        end

        # Convert dL → dE for triples. dL is constant on orbits (host
        # symmetry); pick the canonical rep, divide by stab(canon).
        du[iIII] = dL[_enc3([:I,:I,:I]) + 1] / stab_of[_enc3([:I,:I,:I]) + 1]
        du[iIIS] = dL[_enc3([:I,:I,:S]) + 1] / stab_of[_enc3([:I,:I,:S]) + 1]
        du[iISI] = dL[_enc3([:I,:S,:I]) + 1] / stab_of[_enc3([:I,:S,:I]) + 1]
        du[iISS] = dL[_enc3([:I,:S,:S]) + 1] / stab_of[_enc3([:I,:S,:S]) + 1]
        du[iSIS] = dL[_enc3([:S,:I,:S]) + 1] / stab_of[_enc3([:S,:I,:S]) + 1]
        du[iSSS] = dL[_enc3([:S,:S,:S]) + 1] / stab_of[_enc3([:S,:S,:S]) + 1]

        # Pair derivatives — exact marginals of triple transitions.
        E_ISS = u[iISS]
        E_ISI = u[iISI]
        du[iSS] = -β * E_ISS + γ * SI
        du[iIS] = -γ * SI - β * SI - 2β * E_ISI + β * E_ISS + 2γ * II
        du[iII] = -2γ * II + β * SI + 2β * E_ISI

        # Singleton derivatives — exact marginals of pair transitions.
        du[iS]  =  γ * Ival - β * SI
        du[iI]  = -γ * Ival + β * SI
        return nothing
    end
    return rhs!
end

# ─── Initial conditions for (k=2, m=3) ─────────────────────────────────────
#
# Random-mixing IC: each host vertex independently has state I with
# probability ε, S with probability 1-ε. For a 2-regular ring on N
# nodes (assume N ≥ 4) there are N edges and N induced P₃'s (one
# centred at each vertex). Counts:
#
#   ⟨X⟩         = N · P(X)
#   E_pair_c    = N · |orbit(c)| · ∏ᵢ P(c[i])    (1 edge per ring vertex)
#   E_triple_c  = N · |orbit(c)| · ∏ᵢ P(c[i])    (1 P₃ per ring vertex)
#
# Pair conservation: E_SS + E_SI + E_II = N · [(1-ε)² + 2ε(1-ε) + ε²] = N.
# Triple conservation: Σ_c E_c = N · (ε + (1-ε))³ = N.

function _build_sis_k2_m3_ic(idx, N::Float64, ε::Float64, k::Int)
    nvars = maximum(values(idx))
    u0 = zeros(Float64, nvars)
    pS = 1 - ε; pI = ε

    # Singletons
    u0[idx[(:singleton, [:S])]] = N * pS
    u0[idx[(:singleton, [:I])]] = N * pI

    # Pairs (k=2 ring → N edges)
    Etot = N
    u0[idx[(:P2, [:S, :S])]] = Etot * pS^2
    u0[idx[(:P2, [:I, :S])]] = Etot * 2 * pI * pS
    u0[idx[(:P2, [:I, :I])]] = Etot * pI^2

    # Triples (k=2 ring → N induced P₃'s)
    pofstate(s::Symbol) = (s === :I) ? pI : pS
    for (c, osz) in _P3_TRIPLE_STATES
        prod = pofstate(c[1]) * pofstate(c[2]) * pofstate(c[3])
        u0[idx[(:P3, c)]] = N * osz * prod
    end
    return u0
end

# ─── Solver ────────────────────────────────────────────────────────────────

"""
    solve_motif(sys::MotifSystem; saveat=nothing, alg=Tsit5(),
                reltol=1e-8, abstol=1e-10, kwargs...)

Solve the motif-closure ODE system. Returns the `OrdinaryDiffEq` solution
object. Use [`compartment`](@ref) to extract per-compartment trajectories.

Default tolerances are tightened from `OrdinaryDiffEq`'s defaults
(`reltol=1e-3, abstol=1e-6`) to `1e-8`/`1e-10`; closure ratios in the
motif RHS amplify integration noise, so loose tolerances can let
components drift negative. Pass `reltol`/`abstol` explicitly to override.
"""
function solve_motif(sys::MotifSystem; saveat = nothing,
                     alg = nothing,
                     reltol = 1e-8, abstol = 1e-10,
                     kwargs...)
    prob = OrdinaryDiffEqDefault.ODEProblem(sys.rhs!, sys.u0, sys.tspan, sys.params)
    if isnothing(alg)
        if saveat === nothing
            return OrdinaryDiffEqDefault.solve(prob; reltol = reltol, abstol = abstol, kwargs...)
        else
            return OrdinaryDiffEqDefault.solve(prob; saveat = saveat,
                                               reltol = reltol, abstol = abstol, kwargs...)
        end
    elseif saveat === nothing
        return OrdinaryDiffEqDefault.solve(prob, alg; reltol = reltol, abstol = abstol, kwargs...)
    else
        return OrdinaryDiffEqDefault.solve(prob, alg; saveat = saveat,
                                           reltol = reltol, abstol = abstol, kwargs...)
    end
end

# ─── Aggregation / accessors ───────────────────────────────────────────────

"""
    compartment(sys::MotifSystem, sol, base::Symbol) -> Vector{Float64}

Return the population-level trajectory of base state `base` (e.g. `:S`,
`:I`). Aggregated from singleton variables.
"""
function compartment(sys::MotifSystem, sol, base::Symbol)
    key = (:singleton, [base])
    haskey(sys.index, key) ||
        throw(ArgumentError("unknown base compartment :$base; available: $(sort([s[1] for ((sh,s),_) in sys.index if sh == :singleton]))"))
    i = sys.index[key]
    return [u[i] for u in sol.u]
end

# ─── Generic chain RHS for SIS, (k=2, m ≥ 3) ───────────────────────────────
#
# Variables:
#   singletons:  ⟨I⟩, ⟨S⟩
#   P₂ pairs  :  E_II, E_IS, E_SS
#   P_m       :  one variable per canonical state class on P_m.
#
# Number of canonical P_m states (Burnside under {id, reflection}):
#   |C_m| = (2^m + 2^⌈m/2⌉) / 2
# For m = 3, 4, 5, 6 this gives 6, 10, 20, 36.
#
# State encoding: a labelled m-tuple σ ∈ {:S,:I}^m is encoded as the integer
#   e = Σ_{i=1..m} bit(σ_i) · 2^(m-i),   bit(:I)=1, bit(:S)=0.
# So σ[1] sits in the highest (MSB) bit-position, σ[m] in the LSB.
# Reflection of e: bit-reverse the m-bit pattern.
#
# Labelled count for the canonical class containing σ:
#   L_σ_P_m = E_canon · stab(canon)        with stab = |G| / |orbit|.
# Here |G| = 2 so stab ∈ {1, 2}: stab=2 for palindromic states, 1 otherwise.
#
# ─── Closure (the generalised order-m path Kirkwood) ─────────────────────
#
# For external infection at vertex 1 of an m-tuple σ (σ_1=:S):
#   L_{(I, σ_1, …, σ_m)}_{P_{m+1}}
#       ≈ L_{(I, σ_1, …, σ_{m-1})}_{P_m} · L_σ_{P_m}
#            / L_{(σ_1, …, σ_{m-1})}_{P_{m-1}}
# Symmetric form for vertex m. The denominator P_{m-1} count is computed
# on the fly by marginalising P_m over the dropped position.
#
# At m = 3 this is L_(I,σ₁,σ₂)·L_σ / L_(σ₁,σ₂) which differs from the
# specialised B(a2) form L_σ·L_(I,σ₁)/⟨σ₁⟩. The two coincide at the
# random-mixing IC (both reduce to N·ε·∏ p(σ_i)) but in general the
# generic form is a strictly higher-order closure.
#
# ─── Pair / singleton derivatives ────────────────────────────────────────
#
# Singleton equations are model-exact:
#   d⟨S⟩/dt =  γ·⟨I⟩ - β·E_IS
#   d⟨I⟩/dt = -γ·⟨I⟩ + β·E_IS
#
# Pair equations are exact marginals of P_3-level dynamics; for m > 3
# we obtain L_{(I,S,s)}_{P_3} (and L_{(s,S,I)}_{P_3}) on the fly as
# marginals of the tracked P_m by summing over the (m-3) free positions.
#
# Concretely (with L_{(S,S)}=2·E_SS, L_{(I,I)}=2·E_II, L_{(I,S)}=L_{(S,I)}=E_IS):
#   dE_SS/dt = -β·L_{(I,S,S)}_{P_3}                 + γ·E_IS
#   dE_IS/dt =  β·L_{(I,S,S)}_{P_3} - β·L_{(I,S,I)}_{P_3} - β·E_IS - γ·E_IS + 2γ·E_II
#   dE_II/dt =  β·L_{(I,S,I)}_{P_3} + β·E_IS - 2γ·E_II
# These reduce to the specialised m=3 pair equations when m = 3 (then
# L_{(I,S,S)}_{P_3} = E_ISS and L_{(I,S,I)}_{P_3} = E_ISI exactly).

# Helper: enumerate canonical encodings (one per canonical state class).
# Returns Vector{Int}: encoded labelled states e such that decode(e) is
# lex-min among {decode(e), reverse(decode(e))}.
function _canonical_encodings(m::Int)
    canon = Int[]
    for e in 0:((1 << m) - 1)
        # bit-reverse e (m bits)
        r = 0
        for i in 0:(m-1)
            if ((e >> i) & 1) == 1
                r |= 1 << (m - 1 - i)
            end
        end
        # σ ≤ reverse(σ) iff e (read MSB-first) ≤ reverse-encoded;
        # since bits map directly to σ (MSB=σ_1), comparing σ
        # lex-order is equivalent to comparing the integer encodings.
        if e <= r
            push!(canon, e)
        end
    end
    return canon
end

@inline function _bit_reverse(e::Int, m::Int)
    r = 0
    for i in 0:(m-1)
        if ((e >> i) & 1) == 1
            r |= 1 << (m - 1 - i)
        end
    end
    return r
end

# Decode encoded e into a Vector{Symbol} of length m (σ[1] = MSB).
function _decode_state(e::Int, m::Int)
    σ = Vector{Symbol}(undef, m)
    for i in 1:m
        σ[i] = (((e >> (m - i)) & 1) == 1) ? :I : :S
    end
    return σ
end

function _build_sis_k2_chain_rhs(m::Int, idx::Dict{Tuple{Symbol,Vector{Symbol}},Int})
    @assert m >= 3 "_build_sis_k2_chain_rhs requires m ≥ 3 (got m=$m)"

    pm_name = m == 2 ? :P2 : (m == 3 ? :P3 : Symbol("P", m))

    # Singleton + pair indices
    iI  = idx[(:singleton, [:I])]
    iS  = idx[(:singleton, [:S])]
    iII = idx[(:P2, [:I, :I])]
    iIS = idx[(:P2, [:I, :S])]
    iSS = idx[(:P2, [:S, :S])]

    NLAB = 1 << m   # 2^m labelled m-tuples

    # Per-encoded-state lookup tables
    canon_idx_of = Vector{Int}(undef, NLAB)  # u-index of canonical class
    stab_of      = Vector{Int}(undef, NLAB)  # 2 / orbit_size (∈ {1,2})

    for e in 0:NLAB-1
        σ = _decode_state(e, m)
        rσ = reverse(σ)
        canon = (σ <= rσ) ? σ : rσ
        orbit = (σ == rσ) ? 1 : 2
        canon_idx_of[e + 1] = idx[(pm_name, canon)]
        stab_of[e + 1]      = 2 ÷ orbit
    end

    # List of canonical encodings (for converting dL → dE without double counting)
    canon_encodings = _canonical_encodings(m)

    # Pre-allocated working buffers (captured in the closure)
    Lvals = Vector{Float64}(undef, NLAB)
    dL    = Vector{Float64}(undef, NLAB)

    # Pre-compute, for each (a, b, c) ∈ {0,1}^3, the list of encodings of
    # m-tuples whose first three (resp. last three) bits equal (a, b, c).
    # We only need the case L_{(I, S, x)} and L_{(x, S, I)} for x ∈ {S,I},
    # so build the four lists explicitly.
    function _enc_list_left(a::Int, b::Int, c::Int)
        base = (a << (m-1)) | (b << (m-2)) | (c << (m-3))
        nfree = m - 3
        if nfree == 0
            return [base]
        end
        return [base | x for x in 0:((1 << nfree) - 1)]
    end
    function _enc_list_right(a::Int, b::Int, c::Int)
        suffix = (a << 2) | (b << 1) | c
        nfree = m - 3
        if nfree == 0
            return [suffix]
        end
        return [(x << 3) | suffix for x in 0:((1 << nfree) - 1)]
    end

    # We need:
    #   L_{(I,S,S)}_{P_3} via right-extension:  encodings starting (1,0,0)
    #   L_{(I,S,I)}_{P_3} via right-extension:  encodings starting (1,0,1)
    # (Reflection symmetry of the dynamics ensures left-extension counts agree.)
    enc_ISS_left = _enc_list_left(1, 0, 0)
    enc_ISI_left = _enc_list_left(1, 0, 1)

    # Bit mask of low (m-1) bits, used for endpoint extension encoding.
    lowmask = (1 << (m - 1)) - 1

    function rhs!(du, u, p, t)
        β = p.β; γ = p.γ
        S    = u[iS]; Ival = u[iI]
        SSv  = u[iSS]; SIv  = u[iIS]; IIv  = u[iII]

        # Materialise labelled P_m counts L_σ for every encoded state.
        @inbounds for e in 0:NLAB-1
            Lvals[e + 1] = u[canon_idx_of[e + 1]] * stab_of[e + 1]
        end

        fill!(dL, 0.0)

        @inbounds for e in 0:NLAB-1
            Lσ = Lvals[e + 1]
            for i in 1:m
                bit_i = (e >> (m - i)) & 1
                if bit_i == 1
                    # Recovery at vertex i: σ[i] :I → :S
                    e_new = e ⊻ (1 << (m - i))
                    flow = γ * Lσ
                    dL[e + 1]      -= flow
                    dL[e_new + 1]  += flow
                else
                    # Infection at vertex i: σ[i] :S → :I
                    n_int = 0
                    if i > 1 && (((e >> (m - (i - 1))) & 1) == 1)
                        n_int += 1
                    end
                    if i < m && (((e >> (m - (i + 1))) & 1) == 1)
                        n_int += 1
                    end
                    flow_int = β * n_int * Lσ

                    flow_ext = 0.0
                    if i == 1
                        # Closure for L_{(I, σ_1=S, σ_2, …, σ_m)}_{P_{m+1}}
                        #   ≈ L_{(I, σ_1, …, σ_{m-1})}_{P_m} · Lσ
                        #        / L_{(σ_1, …, σ_{m-1})}_{P_{m-1}}
                        # Numerator labelled state: bits (1, σ_1, σ_2, …, σ_{m-1})
                        e_num = (1 << (m - 1)) | (e >> 1)
                        # Denominator: marginalise over last bit of (σ_1..σ_{m-1})-prefix.
                        # The (m-1)-prefix is e >> 1; extend it by both possible final bits.
                        eprefix = e >> 1
                        eA = eprefix << 1
                        denom = Lvals[eA + 1] + Lvals[(eA | 1) + 1]
                        flow_ext = β * safe_ratio(Lvals[e_num + 1] * Lσ, denom)
                    elseif i == m
                        # Closure for L_{(σ_1, …, σ_{m-1}, σ_m=S, I)}_{P_{m+1}}
                        #   ≈ Lσ · L_{(σ_2, …, σ_m, I)}_{P_m}
                        #        / L_{(σ_2, …, σ_m)}_{P_{m-1}}
                        esuf = e & lowmask
                        e_num = (esuf << 1) | 1
                        # Denominator: marginalise over first bit of (σ_2..σ_m)-suffix.
                        # The (m-1)-suffix is esuf; prepend both possible first bits.
                        denom = Lvals[esuf + 1] + Lvals[((1 << (m - 1)) | esuf) + 1]
                        flow_ext = β * safe_ratio(Lσ * Lvals[e_num + 1], denom)
                    end

                    e_new = e | (1 << (m - i))
                    flow = flow_int + flow_ext
                    dL[e + 1]     -= flow
                    dL[e_new + 1] += flow
                end
            end
        end

        # Convert dL → dE for P_m canonicals (one assignment per canonical class).
        @inbounds for ec in canon_encodings
            du[canon_idx_of[ec + 1]] = dL[ec + 1] / stab_of[ec + 1]
        end

        # Marginalise the tracked P_m to obtain L_{(I,S,S)}_P_3 and L_{(I,S,I)}_P_3.
        L_ISS_P3 = 0.0
        @inbounds for e in enc_ISS_left
            L_ISS_P3 += Lvals[e + 1]
        end
        L_ISI_P3 = 0.0
        @inbounds for e in enc_ISI_left
            L_ISI_P3 += Lvals[e + 1]
        end

        # Pair derivatives (exact marginals at the P_3 level — closure
        # already absorbed into the P_m state through L_*_P_3 being a
        # marginal of the closed-evolved P_m).
        du[iSS] = -β * L_ISS_P3 + γ * SIv
        du[iIS] =  β * L_ISS_P3 - β * L_ISI_P3 - β * SIv - γ * SIv + 2γ * IIv
        du[iII] =  β * L_ISI_P3 + β * SIv - 2γ * IIv

        # Singleton derivatives (model-exact)
        du[iS]  =  γ * Ival - β * SIv
        du[iI]  = -γ * Ival + β * SIv

        return nothing
    end
    return rhs!
end

# ─── Generic chain initial conditions (random-mixing independence) ─────────
#
# For a 2-regular ring on N nodes (assume N ≥ m + 1) there are N induced
# P_m's (one per starting vertex). With independent vertex states
# P(:S) = 1-ε, P(:I) = ε:
#   E_σ(P_m, canonical) = N · orbit_size(σ) · ∏ᵢ p(σ_i)
#
# Conservation at IC:
#   ⟨S⟩+⟨I⟩ = N · 1 = N
#   E_SS+E_IS+E_II = N · ((1-ε)² + 2ε(1-ε) + ε²) = N
#   Σ_c E_c(P_m) = N · (ε + (1-ε))^m = N
function _build_sis_k2_chain_ic(m::Int, idx::Dict, N::Float64, ε::Float64, k::Int)
    nvars = maximum(values(idx))
    u0 = zeros(Float64, nvars)
    pS = 1 - ε; pI = ε
    pofstate(s::Symbol) = (s === :I) ? pI : pS

    # Singletons
    u0[idx[(:singleton, [:S])]] = N * pS
    u0[idx[(:singleton, [:I])]] = N * pI

    # Pairs (k=2 ring → N edges)
    Etot = N
    u0[idx[(:P2, [:S, :S])]] = Etot * pS^2
    u0[idx[(:P2, [:I, :S])]] = Etot * 2 * pI * pS
    u0[idx[(:P2, [:I, :I])]] = Etot * pI^2

    # P_m motifs
    pm_name = m == 2 ? :P2 : (m == 3 ? :P3 : Symbol("P", m))
    for e in 0:((1 << m) - 1)
        σ = _decode_state(e, m)
        rσ = reverse(σ)
        if σ != minimum((σ, rσ))
            continue
        end
        orbit = (σ == rσ) ? 1 : 2
        prob = 1.0
        for s in σ
            prob *= pofstate(s)
        end
        u0[idx[(pm_name, σ)]] = N * orbit * prob
    end
    return u0
end

# ─── RHS for SIS, (k=3, m=3) — Phase B(b) ──────────────────────────────────
#
# Variables (in the order produced by `_build_variables`):
#
#   singletons:  ⟨I⟩, ⟨S⟩
#   P₂ pairs  :  E_II, E_IS, E_SS                                   (3)
#   P₃ triples:  E_(I,I,I), E_(I,I,S), E_(I,S,I),
#                E_(I,S,S), E_(S,I,S), E_(S,S,S)                    (6)
#   C₃ triples:  E_(I,I,I), E_(I,I,S), E_(I,S,S), E_(S,S,S)         (4)
#
# Total = 15 ODE variables.
#
# Counting conventions (LOCKED, same as elsewhere):
#   * E_c (canonical) counts unordered induced subgraphs in canonical
#     state class c.
#   * Labelled count L_σ (ordered embedding (1=σ₁, 2=σ₂, …)):
#       L_σ = E_canon(σ) · |stab(canon(σ))|
#     For C₃ |G|=6:
#       canonical [SSS], [III] → orbit 1, stab 6
#       canonical [ISS], [IIS] → orbit 3, stab 2
#
# Closure for triple-RHS at the 4-vertex level (Markov-conditional on the
# vertex hosting the external pendant; multiplied by (n_ext/k) to get the
# expected external-:I-neighbour count per induced motif):
#   * P₃ extension at endpoint i ∈ {1,3} (n_ext = k-1 = 2):
#       flow_ext ≈ β · (k-1)/k · L_σ · L_(σᵢ,:I) / ⟨σᵢ⟩
#   * P₃ extension at middle vertex 2 (n_ext = k-2 = 1):
#       flow_ext ≈ β · (k-2)/k · L_σ · L_(σ₂,:I) / ⟨σ₂⟩
#   * C₃ extension at any vertex i (n_ext = k-2 = 1):
#       flow_ext ≈ β · (k-2)/k · L_σ · L_(σᵢ,:I) / ⟨σᵢ⟩
#
# Derivation: per induced motif of state σ, the number of ext :I-neighbours
# at vertex i ≈ n_ext · P(nb is :I | vᵢ in σᵢ), and
# P(nb is :I | vᵢ in σᵢ) ≈ L_(σᵢ,:I) / [k · ⟨σᵢ⟩].
#
# Pair derivatives are exact marginals of triple transitions (sum over
# both shapes):
#
#   L_(I,S,S)_total = E_(I,S,S)(P₃)            + 2·E_(I,S,S)(C₃)
#   L_(I,S,I)_total = 2·E_(I,S,I)(P₃)          + 2·E_(I,I,S)(C₃)
#
#   dE_SS/dt = -β·L_(I,S,S)_total + γ·E_SI
#   dE_SI/dt = -γ·E_SI - β·E_SI - β·L_(I,S,I)_total
#              + β·L_(I,S,S)_total + 2γ·E_II
#   dE_II/dt = -2γ·E_II + β·E_SI + β·L_(I,S,I)_total
#
# Singleton derivatives are exact marginals of pair transitions.
#
# C₃ canonical states (for orbit-size lookup).
const _C3_TRIPLE_STATES = [
    ([:I, :I, :I], 1),
    ([:I, :I, :S], 3),
    ([:I, :S, :S], 3),
    ([:S, :S, :S], 1),
]

@inline function _c3_orbit_size(canon::Vector{Symbol})
    if canon == [:I,:I,:I] || canon == [:S,:S,:S]
        return 1
    else
        return 3
    end
end

function _build_sis_k3_m3_rhs(idx::Dict{Tuple{Symbol,Vector{Symbol}},Int})
    # Singleton + pair indices
    iI  = idx[(:singleton, [:I])]
    iS  = idx[(:singleton, [:S])]
    iII = idx[(:P2, [:I, :I])]
    iIS = idx[(:P2, [:I, :S])]
    iSS = idx[(:P2, [:S, :S])]

    # P₃ canonical indices
    iP3_III = idx[(:P3, [:I, :I, :I])]
    iP3_IIS = idx[(:P3, [:I, :I, :S])]
    iP3_ISI = idx[(:P3, [:I, :S, :I])]
    iP3_ISS = idx[(:P3, [:I, :S, :S])]
    iP3_SIS = idx[(:P3, [:S, :I, :S])]
    iP3_SSS = idx[(:P3, [:S, :S, :S])]

    # C₃ canonical indices
    iC3_III = idx[(:C3, [:I, :I, :I])]
    iC3_IIS = idx[(:C3, [:I, :I, :S])]
    iC3_ISS = idx[(:C3, [:I, :S, :S])]
    iC3_SSS = idx[(:C3, [:S, :S, :S])]

    # P₃ labelled-state → (canonical_var_index, stab)
    p3_canon_idx = Vector{Int}(undef, 8)
    p3_stab      = Vector{Int}(undef, 8)
    for σ in _P3_LABELLED
        rev = reverse(σ)
        canon = σ <= rev ? σ : rev
        osz   = (σ == rev) ? 1 : 2
        e = _enc3(σ)
        p3_canon_idx[e + 1] =
            canon == [:I,:I,:I] ? iP3_III :
            canon == [:I,:I,:S] ? iP3_IIS :
            canon == [:I,:S,:I] ? iP3_ISI :
            canon == [:I,:S,:S] ? iP3_ISS :
            canon == [:S,:I,:S] ? iP3_SIS :
                                  iP3_SSS
        p3_stab[e + 1] = 2 ÷ osz
    end

    # C₃ labelled-state → (canonical_var_index, stab)
    c3_canon_idx = Vector{Int}(undef, 8)
    c3_stab      = Vector{Int}(undef, 8)
    for σ in _P3_LABELLED  # reuse the 8-element labelled enumeration
        canon = sort(σ)
        e = _enc3(σ)
        c3_canon_idx[e + 1] =
            canon == [:I,:I,:I] ? iC3_III :
            canon == [:I,:I,:S] ? iC3_IIS :
            canon == [:I,:S,:S] ? iC3_ISS :
                                  iC3_SSS
        c3_stab[e + 1] = 6 ÷ _c3_orbit_size(canon)
    end

    function rhs!(du, u, p, t)
        β = p.β; γ = p.γ
        S  = u[iS];  Ival = u[iI]
        SS = u[iSS]; SI = u[iIS]; II = u[iII]

        @inline function Lpair(X::Symbol, Y::Symbol)
            if X === :S && Y === :S
                return 2.0 * SS
            elseif X === :I && Y === :I
                return 2.0 * II
            else
                return SI
            end
        end
        @inline single(X::Symbol) = (X === :I) ? Ival : S

        dLP3 = zeros(Float64, 8)
        dLC3 = zeros(Float64, 8)

        # ───── P₃ transitions ─────
        @inbounds for σ in _P3_LABELLED
            eσ = _enc3(σ)
            Lσ = u[p3_canon_idx[eσ + 1]] * p3_stab[eσ + 1]

            for i in 1:3
                if σ[i] === :I
                    σp = (i == 1) ? [:S, σ[2], σ[3]] :
                         (i == 2) ? [σ[1], :S, σ[3]] :
                                    [σ[1], σ[2], :S]
                    flow = γ * Lσ
                    dLP3[eσ + 1]       -= flow
                    dLP3[_enc3(σp) + 1] += flow
                else
                    # σ[i] === :S; infection of vertex i
                    n_int = 0
                    if i == 1 && σ[2] === :I
                        n_int += 1
                    elseif i == 2
                        if σ[1] === :I; n_int += 1; end
                        if σ[3] === :I; n_int += 1; end
                    elseif i == 3 && σ[2] === :I
                        n_int += 1
                    end
                    flow_int = β * n_int * Lσ

                    flow_ext = 0.0
                    if i == 1
                        # P₃ → P₄ at endpoint 1; (k-1)/k = 2/3 effective slot fraction.
                        flow_ext = β * (2.0/3.0) *
                            safe_ratio(Lσ * Lpair(:I, :S), single(:S))
                    elseif i == 2
                        # P₃ → K_{1,3} at middle; (k-2)/k = 1/3 effective slot fraction.
                        # Use simple Markov closure: P(ext nb is :I | v_2 in :S).
                        flow_ext = β * (1.0/3.0) *
                            safe_ratio(Lσ * Lpair(:S, :I), single(:S))
                    else  # i == 3
                        # P₃ → P₄ at endpoint 3; (k-1)/k = 2/3 effective slot fraction.
                        flow_ext = β * (2.0/3.0) *
                            safe_ratio(Lσ * Lpair(:S, :I), single(:S))
                    end

                    σp = (i == 1) ? [:I, σ[2], σ[3]] :
                         (i == 2) ? [σ[1], :I, σ[3]] :
                                    [σ[1], σ[2], :I]
                    flow = flow_int + flow_ext
                    dLP3[eσ + 1]       -= flow
                    dLP3[_enc3(σp) + 1] += flow
                end
            end
        end

        # ───── C₃ transitions ─────
        @inbounds for σ in _P3_LABELLED
            eσ = _enc3(σ)
            Lσ = u[c3_canon_idx[eσ + 1]] * c3_stab[eσ + 1]

            for i in 1:3
                if σ[i] === :I
                    σp = (i == 1) ? [:S, σ[2], σ[3]] :
                         (i == 2) ? [σ[1], :S, σ[3]] :
                                    [σ[1], σ[2], :S]
                    flow = γ * Lσ
                    dLC3[eσ + 1]       -= flow
                    dLC3[_enc3(σp) + 1] += flow
                else
                    # σ[i] === :S; infection. C₃: every vertex has internal
                    # degree 2 (the other 2 vertices are neighbours).
                    n_int = 0
                    for j in 1:3
                        if j != i && σ[j] === :I
                            n_int += 1
                        end
                    end
                    flow_int = β * n_int * Lσ

                    # C₃ → paw at vertex i; (k-2)/k = 1/3 effective slot fraction.
                    flow_ext = β * (1.0/3.0) *
                        safe_ratio(Lσ * Lpair(:S, :I), single(:S))

                    σp = (i == 1) ? [:I, σ[2], σ[3]] :
                         (i == 2) ? [σ[1], :I, σ[3]] :
                                    [σ[1], σ[2], :I]
                    flow = flow_int + flow_ext
                    dLC3[eσ + 1]       -= flow
                    dLC3[_enc3(σp) + 1] += flow
                end
            end
        end

        # Convert dL → dE for P₃
        du[iP3_III] = dLP3[_enc3([:I,:I,:I]) + 1] / p3_stab[_enc3([:I,:I,:I]) + 1]
        du[iP3_IIS] = dLP3[_enc3([:I,:I,:S]) + 1] / p3_stab[_enc3([:I,:I,:S]) + 1]
        du[iP3_ISI] = dLP3[_enc3([:I,:S,:I]) + 1] / p3_stab[_enc3([:I,:S,:I]) + 1]
        du[iP3_ISS] = dLP3[_enc3([:I,:S,:S]) + 1] / p3_stab[_enc3([:I,:S,:S]) + 1]
        du[iP3_SIS] = dLP3[_enc3([:S,:I,:S]) + 1] / p3_stab[_enc3([:S,:I,:S]) + 1]
        du[iP3_SSS] = dLP3[_enc3([:S,:S,:S]) + 1] / p3_stab[_enc3([:S,:S,:S]) + 1]

        # Convert dL → dE for C₃
        du[iC3_III] = dLC3[_enc3([:I,:I,:I]) + 1] / c3_stab[_enc3([:I,:I,:I]) + 1]
        du[iC3_IIS] = dLC3[_enc3([:I,:I,:S]) + 1] / c3_stab[_enc3([:I,:I,:S]) + 1]
        du[iC3_ISS] = dLC3[_enc3([:I,:S,:S]) + 1] / c3_stab[_enc3([:I,:S,:S]) + 1]
        du[iC3_SSS] = dLC3[_enc3([:S,:S,:S]) + 1] / c3_stab[_enc3([:S,:S,:S]) + 1]

        # ───── Pair derivatives — exact marginals of triple transitions ─
        L_ISS_total = u[iP3_ISS] + 2.0 * u[iC3_ISS]
        L_ISI_total = 2.0 * u[iP3_ISI] + 2.0 * u[iC3_IIS]

        du[iSS] = -β * L_ISS_total + γ * SI
        du[iIS] = -γ * SI - β * SI - β * L_ISI_total +
                  β * L_ISS_total + 2γ * II
        du[iII] = -2γ * II + β * SI + β * L_ISI_total

        # ───── Singleton derivatives — exact marginals of pair transitions
        du[iS]  =  γ * Ival - β * SI
        du[iI]  = -γ * Ival + β * SI
        return nothing
    end
    return rhs!
end

# ─── Initial conditions for (k=3, m=3) ─────────────────────────────────────
#
# Random-mixing IC at infected fraction ε on a 3-regular host with
#   N        host nodes               → ⟨X⟩         = N · P(X)
#   3N/2     edges                    → E_pair_c    = (3N/2) · |orbit| · ∏P
#   n_p3     induced P₃ motifs        → E_(P₃,c)    = n_p3 · |orbit| · ∏P
#   n_c3     induced C₃ motifs        → E_(C₃,c)    = n_c3 · |orbit| · ∏P
#
# Conservation:
#   Σ_c E_pair_c = 3N/2,  Σ_c E_(P₃,c) = n_p3,  Σ_c E_(C₃,c) = n_c3.

function _build_sis_k3_m3_ic(idx, N::Float64, ε::Float64,
                              n_p3::Float64, n_c3::Float64)
    nvars = maximum(values(idx))
    u0 = zeros(Float64, nvars)
    pS = 1 - ε; pI = ε
    pofstate(s::Symbol) = (s === :I) ? pI : pS

    u0[idx[(:singleton, [:S])]] = N * pS
    u0[idx[(:singleton, [:I])]] = N * pI

    Etot = N * 3 / 2
    u0[idx[(:P2, [:S, :S])]] = Etot * pS^2
    u0[idx[(:P2, [:I, :S])]] = Etot * 2 * pI * pS
    u0[idx[(:P2, [:I, :I])]] = Etot * pI^2

    for (c, osz) in _P3_TRIPLE_STATES
        u0[idx[(:P3, c)]] = n_p3 * osz *
            pofstate(c[1]) * pofstate(c[2]) * pofstate(c[3])
    end
    for (c, osz) in _C3_TRIPLE_STATES
        u0[idx[(:C3, c)]] = n_c3 * osz *
            pofstate(c[1]) * pofstate(c[2]) * pofstate(c[3])
    end
    return u0
end

# ─── RHS for SIS, (k=3, m=4) — Phase B(c) ──────────────────────────────────
#
# Variables (added on top of the B(b) layout):
#
#   singletons:  ⟨I⟩, ⟨S⟩                                            (2)
#   P₂ pairs  :  E_II, E_IS, E_SS                                    (3)
#   P₃ triples:  6 canonical states                                  (6)
#   C₃ triples:  4 canonical states                                  (4)
#   P_4       :  10 canonical states                                 (10)
#   K_{1,3}   :  8  canonical states                                 (8)
#   paw       :  12 canonical states                                 (12)
#   C_4       :  6  canonical states                                 (6)
#   K_4 − e   :  9  canonical states                                 (9)
#   K_4       :  5  canonical states                                 (5)
#
# Total = 65 ODE variables.
#
# Closure choice for the 5-vertex external infection on each 4-vertex
# motif σ: at every vertex i with n_ext_i = k − deg_inside_σ(i) ≥ 1 we
# use a *per-shape higher-order Kirkwood* closure that drops a chosen
# vertex w (typically diametrically opposite to the extension vertex i)
# and factorises the 5-vertex labelled count through the resulting
# 3-vertex induced subgraph:
#
#   L_(e, σ) ≈ L_(σ-{w}∪{e}) · L_σ / L_(σ-{w})
#
# where σ-{w} is always a tracked 3-vertex shape (P_3 or C_3) and
# σ-{w}∪{e} is always a tracked 4-vertex shape (P_4, K_{1,3}, paw, C_4).
# The drop vertex w is selected per (shape, ext_vertex) pair via the
# `_CLOSURE_RULES_4V` registry below; see `motif_symbolic.jl` for the
# matching symbolic closure rule (must agree element-wise).
#
# The contribution to dL_σ at the (S → I) flip of vertex i is then
#
#   flow_ext_i = β · (n_ext_i / k) · safe_ratio(L_(σ-{w}∪{e=I}) · L_σ,
#                                                L_(σ-{w}))
#
# This generalises the simple single-vertex Markov closure (which is
# recovered by the `:uniform_anchor` legacy variant, kept as a private
# diagnostic via the `closure_kind` kwarg of `_build_sis_k3_m4_rhs`).

# ─── Per-shape Kirkwood closure registry ──────────────────────────────────
#
# For every (shape_name, ext_vertex_index) with n_ext ≥ 1 we list:
#
#   * target3 : the 3-vertex shape obtained by deleting drop vertex w.
#   * perm3   : NTuple{3,Int} of σ-vertex indices giving the labelled
#               state of σ-{w} at canonical positions 1..3 of `target3`.
#   * target4 : the 4-vertex shape obtained by replacing w with the
#               extension vertex e.
#   * perm4   : NTuple{4,Int} of σ-vertex indices (or 0 marking the
#               position of the extension vertex e, whose state is :I)
#               giving the labelled state of σ-{w}∪{e} at canonical
#               positions 1..4 of `target4`. Canonicalisation is then
#               applied via `canonical_state` to look up the correct
#               variable index.
#
# Verified per-shape derivations are documented in the design spec; the
# symbolic oracle in `motif_symbolic.jl` uses the same registry.
const _CLOSURE_RULES_4V = Dict{Tuple{Symbol,Int}, NamedTuple}(
    # P_4 (1-2-3-4): drop opposite endpoint.
    (:P4, 1) => (target3=:P3, perm3=(1,2,3), target4=:P4,  perm4=(0,1,2,3)),
    (:P4, 2) => (target3=:P3, perm3=(1,2,3), target4=:K13, perm4=(2,1,3,0)),
    (:P4, 3) => (target3=:P3, perm3=(2,3,4), target4=:K13, perm4=(3,2,4,0)),
    (:P4, 4) => (target3=:P3, perm3=(2,3,4), target4=:P4,  perm4=(2,3,4,0)),
    # K_{1,3} (centre=1, leaves=2,3,4): drop another leaf.
    (:K13, 2) => (target3=:P3, perm3=(2,1,4), target4=:P4, perm4=(4,1,2,0)),
    (:K13, 3) => (target3=:P3, perm3=(3,1,4), target4=:P4, perm4=(4,1,3,0)),
    (:K13, 4) => (target3=:P3, perm3=(3,1,4), target4=:P4, perm4=(3,1,4,0)),
    # paw (triangle {1,2,3} + leaf 4 at apex 1): triangle vertices drop
    # the leaf w=4 (anchor C_3, target paw); the leaf v=4 drops a
    # triangle vertex (anchor P_3 path, target P_4).
    (:paw, 2) => (target3=:C3, perm3=(1,2,3), target4=:paw, perm4=(2,1,3,0)),
    (:paw, 3) => (target3=:C3, perm3=(1,2,3), target4=:paw, perm4=(3,1,2,0)),
    (:paw, 4) => (target3=:P3, perm3=(3,1,4), target4=:P4,  perm4=(3,1,4,0)),
    # C_4 (1-2-3-4-1): drop the diametrically opposite vertex.
    (:C4, 1) => (target3=:P3, perm3=(2,1,4), target4=:K13, perm4=(1,2,4,0)),
    (:C4, 2) => (target3=:P3, perm3=(1,2,3), target4=:K13, perm4=(2,1,3,0)),
    (:C4, 3) => (target3=:P3, perm3=(2,3,4), target4=:K13, perm4=(3,2,4,0)),
    (:C4, 4) => (target3=:P3, perm3=(1,4,3), target4=:K13, perm4=(4,1,3,0)),
    # K_4 - e (edges (1,2),(2,3),(3,4),(4,1),(1,3); deg-3 at 1,3, deg-2
    # at 2,4): degree-2 vertices drop the diagonal partner (anchor C_3,
    # target paw with leaf at the extension vertex).
    (:K4me, 2) => (target3=:C3, perm3=(1,2,3), target4=:paw, perm4=(2,1,3,0)),
    (:K4me, 4) => (target3=:C3, perm3=(1,3,4), target4=:paw, perm4=(4,1,3,0)),
    # K_4: every vertex has n_ext = 0; no closure needed.
)
#
# Lower-order ODEs are EXACT marginals of the highest-tracked-shape
# transitions in the B(b) sense:
#
#   * Singleton derivatives = pair-flow marginals (model-exact).
#   * Pair derivatives      = labelled-triple flow marginals
#                             (sum over P₃ + C₃ contributions, same as B(b)).
#   * Triple derivatives    = labelled-state flow on each triple shape
#                             with the order-3 single-anchor closure
#                             (same as B(b)).
#   * 4-vertex derivatives  = labelled-state flow on each 4-vertex shape
#                             with the order-4 single-anchor closure
#                             described above.

# Helper: build per-shape labelled-state lookup tables (canon var index,
# stab factor, internal degrees, neighbour lists).
function _shape_lookup(sh::MotifShape, idx::Dict)
    n      = sh.n_nodes
    nlab   = 1 << n
    canon_idx = Vector{Int}(undef, nlab)
    stab      = Vector{Int}(undef, nlab)
    Gsz       = length(sh.automorphisms)
    for e in 0:nlab-1
        σ = Vector{Symbol}(undef, n)
        for i in 1:n
            σ[i] = (((e >> (n - i)) & 1) == 1) ? :I : :S
        end
        canon, osz = canonical_state(sh, σ)
        canon_idx[e + 1] = idx[(sh.name, canon)]
        stab[e + 1]      = Gsz ÷ osz
    end
    deg_in = zeros(Int, n)
    for (a, b) in sh.edges
        deg_in[a] += 1
        deg_in[b] += 1
    end
    nbrs = [Int[] for _ in 1:n]
    for (a, b) in sh.edges
        push!(nbrs[a], b)
        push!(nbrs[b], a)
    end
    return (n = n, nlab = nlab, canon_idx = canon_idx, stab = stab,
            deg_in = deg_in, nbrs = nbrs)
end

# Decode encoded 4-vertex labelled state e into Vector{Symbol}.
@inline function _decode4(e::Int)
    return Symbol[
        (((e >> 3) & 1) == 1) ? :I : :S,
        (((e >> 2) & 1) == 1) ? :I : :S,
        (((e >> 1) & 1) == 1) ? :I : :S,
        ((e & 1) == 1) ? :I : :S,
    ]
end

# Build closure data: for each shape, the precomputed per-state σ vectors
# (so we don't reallocate per RHS call) and per-vertex slot factors
# (n_ext_i / k as Float64).
function _shape4_closure_data(sh::MotifShape, k::Int)
    n     = sh.n_nodes
    nlab  = 1 << n
    σ_of  = Vector{Vector{Symbol}}(undef, nlab)
    for e in 0:nlab-1
        σ_of[e + 1] = _decode4(e)
    end
    deg_in = zeros(Int, n)
    for (a, b) in sh.edges
        deg_in[a] += 1; deg_in[b] += 1
    end
    n_ext  = [k - deg_in[i] for i in 1:n]
    slot   = [n_ext[i] / k for i in 1:n]
    nbrs   = [Int[] for _ in 1:n]
    for (a, b) in sh.edges
        push!(nbrs[a], b); push!(nbrs[b], a)
    end
    return (σ_of = σ_of, n_ext = n_ext, slot = slot, nbrs = nbrs)
end

@inline function _enc4(σ::Vector{Symbol})
    e = 0
    for i in 1:4
        if σ[i] === :I
            e |= 1 << (4 - i)
        end
    end
    return e
end

# ─── 3-from-4 marginalisation (locked semantic #5) ─────────────────────────
#
# The P_3 / C_3 derivatives are EXACT marginals of the 4-vertex shape
# derivatives:
#
#   dE_(σ_3)/dt = Σ_(shape_4, σ_4_canon) M[(σ_3), (σ_4_canon)] · dE_(σ_4_canon)/dt
#
# where M is built from the shape-4 sub-3-set topology only and is
# host-independent. For a `k`-regular host the asymptotic external
# multiplicity is `ext = k·(k-1)·(k-2)/(deg_pattern)` — in the 3-regular
# case this is `5` per induced P_3 instance and `3` per induced C_3
# instance (each P_3 has 5 external slots; each C_3 has 3). Dividing by
# this ext factor makes M an exact marginalisation map at the random-
# mixing IC for asymptotic 3-regular hosts; for finite hosts the identity
# `M·E_4 = E_3` holds approximately (within O(1/N) for sparse hosts) and
# for arbitrary user-supplied (n_p3, n_c3, n_p4, …) counts the IC may
# deviate. The conservation law `Σ_c E_(P_3,c) = n_p3` (and analogously
# for C_3) is preserved exactly because the row-sum of M (per shape_4) is
# a constant that multiplies the 4-vertex conservation `Σ_c4 dE_4 = 0`.
#
# Bookkeeping: for each canonical 4-vertex variable `(shape_4, σ_4_canon)`
# we enumerate the four positional 3-subsets `T ⊂ {1,2,3,4}`. If the
# induced subgraph on `T` has 2 edges → P_3 contribution (middle vertex =
# the deg-2 vertex), 3 edges → C_3 contribution. The labelled 3-state is
# read off `σ_4_canon` at the appropriate positions, then canonicalised
# via `canonical_state` on `_P3_SHAPE` / `_C3_SHAPE`. The accumulated
# integer count is divided by the asymptotic ext factor.

const _3SUBS_OF_4 = ((1,2,3), (1,2,4), (1,3,4), (2,3,4))

# For a 4-vertex shape, enumerate the per-positional-3-subset
# specifications: for each `T` whose induced subgraph is connected,
# return `(:P3 or :C3, ordering)` where `ordering[j]` gives the
# position in {1..4} of the j-th vertex in the target 3-shape's
# canonical position layout (P_3: position 2 = middle; C_3: any).
function _shape4_3subset_specs(sh::MotifShape)
    specs = Tuple{Symbol, NTuple{3,Int}}[]
    for T in _3SUBS_OF_4
        ne = 0
        deg = Dict{Int,Int}(t => 0 for t in T)
        for (a, b) in sh.edges
            if a in T && b in T
                ne += 1
                deg[a] += 1
                deg[b] += 1
            end
        end
        if ne == 2
            mid = 0
            ends = Int[]
            for v in T
                if deg[v] == 2
                    mid = v
                else
                    push!(ends, v)
                end
            end
            mid == 0 && continue   # disconnected (shouldn't happen with ne==2)
            sort!(ends)
            push!(specs, (:P3, (ends[1], mid, ends[2])))
        elseif ne == 3
            push!(specs, (:C3, (T[1], T[2], T[3])))
        end
    end
    return specs
end

# Build the 3-from-4 marginalisation contribution list:
#   contribs[i_3] :: Vector{Tuple{Int,Float64}}
# where each `(i_4, coeff)` says `du[i_3] += coeff · du[i_4]`.
# `k` is the host regularity (used to set the ext-factor); only k=3 is
# implemented here.
function _build_mat_3from4(idx::Dict{Tuple{Symbol,Vector{Symbol}},Int};
                          k::Int = 3)
    k == 3 || throw(ArgumentError(
        "_build_mat_3from4: only k=3 supported (got k=$k)"))
    contribs = Dict{Int, Vector{Tuple{Int,Float64}}}()
    ext = Dict(:P3 => 5.0, :C3 => 3.0)   # asymptotic 3-regular multiplicity
    sh3_for = Dict(:P3 => _P3_SHAPE, :C3 => _C3_SHAPE)
    for sh4 in _SHAPES_4V_K3
        specs = _shape4_3subset_specs(sh4)
        seen_canon = Set{Vector{Symbol}}()
        for e4 in 0:15
            σ4 = _decode4(e4)
            canon4, _ = canonical_state(sh4, σ4)
            canon4 in seen_canon && continue
            push!(seen_canon, canon4)
            i4 = idx[(sh4.name, canon4)]
            counts = Dict{Tuple{Symbol,Vector{Symbol}}, Int}()
            for (sh3_name, ord) in specs
                state3_lab = Symbol[canon4[ord[1]],
                                    canon4[ord[2]],
                                    canon4[ord[3]]]
                canon3, _ = canonical_state(sh3_for[sh3_name], state3_lab)
                key = (sh3_name, canon3)
                counts[key] = get(counts, key, 0) + 1
            end
            for ((sh3_name, canon3), c) in counts
                i3 = idx[(sh3_name, canon3)]
                coef = c / ext[sh3_name]
                push!(get!(contribs, i3, Tuple{Int,Float64}[]), (i4, coef))
            end
        end
    end
    return contribs
end

# NOTE: a per-event 3-from-4 accumulation table was prototyped here
# (`_shape4_subset_per_vertex`, `_build_3from4_event_table`) as an
# attempt to make the 3-from-4 marginalisation dynamically exact on
# finite hosts. It was removed because the natural divisor formula
# (|Aut_3| · ext) does not reduce to the constant snapshot matrix
# `_build_mat_3from4` even at the factorising IC where the snapshot
# identity is exact. Resolving this requires further user input on
# the intended semantic.

function _build_sis_k3_m4_rhs(idx::Dict{Tuple{Symbol,Vector{Symbol}},Int};
                              closure_kind::Symbol = :kirkwood)
    closure_kind === :kirkwood || closure_kind === :uniform_anchor ||
        throw(ArgumentError("_build_sis_k3_m4_rhs: closure_kind must be " *
                            ":kirkwood or :uniform_anchor (got :$closure_kind)"))
    # ── Singleton + pair indices ──────────────────────────────────────────
    iI  = idx[(:singleton, [:I])]
    iS  = idx[(:singleton, [:S])]
    iII = idx[(:P2, [:I, :I])]
    iIS = idx[(:P2, [:I, :S])]
    iSS = idx[(:P2, [:S, :S])]

    # ── P₃ canonical indices (B(b)) ───────────────────────────────────────
    iP3_III = idx[(:P3, [:I, :I, :I])]
    iP3_IIS = idx[(:P3, [:I, :I, :S])]
    iP3_ISI = idx[(:P3, [:I, :S, :I])]
    iP3_ISS = idx[(:P3, [:I, :S, :S])]
    iP3_SIS = idx[(:P3, [:S, :I, :S])]
    iP3_SSS = idx[(:P3, [:S, :S, :S])]

    # ── C₃ canonical indices (B(b)) ───────────────────────────────────────
    iC3_III = idx[(:C3, [:I, :I, :I])]
    iC3_IIS = idx[(:C3, [:I, :I, :S])]
    iC3_ISS = idx[(:C3, [:I, :S, :S])]
    iC3_SSS = idx[(:C3, [:S, :S, :S])]

    # P₃ / C₃ labelled-state lookup (8 entries each)
    p3_canon_idx = Vector{Int}(undef, 8)
    p3_stab      = Vector{Int}(undef, 8)
    for σ in _P3_LABELLED
        rev = reverse(σ)
        canon = σ <= rev ? σ : rev
        osz   = (σ == rev) ? 1 : 2
        e = _enc3(σ)
        p3_canon_idx[e + 1] =
            canon == [:I,:I,:I] ? iP3_III :
            canon == [:I,:I,:S] ? iP3_IIS :
            canon == [:I,:S,:I] ? iP3_ISI :
            canon == [:I,:S,:S] ? iP3_ISS :
            canon == [:S,:I,:S] ? iP3_SIS :
                                  iP3_SSS
        p3_stab[e + 1] = 2 ÷ osz
    end
    c3_canon_idx = Vector{Int}(undef, 8)
    c3_stab      = Vector{Int}(undef, 8)
    for σ in _P3_LABELLED
        canon = sort(σ)
        e = _enc3(σ)
        c3_canon_idx[e + 1] =
            canon == [:I,:I,:I] ? iC3_III :
            canon == [:I,:I,:S] ? iC3_IIS :
            canon == [:I,:S,:S] ? iC3_ISS :
                                  iC3_SSS
        c3_stab[e + 1] = 6 ÷ _c3_orbit_size(canon)
    end

    # ── 4-vertex shape lookups ────────────────────────────────────────────
    shapes_4v = _SHAPES_4V_K3
    lookups   = ntuple(i -> _shape_lookup(shapes_4v[i], idx), length(shapes_4v))
    closures  = ntuple(i -> _shape4_closure_data(shapes_4v[i], 3),
                       length(shapes_4v))

    # ── Kirkwood closure target lookups (3-vertex anchor / 4-vertex
    # target). For each (shape, ext-vertex) pair with n_ext > 0, pre-
    # resolve the registry rule to direct array references, so the inner
    # loop avoids any Dict access. Vertices with n_ext = 0 store
    # `nothing` (no external flow contribution).
    target3_arrays = Dict{Symbol, Tuple{Vector{Int}, Vector{Int}}}(
        :P3 => (p3_canon_idx, p3_stab),
        :C3 => (c3_canon_idx, c3_stab),
    )
    shape4_pos = Dict{Symbol, Int}(s.name => si
                                   for (si, s) in enumerate(shapes_4v))
    target4_arrays = Dict{Symbol, Tuple{Vector{Int}, Vector{Int}}}(
        s.name => (lookups[si].canon_idx, lookups[si].stab)
        for (si, s) in enumerate(shapes_4v))

    # `closure_rules[si][i]` = NamedTuple of resolved arrays + perms, or
    # `nothing` if vertex i of shape si has n_ext == 0 (no external slot).
    closure_rules = Vector{Vector{Any}}(undef, length(shapes_4v))
    for (si, sh) in enumerate(shapes_4v)
        n  = sh.n_nodes
        cl = closures[si]
        per_vertex = Vector{Any}(undef, n)
        for i in 1:n
            if cl.n_ext[i] == 0
                per_vertex[i] = nothing
                continue
            end
            rule = get(_CLOSURE_RULES_4V, (sh.name, i), nothing)
            rule === nothing && error(
                "Missing Kirkwood closure rule for ($(sh.name), vertex $i)")
            t3_idx, t3_stab = target3_arrays[rule.target3]
            t4_idx, t4_stab = target4_arrays[rule.target4]
            per_vertex[i] = (perm3 = rule.perm3, perm4 = rule.perm4,
                             t3_idx = t3_idx, t3_stab = t3_stab,
                             t4_idx = t4_idx, t4_stab = t4_stab)
        end
        closure_rules[si] = per_vertex
    end
    canon_encs_4v = ntuple(length(shapes_4v)) do si
        sh = shapes_4v[si]
        # canonical encoding = first labelled state in each orbit
        seen = Set{Vector{Symbol}}()
        out  = Int[]
        for e in 0:15
            σ = closures[si].σ_of[e + 1]
            canon, _ = canonical_state(sh, σ)
            if !(canon in seen)
                push!(seen, canon)
                push!(out, _enc4(canon))
            end
        end
        out
    end

    # ── 3-from-4 marginalisation contributions (locked semantic #5) ──────
    # For each P_3 / C_3 canonical variable index, a list of
    # `(4-vertex var index, coefficient)` pairs such that
    #   du[i_3] = Σ coef · du[i_4]
    # implements a snapshot-exact marginal of the 4-vertex derivatives:
    # `Mat · u_4 = u_3` at any factorising IC, and time-differentiating
    # gives `Mat · du_4 = du_3` at that IC. See `_build_mat_3from4`.
    #
    # IMPORTANT: this identity is exact only at the asymptotic
    # random-3-regular factorising IC; on a finite host with non-
    # factorising state distributions, the constant linear map is no
    # longer the correct dynamic marginal. (The would-be correction is
    # a per-event accumulation inside the 4-vertex flow loop, which we
    # explored but could not reduce to a closed form that matches the
    # snapshot marginal at IC. Pending user input on intended semantic.)
    mat_3from4_dict = _build_mat_3from4(idx; k = 3)
    triple_targets = collect(pairs(mat_3from4_dict))
    triple_target_idx = Int[p.first for p in triple_targets]
    triple_target_contribs = Vector{Vector{Tuple{Int,Float64}}}(
        undef, length(triple_targets))
    for (j, p) in enumerate(triple_targets)
        triple_target_contribs[j] = p.second
    end

    function rhs!(du, u, p, t)
        β = p.β; γ = p.γ
        S  = u[iS];  Ival = u[iI]
        SS = u[iSS]; SI = u[iIS]; II = u[iII]

        @inline function Lpair(X::Symbol, Y::Symbol)
            if X === :S && Y === :S
                return 2.0 * SS
            elseif X === :I && Y === :I
                return 2.0 * II
            else
                return SI
            end
        end
        @inline single(X::Symbol) = (X === :I) ? Ival : S

        # ───── P₃ and C₃ derivatives are computed as exact marginals
        # of the 4-vertex shape derivatives (locked semantic #5). The
        # assignment is deferred until after the 4-vertex transition
        # block below, which fills in `du[i_4]` for every 4-vertex
        # canonical variable.

        # ───── 4-vertex shape transitions ──────────────────────────────
        @inbounds for si in 1:length(shapes_4v)
            lk = lookups[si]
            cl = closures[si]
            rules_si = closure_rules[si]
            n  = lk.n
            dL = zeros(Float64, lk.nlab)
            for e in 0:lk.nlab-1
                σ  = cl.σ_of[e + 1]
                Lσ = u[lk.canon_idx[e + 1]] * lk.stab[e + 1]
                for i in 1:n
                    if σ[i] === :I
                        # Recovery
                        e_new = e ⊻ (1 << (n - i))
                        flow = γ * Lσ
                        dL[e + 1]     -= flow
                        dL[e_new + 1] += flow
                    else
                        # Internal infection contribution
                        n_int = 0
                        for j in cl.nbrs[i]
                            if σ[j] === :I; n_int += 1; end
                        end
                        flow_int = β * n_int * Lσ
                        # External (5-vertex closure) — only if slot > 0
                        flow_ext = 0.0
                        rule = rules_si[i]
                        if rule !== nothing
                            if closure_kind === :uniform_anchor
                                flow_ext = β * cl.slot[i] *
                                    safe_ratio(Lσ * Lpair(σ[i], :I),
                                               single(σ[i]))
                            else  # :kirkwood (per-shape higher-order)
                                p3 = rule.perm3
                                e3 = (_bit(σ[p3[1]]) << 2) |
                                     (_bit(σ[p3[2]]) << 1) |
                                      _bit(σ[p3[3]])
                                L3 = u[rule.t3_idx[e3 + 1]] *
                                     rule.t3_stab[e3 + 1]
                                p4 = rule.perm4
                                # state4: positions inherit from σ
                                # except the e-position (perm4[j] == 0)
                                # which holds :I.
                                b1 = p4[1] == 0 ? 1 : _bit(σ[p4[1]])
                                b2 = p4[2] == 0 ? 1 : _bit(σ[p4[2]])
                                b3 = p4[3] == 0 ? 1 : _bit(σ[p4[3]])
                                b4 = p4[4] == 0 ? 1 : _bit(σ[p4[4]])
                                e4 = (b1 << 3) | (b2 << 2) | (b3 << 1) | b4
                                L4 = u[rule.t4_idx[e4 + 1]] *
                                     rule.t4_stab[e4 + 1]
                                flow_ext = β * cl.slot[i] *
                                    safe_ratio(L4 * Lσ, L3)
                            end
                        end
                        e_new = e | (1 << (n - i))
                        flow = flow_int + flow_ext
                        dL[e + 1]     -= flow
                        dL[e_new + 1] += flow
                    end
                end
            end
            # Convert dL → dE (one assignment per canonical class)
            for ec in canon_encs_4v[si]
                du[lk.canon_idx[ec + 1]] = dL[ec + 1] / lk.stab[ec + 1]
            end
        end

        # ───── 3-from-4 marginalisation (locked semantic #5) ─────────────
        # P_3 / C_3 derivatives = exact snapshot marginals of the 4-vertex
        # derivatives. See `_build_mat_3from4` and the precompute above
        # for the structure and asymptotic-IC validity caveat.
        @inbounds for j in 1:length(triple_target_idx)
            i3 = triple_target_idx[j]
            s = 0.0
            for (i4, coef) in triple_target_contribs[j]
                s += coef * du[i4]
            end
            du[i3] = s
        end

        # ───── Pair derivatives — exact marginals of triple flow (B(b)) ─
        L_ISS_total = u[iP3_ISS] + 2.0 * u[iC3_ISS]
        L_ISI_total = 2.0 * u[iP3_ISI] + 2.0 * u[iC3_IIS]
        du[iSS] = -β * L_ISS_total + γ * SI
        du[iIS] = -γ * SI - β * SI - β * L_ISI_total +
                  β * L_ISS_total + 2γ * II
        du[iII] = -2γ * II + β * SI + β * L_ISI_total

        # ───── Singleton derivatives — exact marginals of pair flow ─────
        du[iS]  =  γ * Ival - β * SI
        du[iI]  = -γ * Ival + β * SI
        return nothing
    end
    return rhs!
end

# ─── Initial conditions for (k=3, m=4) ─────────────────────────────────────
#
# Random-mixing IC at infected fraction ε:
#
#   ⟨X⟩         = N · P(X)
#   E_pair_c    = (3N/2) · |orbit| · ∏ P(c[i])
#   E_(P₃,c)    = n_p3   · |orbit| · ∏ P(c[i])
#   E_(C₃,c)    = n_c3   · |orbit| · ∏ P(c[i])
#   E_(shape4,c)= n_shape4 · |orbit| · ∏ P(c[i])         for each 4-vertex shape
#
# Conservation of canonical sums per shape:  Σ_c E_(shape,c) = n_shape.

function _build_sis_k3_m4_ic(idx, N::Float64, ε::Float64,
                              n_p3::Float64, n_c3::Float64,
                              n_p4::Float64, n_k13::Float64,
                              n_paw::Float64, n_c4::Float64,
                              n_k4me::Float64, n_k4::Float64)
    nvars = maximum(values(idx))
    u0 = zeros(Float64, nvars)
    pS = 1 - ε; pI = ε
    pofstate(s::Symbol) = (s === :I) ? pI : pS

    # Singletons
    u0[idx[(:singleton, [:S])]] = N * pS
    u0[idx[(:singleton, [:I])]] = N * pI

    # Pairs (3N/2 edges)
    Etot = N * 3 / 2
    u0[idx[(:P2, [:S, :S])]] = Etot * pS^2
    u0[idx[(:P2, [:I, :S])]] = Etot * 2 * pI * pS
    u0[idx[(:P2, [:I, :I])]] = Etot * pI^2

    # P₃ / C₃
    for (c, osz) in _P3_TRIPLE_STATES
        u0[idx[(:P3, c)]] = n_p3 * osz *
            pofstate(c[1]) * pofstate(c[2]) * pofstate(c[3])
    end
    for (c, osz) in _C3_TRIPLE_STATES
        u0[idx[(:C3, c)]] = n_c3 * osz *
            pofstate(c[1]) * pofstate(c[2]) * pofstate(c[3])
    end

    # 4-vertex shapes
    base_states = [:S, :I]
    counts_4v = (n_p4, n_k13, n_paw, n_c4, n_k4me, n_k4)
    for (sh, ncount) in zip(_SHAPES_4V_K3, counts_4v)
        for (c, osz) in enumerate_state_classes(sh, base_states)
            prob = pofstate(c[1]) * pofstate(c[2]) * pofstate(c[3]) * pofstate(c[4])
            u0[idx[(sh.name, c)]] = ncount * osz * prob
        end
    end
    return u0
end

# ─── Helper: count induced 4-vertex subgraphs in a Graphs.jl host graph ────
"""
    induced_subgraph_counts_4vertex(g::AbstractGraph) -> NamedTuple

Brute-force enumerate all 4-element vertex subsets of `g` and classify
each induced subgraph by the number of internal edges and degree
sequence. Returns a `NamedTuple` with fields
`(:p4, :k13, :paw, :c4, :k4me, :k4)` containing the count of induced
copies of each connected 4-vertex shape on a `k`-regular host
(`k = 3` is the typical use case but the helper makes no such
assumption — it just classifies whatever 4-vertex induced subgraphs
appear).

Time complexity is `O(N^4)`, fine for `N ≲ 500`.
"""
function induced_subgraph_counts_4vertex(g::Graphs.AbstractGraph)
    n   = Graphs.nv(g)
    cnt = (p4 = 0, k13 = 0, paw = 0, c4 = 0, k4me = 0, k4 = 0)
    np4 = 0; nk13 = 0; npaw = 0; nc4 = 0; nk4me = 0; nk4 = 0
    @inbounds for a in 1:n-3, b in a+1:n-2, c in b+1:n-1, d in c+1:n
        nodes = (a, b, c, d)
        # Build degree sequence and edge count of induced subgraph.
        deg = (0, 0, 0, 0)
        m   = 0
        # Manual edge tests for each of the 6 vertex pairs.
        e_ab = Graphs.has_edge(g, a, b)
        e_ac = Graphs.has_edge(g, a, c)
        e_ad = Graphs.has_edge(g, a, d)
        e_bc = Graphs.has_edge(g, b, c)
        e_bd = Graphs.has_edge(g, b, d)
        e_cd = Graphs.has_edge(g, c, d)
        m = (e_ab ? 1 : 0) + (e_ac ? 1 : 0) + (e_ad ? 1 : 0) +
            (e_bc ? 1 : 0) + (e_bd ? 1 : 0) + (e_cd ? 1 : 0)
        # Skip disconnected 4-vertex subgraphs (m < 3 cannot connect 4 nodes).
        m < 3 && continue
        da = (e_ab ? 1 : 0) + (e_ac ? 1 : 0) + (e_ad ? 1 : 0)
        db = (e_ab ? 1 : 0) + (e_bc ? 1 : 0) + (e_bd ? 1 : 0)
        dc = (e_ac ? 1 : 0) + (e_bc ? 1 : 0) + (e_cd ? 1 : 0)
        dd = (e_ad ? 1 : 0) + (e_bd ? 1 : 0) + (e_cd ? 1 : 0)
        ds = sort([da, db, dc, dd])
        # Connectivity check: ensure subgraph connected on these 4 nodes
        # via simple BFS from `a`.
        adj = (
            (false, e_ab, e_ac, e_ad),
            (e_ab, false, e_bc, e_bd),
            (e_ac, e_bc, false, e_cd),
            (e_ad, e_bd, e_cd, false),
        )
        seen = (true, false, false, false)
        # iterative BFS up to 4 nodes
        stack = (1,)
        seen_arr = [true, false, false, false]
        st = [1]
        while !isempty(st)
            u = pop!(st)
            for v in 1:4
                if adj[u][v] && !seen_arr[v]
                    seen_arr[v] = true
                    push!(st, v)
                end
            end
        end
        all(seen_arr) || continue
        # Classify by (m, degree-sequence)
        if m == 3
            if ds == [1, 1, 1, 3]
                nk13 += 1
            elseif ds == [1, 1, 2, 2]
                np4 += 1
            end
            # m==3 connected 4-vertex graphs: only K_{1,3} or P_4.
        elseif m == 4
            # paw (triangle + pendant): degrees [1,2,2,3]
            # C_4: degrees [2,2,2,2]
            if ds == [1, 2, 2, 3]
                npaw += 1
            elseif ds == [2, 2, 2, 2]
                nc4 += 1
            end
        elseif m == 5
            # K_4 - e (diamond): degrees [2,2,3,3]
            if ds == [2, 2, 3, 3]
                nk4me += 1
            end
        elseif m == 6
            # K_4: degrees [3,3,3,3]
            if ds == [3, 3, 3, 3]
                nk4 += 1
            end
        end
    end
    return (p4 = np4, k13 = nk13, paw = npaw, c4 = nc4, k4me = nk4me, k4 = nk4)
end
