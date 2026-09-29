# motif.jl — SIS motif closures (`motif_based_sis`): exactness, host bookkeeping and SSA validation.
#
# Owned by WP37 (DESIGN_NetworkEpiCore.md §G.2). The legacy tests of this area stay in
# 00_legacy_motif.jl; this suite adds the regression tests for VERIFIED_ISSUES.md B05 (k = 3, m = 4
# per-shape Kirkwood slot factor) and for the k = 2, m = 3 slot factor found in WP37, and validates
# the closures against NetworkOutbreaks.
#
# 1. Exactness on tree-indexed Markov states. On an infinite k-regular tree whose vertex states form a
#    stationary reversible two-state Markov chain along every edge (P(I|S) = a, P(I|I) = b), the state
#    of a vertex outside a motif depends on the motif only through the motif vertex it is attached to.
#    Every closure in `motif_based_sis` (Keeling κ for pairs, the single-vertex anchor with slot factor
#    n_ext/k, the chain Kirkwood, the per-shape Kirkwood) is exact there, so each right-hand side must
#    equal the master-equation derivative computed independently below. a = b is the factorising
#    (Bernoulli) initial condition; a ≠ b tests non-factorising, correlated states.
# 2. The per-shape Kirkwood registry and its anchor completions, checked structurally (the tree test
#    reaches only the P₄ and K₁,₃ rules).
# 3. B05 closed forms at the Bernoulli initial condition and the P₃ derivatives of the m = 4 system;
#    the k = 2, m = 3 closed form.
# 4. Finite hosts: the fast induced 4-vertex subgraph counter against brute force; every k = 3 closure
#    is exact at the Bernoulli state of 3-regular graphs with short cycles given their own counts; with
#    host multiplicities the P₃ / C₃ variables stay the exact image Mat·E₄ of the 4-vertex variables.
# 5. Validation against NetworkOutbreaks (NextReaction, NetworkOutbreaks.stable_rng streams, SIS
#    conditioned on survival as in DESIGN §E.2): a 3-regular host with N = 1000 and a fresh graph per
#    run, the vignette-10 host (N = 500, fixed), and a ring (N = 1000). Motif closures are
#    approximations (`:approximate` in §E.2); the assertions pin the ordering and the former
#    `@test_broken` bounds of 00_legacy_motif.jl, with margins far above the Monte Carlo SE.

using NodeBasedModels
using NodeBasedModels: _build_sis_k3_m4_rhs, _mat_3from4_multiplicities, _build_mat_3from4,
                       induced_subgraph_counts_4vertex, canonical_state
import NetworkOutbreaks as NO
using Graphs
using Random
using Statistics
using Printf
using Test

# ─── Tree-indexed Markov states and the exact master equation ──────────────────

"""Stationary reversible two-state chain on host edges: `a = P(I|S)`, `b = P(I|I)`."""
markov_chain(a, b) = (a = a, b = b, πI = a / (1 - b + a))
p_state(ch, x) = x === :I ? ch.πI : 1 - ch.πI
p_cond(ch, y, x) = (q = x === :I ? ch.b : ch.a; y === :I ? q : 1 - q)   # P(y | neighbour x)

function shape_neighbours(sh)
    nb = [Int[] for _ in 1:sh.n_nodes]
    for (a, b) in sh.edges
        push!(nb[a], b); push!(nb[b], a)
    end
    return nb
end

"""BFS order of a tree shape from vertex 1, as (vertex, parent) pairs; parent 0 for the root."""
function bfs_parents(sh)
    nb = shape_neighbours(sh)
    order = [(1, 0)]; seen = falses(sh.n_nodes); seen[1] = true
    head = 1
    while head <= length(order)
        v = order[head][1]; head += 1
        for w in nb[v]
            seen[w] && continue
            seen[w] = true; push!(order, (w, v))
        end
    end
    return order
end

"""Labelled embeddings per host vertex of the connected shape `sh` into the k-regular tree (0 unless a tree)."""
function embeddings_per_node(sh, k)
    length(sh.edges) == sh.n_nodes - 1 || return 0
    nb = shape_neighbours(sh)
    c = 1
    for (v, parent) in bfs_parents(sh)
        children = count(w -> w != parent, nb[v])
        free = k - (parent == 0 ? 0 : 1)
        for j in 0:children-1
            c *= free - j
        end
    end
    return c
end

function tree_probability(sh, σ, ch)
    p = 1.0
    for (v, parent) in bfs_parents(sh)
        p *= parent == 0 ? p_state(ch, σ[v]) : p_cond(ch, σ[v], σ[parent])
    end
    return p
end

"""
The motif state vector of `sys` and the exact master-equation derivative there (per-edge rate β, recovery
rate γ) on a k-regular host with `counts[shape]` induced copies of each tracked shape, when the state of a
vertex outside a motif depends on the motif only through its attachment vertex, with P(I | S) = `ch.a`.
This holds on the k-regular tree for any chain `ch`, and on any k-regular graph when the vertex states are
independent (`ch.a == ch.b`); motif state probabilities come from a spanning tree of the shape, which is
exact in both cases (shapes with cycles have no copies on a tree).
"""
function motif_state_and_exact_rhs(sys, k, counts, ch, β, γ)
    u = zeros(length(sys.u0)); du = zeros(length(sys.u0))
    for sh in sys.shapes
        n = sh.n_nodes
        G = length(sh.automorphisms)
        nb = shape_neighbours(sh)
        n_ext = [k - length(nb[i]) for i in 1:n]
        labelled = [collect(t) for t in Iterators.product(ntuple(_ -> (:S, :I), n)...)]
        L = Dict(σ => counts[sh.name] * G * tree_probability(sh, σ, ch) for σ in labelled)
        dL = Dict(σ => 0.0 for σ in labelled)
        for σ in labelled, i in 1:n
            rate = σ[i] === :I ? γ :
                   β * (count(j -> σ[j] === :I, nb[i]) + n_ext[i] * p_cond(ch, :I, :S))
            τ = copy(σ); τ[i] = σ[i] === :I ? :S : :I
            dL[σ] -= rate * L[σ]
            dL[τ] += rate * L[σ]
        end
        for σ in labelled
            canon, orbit = canonical_state(sh, σ)
            canon == σ || continue
            stab = G ÷ orbit
            j = sys.index[(sh.name, canon)]
            u[j] = L[σ] / stab
            du[j] = dL[σ] / stab
        end
    end
    return u, du
end

"""Induced copies of each shape of `sys` on the k-regular tree with `N` vertices."""
tree_counts(sys, k, N) =
    Dict(sh.name => N * embeddings_per_node(sh, k) / length(sh.automorphisms) for sh in sys.shapes)

markov_state_and_exact_rhs(sys, k, N::Real, ch, β, γ) =
    motif_state_and_exact_rhs(sys, k, tree_counts(sys, k, N), ch, β, γ)

rhs_at(sys, u) = (du = zeros(length(u)); sys.rhs!(du, u, sys.params, 0.0); du)
with_rhs(sys, rhs!) = MotifSystem(sys.shapes, sys.variables, sys.index, rhs!, sys.u0, sys.tspan,
                                  sys.params, sys.model, sys.network, sys.closure)
maxrelerr(x, y) = maximum(abs.(x .- y)) / max(maximum(abs.(y)), 1e-300)

@testset "Motif RHS is exact on tree-indexed Markov states" begin
    β, γ, N = 0.6, 0.4, 1000.0
    chains = [("Bernoulli ε = 0.05", markov_chain(0.05, 0.05)),
              ("correlated P(I|S) = 0.2, P(I|I) = 0.5", markov_chain(0.2, 0.5)),
              ("anti-correlated P(I|S) = 0.6, P(I|I) = 0.1", markov_chain(0.6, 0.1))]
    cases = [(2, 2, (;)), (2, 3, (;)), (2, 3, (_use_generic_chain_builder = true,)),
             (2, 4, (;)), (2, 5, (;)), (2, 6, (;)), (3, 2, (;)), (3, 3, (;)), (3, 4, (;))]
    for (k, m, kw) in cases, (label, ch) in chains
        sys = motif_based_sis(; β, γ, k, m, N, ε = ch.πI, kw...)
        u, exact = markov_state_and_exact_rhs(sys, k, N, ch, β, γ)
        @testset "k = $k, m = $m $(isempty(kw) ? "" : "(generic chain) ")$label" begin
            @test maxrelerr(rhs_at(sys, u), exact) < 1e-12
            if k == 3 && m == 4
                ua = with_rhs(sys, _build_sis_k3_m4_rhs(sys.index; closure_kind = :uniform_anchor))
                @test maxrelerr(rhs_at(ua, u), exact) < 1e-12
            end
        end
    end
    # The Bernoulli state is the package's own initial condition.
    for (k, m) in ((2, 3), (3, 3), (3, 4))
        sys = motif_based_sis(; β, γ, k, m, N, ε = 0.05)
        u, _ = markov_state_and_exact_rhs(sys, k, N, markov_chain(0.05, 0.05), β, γ)
        @test maxrelerr(sys.u0, u) < 1e-13
    end
end

@testset "Per-shape Kirkwood registry (k = 3, m = 4) is structurally consistent" begin
    # The tree-Markov test above only reaches the P₄ and K₁,₃ rules (the other shapes have no copies
    # on a tree). Here every rule is checked: the dropped vertex w is not adjacent to the extended
    # vertex i (so i keeps its n_ext slots in the anchor), perm3 maps the target 3-shape onto the
    # induced anchor σ − {w}, and perm4 maps the target 4-shape onto the anchor plus an external
    # vertex e adjacent to i only.
    rules = NodeBasedModels._CLOSURE_RULES_4V
    shapes = Dict(s.name => s for s in motif_based_sis(β = 0.5, γ = 0.3, k = 3, m = 4).shapes)
    edgeset(pairs) = Set(Set(p) for p in pairs)
    for name in (:P4, :K13, :paw, :C4, :K4me, :K4)
        sh = shapes[name]
        nb = shape_neighbours(sh)
        for i in 1:4
            @test haskey(rules, (name, i)) == (3 - length(nb[i]) > 0)
            haskey(rules, (name, i)) || continue
            rule = rules[(name, i)]
            anchor = collect(rule.perm3)
            w = only(setdiff(1:4, anchor))
            @test i in anchor
            @test !(w in nb[i])
            induced = [(a, b) for (a, b) in sh.edges if a in anchor && b in anchor]
            t3 = shapes[rule.target3]
            @test edgeset((rule.perm3[a], rule.perm3[b]) for (a, b) in t3.edges) == edgeset(induced)
            @test sort(filter(!=(0), collect(rule.perm4))) == sort(anchor)
            t4 = shapes[rule.target4]
            host = vcat([(a, b) for (a, b) in induced], [(i, 0)])
            @test edgeset((rule.perm4[a], rule.perm4[b]) for (a, b) in t4.edges) == edgeset(host)
            # All completions of the anchor by an external neighbour e of i: e adjacent to i and to
            # the subsets S = ∅, {o₁}, {o₂}, {o₁, o₂} of the other anchor vertices, in that order.
            comps = NodeBasedModels._kirkwood_completions(sh, i, rule)
            others = [a for a in anchor if a != i]
            subsets = ([], [others[1]], [others[2]], others)
            @test length(comps) == 4
            @test comps[1] == (rule.target4, rule.perm4)
            for ((target4, perm4), S) in zip(comps, subsets)
                hostS = vcat(host, [(s, 0) for s in S])
                @test edgeset((perm4[a], perm4[b]) for (a, b) in shapes[target4].edges) == edgeset(hostS)
            end
        end
    end
end

@testset "B05: k = 3, m = 4 closed forms at the Bernoulli initial condition" begin
    β, γ, N, ε = 0.6, 0.4, 1000.0, 0.05
    sys = motif_based_sis(; β, γ, k = 3, m = 4, tspan = (0.0, 25.0), N, ε)
    du = rhs_at(sys, sys.u0)
    # 6N induced P₄. SSSS gains by recovery from the four one-I states and loses by infection of its
    # vertices from their external neighbours (2 + 1 + 1 + 2 = 6 slots, each I with probability ε);
    # package before the fix: -77.16.
    @test du[sys.index[(:P4, [:S, :S, :S, :S])]] ≈ 6N * (4γ * ε * (1 - ε)^3 - 6β * ε * (1 - ε)^4) rtol = 1e-12
    # P₃ derivatives of the m = 4 system (Mat·dE₄) equal those of the m = 3 system at this state
    # (package before the fix: ISS -68.2 instead of +24.94).
    sys3 = motif_based_sis(; β, γ, k = 3, m = 3, N, ε)
    d3 = rhs_at(sys3, sys3.u0)
    for c in ([:S, :S, :S], [:I, :S, :S], [:S, :I, :S], [:I, :S, :I], [:I, :I, :S], [:I, :I, :I])
        @test du[sys.index[(:P3, c)]] ≈ d3[sys3.index[(:P3, c)]] rtol = 1e-12
    end
    @test du[sys.index[(:P3, [:I, :S, :S])]] ≈ 24.9375 rtol = 1e-12
    # The Kirkwood and single-vertex-anchor closures agree on a factorising state.
    du_u = rhs_at(with_rhs(sys, _build_sis_k3_m4_rhs(sys.index; closure_kind = :uniform_anchor)), sys.u0)
    @test du ≈ du_u rtol = 1e-12
    # The trajectory stays in the physical simplex (package before the fix: I/N reached 1.105 and a
    # motif count reached -185).
    sol = solve_motif(sys; reltol = 1e-10, abstol = 1e-12)
    iI = sys.index[(:singleton, [:I])]
    @test all(0 .<= getindex.(sol.u, iI) .<= N)
    @test minimum(minimum, sol.u) > -1e-6 * N
    @test 0.74 < sol.u[end][iI] / N < 0.76
    # No warning any more: m = 4 is an ordinary approximation, validated below.
    @test_logs motif_based_sis(; β, γ, k = 3, m = 4, N, ε)
end

@testset "k = 2, m = 3 single-vertex anchor carries the slot factor 1/2" begin
    β, γ, N, ε = 0.6, 0.4, 1000.0, 0.05
    sys = motif_based_sis(; β, γ, k = 2, m = 3, N, ε)
    du = rhs_at(sys, sys.u0)
    # Each end of an SSS path has one external neighbour, infected with probability ε
    # (package before the fix: -48.735, i.e. twice the external infection pressure).
    @test du[sys.index[(:P3, [:S, :S, :S])]] ≈ N * (3γ * ε * (1 - ε)^2 - 2β * ε * (1 - ε)^3) rtol = 1e-12
    gen = motif_based_sis(; β, γ, k = 2, m = 3, N, ε, _use_generic_chain_builder = true)
    @test du ≈ rhs_at(gen, gen.u0) rtol = 1e-12
end

# ─── Finite-host bookkeeping for k = 3, m = 4 ──────────────────────────────────

"""Induced connected 4-vertex subgraph counts by brute force over all 4-subsets (reference)."""
function brute_force_counts(g)
    c = Dict(s => 0 for s in (:p4, :k13, :paw, :c4, :k4me, :k4))
    n = nv(g)
    for a in 1:n-3, b in a+1:n-2, cc in b+1:n-1, d in cc+1:n
        sub, _ = induced_subgraph(g, [a, b, cc, d])
        is_connected(sub) || continue
        ds = sort(degree(sub)); m = ne(sub)
        s = m == 3 ? (ds == [1, 1, 1, 3] ? :k13 : :p4) :
            m == 4 ? (ds == [2, 2, 2, 2] ? :c4 : :paw) : m == 5 ? :k4me : :k4
        c[s] += 1
    end
    return (p4 = c[:p4], k13 = c[:k13], paw = c[:paw], c4 = c[:c4], k4me = c[:k4me], k4 = c[:k4])
end

@testset "induced_subgraph_counts_4vertex" begin
    graphs = [random_regular_graph(30, 3; rng = Xoshiro(1)), smallgraph(:petersen), complete_graph(4),
              complete_graph(6), complete_bipartite_graph(3, 3), smallgraph(:house), wheel_graph(7),
              erdos_renyi(25, 0.2; rng = Xoshiro(2)), path_graph(4), star_graph(5), cycle_graph(4),
              SimpleGraph(3)]
    for g in graphs
        @test induced_subgraph_counts_4vertex(g) == brute_force_counts(g)
    end
    @test_throws ArgumentError induced_subgraph_counts_4vertex(SimpleDiGraph(4))
    # The vignette-10 host is fixed by its seed; its counts are part of the documented example.
    gv = random_regular_graph(500, 3; rng = MersenneTwister(20))
    @test induced_subgraph_counts_4vertex(gv) == (p4 = 2958, k13 = 494, paw = 6, c4 = 6, k4me = 0, k4 = 0)
end

"""Induced copies of every tracked shape on the graph `g`."""
function host_counts(g)
    n_c3 = sum(triangles(g)) ÷ 3
    n_p3 = sum(d * (d - 1) ÷ 2 for d in degree(g)) - 3n_c3
    c4 = induced_subgraph_counts_4vertex(g)
    return Dict(:singleton => nv(g), :P2 => ne(g), :P3 => n_p3, :C3 => n_c3, :P4 => c4.p4, :K13 => c4.k13,
                :paw => c4.paw, :C4 => c4.c4, :K4me => c4.k4me, :K4 => c4.k4)
end

@testset "k = 3 closures are exact at the Bernoulli state of finite hosts with short cycles" begin
    # With independent vertex states the expected number of infected external neighbours of a motif
    # vertex is n_ext·ε on any 3-regular graph, and every closure reproduces it from the host's own
    # counts: Keeling's pair closure, the single-vertex anchor, and the per-shape Kirkwood closure once
    # all completions of its anchor are summed and the 3-from-4 multiplicities are the host averages.
    # (Before WP37 the m = 4 system failed here on every host listed. With the slot factor and the
    # multiplicities fixed but only the tree-like completion of each anchor, it still fails on every
    # host below that has a C₃ or C₄ and external slots: all but K₄ and the Petersen graph.)
    β, γ, ε = 0.6, 0.4, 0.1
    hosts = [("K₄", complete_graph(4)), ("K₃,₃", complete_bipartite_graph(3, 3)),
             ("prism", circular_ladder_graph(3)), ("cube", smallgraph(:cubical)),
             ("Petersen", smallgraph(:petersen)), ("3-regular, N = 16", random_regular_graph(16, 3; rng = Xoshiro(3))),
             ("vignette-10 host", random_regular_graph(500, 3; rng = MersenneTwister(20)))]
    for (label, g) in hosts, m in 2:4
        c = host_counts(g)
        kw = m == 2 ? (;) :
             m == 3 ? (n_p3 = c[:P3], n_c3 = c[:C3]) :
                      (n_p3 = c[:P3], n_c3 = c[:C3], n_p4 = c[:P4], n_k13 = c[:K13], n_paw = c[:paw],
                       n_c4 = c[:C4], n_k4me = c[:K4me], n_k4 = c[:K4])
        sys = motif_based_sis(; β, γ, k = 3, m, N = Float64(nv(g)), ε, map(Float64, kw)...)
        u, exact = motif_state_and_exact_rhs(sys, 3, c, markov_chain(ε, ε), β, γ)
        @testset "$label, m = $m" begin
            @test maxrelerr(sys.u0, u) < 1e-13
            @test maxrelerr(rhs_at(sys, u), exact) < 1e-12
        end
    end
end

@testset "3-from-4 multiplicities follow the host counts" begin
    # Triangles without any triangle-containing 4-vertex shape are inconsistent on a 3-regular
    # host (the C₃ variables would be frozen): an error, not a silent fallback.
    @test_throws ArgumentError motif_based_sis(; β = 0.6, γ = 0.4, k = 3, m = 4, N = 100.0,
                                               n_c3 = 5.0)
    @test motif_based_sis(; β = 0.6, γ = 0.4, k = 3, m = 4, N = 100.0, n_c3 = 5.0,
                          n_paw = 15.0) isa MotifSystem
    # Asymptotic 3-regular host: every P₃ has 5 connected 4-vertex supersets, every C₃ has 3.
    @test _mat_3from4_multiplicities(n_p3 = 3000.0, n_c3 = 0.0, n_p4 = 6000.0, n_k13 = 1000.0) ==
          (P3 = 5.0, C3 = 3.0)
    # Vignette-10 host: (2·2958 + 3·494 + 2·6 + 4·6)/1494 and 6/2.
    counts = (n_p3 = 1494.0, n_c3 = 2.0, n_p4 = 2958.0, n_k13 = 494.0, n_paw = 6.0, n_c4 = 6.0,
              n_k4me = 0.0, n_k4 = 0.0)
    mult = _mat_3from4_multiplicities(; counts...)
    @test mult.P3 ≈ 7434 / 1494 rtol = 1e-15
    @test mult.C3 == 3.0
    # K₄ host: 4 triangles, one K₄; every C₃ lies in exactly one K₄.
    @test _mat_3from4_multiplicities(n_p3 = 0.0, n_c3 = 4.0, n_p4 = 0.0, n_k13 = 0.0, n_k4 = 1.0) ==
          (P3 = 5.0, C3 = 1.0)

    β, γ, ε = 0.6, 0.4, 0.05
    sys = motif_based_sis(; β, γ, k = 3, m = 4, tspan = (0.0, 25.0), N = 500.0, ε, counts...)
    M = _build_mat_3from4(sys.index; k = 3, ext_p3 = mult.P3, ext_c3 = mult.C3)
    image(u) = Dict(i3 => sum(cf * u[i4] for (i4, cf) in lst) for (i3, lst) in M)
    # At the factorising initial condition the tracked P₃/C₃ counts are the image of the 4-vertex
    # counts (with the asymptotic multiplicity 5 the P₃ total would be 1486.8 instead of 1494) …
    for (i3, v) in image(sys.u0)
        @test v ≈ sys.u0[i3] rtol = 1e-12
    end
    # … and they stay the image along the solution, because dE₃ = Mat·dE₄.
    sol = solve_motif(sys; saveat = 5.0, reltol = 1e-10, abstol = 1e-12)
    for u in sol.u, (i3, v) in image(u)
        @test v ≈ u[i3] atol = 1e-7 * 500
    end
    # Conservation: every shape keeps its host count.
    for (shape, n) in ((:P3, 1494.0), (:C3, 2.0), (:P4, 2958.0), (:K13, 494.0), (:paw, 6.0), (:C4, 6.0))
        tot = sum(sol.u[end][sys.index[(v.shape.name, v.state)]] for v in sys.variables if v.shape.name == shape)
        @test tot ≈ n rtol = 1e-8
    end
end

# ─── Validation against NetworkOutbreaks ────────────────────────────────────────

const BASE_SEED = 20260926          # DESIGN §E.2
const TGRID = collect(0.0:0.5:25.0)
const NSIMS = 200

sis_outbreak_model(β, γ) = NO.OutbreakModel([:S, :I], [false, true],
    [NO.OutbreakTransition(:S, :I, β, :infection), NO.OutbreakTransition(:I, :S, γ, :spontaneous)];
    name = :sis)

"""
Mean and SE of the prevalence I/N on `TGRID` over `NSIMS` NextReaction runs, conditioned on survival
(I(t_end) > 0). Run r uses the host `graph_of(r)` and SSA seed `BASE_SEED + 2^32 + stream + r`
(DESIGN §J.7; `simulate(spec; seed = s)` draws from `NetworkOutbreaks.stable_rng(s)`). Exactly
`round(ε N)` nodes are seeded, uniformly without replacement.
"""
function ssa_prevalence(graph_of; β, γ, ε, stream)
    X = zeros(length(TGRID), NSIMS)
    for r in 1:NSIMS
        g = graph_of(r)
        spec = NO.OutbreakSpec(model = sis_outbreak_model(β, γ), network = g,
                               initial = NO.SeedFraction(:I => ε), tspan = (0.0, TGRID[end]))
        tr = NO.simulate(spec; algorithm = NO.NextReaction(), seed = BASE_SEED + 2^32 + stream + r)
        I = NO.compartment_series(tr, :I)
        X[:, r] = [I[max(searchsortedlast(tr.times, t), 1)] / nv(g) for t in TGRID]
    end
    keep = X[end, :] .> 0
    Xs = X[:, keep]
    return (mean = vec(mean(Xs; dims = 2)), se = vec(std(Xs; dims = 2)) ./ sqrt(count(keep)),
            runs = count(keep))
end

function prevalence_curve(sys)
    sol = solve_motif(sys; reltol = 1e-10, abstol = 1e-12)
    iI = sys.index[(:singleton, [:I])]
    return [sol(t)[iI] / sys.params.N for t in TGRID]
end

"""D∞ = max_t |x − x̄| of each curve against the SSA mean, logged with the Monte Carlo SE."""
function sup_errors(label, ref, curves)
    D = Dict(name => maximum(abs.(x .- ref.mean)) for (name, x) in curves)
    @info "$label: $(ref.runs)/$NSIMS runs survived; SE ≤ $(round(maximum(ref.se); sigdigits = 2))" *
          join([@sprintf("\n  %-28s D∞ = %.4f", n, D[n]) for (n, _) in curves])
    return D
end

@testset "Validation against NetworkOutbreaks SSA" begin
    β, γ, ε = 0.6, 0.4, 0.05

    @testset "3-regular host, N = 1000, fresh graph per run" begin
        N = 1000
        ref = ssa_prevalence(r -> random_regular_graph(N, 3; rng = NO.stable_rng(BASE_SEED + r));
                             β, γ, ε, stream = 0)
        @test ref.runs == NSIMS
        curves = [("m = $m", prevalence_curve(motif_based_sis(; β, γ, k = 3, m, tspan = (0.0, 25.0),
                                                               N = Float64(N), ε))) for m in 2:4]
        D = sup_errors("RRG(3), N = $N", ref, curves)
        @test maximum(ref.se) < 0.005
        @test D["m = 4"] < 0.015                 # package before the fix: 0.359
        @test D["m = 4"] < D["m = 3"] - 0.01     # D∞(m = 3) ≈ 0.025
        @test D["m = 3"] < D["m = 2"] - 0.03     # D∞(m = 2) ≈ 0.072
    end

    @testset "vignette-10 host (N = 500, MersenneTwister(20)), quenched" begin
        g = random_regular_graph(500, 3; rng = MersenneTwister(20))
        cnt = induced_subgraph_counts_4vertex(g)
        n_c3 = sum(triangles(g)) ÷ 3
        n_p3 = sum(d * (d - 1) ÷ 2 for d in degree(g)) - 3n_c3
        ref = ssa_prevalence(r -> g; β, γ, ε, stream = 10^6)
        @test ref.runs == NSIMS
        common = (; β, γ, k = 3, tspan = (0.0, 25.0), N = 500.0, ε)
        curves = [("m = 2", prevalence_curve(motif_based_sis(; common..., m = 2))),
                  ("m = 3", prevalence_curve(motif_based_sis(; common..., m = 3, n_p3 = Float64(n_p3),
                                                              n_c3 = Float64(n_c3)))),
                  ("m = 4", prevalence_curve(motif_based_sis(; common..., m = 4, n_p3 = Float64(n_p3),
                                                              n_c3 = Float64(n_c3), n_p4 = Float64(cnt.p4),
                                                              n_k13 = Float64(cnt.k13), n_paw = Float64(cnt.paw),
                                                              n_c4 = Float64(cnt.c4), n_k4me = Float64(cnt.k4me),
                                                              n_k4 = Float64(cnt.k4))))]
        D = sup_errors("vignette-10 host", ref, curves)
        # The former @test_broken bounds of 00_legacy_motif.jl (package before the fix: 0.348).
        @test D["m = 4"] ≤ D["m = 3"] + 0.02
        @test D["m = 4"] ≤ 0.05
        @test D["m = 4"] < D["m = 3"] - 0.01
        @test D["m = 3"] < D["m = 2"] - 0.03
    end

    @testset "ring, N = 1000" begin
        N = 1000
        ring = cycle_graph(N)
        ref = ssa_prevalence(r -> ring; β, γ, ε, stream = 2 * 10^6)
        @test ref.runs == NSIMS
        common = (; β, γ, k = 2, tspan = (0.0, 25.0), N = Float64(N), ε)
        curves = [("m = $m", prevalence_curve(motif_based_sis(; common..., m))) for m in 2:6]
        push!(curves, ("m = 3 (generic chain)",
                       prevalence_curve(motif_based_sis(; common..., m = 3, _use_generic_chain_builder = true))))
        D = sup_errors("ring, N = $N", ref, curves)
        # Package before the fix: D∞(m = 3) = 0.318 > D∞(m = 2) = 0.305.
        @test D["m = 3"] < D["m = 2"] - 0.1
        for m in 4:6
            @test D["m = $m"] < D["m = 3"]
        end
    end
end
