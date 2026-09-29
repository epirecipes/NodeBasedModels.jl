# 00_legacy_closures.jl — legacy NodeBasedModels tests:
# closure method types.
#
# Moved verbatim (dedented one level) from the pre-0.2 test/runtests.jl by WP2
# (DESIGN_NetworkEpiCore.md §G.2). Owned by the work package that replaces this
# area; see §G.1. The imports are the original file's, so name resolution is unchanged.

using NodeBasedModels
using Test
using OrdinaryDiffEqDefault
using ModelingToolkit
using Graphs
using Random
using Catalyst
using Symbolics

# ─── Closure Methods ──────────────────────────────────────────────────
@testset "Closure Methods" begin
    @test BernoulliClosure() isa ClosureMethod
    @test KeelingClosure() isa ClosureMethod
    @test BarnardClosure() isa ClosureMethod
    @test PowerClosure(1.5).p == 1.5
    @test KirkwoodClosure() isa ClosureMethod
    @test KirkwoodClosure() isa KirkwoodClosure
end

# ─── WP24: Keeling's N is the population of the state (verified issue B03) ────────
# Keeling (1999) and Barnard's thesis (§4.2, C_AB = N[AB]/(n[A][B])) define the correlation
# factor with N the number of nodes in the units of the state, N = Σ_X [X]. 0.1 used the `N`
# field of the network structure, which need not match the `N` keyword that scales the state;
# the ϕ-weighted triangle term was then N_net/N_kw times too large or too small.
@testset "Keeling closure independent of the population scale (B03)" begin
    m = sir_model()
    p = Dict(:τ => 1 / 6, :γ => 0.25)
    function Rfrac(net, N)
        ps = generate_pairwise(m, net, KeelingClosure(); N = N, ε = 0.01, tspan = (0.0, 80.0))
        sol = solve_pairwise(ps, p; saveat = 0:1.0:80)
        sol[ps.singles[:R]] ./ N
    end
    ref = Rfrac(regular_network(6; ϕ = 0.2), 1.0)
    @test Rfrac(regular_network(6; ϕ = 0.2, N = 1000), 1000.0) ≈ ref rtol = 1e-6
    @test Rfrac(regular_network(6; ϕ = 0.2), 1000.0) ≈ ref rtol = 1e-6          # 0.1: 0.9846 vs 0.9294
    @test Rfrac(regular_network(6; ϕ = 0.2, N = 1000), 1.0) ≈ ref rtol = 1e-6   # 0.1: 0.0103
    pk = zeros(10); pk[4] = 0.5; pk[10] = 0.5
    @test Rfrac(degree_distribution_network(pk; ϕ = 0.2), 1000.0) ≈
          Rfrac(degree_distribution_network(pk; ϕ = 0.2), 1.0) rtol = 1e-6       # 0.1: 0.9190 vs 0.8426
    # the closure is homogeneous of degree 1 in (singles, pairs)
    s = Dict(:S => 0.7, :I => 0.2, :R => 0.1)
    q = Dict((:S, :S) => 2.4, (:S, :I) => 0.55, (:S, :R) => 0.25, (:I, :I) => 0.5, (:I, :R) => 0.2,
             (:R, :R) => 0.3)
    for net in (regular_network(6; ϕ = 0.2), degree_distribution_network(pk; ϕ = 0.2)),
        (A, B, C) in ((:S, :S, :I), (:I, :S, :I), (:I, :S, :R))
        @test triple_closure(A, B, C, Dict(k => 1000v for (k, v) in q), Dict(k => 1000v for (k, v) in s),
                             net, KeelingClosure()) ≈
              1000 * triple_closure(A, B, C, q, s, net, KeelingClosure()) rtol = 1e-12
    end

    # An independent transcription of Keeling's (1999) clustered SIR pairwise model on an
    # n-regular network (mixed convention: [XX] counts each XX edge twice), in counts, with
    # N = Σ[X]: the 0.1 mismatch case (network N = 1, keyword N = 1000) must reproduce it.
    n, ϕ, τ, γ, N, ε = 6, 0.2, 1 / 6, 0.25, 1000.0, 0.01
    ξ = (n - 1) / n
    # the correlation term is 0 while [A][C] = 0 (no R yet), as in the package's closure
    tri(AB, BC, B_, AC, A_, C_) =
        ξ * AB * BC / B_ * ((1 - ϕ) + (A_ * C_ == 0 ? 0.0 : ϕ * N * AC / (n * A_ * C_)))
    function keeling!(du, u, _, t)
        S, I, R, SS, SI, SR, II, IR, RR = u
        SSI = tri(SS, SI, S, SI, S, I)
        ISI = tri(SI, SI, S, II, I, I)
        ISR = tri(SI, SR, S, IR, I, R)
        du[1] = -τ * SI
        du[2] = τ * SI - γ * I
        du[3] = γ * I
        du[4] = -2τ * SSI
        du[5] = τ * (SSI - ISI - SI) - γ * SI
        du[6] = -τ * ISR + γ * SI
        du[7] = 2τ * (ISI + SI) - 2γ * II
        du[8] = τ * ISR + γ * II - γ * IR
        du[9] = 2γ * IR
        nothing
    end
    x = 1 - ε
    u0 = [N * x, N * ε, 0.0, n * N * x^2, n * N * x * ε, 0.0, n * N * ε^2, 0.0, 0.0]
    hand = OrdinaryDiffEqDefault.solve(ODEProblem(keeling!, u0, (0.0, 80.0)); saveat = 0:1.0:80,
                                        reltol = 1e-11, abstol = 1e-10)
    @test Rfrac(regular_network(6; ϕ = 0.2), 1000.0) ≈ [u[3] for u in hand.u] ./ N rtol = 1e-6
end
