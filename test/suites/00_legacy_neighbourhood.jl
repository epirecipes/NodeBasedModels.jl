# 00_legacy_neighbourhood.jl — legacy NodeBasedModels tests:
# the n = 2 neighbourhood model (Phase C).
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

# ─── Neighbourhood model (Phase C; Keeling et al. 2016, Approx 3, n=2) ──
@testset "Neighbourhood model (Phase C, n=2)" begin
    @testset "Layout, IC, and conservation at t=0" begin
        sys = generate_neighbourhood(sis_model(), 3, 2;
                                      β = 0.6, γ = 0.4,
                                      N = 1.0, ε = 0.05,
                                      tspan = (0.0, 50.0))
        @test sys.k == 3 && sys.n == 2
        @test length(sys.u0) == 2 * (3 + 1)
        @test sys.var_names ==
              [:S_0, :S_1, :S_2, :S_3, :I_0, :I_1, :I_2, :I_3]
        @test isapprox(sum(sys.u0), 1.0; atol = 1e-12)
        k = sys.k
        ed_S = sum(y * sys.u0[sys.index[(:S, y)]] for y in 0:k)
        ed_I = sum((k - y) * sys.u0[sys.index[(:I, y)]] for y in 0:k)
        @test isapprox(ed_S, ed_I; atol = 1e-12)
    end

    @testset "Conservation along trajectory (k=3)" begin
        sys = generate_neighbourhood(sis_model(), 3, 2;
                                      β = 0.7, γ = 0.5,
                                      N = 1.0, ε = 0.05,
                                      tspan = (0.0, 60.0))
        sol = solve_neighbourhood(sys; reltol = 1e-10, abstol = 1e-12,
                                   saveat = 5.0)
        k = sys.k
        for u in sol.u
            @test isapprox(sum(u), 1.0; atol = 1e-8)
            ed_S = sum(y * u[sys.index[(:S, y)]] for y in 0:k)
            ed_I = sum((k - y) * u[sys.index[(:I, y)]] for y in 0:k)
            @test isapprox(ed_S, ed_I; atol = 1e-8)
        end
    end

    @testset "Symbolic validator agreement (k=3)" begin
        sys = generate_neighbourhood(sis_model(), 3, 2;
                                      β = 0.7, γ = 0.4,
                                      N = 1.0, ε = 0.1)
        rhs_sym!, var_keys, _ = build_neighbourhood_symbolic_rhs(3)
        @test var_keys ==
              [(:S, 0), (:S, 1), (:S, 2), (:S, 3),
               (:I, 0), (:I, 1), (:I, 2), (:I, 3)]
        n = length(sys.u0)
        du_n = zeros(n); du_s = zeros(n)
        p_vec = (sys.params.β, sys.params.γ)
        sys.rhs!(du_n, sys.u0, sys.params, 0.0)
        rhs_sym!(du_s, sys.u0, p_vec, 0.0)
        @test isapprox(du_n, du_s; atol = 1e-12, rtol = 0)
        rng = MersenneTwister(20240321)
        for _ in 1:5
            u = (0.1 .+ 0.9 .* rand(rng, n)) .* sys.u0
            sys.rhs!(du_n, u, sys.params, 0.0)
            rhs_sym!(du_s, u, p_vec, 0.0)
            @test isapprox(du_n, du_s; atol = 1e-10, rtol = 0)
        end
        u_dfe = 1e-15 .* sys.u0
        sys.rhs!(du_n, u_dfe, sys.params, 0.0)
        rhs_sym!(du_s, u_dfe, p_vec, 0.0)
        @test isapprox(du_n, du_s; atol = 1e-12, rtol = 0)
    end

    @testset "Symbolic validator at k=2" begin
        sys = generate_neighbourhood(sis_model(), 2, 2;
                                      β = 1.5, γ = 0.5, N = 1.0, ε = 0.1)
        rhs_sym!, _, _ = build_neighbourhood_symbolic_rhs(2)
        n = length(sys.u0)
        du_n = zeros(n); du_s = zeros(n)
        p_vec = (sys.params.β, sys.params.γ)
        sys.rhs!(du_n, sys.u0, sys.params, 0.0)
        rhs_sym!(du_s, sys.u0, p_vec, 0.0)
        @test isapprox(du_n, du_s; atol = 1e-12, rtol = 0)
    end

    @testset "Reduces toward mean-field at high β" begin
        β, γ, k = 10.0, 1.0, 3
        sys = generate_neighbourhood(sis_model(), k, 2;
                                      β = β, γ = γ,
                                      N = 1.0, ε = 0.05,
                                      tspan = (0.0, 30.0))
        sol = solve_neighbourhood(sys; reltol = 1e-10, abstol = 1e-12)
        nb_prev = sum(sol.u[end][sys.index[(:I, y)]] for y in 0:k)
        mf_prev = 1.0 - γ / (β * k)
        @test isapprox(nb_prev, mf_prev; atol = 5e-3)
    end

    @testset "Endemic prevalence near paper Fig 5 (k=3, γ=1)" begin
        # τ values where the paper reports a strongly endemic regime
        # (well above τ_C ≈ 0.544).  The neighbourhood (n=2) curve in
        # Fig 5B passes through (τ=1.0, prev≈0.59) and (τ=1.5,
        # prev≈0.75) within visual reading accuracy.
        cases = [(1.0, 0.59), (1.5, 0.75)]
        for (τ, ref) in cases
            sys = generate_neighbourhood(sis_model(), 3, 2;
                                          β = τ, γ = 1.0,
                                          N = 1.0, ε = 0.05,
                                          tspan = (0.0, 200.0))
            sol = solve_neighbourhood(sys; reltol = 1e-10, abstol = 1e-12)
            prev = sum(sol.u[end][sys.index[(:I, y)]] for y in 0:3)
            @test isapprox(prev, ref; atol = 0.05)
        end
    end

    @testset "Throws on unsupported n / non-SIS model" begin
        @test_throws ArgumentError generate_neighbourhood(
            sis_model(), 3, 3; β = 1.0, γ = 1.0)
        @test_throws ArgumentError generate_neighbourhood(
            sis_model(), 3, 1; β = 1.0, γ = 1.0)
        @test_throws ArgumentError generate_neighbourhood(
            sir_model(), 3, 2; β = 1.0, γ = 1.0)
    end

    @testset "Gillespie comparison (random 3-regular, N=500)" begin
        using StableRNGs
        rng = StableRNG(20240301)
        N = 500
        g = random_regular_graph(N, 3; rng = rng)
        net = GraphNetwork(g)
        β, γ = 0.6, 0.4
        avg = gillespie_sis_average(net; nruns = 48, dt = 1.0,
                                     tmax_grid = 80.0,
                                     infection_rate = β,
                                     recovery_rate = γ,
                                     initial_infected = collect(1:25))
        gill_prev = avg.I_mean[end] / N

        sys = generate_neighbourhood(sis_model(), 3, 2;
                                      β = β, γ = γ,
                                      N = 1.0, ε = 25.0 / N,
                                      tspan = (0.0, 80.0))
        sol = solve_neighbourhood(sys; reltol = 1e-10, abstol = 1e-12)
        nb_prev = sum(sol.u[end][sys.index[(:I, y)]] for y in 0:3)
        @info "Phase C Gillespie comparison" β γ N gill_prev nb_prev abs_diff =
            abs(nb_prev - gill_prev)
        @test abs(nb_prev - gill_prev) ≤ 0.05
    end
end

# ─── Golden: frozen numbers of this area (test/golden/neighbourhood/) ─────────────────
include(joinpath(@__DIR__, "..", "golden", "GoldenIO.jl"))
GoldenIO.check_area("neighbourhood")
