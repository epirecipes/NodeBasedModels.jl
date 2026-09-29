# 00_legacy_analysis.jl — legacy NodeBasedModels tests:
# R0, epidemic threshold, early growth rate and disease-free equilibrium (analysis.jl).
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

# ─── R₀ Computation ──────────────────────────────────────────────────
@testset "R₀ Computation" begin
    @testset "Homogeneous Bernoulli SIR" begin
        net = regular_network(6)
        R0 = basic_reproduction_number(net, BernoulliClosure(), 0.5, 0.1)
        @test isapprox(R0, 0.5 * 4 / 0.1)  # τ(n-2)/γ = 20
    end

    @testset "Model-based symbolic Bernoulli SIR" begin
        expr = basic_reproduction_number(sir_model(), regular_network(6), BernoulliClosure())
        @test !(expr isa Pair)
        @test occursin("τ", string(expr))
        @test occursin("γ", string(expr))
        @test occursin("4", string(expr))
    end

    @testset "Model-based symbolic zero remains symbolic" begin
        # WP24 (B01): the clustered closures have no closed-form threshold ratio, so the
        # symbolic method throws for them (0.1 returned ad hoc corrections).
        expr = basic_reproduction_number(sir_model(), regular_network(2; ϕ=0.3), BernoulliClosure())
        @test expr isa Symbolics.Num
        @test string(expr) == "0"
        for closure in (KeelingClosure(), BarnardClosure())
            @test_throws ArgumentError basic_reproduction_number(sir_model(), regular_network(2; ϕ=0.3), closure)
        end

        zero_degree = degree_distribution_network([1.0])
        expr = basic_reproduction_number(sir_model(), zero_degree, BernoulliClosure())
        @test expr isa Symbolics.Num
        @test string(expr) == "0"
    end

    @testset "Homogeneous Keeling SIR" begin
        net = regular_network(6; ϕ=0.3)
        R0 = basic_reproduction_number(net, KeelingClosure(), 0.5, 0.1)
        @test R0 < 0.5 * 4 / 0.1  # clustering reduces R₀
        @test R0 > 0
    end

    @testset "Heterogeneous Bernoulli" begin
        # WP24 (B01): a heterogeneous network needs the dynamics (0.1 returned the SIS value)
        net = erdos_renyi_network(5.0)
        @test_throws ArgumentError basic_reproduction_number(net, BernoulliClosure(), 0.5, 0.1)
        for dynamics in (:SIR, :SIS)
            R0 = basic_reproduction_number(net, BernoulliClosure(), 0.5, 0.1; dynamics)
            @test R0 > 0
            @test isfinite(R0)
        end
    end

    @testset "Epidemic threshold" begin
        net = regular_network(6)
        τ_c = epidemic_threshold(net, BernoulliClosure(), 0.1)
        @test isapprox(τ_c, 0.1 / 4)  # γ/(n-2)
    end

    @testset "Early growth rate" begin
        net = regular_network(6)
        r0 = early_growth_rate(net, BernoulliClosure(), 0.5, 0.1)
        @test isapprox(r0, 0.5 * 4 - 0.1)  # τ(n-2) - γ = 1.9
    end

    @testset "Heterogeneous early growth rate" begin
        net = erdos_renyi_network(5.0)
        r0 = early_growth_rate(net, BernoulliClosure(), 0.5, 0.1; dynamics = :SIR)
        expected = 0.5 * (net.second_moment - 2net.mean_degree) / net.mean_degree - 0.1
        @test isapprox(r0, expected)
    end
end

# ─── Disease-free Equilibrium ─────────────────────────────────────────
@testset "Disease-free Equilibrium" begin
    m = sir_model()
    net = regular_network(6)
    dfe = disease_free_equilibrium(m, net; N=100.0)
    @test dfe["[S]"] == 100.0
    @test dfe["[I]"] == 0.0
    @test dfe["[R]"] == 0.0
    @test dfe["[SS]"] == 600.0  # n * N
    @test dfe["[SI]"] == 0.0
end

@testset "Unsupported threshold combinations throw" begin
    net = HeterogeneousNetwork([0.0, 0.5, 0.5])
    @test_throws ArgumentError epidemic_threshold(net, BarnardClosure(), 0.1)
    @test_throws ArgumentError epidemic_threshold(net, BarnardClosure(), 0.1; dynamics = :SIR)
end

# ─── Golden: frozen numbers of this area (test/golden/analysis/) ─────────────────
include(joinpath(@__DIR__, "..", "golden", "GoldenIO.jl"))
GoldenIO.check_area("analysis")
