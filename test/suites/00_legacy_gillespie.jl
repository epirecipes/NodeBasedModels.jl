# 00_legacy_gillespie.jl — legacy NodeBasedModels tests:
# legacy gillespie_sir / gillespie_sis stochastic simulation.
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
using JumpProcesses   # gillespie_sir lives in the JumpProcesses extension (WP24)

# ─── Gillespie Stochastic Simulation ──────────────────────────────────
@testset "Gillespie Simulation" begin
    g = random_regular_graph(50, 4; seed=99)
    net = GraphNetwork(g)

    @testset "Single run" begin
        r = gillespie_sir(net; infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1], tmax=100.0, seed=42)
        @test r isa GillespieResult
        @test r.N == 50
    end

    @testset "Conservation" begin
        r = gillespie_sir(net; infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1,2], tmax=100.0, seed=42)
        ts, S = aggregate(r, :S; saveat=5.0)
        _, I = aggregate(r, :I; saveat=5.0)
        _, R = aggregate(r, :R; saveat=5.0)
        for i in eachindex(ts)
            @test S[i] + I[i] + R[i] == 50
        end
    end

    @testset "Initial conditions" begin
        r = gillespie_sir(net; infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[3, 7], tmax=0.001, seed=42)
        ts, S = aggregate(r, :S; saveat=0.001)
        _, I = aggregate(r, :I; saveat=0.001)
        @test S[1] == 48
        @test I[1] == 2
    end

    @testset "Below threshold: no epidemic" begin
        # τ/γ * (k-1) < 1 → subcritical
        r = gillespie_sir(net; infection_rate=0.01, recovery_rate=0.5,
            initial_infected=[1], tmax=200.0, seed=42)
        ts, R = aggregate(r, :R; saveat=200.0)
        # Should not infect most of the population
        @test R[end] < 20
    end

    @testset "Convenience wrappers" begin
        r = gillespie_sir(net; infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1,2], tmax=5.0, seed=42)
        S1 = aggregate(r, :S; saveat=1.0)
        S2 = compartment(r, :S; saveat=1.0)
        fractions = population_fraction(r, :S; saveat=1.0)
        bundle = compartments(r, [:S, :I]; saveat=1.0)
        @test S1 == S2
        @test fractions.times == S1.times
        @test fractions.counts ≈ S1.counts ./ r.N
        @test haskey(bundle, :S)
        @test haskey(bundle, :I)
    end

    @testset "Averaged runs" begin
        small_g = random_regular_graph(20, 4; seed=11)
        small_net = GraphNetwork(small_g)
        avg = gillespie_sir_average(small_net; nruns=10,
            infection_rate=0.3, recovery_rate=0.1,
            initial_infected=[1], tmax_grid=50.0, dt=5.0)
        @test length(avg.t_grid) == 11  # 0, 5, 10, ..., 50
        @test length(avg.S_mean) == 11
        @test avg.S_mean[1] ≈ 19.0  # 20 - 1 initial infected
        @test all(avg.I_mean .>= 0)
    end
end
