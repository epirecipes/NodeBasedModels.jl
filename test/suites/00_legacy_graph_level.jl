# 00_legacy_graph_level.jl — legacy NodeBasedModels tests:
# individual-based (NIMFA) and pair-based (Kirkwood) models on explicit graphs.
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

# ─── Individual-based Model (order 1) ─────────────────────────────────
@testset "Individual-based Model" begin
    g = random_regular_graph(30, 4; seed=123)
    net = GraphNetwork(g)

    @testset "SIR basic run" begin
        r = generate_individual_based(sir_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1], tspan=(0.0, 80.0), saveat=1.0)
        @test r isa IndividualBasedResult
        @test r.N == 30
        @test r.K == 2  # S, I tracked; R derived
        @test r.state_names == [:S, :I]
    end

    @testset "Conservation law" begin
        r = generate_individual_based(sir_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1,2], tspan=(0.0, 50.0), saveat=1.0)
        S = aggregate(r, :S)
        I = aggregate(r, :I)
        R = aggregate(r, :R)
        # S+I+R = N at all times
        for t_idx in 1:length(r.sol.t)
            @test isapprox(S[t_idx] + I[t_idx] + R[t_idx], 30.0; atol=1e-6)
        end
    end

    @testset "Initial conditions" begin
        r = generate_individual_based(sir_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[5, 10], tspan=(0.0, 1.0), saveat=0.5)
        @test isapprox(node_state(r, 5, :I, 1), 1.0; atol=1e-10)
        @test isapprox(node_state(r, 1, :S, 1), 1.0; atol=1e-10)
        @test isapprox(aggregate(r, :I)[1], 2.0; atol=1e-10)
    end

    @testset "Random seeding (ε)" begin
        r = generate_individual_based(sir_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            ε=0.05, tspan=(0.0, 1.0), saveat=0.5)
        I0 = aggregate(r, :I)[1]
        @test isapprox(I0, 30 * 0.05; atol=0.01)
    end

    @testset "Random seeding (seed_fraction)" begin
        r = generate_individual_based(sir_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            seed_fraction=0.05, tspan=(0.0, 1.0), saveat=0.5)
        I0 = aggregate(r, :I)[1]
        @test isapprox(I0, 30 * 0.05; atol=0.01)
    end

    @testset "SIS model" begin
        r = generate_individual_based(sis_model(), net;
            infection_rate=0.3, recovery_rate=0.1,
            initial_infected=[1], tspan=(0.0, 50.0), saveat=1.0)
        @test r.K == 1  # Only S tracked; I derived
        S = aggregate(r, :S)
        I = aggregate(r, :I)
        for t_idx in 1:length(r.sol.t)
            @test isapprox(S[t_idx] + I[t_idx], 30.0; atol=1e-6)
        end
    end

    @testset "Directed graph respects infection direction" begin
        g_dir = SimpleDiGraph(2)
        add_edge!(g_dir, 1, 2)
        net_dir = GraphNetwork(g_dir)
        r = generate_individual_based(sir_model(), net_dir;
            infection_rate=1.0, recovery_rate=0.0,
            initial_infected=[2], tspan=(0.0, 2.0), saveat=1.0)
        @test isapprox(node_state(r, 1, :S, length(r.sol.t)), 1.0; atol=1e-8)
    end

    @testset "Complete graph upper bound" begin
        # On complete graph, individual-based should overestimate infection
        g_full = complete_graph(20)
        net_full = GraphNetwork(g_full)
        r = generate_individual_based(sir_model(), net_full;
            infection_rate=0.1, recovery_rate=0.1,
            initial_infected=[1], tspan=(0.0, 50.0), saveat=1.0)
        R_final = aggregate(r, :R)[end]
        # Epidemic should occur (R₀ = τ(N-1)/γ = 0.1*19/0.1 = 19 >> 1)
        @test R_final > 10.0
    end

    @testset "Convenience wrappers" begin
        r = generate_individual_based(sir_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1,2], tspan=(0.0, 10.0), saveat=1.0)
        S = compartment(r, :S)
        bundle = compartments(r, [:S, :I, :R])
        @test S == aggregate(r, :S)
        @test population_fraction(r, :S) ≈ aggregate(r, :S) ./ r.N
        @test haskey(bundle, :S)
        @test haskey(bundle, :I)
        @test haskey(bundle, :R)
    end
end

# ─── Pair-based Model (order 2) ───────────────────────────────────────
@testset "Pair-based Model" begin
    g = random_regular_graph(30, 4; seed=123)
    net = GraphNetwork(g)

    @testset "SIR basic run" begin
        r = generate_pair_based(sir_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1], tspan=(0.0, 80.0), saveat=1.0)
        @test r isa PairBasedResult
        @test r.N == 30
        @test r.n_directed_edges == 2 * ne(g)
    end

    @testset "Conservation law" begin
        r = generate_pair_based(sir_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1,2], tspan=(0.0, 50.0), saveat=1.0)
        S = aggregate(r, :S)
        I = aggregate(r, :I)
        R = aggregate(r, :R)
        for t_idx in 1:length(r.sol.t)
            @test isapprox(S[t_idx] + I[t_idx] + R[t_idx], 30.0; atol=1e-4)
        end
    end

    @testset "Initial conditions" begin
        r = generate_pair_based(sir_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[5], tspan=(0.0, 1.0), saveat=0.5)
        @test isapprox(node_state(r, 5, :I, 1), 1.0; atol=1e-10)
        @test isapprox(node_state(r, 1, :S, 1), 1.0; atol=1e-10)
    end

    @testset "Random seeding (seed_fraction)" begin
        r = generate_pair_based(sir_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            seed_fraction=0.05, tspan=(0.0, 1.0), saveat=0.5)
        I0 = aggregate(r, :I)[1]
        @test isapprox(I0, 30 * 0.05; atol=0.01)
    end

    @testset "Unsupported models and closures" begin
        @test_throws ArgumentError generate_pair_based(sis_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1], tspan=(0.0, 5.0), saveat=1.0)
        @test_throws ArgumentError generate_pair_based(sir_model(), net;
            closure=BernoulliClosure(),
            infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1], tspan=(0.0, 5.0), saveat=1.0)

        g_dir = SimpleDiGraph(2)
        add_edge!(g_dir, 1, 2)
        net_dir = GraphNetwork(g_dir)
        @test_throws ArgumentError generate_pair_based(sir_model(), net_dir;
            infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1], tspan=(0.0, 5.0), saveat=1.0)
    end

    @testset "Pair-based ≤ Individual-based final size" begin
        # Pair-based should generally predict less infection than individual-based
        # (it accounts for correlations that individual-based ignores)
        r_ib = generate_individual_based(sir_model(), net;
            infection_rate=0.15, recovery_rate=0.1,
            initial_infected=[1], tspan=(0.0, 80.0), saveat=1.0)
        r_pb = generate_pair_based(sir_model(), net;
            infection_rate=0.15, recovery_rate=0.1,
            initial_infected=[1], tspan=(0.0, 80.0), saveat=1.0)
        R_ib = aggregate(r_ib, :R)[end]
        R_pb = aggregate(r_pb, :R)[end]
        @test R_pb ≤ R_ib + 1.0  # pair-based should be less (allow small tolerance)
    end

    @testset "Tree graph exactness" begin
        # On a tree (no cycles), pair-based should be very close to Gillespie mean
        tree = prufer_decode(rand(MersenneTwister(42), 1:20, 18))  # random tree on 20 nodes
        net_tree = GraphNetwork(tree)
        r = generate_pair_based(sir_model(), net_tree;
            infection_rate=0.3, recovery_rate=0.1,
            initial_infected=[1], tspan=(0.0, 60.0), saveat=1.0)
        S = aggregate(r, :S)
        I = aggregate(r, :I)
        R = aggregate(r, :R)
        # Just check it runs and conserves
        @test isapprox(S[1] + I[1] + R[1], 20.0; atol=1e-6)
        @test isapprox(S[end] + I[end] + R[end], 20.0; atol=1e-4)
    end

    @testset "Convenience wrappers" begin
        r = generate_pair_based(sir_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1,2], tspan=(0.0, 10.0), saveat=1.0)
        I = compartment(r, :I)
        bundle = compartments(r, [:S, :I, :R])
        @test I == aggregate(r, :I)
        @test population_fraction(r, :I) ≈ aggregate(r, :I) ./ r.N
        @test haskey(bundle, :S)
        @test haskey(bundle, :I)
        @test haskey(bundle, :R)
    end

    @testset "RS pair probability" begin
        r = generate_pair_based(sir_model(), net;
            infection_rate=0.2, recovery_rate=0.1,
            initial_infected=[1], tspan=(0.0, 5.0), saveat=1.0)
        i, j = r.directed_edges[1]
        @test pair_prob(r, i, j, :R, :S, 1) ≥ -1e-10
    end
end

# ─── Cross-level Hierarchy Validation ─────────────────────────────────
@testset "Hierarchy Ordering" begin
    g = random_regular_graph(40, 4; seed=77)
    net = GraphNetwork(g)
    τ, γ = 0.2, 0.1

    r_ib = generate_individual_based(sir_model(), net;
        infection_rate=τ, recovery_rate=γ,
        initial_infected=[1,2,3], tspan=(0.0, 60.0), saveat=2.0)
    r_pb = generate_pair_based(sir_model(), net;
        infection_rate=τ, recovery_rate=γ,
        initial_infected=[1,2,3], tspan=(0.0, 60.0), saveat=2.0)

    R_ib_final = aggregate(r_ib, :R)[end]
    R_pb_final = aggregate(r_pb, :R)[end]

    # Individual-based overestimates epidemic size vs pair-based
    @test R_ib_final >= R_pb_final - 1.0

    # Both should show an epidemic (R₀ >> 1 for these params)
    @test R_ib_final > 20.0
    @test R_pb_final > 10.0
end

@testset "Graph transmission matrix honored" begin
    g = path_graph(3)
    net_zero = GraphNetwork(g; transmission_rate=0.0)

    r_ib = generate_individual_based(sir_model(), net_zero;
        infection_rate=1.0, recovery_rate=0.1,
        initial_infected=[1], tspan=(0.0, 10.0), saveat=1.0)
    @test all(x -> isapprox(x, 2.0; atol=1e-8), aggregate(r_ib, :S))

    r_pb = generate_pair_based(sir_model(), net_zero;
        infection_rate=1.0, recovery_rate=0.1,
        initial_infected=[1], tspan=(0.0, 10.0), saveat=1.0)
    @test all(x -> isapprox(x, 2.0; atol=1e-6), aggregate(r_pb, :S))

    r_ssa = gillespie_sir(net_zero;
        infection_rate=1.0, recovery_rate=0.1,
        initial_infected=[1], tmax=10.0, seed=42)
    ts, S_counts = aggregate(r_ssa, :S; saveat=1.0)
    @test length(ts) == length(S_counts)
    @test all(S_counts .== 2)
end

# ─── Golden: frozen numbers of this area (test/golden/graph_level/) ─────────────────
include(joinpath(@__DIR__, "..", "golden", "GoldenIO.jl"))
GoldenIO.check_area("graph_level")
