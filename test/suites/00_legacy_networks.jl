# 00_legacy_networks.jl — legacy NodeBasedModels tests:
# HomogeneousNetwork, HeterogeneousNetwork and GraphNetwork structures.
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

# ─── Network Structures ───────────────────────────────────────────────
@testset "Network Structures" begin
    @testset "Homogeneous network" begin
        net = regular_network(6)
        @test mean_degree(net) == 6.0
        @test excess_degree(net) == 5.0
        @test clustering(net) == 0.0
    end

    @testset "Clustered network" begin
        net = regular_network(6; ϕ=0.3)
        @test clustering(net) == 0.3
    end

    @testset "Erdos-Renyi network" begin
        net = erdos_renyi_network(5.0)
        @test isapprox(mean_degree(net), 5.0; atol=0.1)
        @test net.max_degree > 5
    end

    @testset "Custom degree distribution" begin
        probs = zeros(8)
        probs[4] = 0.5  # k=3
        probs[8] = 0.5  # k=7
        net = degree_distribution_network(probs)
        @test isapprox(mean_degree(net), 5.0)
        @test net.max_degree == 7
        @test net.second_moment == 0.5 * 9 + 0.5 * 49  # 29.0
    end

    @testset "GraphNetwork" begin
        g = random_regular_graph(20, 4; seed=42)
        net = GraphNetwork(g)
        @test mean_degree(net) == 4.0
        @test net.graph === g
        @test isnothing(net.transmission_matrix)
    end

    @testset "GraphNetwork custom transmission rate" begin
        g = complete_graph(5)
        net = GraphNetwork(g; transmission_rate=2.0)
        @test !isnothing(net.transmission_matrix)
        @test net.transmission_matrix[1,2] == 2.0
        @test net.transmission_matrix[1,1] == 0.0  # no self-loops
    end

    @testset "Directed GraphNetwork preserves direction" begin
        g = SimpleDiGraph(2)
        add_edge!(g, 1, 2)
        net = GraphNetwork(g; transmission_rate=3.0)
        @test mean_degree(net) == 0.5
        @test net.transmission_matrix[2,1] == 3.0
        @test net.transmission_matrix[1,2] == 0.0
    end
end
