# 00_legacy_reinfection.jl — legacy NodeBasedModels tests:
# reinfection counting lifts (Keeling et al. 2016, Approximation 1).
#
# Moved verbatim (dedented one level) from the pre-0.2 test/runtests.jl by WP2
# (DESIGN_NetworkEpiCore.md §G.2). Owned by the work package that replaces this
# area; see §G.1. The imports are the original file's, so name resolution is unchanged.
#
# WP15 (NodeBasedModels 0.2): `sis_model()` is a NetworkEpiCore ContactModel, so
# `with_reinfection_counting(sis_model(), L)` is NetworkEpiCore's lift (a ContactModel). The
# structural assertions of 0.1 are applied, unchanged, to `CompartmentalModel(lift)`, the model
# the pairwise builder uses; the dynamics tests are unchanged.

using NodeBasedModels
using Test
using OrdinaryDiffEqDefault
using ModelingToolkit
using Graphs
using Random
using Catalyst
using Symbolics

@testset "Reinfection counting (Keeling et al. 2016, Approx. 1)" begin
    @testset "Lifting structure (SIS)" begin
        base = sis_model()

        # L=0 collapses everything to p=0 (no I_0 pruning at L=0)
        m0 = CompartmentalModel(with_reinfection_counting(base, 0))
        @test sort(m0.compartment_names) == [:I_0, :S_0]
        @test m0.infectious_compartments == [:I_0]
        @test length(m0.transitions) == 2
        @test any(t -> t.from == :S_0 && t.to == :I_0 && t.type == :infection,
                  m0.transitions)
        @test any(t -> t.from == :I_0 && t.to == :S_0 && t.type == :spontaneous,
                  m0.transitions)

        # L=1: I_0 is pruned, infection caps at p=1
        m1 = CompartmentalModel(with_reinfection_counting(base, 1))
        @test sort(m1.compartment_names) == [:I_1, :S_0, :S_1]
        @test m1.infectious_compartments == [:I_1]
        @test !(:I_0 in m1.compartment_names)        # no unphysical I_0
        @test any(t -> t.from == :S_0 && t.to == :I_1, m1.transitions)
        @test any(t -> t.from == :S_1 && t.to == :I_1, m1.transitions)
        @test any(t -> t.from == :I_1 && t.to == :S_1, m1.transitions)

        # L=3: caps and increments work; only the highest p target is repeated
        m3 = CompartmentalModel(with_reinfection_counting(base, 3))
        @test length([c for c in m3.compartment_names if startswith(string(c), "S_")]) == 4
        @test length([c for c in m3.compartment_names if startswith(string(c), "I_")]) == 3
        inf_targets = sort([t.to for t in m3.transitions if t.type == :infection])
        @test inf_targets == [:I_1, :I_2, :I_3, :I_3]   # S_2→I_3 and S_3→I_3 both cap at I_3

        # Parameter symbols are preserved
        @test all(t -> t.rate in (:τ, :γ), m3.transitions)
    end

    @testset "Lifting (SIRS) preserves p across recovery and waning" begin
        base = sirs_model()
        m = CompartmentalModel(with_reinfection_counting(base, 2))
        # S, I, R compartments at appropriate p_min
        @test :S_0 in m.compartment_names
        @test :S_1 in m.compartment_names
        @test :S_2 in m.compartment_names
        @test :I_1 in m.compartment_names
        @test :I_2 in m.compartment_names
        @test :R_1 in m.compartment_names
        @test :R_2 in m.compartment_names
        @test !(:I_0 in m.compartment_names)
        @test !(:R_0 in m.compartment_names)
        # Recovery preserves p, infection increments p
        @test any(t -> t.from == :I_2 && t.to == :R_2 && t.type == :spontaneous, m.transitions)
        @test any(t -> t.from == :R_2 && t.to == :S_2 && t.type == :spontaneous, m.transitions)
        @test any(t -> t.from == :S_1 && t.to == :I_2 && t.type == :infection, m.transitions)
    end

    @testset "L=0 reproduces base SIS dynamics" begin
        base = sis_model()
        net  = regular_network(3)
        tspan = (0.0, 30.0)
        params = Dict(:τ => 0.6, :γ => 1.0)

        psys_base = generate_pairwise(base, net, BernoulliClosure();
                                       tspan = tspan)
        sol_base  = solve_pairwise(psys_base, params)

        psys_lift = generate_pairwise(with_reinfection_counting(base, 0),
                                       net, BernoulliClosure();
                                       tspan = tspan)
        sol_lift  = solve_pairwise(psys_lift, params)

        ts = range(tspan[1], tspan[2]; length = 11)
        for t in ts
            S_base = sol_base(t; idxs = psys_base.singles[:S])
            I_base = sol_base(t; idxs = psys_base.singles[:I])
            S_lift = sol_lift(t; idxs = psys_lift.singles[:S_0])
            I_lift = sol_lift(t; idxs = psys_lift.singles[:I_0])
            @test isapprox(S_lift, S_base; atol = 1e-7, rtol = 1e-6)
            @test isapprox(I_lift, I_base; atol = 1e-7, rtol = 1e-6)
        end
    end

    @testset "L=4 changes transient aggregate before saturation" begin
        base = sis_model(τ = :β)
        net  = regular_network(3)
        params = Dict(:β => 0.6, :γ => 0.4)
        tspan = (0.0, 20.0)

        psys_base = generate_pairwise(base, net, KeelingClosure();
                                      tspan = tspan,
                                      seed_fraction = 0.05)
        sol_base  = solve_pairwise(psys_base, params; saveat = 1.0)
        I_base    = sol_base[psys_base.singles[:I]]

        psys_lift = generate_pairwise(with_reinfection_counting(base, 4),
                                       net, KeelingClosure();
                                       tspan = tspan,
                                       seed_fraction = 0.05)
        sol_lift  = solve_pairwise(psys_lift, params; saveat = 1.0)
        I_lift    = reinfection_totals(psys_lift, sol_lift)[:I]

        @test maximum(abs.(I_base .- I_lift)) > 0.05
        @test I_lift[findfirst(==(6.0), sol_lift.t)] <
              I_base[findfirst(==(6.0), sol_base.t)]
        @test abs(I_base[end] - I_lift[end]) < 1e-3
    end

    @testset "Conservation invariants (SIS, L=4)" begin
        base = sis_model()
        net  = regular_network(3)
        psys = generate_pairwise(with_reinfection_counting(base, 4),
                                  net, BernoulliClosure();
                                  tspan = (0.0, 40.0))
        sol  = solve_pairwise(psys, Dict(:τ => 0.6, :γ => 1.0))
        totals = reinfection_totals(psys, sol)
        for ti in eachindex(sol.t)
            tot = totals[:S][ti] + totals[:I][ti]
            @test isapprox(tot, 1.0; atol = 1e-6)
        end
        # No occupancy of unphysical I_0 (it should not exist in the system)
        @test !haskey(psys.singles, :I_0)

        # Mixed-convention pair conservation: 2·cross + self ≈ k·N
        k = mean_degree(net)
        N = 1.0   # default population fraction
        for ti in eachindex(sol.t)
            cross = 0.0
            self  = 0.0
            for ((a, b), v) in psys.pairs
                val = sol[v][ti]
                if a == b
                    self += val
                else
                    cross += val
                end
            end
            @test isapprox(2 * cross + self, k * N; atol = 1e-4, rtol = 1e-5)
        end
    end

    @testset "Helpers: base_compartment_of / infection_count_of" begin
        @test base_compartment_of(:S_3) == :S
        @test base_compartment_of(:I_0) == :I
        @test base_compartment_of(:plain) == :plain
        @test infection_count_of(:S_3) == 3
        @test infection_count_of(:I_0) == 0
        @test infection_count_of(:plain) === nothing
    end

    @testset "Gillespie SIS basic correctness" begin
        g = random_regular_graph(200, 3, rng = MersenneTwister(7))
        net = GraphNetwork(g)
        res = gillespie_sis(net; infection_rate = 1.5, recovery_rate = 1.0,
                             initial_infected = collect(1:20),
                             tmax = 20.0, seed = 11)
        # Conservation: state vector always sums to N
        @test all(count(s) + count(.!(s)) == 200 for s in res.states)
        # Initial infections at t=0 are recorded as p=1
        for i in 1:20
            @test 1 in res.infection_times[i] .|> (x -> x == 0.0)
        end
        # Infection counts are monotonically non-decreasing in time
        for i in 1:200
            ts = sort(res.infection_times[i])
            @test issorted(ts)
        end
        # Histogram totals match population at every recorded time grid pt
        for t in (0.0, 5.0, 10.0, 20.0)
            h = reinfection_histogram(res, t, 4)
            @test sum(h.S) + sum(h.I) == 200
        end
    end

    @testset "Gillespie SIS reproduces lifted pairwise prediction (mean over runs)" begin
        n = 1000
        g = random_regular_graph(n, 3, rng = MersenneTwister(13))
        net = GraphNetwork(g)
        τ, γ = 0.6, 1.0
        avg = gillespie_sis_average(net;
                                      nruns = 100,
                                      dt = 1.0,
                                      tmax_grid = 30.0,
                                      infection_rate = τ,
                                      recovery_rate = γ,
                                      initial_infected = collect(1:50),
                                      seed = 1)
        base = sis_model()
        # Compare total prevalence (sum over p) with the lifted pairwise model
        psys = generate_pairwise(with_reinfection_counting(base, 4),
                                  regular_network(3),
                                  BernoulliClosure(); tspan = (0.0, 30.0),
                                  seed_fraction = 50/n)
        sol  = solve_pairwise(psys, Dict(:τ => τ, :γ => γ))
        totals = reinfection_totals(psys, sol)
        # Endpoint prevalence: stochastic vs ODE — the L=4 reinfection
        # lift still has a residual closure gap on this k=3 endemic
        # benchmark (~7%), consistent with Keeling et al. (2016) Fig 4.
        stoch_I = avg.I_mean[end] / n
        ode_I   = totals[:I][end]
        @test abs(stoch_I - ode_I) < 0.10
    end
end

# ─── Golden: frozen numbers of this area (test/golden/reinfection/) ─────────────────
include(joinpath(@__DIR__, "..", "golden", "GoldenIO.jl"))
GoldenIO.check_area("reinfection")
