# 00_legacy_networkoutbreaks.jl — legacy NodeBasedModels tests:
# integration with NetworkOutbreaks.jl (OutbreakModel from a CompartmentalModel).
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

if Base.find_package("NetworkOutbreaks") === nothing
    @info "Skipping NetworkOutbreaks integration tests; NetworkOutbreaks is not available"
else
    @testset "NetworkOutbreaks integration" begin
        using NetworkOutbreaks
        using Graphs
        using StableRNGs
        using Statistics: mean

        nbm = sir_model()
        model = OutbreakModel(nbm, Dict(:τ => 1.5, :γ => 1.0))
        @test model.compartments == [:S, :I, :R]
        @test model.infectious == [false, true, false]

        g = random_regular_graph(400, 6; rng = StableRNG(11))
        spec = OutbreakSpec(model = model, network = g,
                            initial = SeedFraction(:I => 0.05),
                            tspan = (0.0, 60.0))
        ens = simulate_ensemble(spec; nsims = 8, seed = 321)
        fs = mean(NetworkOutbreaks.final_size(t; recovered = :R) for t in ens.trajectories)
        @test 0.10 < fs <= 1.0

        # Cross-validate gillespie_sis (legacy) against NetworkOutbreaks SSA
        # for SIS dynamics. Both engines should produce comparable mean
        # prevalence trajectories on the same network/parameters.
        @testset "gillespie_sis vs NetworkOutbreaks (SIS)" begin
            N      = 500
            β, γ   = 0.6, 0.4   # supercritical for ⟨k⟩=6
            tspan  = (0.0, 30.0)
            t_meas = 25.0
            g = random_regular_graph(N, 6; rng = StableRNG(99))
            net = GraphNetwork(g)

            # Legacy engine: aggregate I(t) at t_meas across nsims runs.
            nsims = 12
            I_legacy = Float64[]
            for k in 1:nsims
                res = gillespie_sis(net;
                    infection_rate = β, recovery_rate = γ,
                    initial_infected = collect(1:25),
                    tmax = tspan[2], seed = 1000 + k)
                push!(I_legacy, count(res(t_meas)))
            end

            # NetworkOutbreaks engine: same setup, ensemble interpolated.
            sis = sis_model()
            om  = OutbreakModel(sis, Dict(:τ => β, :γ => γ))
            spec = OutbreakSpec(model = om, network = g,
                initial = SeedNodes(:I => collect(1:25)), tspan = tspan)
            ens = simulate_ensemble(spec; nsims = nsims, seed = 7)
            I_no = Float64[]
            for tr in ens.trajectories
                # Interpolate: piecewise-constant I count at t_meas.
                Is = compartment_series(tr, :I)
                k  = searchsortedlast(tr.times, t_meas)
                push!(I_no, k == 0 ? 0.0 : Float64(Is[k]))
            end

            μ_legacy = mean(I_legacy) / N
            μ_no     = mean(I_no)     / N
            # Both should produce non-trivial outbreaks (R₀ ≈ 9 here),
            # and their prevalences should agree to within Monte-Carlo
            # noise on a small ensemble.
            @test μ_legacy > 0.05
            @test μ_no     > 0.05
            @test isapprox(μ_legacy, μ_no; atol = 0.10)
        end
    end
end
