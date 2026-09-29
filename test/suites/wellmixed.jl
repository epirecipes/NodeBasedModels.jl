# wellmixed.jl — MeanFieldClosure on WellMixed(κ): mass action (WP23; DESIGN_NetworkEpiCore.md
# §A.4, §B.6, §C.2, §C.3, §D.5 M1/M14, §E.3 :sir_wm5).
#
# - `default_closure(WellMixed(κ))` is `MeanFieldClosure()`, and `node_based` on a well-mixed
#   population returns a `MeanFieldSystem`;
# - its vector field is NetworkEpiCore's `mass_action(cm; κ)` (symbolically, for SIR, SEIR, SIS,
#   SIRS, SEAIR, two strains and vaccination, and for frequency-dependent rates at every κ);
# - trajectories equal an independent mass-action ODE, EdgeBasedModels' lift on `WellMixed(κ)`
#   (the unit law M1, when EdgeBasedModels can be loaded) and, stochastically, NetworkOutbreaks'
#   `MassActionSSA` ensemble of `:sir_wm5` (N = 10⁴, 200 runs, conditioned on major outbreaks).

using NodeBasedModels
using NetworkEpiCore
using Test
using ModelingToolkit
using OrdinaryDiffEqDefault
import NetworkOutbreaks

const HAVE_EBM = try
    @eval import EdgeBasedModels
    true
catch err
    err isa ArgumentError || rethrow()          # "Package EdgeBasedModels not found in current path"
    false
end
HAVE_EBM || @info "wellmixed: EdgeBasedModels is not loadable here; its cross-checks are skipped"

const TOL = (reltol = 1e-11, abstol = 1e-13)

maxdiff(a, b) = maximum(abs.(a .- b))

initial_for(cm) = nameof(cm) === :twostrain ? SeedFraction(:I1 => 0.005, :I2 => 0.005) :
                  default_seed(cm, 0.01)

@testset "MeanFieldClosure: the default on WellMixed; the field is mass action" begin
    @test default_closure(WellMixed(5)) isa MeanFieldClosure
    sc = scenario(:sir_wm5)
    sys = node_based(sc)
    @test sys isa MeanFieldSystem && sys.closure isa MeanFieldClosure
    @test sys.network === sc.network && sys.metadata[:κ] == 5.0
    @test occursin("κ = 5.0", sprint(show, sys))
    for κ in (5.0, 2.5), cm in (sir_model(), seir_model(), sis_model(), sirs_model(), seair_model(),
                                twostrain_model(), sirv_model())
        mf = node_based(cm, WellMixed(κ); initial = initial_for(cm))
        @test mf isa MeanFieldSystem
        @test Set(state_names(symbolic_ode(mf))) == Set(species_names(cm))
        @test vector_fields_equal(symbolic_ode(mf), mass_action(cm; κ))
    end
    @test !vector_fields_equal(symbolic_ode(node_based(sir_model(), WellMixed(5))),
                               mass_action(sir_model(); κ = 4))
    # frequency-dependent rates (β): τ = β/κ, so the field is MA(β) for every κ (§B.6)
    sirF = ContactModel(:sirF; contacts = [Contact(:S, :I, :I, :β)],
                        transitions = [NodeTransition(:I, :R, :γ)],
                        convention = FrequencyDependent())
    for κ in (2.0, 5.0, 12.0)
        @test vector_fields_equal(symbolic_ode(node_based(sirF, WellMixed(κ))),
                                  mass_action(sirF; κ = 1))
    end
end

@testset ":sir_wm5: the trajectories are those of mass action MA(1/2, 1/4)" begin
    sc = scenario(:sir_wm5)
    sys = node_based(sc)
    sol = solve_epidemic(sys, sc; TOL...)
    mc = model_curves(sys, sol; t = sc.tgrid)
    @test mc.representation === :mean_field && mc.label == "mean field (κ = 5.0)"
    @test Set(sc.observables) ⊆ Set(keys(mc.values))
    β, γ = 5 * sc.params[:τ], sc.params[:γ]
    @test β ≈ 0.5
    function ma!(du, u, _, t)
        S, I = u[1], u[2]
        du[1] = -β * S * I
        du[2] = β * S * I - γ * I
        du[3] = γ * I
        du[4] = β * S * I
    end
    ref = OrdinaryDiffEqDefault.solve(ODEProblem(ma!, [0.99, 0.01, 0.0, 0.01], (0.0, 60.0));
                                      saveat = sc.tgrid, TOL...)
    U = reduce(hcat, ref.u)
    for (row, X) in enumerate((:S, :I, :R, :cumulative))
        @test maxdiff(mc[X], U[row, :]) < 1e-8
    end
    @test mc[:infectious] == mc[:I]
    # the peak (DESIGN §E.3: 0.158 at t ≈ 17.5) and the final size (NetworkEpiCore's, t → ∞)
    @test abs(maximum(mc[:I]) - 0.158) < 1e-3
    @test abs(sc.tgrid[argmax(mc[:I])] - 17.5) <= 0.25
    long = solve_epidemic(sys; tspan = (0.0, 400.0), TOL...)
    @test abs(compartment(sys, long, :cumulative)[end] - sc.expected[:final_size]) < 1e-6
    # M1: the edge-based lift on WellMixed(κ) is the same model (with an exit: ξ)
    if HAVE_EBM
        for (cm, p) in ((sc.model, sc.params),
                        (seir_model(), Dict(:τ => 0.1, :σ => 0.2, :γ => 0.25)),
                        (sirv_model(), Dict(:τ => 0.1, :γ => 0.25, :ν => 0.02)))
            eb = EdgeBasedModels.edge_based(cm, WellMixed(5))
            mf = node_based(cm, WellMixed(5); p)
            kw = (; p, initial = default_seed(cm, 0.01), tspan = (0.0, 60.0), saveat = sc.tgrid)
            a = model_curves(eb, solve_epidemic(eb; kw..., TOL...); t = sc.tgrid)
            b = model_curves(mf, solve_epidemic(mf; kw..., TOL...); t = sc.tgrid)
            for X in vcat(species_names(cm), [:infectious, :cumulative])
                @test maxdiff(a[X], b[X]) < 1e-8
            end
        end
    else
        @test_skip "EdgeBasedModels is not loadable: the unit law M1 is not cross-checked"
    end
end

@testset "Mean field beyond T_EB, removals and the population scale" begin
    # SIS: the mass-action endemic state I* = 1 − γ/(κτ)
    sis = node_based(sis_model(), WellMixed(5); p = Dict(:τ => 0.1, :γ => 0.25),
                     seed_fraction = 0.01)
    sol = solve_epidemic(sis; tspan = (0.0, 400.0), TOL...)
    @test abs(compartment(sis, sol, :I)[end] - 0.5) < 1e-8
    # a removal I → ∅ goes to the absorbing compartment `removed`; counts on the scale N
    rem = ContactModel(:sir_removal; contacts = [Contact(:S, :I, :I, :τ)],
                       transitions = [NodeTransition(:I, nothing, :γ)])
    p = Dict(:τ => 0.1, :γ => 0.25)
    unit = node_based(rem, WellMixed(4); p, seed_fraction = 0.01)
    counts = node_based(rem, WellMixed(4); p, seed_fraction = 0.01, N = 1000)
    @test haskey(unit.singles, :removed)
    tt = 0.0:1.0:60.0
    a = solve_epidemic(unit; saveat = tt, TOL...)
    b = solve_epidemic(counts; saveat = tt, TOL...)
    total = sum(compartment(counts, b, X) for X in (:S, :I, :removed))
    @test maxdiff(total, fill(1000.0, length(tt))) < 1e-8
    ca, cb = model_curves(unit, a), model_curves(counts, b)
    for X in (:S, :I, :removed, :cumulative)
        @test maxdiff(ca[X], cb[X]) < 1e-10
    end
    @test maxdiff(population_fraction(counts, b, :I), compartment(unit, a, :I)) < 1e-10
    # seeding: an explicit SeedFraction; seeded R is not an infection (DESIGN §J.8)
    u = default_initial_conditions(node_based(sir_model(), WellMixed(5));
                                   initial = SeedFraction(:I => 0.02, :R => 0.1))
    mf = node_based(sir_model(), WellMixed(5))
    @test u[mf.singles[:S]] ≈ 0.88 && u[mf.metadata[:cumulative]] ≈ 0.02
    @test default_initial_conditions(mf) === mf.u0
    @test node_variables(mf) === mf.singles
end

@testset "MeanFieldClosure: errors" begin
    pois = ConfigurationNetwork(PoissonDegree(5))
    @test_throws ArgumentError node_based(sir_model(), pois; closure = MeanFieldClosure())
    @test_throws ArgumentError node_based(sir_model(), ClusteredNetwork(RegularDegree(2),
                                                                        RegularDegree(2));
                                          closure = MeanFieldClosure())
    @test_throws ArgumentError node_based(sir_model(), regular_network(4);
                                          closure = MeanFieldClosure())
    @test_throws ArgumentError generate_pairwise(CompartmentalModel(sir_model()),
                                                 regular_network(4), MeanFieldClosure())
    @test_throws ArgumentError node_based(sir_model(), WellMixed(5); closure = BernoulliClosure())
    @test_throws ArgumentError node_based(sir_model(), WellMixed(5); level = :s_anchored)
    @test_throws ArgumentError node_based(sir_model(), WellMixed(5); level = :individual)
    @test_throws MethodError node_based(sir_model(), WellMixed(5); seedfraction = 0.1)
    @test_throws ArgumentError node_based(twostrain_model(), WellMixed(5))   # two entry states
    @test_throws ArgumentError solve_epidemic(node_based(sir_model(), WellMixed(5));
                                              p = Dict(:β => 0.3))
end

@testset "Against NetworkOutbreaks: MassActionSSA on :sir_wm5" begin
    sc = scenario(:sir_wm5)
    @test sc.sim.algorithm === :mass_action
    ref = NetworkOutbreaks.summarise(NetworkOutbreaks.scenario_ensemble(sc))
    @test ref.N == 10_000 && ref.nsims == 200
    sys = node_based(sc)
    mc = model_curves(sys, solve_epidemic(sys, sc; TOL...); t = sc.tgrid, label = "mean field")
    tab = compare(ref, mc)
    @info "wellmixed: :sir_wm5 against MassActionSSA (N = $(ref.N), $(ref.n_major)/$(ref.nsims) major)" tab
    @test tab["mean field", :I].D∞ < 0.005
    @test abs(tab["mean field", :I].ΔR∞) < 0.005
end
