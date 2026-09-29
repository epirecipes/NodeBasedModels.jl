# pgf_closure.jl — PGFClosure, the S-anchored level and the constant closure (WP23;
# DESIGN_NetworkEpiCore.md §A.4, §D.4, §D.5 M6/M7/M8, §E.2, §E.3, §J.1, §J.2, §J.8).
#
# - PGFClosure (K_ψ(θ) = ψψ''/ψ'² with θ) equals the edge-based model to 1e-8 on :sir_bim, :sir_pl,
#   :seair_pois5 and :sir_vax_pois5, at the S-anchored and at the population level (M6);
# - the constant closure K = closure_constant(d) equals it on the Poisson-type :sir_reg6, :sir_pois5
#   and :sir_nb4 and is biased on :sir_bim and :sir_pl (pinned ΔR∞, M8); the S-anchored subsystem
#   of the constant-K model is the restriction of the population model (M7);
# - M6 as a runtime morphism: `verify` of the map π^PW from EdgeBasedModels' field onto
#   `symbolic_ode` of the S-anchored system; `symbolic_ode` equals EdgeBasedModels'
#   `pairwise_image(ebcm).ode` once that exists (WP18);
# - against NetworkOutbreaks ensembles of the scenarios (N = 10⁴, 200 runs, a fresh graph per run,
#   the stable_rng streams of the scenario, conditioned on major outbreaks; DESIGN §E.2, §J.7).
#
# The edge-based reference `eb_reference` below is the per-reaction field of DESIGN §D.4 written
# out numerically, independent of EdgeBasedModels and of the code under test. EdgeBasedModels is a
# test-only dependency: its checks run when it can be loaded.

using NodeBasedModels
using NetworkEpiCore
using Test
using ModelingToolkit
using OrdinaryDiffEqDefault
using Symbolics
import NetworkOutbreaks

const HAVE_EBM = try
    @eval import EdgeBasedModels
    true
catch err
    err isa ArgumentError || rethrow()          # "Package EdgeBasedModels not found in current path"
    false
end
HAVE_EBM || @info "pgf_closure: EdgeBasedModels is not loadable here; its cross-checks are skipped"

const TOL = (reltol = 1e-11, abstol = 1e-13)

# Infected compartments of the scenario models (DESIGN §J.8), for the reference accumulator.
const INFECTED = Dict(:sir => [:I], :seair => [:E, :I, :A], :sirv => [:I], :seir => [:E, :I])

maxdiff(a, b) = maximum(abs.(a .- b))

# Values of the MTK variable v of `sol` on the grid tt.
at(sol, v, tt) = Float64[sol(t; idxs = v) for t in tt]

"""
The edge-based model of DESIGN §D.4 on ConfigurationNetwork(d), solved on `tgrid`: coordinates
θ, ξ, φ_X and pop_X for the non-susceptible species X (and the sink of removals), and the
accumulator. Returns the curves by name (the susceptible species S = qξψ(θ), the species
X = pop_X, :infectious, :cumulative) and the coordinates (:θ, :ξ, (:φ, X)).
"""
function eb_reference(cm::ContactModel, d, p, initial::SeedFraction, tspan, tgrid; infected)
    s = only(susceptible_species(cm))
    nodes = [X for X in species_names(cm) if X !== s]
    any(t -> t.to === nothing, node_transitions(cm)) && push!(nodes, :removed)
    ix = Dict(X => i for (i, X) in enumerate(nodes))
    n = length(nodes)
    ρ = Dict{Symbol,Float64}(X => v for (X, v) in seed_fractions(initial) if X !== s)
    q = 1 - sum(values(ρ); init = 0.0)
    val(r) = Float64(rate_value(r, p))
    cs = [(c.infector, c.product, val(c.rate)) for c in contacts(cm)]
    ts = [(t.from, something(t.to, :removed), val(t.rate)) for t in node_transitions(cm)]
    ψ(x) = pgf(d, x)
    ψ1(x) = pgf_derivative(d, x, 1)
    ψ2(x) = pgf_derivative(d, x, 2)
    k̄ = mean_degree(d)
    iφ(X) = 2 + ix[X]
    ipop(X) = 2 + n + ix[X]
    function f!(du, u, _, t)
        fill!(du, 0.0)
        θ, ξ = u[1], u[2]
        for (J, P, τ) in cs                                   # contact s + J → P + J
            h = τ * u[iφ(J)]
            du[1] -= h
            du[iφ(J)] -= h
            du[iφ(P)] += h * q * ξ * ψ2(θ) / k̄
            du[ipop(P)] += h * q * ξ * ψ1(θ)
            P in infected && (du[end] += h * q * ξ * ψ1(θ))
        end
        for (W, Z, a) in ts
            if W === s                                        # exit s → Z
                du[2] -= a * ξ
                du[iφ(Z)] += a * q * ξ * ψ1(θ) / k̄
                du[ipop(Z)] += a * q * ξ * ψ(θ)
                Z in infected && (du[end] += a * q * ξ * ψ(θ))
            else                                              # progression or removal
                du[iφ(W)] -= a * u[iφ(W)]
                du[iφ(Z)] += a * u[iφ(W)]
                du[ipop(W)] -= a * u[ipop(W)]
                du[ipop(Z)] += a * u[ipop(W)]
                (!(W in infected) && Z in infected) && (du[end] += a * u[ipop(W)])
            end
        end
        return nothing
    end
    u0 = vcat(1.0, 1.0, [get(ρ, X, 0.0) for X in nodes], [get(ρ, X, 0.0) for X in nodes],
              sum(get(ρ, X, 0.0) for X in infected))
    sol = OrdinaryDiffEqDefault.solve(ODEProblem(f!, u0, Float64.(tspan)); saveat = tgrid, TOL...)
    U = reduce(hcat, sol.u)
    out = Dict{Any,Vector{Float64}}()
    out[:θ], out[:ξ] = U[1, :], U[2, :]
    out[s] = q .* out[:ξ] .* ψ.(out[:θ])
    for X in nodes
        out[(:φ, X)] = U[iφ(X), :]
        out[X] = U[ipop(X), :]
    end
    out[:infectious] = reduce(+, (out[J] for J in infectious_species(cm)))
    out[:cumulative] = U[end, :]
    out[:q] = fill(q, length(sol.t))
    return out
end
eb_reference(sc::Scenario) =
    eb_reference(sc.model, sc.network.degrees, sc.params, sc.initial, sc.tspan, sc.tgrid;
                 infected = INFECTED[nameof(sc.model)])

# EdgeBasedModels' curves of a scenario.
function ebm_curves(sc::Scenario)
    sys = EdgeBasedModels.edge_based(sc.model, sc.network)
    return model_curves(sys, solve_epidemic(sys, sc; TOL...); t = sc.tgrid)
end

function solved_curves(sys, sc; label = nothing)
    sol = solve_epidemic(sys, sc; TOL...)
    mc = label === nothing ? model_curves(sys, sol; t = sc.tgrid) :
         model_curves(sys, sol; t = sc.tgrid, label)
    return sol, mc
end

@testset "PGFClosure is the edge-based model for every degree PGF (M6)" begin
    for id in (:sir_bim, :sir_pl, :seair_pois5, :sir_vax_pois5)
        sc = scenario(id)
        d = sc.network.degrees
        ref = eb_reference(sc)
        ebm = HAVE_EBM ? ebm_curves(sc) : nothing
        # the independent reference is EdgeBasedModels' lift
        if ebm !== nothing
            for X in sc.observables
                @test maxdiff(ebm[X], ref[X]) < 1e-8
            end
        end
        for level in (:s_anchored, :population)
            sys = node_based(sc; closure = PGFClosure(), level)
            @test sys isa SAnchoredSystem
            @test sys.level === level
            s = sys.metadata[:susceptible]
            ncomp = length(sys.singles)
            @test length(sys.pairs) == (level === :s_anchored ? ncomp : ncomp * (ncomp + 1) ÷ 2)
            sol, mc = solved_curves(sys, sc)
            @test mc.representation === (level === :s_anchored ? :s_anchored : :pgf_closure)
            @test Set(sc.observables) ⊆ Set(keys(mc.values))
            for X in sc.observables                          # species, :infectious, :cumulative
                @test maxdiff(mc[X], ref[X]) < 1e-8
                ebm === nothing || @test maxdiff(mc[X], ebm[X]) < 1e-8
            end
            # θ is the edge-based θ, and the S-anchored pairs are the π^PW image of the EB state:
            # [sX] = qξψ'(θ)φ_X, [ss] = q²ξ²ψ'(θ)²/ψ'(1)
            θ = at(sol, sys.metadata[:θ], sc.tgrid)
            @test maxdiff(θ, ref[:θ]) < 1e-8
            @test compartment(sys, sol, :θ) == sol[sys.metadata[:θ]]
            qξψ1 = ref[:q] .* ref[:ξ] .* pgf_derivative.(Ref(d), ref[:θ], 1)
            for ((a, b), v) in sys.pairs
                (a === s || b === s) || continue
                X = a === s ? b : a
                image = X === s ? qξψ1 .^ 2 ./ mean_degree(d) : qξψ1 .* ref[(:φ, X)]
                @test maxdiff(at(sol, v, sc.tgrid), image) < 1e-8
            end
        end
    end
    # the final size of the non-PT scenario is the EB one (DESIGN §E.3: EB R∞ 0.4956)
    sc = scenario(:sir_bim)
    _, mc = solved_curves(node_based(sc; closure = PGFClosure()), sc)
    @test abs(mc[:cumulative][end] - 0.4956) < 5e-4
end

@testset "The constant closure is exact iff the degrees are Poisson type (M8, M7)" begin
    for id in (:sir_reg6, :sir_pois5, :sir_nb4)
        sc = scenario(id)
        d = sc.network.degrees
        @test is_poisson_type(d) !== nothing
        # K_ψ(x) is the constant closure_constant(d) on a Poisson-type network
        @test all(x -> isapprox(NodeBasedModels._pgf_closure_factor(d, x), closure_constant(d);
                                rtol = 1e-12), 0.05:0.05:1.0)
        ref = eb_reference(sc)
        pop = node_based(sc)                                  # the default: constant K
        @test pop isa PairwiseSystem && pop.closure isa BernoulliClosure
        san = node_based(sc; level = :s_anchored)             # constant K on PW^S
        @test san isa SAnchoredSystem && san.closure isa BernoulliClosure
        @test san.metadata[:K] == closure_constant(d) && san.metadata[:θ] === nothing
        pgf = node_based(sc; closure = PGFClosure(), level = :s_anchored)
        for sys in (pop, san, pgf)
            _, mc = solved_curves(sys, sc)
            for X in sc.observables
                @test maxdiff(mc[X], ref[X]) < 1e-8
            end
        end
    end
    # non-PT: the constant closure is biased (pinned, DESIGN §E.2: 0.034 on :sir_bim, 0.068 on
    # :sir_pl), and the S-anchored constant-K model has the node observables of the population one
    for (id, lo, hi) in ((:sir_bim, 0.030, 0.038), (:sir_pl, 0.062, 0.075))
        sc = scenario(id)
        d = sc.network.degrees
        @test is_poisson_type(d) === nothing
        K = [NodeBasedModels._pgf_closure_factor(d, x) for x in 0.3:0.1:1.0]
        @test K[end] ≈ closure_constant(d)
        @test maximum(K) - minimum(K) > 0.1
        ref = eb_reference(sc)
        _, cpop = solved_curves(node_based(sc), sc)
        _, csan = solved_curves(node_based(sc; level = :s_anchored), sc)
        for c in (cpop, csan)
            ΔR = c[:cumulative][end] - ref[:cumulative][end]
            @test ΔR > 0.02                                   # the pinned bias
            @test lo < ΔR < hi
        end
        for X in sc.observables                               # M7: restriction
            @test maxdiff(cpop[X], csan[X]) < 1e-8
        end
        id === :sir_bim && @test 0.008 < maxdiff(cpop[:I], ref[:I]) < 0.011   # §E.3: 0.0094
    end
end

@testset "PGFClosure at the population level: every pair; conservation" begin
    # With K_ψ ≡ K (Poisson type) the full PGF pairwise model is the population pairwise model
    # with the constant closure, pair by pair: an independent check of the non-S pair equations
    # (every triple of a T_EB model is S-centred). Removals go to the sink (sir with I → ∅).
    rem = ContactModel(:sir_removal; contacts = [Contact(:S, :I, :I, :τ)],
                       transitions = [NodeTransition(:I, nothing, :γ), NodeTransition(:S, :V, :ν)])
    values_ = merge(Dict(:τ => 1 / 6, :γ => 1 / 4, :ν => 0.02), scenario(:seair_pois5).params)
    for (cm, net) in ((sir_model(), ConfigurationNetwork(RegularDegree(6))),
                      (sirv_model(), ConfigurationNetwork(PoissonDegree(5))),
                      (seair_model(), ConfigurationNetwork(NegBinDegree(mean = 4, var = 8))),
                      (rem, ConfigurationNetwork(PoissonDegree(5))))
        own = Set(Symbol(string(x)) for x in rate_parameters(cm))
        kw = (; p = Dict(k => v for (k, v) in values_ if k in own),
              initial = SeedFraction((nameof(cm) === :seair ? :E : :I) => 0.01),
              tspan = (0.0, 40.0))
        full = node_based(cm, net; closure = PGFClosure(), kw...)
        cons = node_based(cm, net; kw...)
        @test Set(keys(full.pairs)) == Set(keys(cons.pairs))
        @test Set(keys(full.singles)) == Set(keys(cons.singles))
        nameof(cm) === :sir_removal && @test haskey(full.singles, :removed)
        tt = 0.0:0.5:40.0
        a = solve_epidemic(full; saveat = tt, TOL...)
        b = solve_epidemic(cons; saveat = tt, TOL...)
        for (k, v) in full.pairs
            @test maxdiff(at(a, v, tt), at(b, cons.pairs[k], tt)) < 1e-8
        end
        for (X, v) in full.singles
            @test maxdiff(at(a, v, tt), at(b, cons.singles[X], tt)) < 1e-8
        end
    end
    # Conservation on the bimodal network: Σ_X [X] = N, the ordered pairs sum to ⟨k⟩N, and the
    # stubs of the susceptible nodes are Σ_Y [sY] = [s]θψ'(θ)/ψ(θ) (at both levels, in counts)
    sc = scenario(:sir_bim)
    d = sc.network.degrees
    for level in (:population, :s_anchored)
        sys = node_based(sc; closure = PGFClosure(), level, N = 1000)
        sol = solve_epidemic(sys, sc; TOL...)
        tt = collect(sc.tgrid)
        S = at(sol, sys.singles[:S], tt)
        θ = at(sol, sys.metadata[:θ], tt)
        @test maxdiff(sum(at(sol, v, tt) for v in values(sys.singles)), fill(1000.0, length(tt))) < 1e-7
        # Σ_Y [SY] in the ordered-pair convention: [SS] and every cross pair [SY] once
        stubs = sum(at(sol, v, tt) for ((a, b), v) in sys.pairs if a === :S || b === :S)
        @test maxdiff(stubs, S .* θ .* pgf_derivative.(Ref(d), θ, 1) ./ pgf.(Ref(d), θ)) < 1e-7
        if level === :population
            total = sum((a === b ? 1 : 2) .* at(sol, v, tt) for ((a, b), v) in sys.pairs)
            @test maxdiff(total, fill(1000 * mean_degree(d), length(tt))) < 1e-6
        end
        # counts on the scale N = 1000 give the same fractions as N = 1
        _, c1 = solved_curves(node_based(sc; closure = PGFClosure(), level), sc)
        c1000 = model_curves(sys, sol; t = sc.tgrid)
        for X in sc.observables
            @test maxdiff(c1000[X], c1[X]) < 1e-9
        end
    end
end

@testset "Seeding, parameters and the API of SAnchoredSystem" begin
    net = ConfigurationNetwork(EmpiricalDegree(2 => 5 / 6, 10 => 1 / 6))
    k̄ = mean_degree(net.degrees)
    sys = node_based(sir_model(), net; closure = PGFClosure(), level = :s_anchored,
                     p = Dict(:τ => 1 / 6, :γ => 1 / 4))
    S, I = sys.singles[:S], sys.singles[:I]
    SS, SI = sys.pairs[(:S, :S)], sys.pairs[(:S, :I)]
    θ = sys.metadata[:θ]
    # the default seed: ε = 1e-3 in the entry state; the π^PW image of the EB initial condition
    u = default_initial_conditions(sys)
    @test u === sys.u0
    @test u[S] ≈ 0.999 && u[I] ≈ 1e-3 && u[θ] == 1.0
    @test u[SI] ≈ k̄ * 0.999 * 1e-3 && u[SS] ≈ k̄ * 0.999^2
    @test u[sys.metadata[:cumulative]] ≈ 1e-3
    u = default_initial_conditions(sys; initial = SeedFraction(:I => 0.05, :R => 0.1))
    @test u[S] ≈ 0.85 && u[SI] ≈ k̄ * 0.85 * 0.05 && u[sys.pairs[(:S, :R)]] ≈ k̄ * 0.85 * 0.1
    @test u[sys.metadata[:cumulative]] ≈ 0.05                # seeded R is not infected (§J.8)
    @test default_initial_conditions(sys; seed_fraction = 0.02)[I] ≈ 0.02
    # SEIR seeds E by default (the entry state), or I with seed_state = :first_infectious
    seir = node_based(seir_model(), ConfigurationNetwork(PoissonDegree(5)); closure = PGFClosure(),
                      level = :s_anchored, seed_fraction = 0.01)
    @test seir.u0[seir.singles[:E]] ≈ 0.01 && seir.u0[seir.singles[:I]] == 0.0
    u = default_initial_conditions(seir; seed_state = :first_infectious)
    @test u[seir.singles[:I]] ≈ 0.01 && u[seir.singles[:E]] == 0.0
    # solving: keyword, positional and scenario forms; parameter errors
    tt = 0.0:1.0:30.0
    a = solve_epidemic(sys; saveat = tt, TOL...)
    b = solve_epidemic(sys, Dict(:τ => 1 / 6, :γ => 1 / 4); saveat = tt, TOL...)
    @test compartment(sys, a, :I) == compartment(sys, b, :I)
    c = solve_epidemic(sys; p = Dict(:τ => 0.3), saveat = tt, TOL...)
    @test compartment(sys, c, :I) != compartment(sys, a, :I)
    @test_throws ArgumentError solve_epidemic(sys; p = Dict(:β => 0.3))
    bare = node_based(ContactModel(:sirb; contacts = [Contact(:S, :I, :I, :τb)],
                                   transitions = [NodeTransition(:I, :R, :γb)]),
                      net; closure = PGFClosure())
    @test_throws ArgumentError solve_epidemic(bare; p = Dict(:τb => 0.2))   # γb has no value
    d = solve_epidemic(sys; initial = SeedFraction(:I => 0.05), saveat = tt, TOL...)
    @test compartment(sys, d, :I)[1] ≈ 0.05
    # observing
    θt = compartment(sys, a, :θ)
    @test all(diff(θt) .<= 1e-14) && 0 < minimum(θt) && maximum(θt) == 1.0
    @test population_fraction(sys, a, :I) == compartment(sys, a, :I)
    @test compartment(sys, a, :cumulative)[end] ≈ 1 - compartment(sys, a, :S)[end]
    @test_throws ArgumentError population_fraction(sys, a, :θ)
    @test_throws ArgumentError compartment(sys, a, :E)
    @test Set(keys(compartments(sys, a, [:S, :I]))) == Set([:S, :I])
    @test node_variables(sys) === sys.singles && pair_variables(sys) === sys.pairs
    @test occursin("level = :s_anchored", sprint(show, sys)) && occursin("θ", sprint(show, sys))
    @test model_curves(sys, a).label == "S-anchored pairwise (PGF closure)"
    con = node_based(sir_model(), net; level = :s_anchored, p = Dict(:τ => 1 / 6, :γ => 1 / 4))
    @test model_curves(con, solve_epidemic(con; saveat = tt)).label ==
          "S-anchored pairwise (constant K = 1.5)"
    @test model_curves(sys, a; label = "x").label == "x"
    # the vector field: θ, the singles and the S-anchored pairs (no accumulator)
    ode = symbolic_ode(sys)
    @test ode isa SymbolicODE
    @test Set(state_names(ode)) == Set([:θ, :S, :I, :R, :SS, :SI, :SR])
    @test Set(state_names(symbolic_ode(node_based(sir_model(), net; closure = PGFClosure())))) ==
          Set([:θ, :S, :I, :R, :SS, :SI, :SR, :II, :IR, :RR])
    @test Set(state_names(symbolic_ode(node_based(sir_model(), net; level = :s_anchored)))) ==
          Set([:S, :I, :R, :SS, :SI, :SR])
    # frequency-dependent rates are converted with the mean degree (τ = β/⟨k⟩)
    sirF = ContactModel(:sirF; contacts = [Contact(:S, :I, :I, :β)],
                        transitions = [NodeTransition(:I, :R, :γ)],
                        convention = FrequencyDependent())
    f = node_based(sirF, net; closure = PGFClosure(), p = Dict(:β => k̄ / 6, :γ => 1 / 4))
    g = node_based(sir_model(), net; closure = PGFClosure(), p = Dict(:τ => 1 / 6, :γ => 1 / 4))
    @test maxdiff(compartment(f, solve_epidemic(f; saveat = tt, TOL...), :I),
                  compartment(g, solve_epidemic(g; saveat = tt, TOL...), :I)) < 1e-10
end

@testset "PGFClosure and the S-anchored level: errors" begin
    pois = ConfigurationNetwork(PoissonDegree(5))
    # an arrow back into S breaks the construction (T_EB only): SIS and SIRS
    for cm in (sis_model(), sirs_model())
        @test_throws AdmissibilityError node_based(cm, pois; closure = PGFClosure())
        @test_throws AdmissibilityError node_based(cm, pois; closure = PGFClosure(),
                                                   level = :s_anchored)
        @test_throws AdmissibilityError node_based(cm, pois; level = :s_anchored)
    end
    clus = ClusteredNetwork(RegularDegree(2), RegularDegree(2))
    @test_throws ArgumentError node_based(sir_model(), clus; closure = PGFClosure())
    @test_throws ArgumentError node_based(sir_model(), clus; level = :s_anchored)
    @test_throws ArgumentError node_based(sir_model(), pois; closure = KeelingClosure(),
                                          level = :s_anchored)
    @test_throws ArgumentError node_based(sir_model(), pois; closure = PowerClosure(2.0),
                                          level = :s_anchored)
    @test_throws ArgumentError node_based(sir_model(), WellMixed(5); level = :s_anchored)
    @test_throws ArgumentError node_based(sir_model(), WellMixed(5); closure = PGFClosure())
    @test_throws ArgumentError node_based(sir_model(), HeterogeneousNetwork(pois);
                                          closure = PGFClosure())
    @test_throws ArgumentError generate_pairwise(CompartmentalModel(sir_model()),
                                                 regular_network(4), PGFClosure())
    @test_throws MethodError node_based(sir_model(), pois; closure = PGFClosure(),
                                        seedfraction = 0.1)                 # misspelt keyword
    @variables μ
    @test_throws ArgumentError node_based(sir_model(), ConfigurationNetwork(PoissonDegree(μ));
                                          closure = PGFClosure())
    @test_throws ArgumentError node_based(sir_model(), ConfigurationNetwork(RegularDegree(0));
                                          closure = PGFClosure())
    @test_throws ArgumentError node_based(sir_model(), pois; closure = PGFClosure(),
                                          initial = SeedFraction(:I => 1.0))   # no S left
    @test_throws ArgumentError node_based(twostrain_model(), pois; closure = PGFClosure())
    @test node_based(twostrain_model(), pois; closure = PGFClosure(),
                     initial = SeedFraction(:I1 => 0.005, :I2 => 0.005)) isa SAnchoredSystem
end

# M6 as a runtime morphism: π^PW from EdgeBasedModels' raw field onto `symbolic_ode` of the
# S-anchored system; `verify` checks Dπ·F = G∘π symbolically, with the numeric fallback.
function m6_map(ebsys, nb)
    co = ebsys.metadata[:coords]
    s = nb.metadata[:susceptible]
    q = ebsys.metadata[:q][s].param
    θ = co[:θ]
    ξ = get(co, :ξ, 1)
    d = ebsys.metadata[:network].degrees
    ψ, ψ1 = pgf(d, θ), pgf_derivative(d, θ, 1)
    mp = Pair{Any,Any}[]
    nb.metadata[:θ] === nothing || push!(mp, nb.metadata[:θ] => θ)
    for (X, v) in nb.singles
        push!(mp, v => (X === s ? q * ξ * ψ : co[Symbol(:pop_, X)]))
    end
    for ((a, b), v) in nb.pairs
        X = a === s ? b : a
        push!(mp, v => (X === s ? q^2 * ξ^2 * ψ1^2 / mean_degree(d) : q * ξ * ψ1 * co[Symbol(:φ_, X)]))
    end
    return mp
end

@testset "M6 as a semiconjugacy: verify(EB → PW^S)" begin
    if HAVE_EBM
        # (:sir_nb4 is left out: NetworkEpiCore's `verify` does not finish on the negative
        # binomial PGF; its trajectories are checked above)
        for id in (:sir_bim, :seair_pois5, :sir_vax_pois5, :sir_reg6)
            sc = scenario(id)
            eb = EdgeBasedModels.edge_based(sc.model, sc.network)
            nbs = node_based(sc.model, sc.network; closure = PGFClosure(), level = :s_anchored)
            m = Semiconjugacy(:eb_to_pws, symbolic_ode(eb), symbolic_ode(nbs), m6_map(eb, nbs))
            @test verify(m).ok
            # a perturbed map is not a semiconjugacy ([ss] without the factor 1/ψ'(1))
            bad = [k => (isequal(k, nbs.pairs[(:S, :S)]) ? v * mean_degree(sc.network.degrees) : v)
                   for (k, v) in m6_map(eb, nbs)]
            @test !verify(Semiconjugacy(:perturbed, symbolic_ode(eb), symbolic_ode(nbs), bad)).ok
            # the constant closure: exact iff Poisson type (M8)
            con = node_based(sc.model, sc.network; level = :s_anchored)
            mc = Semiconjugacy(:eb_to_pws_const, symbolic_ode(eb), symbolic_ode(con), m6_map(eb, con))
            @test verify(mc).ok == (is_poisson_type(sc.network.degrees) !== nothing)
        end
    else
        @test_skip "EdgeBasedModels is not loadable: the M6 semiconjugacy is not verified"
    end
end

@testset "symbolic_ode equals pairwise_image(ebcm).ode after renaming (WP18)" begin
    has_image = HAVE_EBM && isdefined(EdgeBasedModels, :pairwise_image) &&
                !isempty(methods(EdgeBasedModels.pairwise_image))
    if has_image
        for id in (:sir_bim, :seair_pois5, :sir_vax_pois5)
            sc = scenario(id)
            eb = EdgeBasedModels.edge_based(sc.model, sc.network)
            img = EdgeBasedModels.pairwise_image(eb)
            ode = img isa SymbolicODE ? img : hasproperty(img, :ode) ? img.ode : first(img)
            nb = node_based(sc.model, sc.network; closure = PGFClosure(), level = :s_anchored)
            @test vector_fields_equal(symbolic_ode(nb), ode)
        end
    else
        @test_skip "EdgeBasedModels.pairwise_image (WP18) is not available yet"
    end
end

@testset "Against NetworkOutbreaks: exact limit for PGFClosure, pinned bias for constant K" begin
    # Local reference ensembles with the scenarios' own SimConfig (N = 10⁴, 200 runs, a fresh
    # graph per run, stable_rng streams, MajorOutbreak(0.05)) until committed summaries exist.
    for id in (:sir_bim, :sir_vax_pois5, :sir_pl)
        sc = scenario(id)
        ref = NetworkOutbreaks.summarise(NetworkOutbreaks.scenario_ensemble(sc))
        @test ref.N == 10_000 && ref.nsims == 200
        _, san = solved_curves(node_based(sc; closure = PGFClosure(), level = :s_anchored), sc;
                               label = "s_anchored")
        _, pop = solved_curves(node_based(sc; closure = PGFClosure()), sc; label = "pgf")
        _, con = solved_curves(node_based(sc), sc; label = "const")
        tab = compare(ref, san, pop, con)
        @info "pgf_closure: $(id) against NetworkOutbreaks (N = $(ref.N), $(ref.n_major)/$(ref.nsims) major)" tab
        for label in ("s_anchored", "pgf")                   # :exact_limit (§E.2 tolerances)
            @test tab[label, :I].D∞ < 0.005
            @test abs(tab[label, :I].ΔR∞) < 0.005
        end
        if sc.backends[:pgf_closure] === :exact_limit && haskey(sc.backends, :pairwise_const) &&
           sc.backends[:pairwise_const] === :biased
            r = tab["const", :I]
            @test abs(r.ΔR∞) > 0.02                          # pinned bias
            @test r.ΔR∞_ci[1] > 0                            # the ensemble excludes it
        end
    end
end
