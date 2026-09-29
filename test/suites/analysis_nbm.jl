# analysis_nbm.jl — WP24: the threshold analysis of the population pairwise models, SIR and SIS
# separated (verified issue B01), and the literature and simulation evidence for the analysis
# goldens (test/golden/analysis/).
#
# Independent references used here:
# - the linearisation of the ODE that generate_pairwise builds (finite-difference Jacobian at the
#   disease-free state, infected block) for the Bernoulli closure;
# - the early exponential phase of that ODE (log-slope of [I] from a tiny seed) for the Keeling
#   and Barnard closures, whose closure terms are 0/0 at the disease-free state;
# - Barnard's thesis (ch. 4, eqs 4.34–4.39): Keeling closure, SIR, n = 6, ϕ = 0.3, γ = 1 gives
#   τ_c = 0.319169;
# - a reference pairwise ODE with Barnard's improved closure written out from the thesis (eq. 4.23,
#   the triple [ASI] with the S–I link known), independent of closures.jl and of the builder;
# - exact stochastic simulation (NetworkOutbreaks) on random regular graphs.

using NodeBasedModels
using NetworkEpiCore
using Test
using ModelingToolkit
using OrdinaryDiffEqDefault
using Symbolics
using LinearAlgebra
using Statistics
using Graphs
using NetworkOutbreaks
import NetworkOutbreaks as NO

const B = BernoulliClosure()

# The pairwise vector field of `psys` as a function of (u by name, τ, γ), and the index of every
# single and pair.
function pairwise_field(psys)
    sys = psys.system
    unk = ModelingToolkit.unknowns(sys)
    names = Dict{Symbol,Int}()
    for (i, u) in enumerate(unk)
        for (k, v) in psys.singles
            isequal(u, v) && (names[k] = i)
        end
        for ((a, b), v) in psys.pairs
            isequal(u, v) && (names[Symbol(a, b)] = i)
        end
    end
    ps = ModelingToolkit.parameters(sys)
    τp = only(filter(p -> Symbol(p) === :τ, ps))
    γp = only(filter(p -> Symbol(p) === :γ, ps))
    prob = ModelingToolkit.ODEProblem(sys, merge(psys.u0, Dict(τp => 0.1, γp => 1.0)), (0.0, 1.0))
    function f(u, τ, γ)
        prob.ps[τp] = τ
        prob.ps[γp] = γ
        du = similar(u)
        prob.f(du, u, prob.p, 0.0)
        return du
    end
    return f, names, length(unk)
end

# The leading eigenvalue of the finite-difference Jacobian at the disease-free state, restricted to
# the variables that involve I (for SIR the recovered block does not feed back).
function dfe_growth(psys, τ, γ; N = 1.0)
    f, names, m = pairwise_field(psys)
    u = zeros(m)
    u[names[:S]] = N
    u[names[:SS]] = mean_degree(psys.network) * N
    inf = sort([i for (k, i) in names if occursin("I", String(k))])
    J = zeros(length(inf), length(inf))
    h = 1e-7
    for (c, j) in enumerate(inf)
        e = zeros(m)
        e[j] = h
        J[:, c] = ((f(u .+ e, τ, γ) .- f(u .- e, τ, γ)) ./ (2h))[inf]
    end
    return maximum(real, eigvals(J))
end

@testset "B01: SIR and SIS pairwise analysis are separate and consistent" begin
    γ = 1.0
    for n in (4, 6)
        hom = regular_network(n)
        het = degree_distribution_network([zeros(n); 1.0])   # the same n-regular network
        @test epidemic_threshold(hom, B, γ) ≈ γ / (n - 2)                    # SIR default
        @test epidemic_threshold(het, B, γ; dynamics = :SIR) ≈ γ / (n - 2)   # 0.1: γ/(n−1)
        for τ in (0.3, 0.42, 0.6)
            @test basic_reproduction_number(het, B, τ, γ; dynamics = :SIR) ≈
                  basic_reproduction_number(hom, B, τ, γ)                    # 0.1: 1.2186 vs 0.84
        end
        @test epidemic_threshold(hom, B, γ; dynamics = :SIS) ≈ γ / (n - 1)
        @test epidemic_threshold(het, B, γ; dynamics = :SIS) ≈ γ / (n - 1)
        @test basic_reproduction_number(hom, B, γ / (n - 1), γ; dynamics = :SIS) ≈ 1
        @test basic_reproduction_number(het, B, γ / (n - 2), γ; dynamics = :SIR) ≈ 1   # 0.1: 1.414
        τ = 0.42
        ev(x, a, b) = Symbolics.build_function(x, Symbolics.variable(a), Symbolics.variable(b);
                                               expression = Val(false))(τ, γ)
        a = τ * (n - 2) - γ
        Rsis = (a + sqrt((2γ - a)^2 + 8γ * (a + τ))) / (2γ)
        for net in (hom, het)
            @test ev(basic_reproduction_number(sir_model(), net, B), :τ, :γ) ≈ τ * (n - 2) / γ
            @test ev(basic_reproduction_number(sis_model(), net, B), :τ, :γ) ≈ Rsis   # 0.1: SIR formula
            @test early_growth_rate(net, B, τ, γ; dynamics = :SIS) ≈ γ * (Rsis - 1)
            @test early_growth_rate(net, B, τ, γ; dynamics = :SIR) ≈ τ * (n - 2) - γ
        end
    end
    net = degree_distribution_network([0, 0, 0.5, 0, 0, 0, 0, 0, 0.5])      # bimodal 2/8
    @test epidemic_threshold(net, B, 1.0; dynamics = :SIR) ≈ 5 / 24         # SIR ODE τc 0.2083333
    @test epidemic_threshold(net, B, 1.0; dynamics = :SIS) ≈ 5 / 29         # SIS ODE τc 0.1724138
    # a heterogeneous network needs the dynamics; the models are classified strictly
    @test_throws ArgumentError epidemic_threshold(net, B, 1.0)
    @test_throws ArgumentError basic_reproduction_number(net, B, 0.3, 1.0)
    @test_throws ArgumentError early_growth_rate(net, B, 0.3, 1.0)
    @test_throws ArgumentError epidemic_threshold(net, B, 1.0; dynamics = :SEIR)
    @test_throws ArgumentError basic_reproduction_number(sirs_model(; ε = :γ), regular_network(4), B)
    @test_throws ArgumentError basic_reproduction_number(seir_model(), regular_network(4), B)
    @test_throws ArgumentError basic_reproduction_number(sirv_model(), regular_network(4), B)
    # clustered closures have no closed-form R; PowerClosure no linearisation
    @test_throws ArgumentError basic_reproduction_number(sir_model(), regular_network(6; ϕ = 0.2),
                                                         KeelingClosure())
    @test_throws ArgumentError basic_reproduction_number(sis_model(), regular_network(6; ϕ = 0.2),
                                                         BarnardClosure())
    @test_throws ArgumentError basic_reproduction_number(sir_model(), regular_network(6), PowerClosure(1.5))
    @test_throws ArgumentError basic_reproduction_number(sir_model(), net, BarnardClosure())   # heterogeneous
    # … but with ϕ = 0 both clustered closures are the Bernoulli closure, and so is the formula
    ev2(x, τ, γ) = Symbolics.build_function(x, Symbolics.variable(:τ), Symbolics.variable(:γ);
                                            expression = Val(false))(τ, γ)
    for cl in (KeelingClosure(), BarnardClosure()), m in (sir_model(), sis_model())
        @test ev2(basic_reproduction_number(m, regular_network(6), cl), 0.3, 0.7) ≈
              ev2(basic_reproduction_number(m, regular_network(6), B), 0.3, 0.7)
    end
    @test ev2(basic_reproduction_number(sir_model(), regular_network(6), KeelingClosure()), 0.3, 0.7) ≈
          4 * 0.3 / 0.7
    @test ev2(basic_reproduction_number(sis_model(), net, KeelingClosure()), 0.3, 0.7) ≈
          ev2(basic_reproduction_number(sis_model(), net, B), 0.3, 0.7)
    @test_throws ArgumentError epidemic_threshold(regular_network(4), PowerClosure(1.5), 1.0)
    @test_throws ArgumentError epidemic_threshold(degree_distribution_network(net.degree_probs),
                                                  BarnardClosure(), 1.0; dynamics = :SIR)
end

@testset "Bernoulli thresholds and growth rates are the linearisation of the built ODE" begin
    γ = 0.25
    for (dyn, model) in ((:SIR, CompartmentalModel(sir_model())), (:SIS, CompartmentalModel(sis_model())))
        for net in (regular_network(4), degree_distribution_network([0, 0, 0.5, 0, 0, 0, 0, 0, 0.5]))
            psys = generate_pairwise(model, net, B)
            for τ in (0.02, 0.05, 0.1, 0.2)
                @test dfe_growth(psys, τ, γ) ≈ early_growth_rate(net, B, τ, γ; dynamics = dyn) atol = 1e-6
            end
            τc = epidemic_threshold(net, B, γ; dynamics = dyn)
            @test dfe_growth(psys, 0.99τc, γ) < 0 < dfe_growth(psys, 1.01τc, γ)
            @test early_growth_rate(psys; p = Dict(:τ => 0.1, :γ => γ)) ==
                  early_growth_rate(net, B, 0.1, γ; dynamics = dyn)
            @test epidemic_threshold(psys; p = Dict(:γ => γ)) == τc
            @test basic_reproduction_number(psys; p = Dict(:τ => τc, :γ => γ)) ≈ 1
        end
    end
end

@testset "SIS on :sis_reg3: τ_c = γ/(k − 1) = 0.125" begin
    sc = scenario(:sis_reg3)
    k = sc.network.degrees.k
    γ = sc.params[:γ]
    @test (k, γ) == (3, 1 / 4)
    sys = node_based(sc)                              # the default closure: Bernoulli
    @test sys.closure isa BernoulliClosure
    @test epidemic_threshold(sys) == γ / (k - 1) == 0.125
    @test epidemic_threshold(sc.model, sc.network, BernoulliClosure(); p = sc.params) == 0.125
    @test epidemic_threshold(sc.model, network_structure(sc.network); p = sc.params) == 0.125
    @test sc.params[:τ] / epidemic_threshold(sys) ≈ 4                  # the scenario's τ/τ_c
    # the linearised growth changes sign between τ = 0.12 and 0.13
    @test early_growth_rate(sys; p = Dict(:τ => 0.12)) < 0 < early_growth_rate(sys; p = Dict(:τ => 0.13))
    @test early_growth_rate(sc.model, sc.network, BernoulliClosure(); p = Dict(:τ => 0.12, :γ => γ)) < 0
    # … and so does the leading eigenvalue of the built system's Jacobian (independent of analysis.jl)
    psys = generate_pairwise(sc.model, sc.network, BernoulliClosure())
    @test dfe_growth(psys, 0.12, γ) < 0 < dfe_growth(psys, 0.13, γ)
    @test dfe_growth(psys, 0.125, γ) ≈ 0 atol = 1e-7
    @test basic_reproduction_number(sys; p = Dict(:τ => 0.125)) ≈ 1
    @test basic_reproduction_number(sys) > 1                            # τ = 4τ_c
    # SIR on the same network has the larger threshold γ/(k − 2)
    @test epidemic_threshold(sir_model(), sc.network, BernoulliClosure(); p = sc.params) == 0.25
end

@testset "Clustered closures: fast-variable quasi-equilibria" begin
    γ = 1.0
    # Barnard's thesis (eq. 4.39 solved numerically): Keeling closure, SIR, n = 6, ϕ = 0.3
    @test epidemic_threshold(regular_network(6; ϕ = 0.3), KeelingClosure(), γ) ≈ 0.319169 rtol = 3e-6
    # the growth rate vanishes at the threshold and R = 1 there, for every closure and dynamics
    for net in (regular_network(4; ϕ = 0.2), regular_network(6; ϕ = 0.3),
                degree_distribution_network([k in (3, 9) ? 0.5 : 0.0 for k in 0:9]; ϕ = 0.1)),
        cl in (KeelingClosure(), BarnardClosure()), dyn in (:SIR, :SIS)
        (net isa HeterogeneousNetwork && cl isa BarnardClosure) && continue
        τc = epidemic_threshold(net, cl, γ; dynamics = dyn)
        @test abs(early_growth_rate(net, cl, τc, γ; dynamics = dyn)) < 1e-9
        @test early_growth_rate(net, cl, 0.95τc, γ; dynamics = dyn) < 0 <
              early_growth_rate(net, cl, 1.05τc, γ; dynamics = dyn)
        @test basic_reproduction_number(net, cl, τc, γ; dynamics = dyn) ≈ 1
    end
    # ϕ = 0 gives the Bernoulli values; clustering raises the threshold
    for cl in (KeelingClosure(), BarnardClosure()), dyn in (:SIR, :SIS)
        @test epidemic_threshold(regular_network(5), cl, γ; dynamics = dyn) ==
              epidemic_threshold(regular_network(5), B, γ; dynamics = dyn)
        @test early_growth_rate(regular_network(5), cl, 0.4, γ; dynamics = dyn) ==
              early_growth_rate(regular_network(5), B, 0.4, γ; dynamics = dyn)
        @test epidemic_threshold(regular_network(5; ϕ = 0.2), cl, γ; dynamics = dyn) >
              epidemic_threshold(regular_network(5), B, γ; dynamics = dyn)
    end
    # the SIS thresholds are closed forms: Keeling's linear condition, and for Barnard's closure
    # δ − 1 = (n − 1)ϕδ/(n + δ) with x = (n − 1)(1 − ϕ + ϕn/(n + δ)), τ_c = γ/x
    n, ϕ = 6, 0.3
    @test epidemic_threshold(regular_network(n; ϕ), KeelingClosure(), γ; dynamics = :SIS) ≈
          γ * (1 - (n - 1) * ϕ / n) / ((n - 1) * (1 - ϕ))
    δ = (-(n - 1) * (1 - ϕ) + sqrt(((n - 1) * (1 - ϕ))^2 + 4n)) / 2
    @test δ - 1 ≈ (n - 1) * ϕ * δ / (n + δ)
    τc_sis = epidemic_threshold(regular_network(n; ϕ), BarnardClosure(), γ; dynamics = :SIS)
    @test τc_sis ≈ γ / ((n - 1) * (1 - ϕ + ϕ * n / (n + δ)))
    @test τc_sis ≈ 0.2109882087 rtol = 1e-9
    # Barnard's improved closure spreads more readily than Keeling's for these parameters
    # (thesis §4.6, stated for its compact variant): a lower threshold, SIR and SIS
    for dyn in (:SIR, :SIS)
        @test epidemic_threshold(regular_network(n; ϕ), BarnardClosure(), γ; dynamics = dyn) <
              epidemic_threshold(regular_network(n; ϕ), KeelingClosure(), γ; dynamics = dyn)
    end
end

# Several of these solves (the subcritical Keeling cases near δ = ∞) are very stiff: the default
# auto-switching algorithm hits maxiters at τ = 0.08 or 0.12 depending on the platform (x86-64
# vs arm64), while Rodas5P completes those but reports Unstable on some non-stiff supercritical
# cases that the default handles. So try the default first, fall back to stiff solvers, and fail
# loudly (never silently truncate) if none completes. The completed solvers agree to ~1e-12.
const FALLBACK_SOLVERS = (nothing, OrdinaryDiffEqDefault.Rodas5P(), OrdinaryDiffEqDefault.FBDF())

function robust_solve(psys, p; kwargs...)
    retcodes = Symbol[]
    for solver in FALLBACK_SOLVERS
        sol = solve_pairwise(psys, p; solver, kwargs...)
        Symbol(sol.retcode) === :Success && return sol
        push!(retcodes, Symbol(sol.retcode))
    end
    error("robust_solve: every solver failed (retcodes $retcodes) for p = $p")
end

# The early exponential phase of the built ODE: log-slope of [I] over [t1, t2] from a seed ε.
function ode_slope(psys, τ, γ; t1, t2, ε = 1e-13)
    sol = robust_solve(psys, Dict(:τ => τ, :γ => γ); saveat = [t1, t2], reltol = 1e-11,
                       abstol = 1e-28, maxiters = 10^6, tspan = (0.0, t2),
                       u0 = default_initial_conditions(psys; seed_fraction = ε))
    I = compartment(psys, sol, :I)
    return (log(I[2]) - log(I[1])) / (t2 - t1)
end

@testset "Keeling closure: the growth rate is the early phase of the built ODE" begin
    γ = 1.0
    for net in (regular_network(6; ϕ = 0.3),
                degree_distribution_network([k in (3, 9) ? 0.5 : 0.0 for k in 0:9]; ϕ = 0.1))
        for (dyn, model) in ((:SIR, CompartmentalModel(sir_model())), (:SIS, CompartmentalModel(sis_model())))
            psys = generate_pairwise(model, net, KeelingClosure(); tspan = (0.0, 100.0))
            τc = epidemic_threshold(net, KeelingClosure(), γ; dynamics = dyn)
            for f in (1.25, 2.0)          # supercritical: an exponential phase with rate r
                r = early_growth_rate(net, KeelingClosure(), f * τc, γ; dynamics = dyn)
                T = min(100.0, 11.0 / r)
                @test ode_slope(psys, f * τc, γ; t1 = 0.7T, t2 = T) ≈ r rtol = 1e-4
            end
            r = early_growth_rate(net, KeelingClosure(), 0.8τc, γ; dynamics = dyn)   # subcritical
            @test ode_slope(psys, 0.8τc, γ; t1 = 20.0, t2 = 30.0) ≈ r atol = 1e-5
            @test ode_slope(psys, 0.99τc, γ; t1 = 60.0, t2 = 100.0) < 0 <
                  ode_slope(psys, 1.01τc, γ; t1 = 60.0, t2 = 100.0)
        end
    end
end

# ─── Keeling's closure with strong clustering (review round 2) ───────────────────
# The fast variables α = [SI]/[I], δ = [II]/[I] and ℓ = log[I] of Keeling's closure, written out
# here from the triples independently of analysis.jl, at leading order at the disease-free state
# ([S] = 1, [SS] = ⟨k⟩ per node): [SSI] = (q/⟨k⟩)[SS][SI](1 − ϕ + ϕ[SI]/(⟨k⟩[I])) and
# [ISI] = (q/⟨k⟩)[SI]²·ϕ[II]/(⟨k⟩[I]²) (its (1 − ϕ) part is of second order in [I]). Unlike the
# built ODE, this stays integrable when [II]/[I] grows without bound.
function keeling_fast!(du, u, (q, mk, ϕ, τ, γ, sis), t)
    ℓ, α, δ = u
    ssi = q * α * (1 - ϕ + ϕ * α / mk)
    isi = q * ϕ / mk^2 * α^2 * δ
    r = τ * α - γ
    du[1] = r
    du[2] = τ * (ssi - isi - α) - γ * α + (sis ? γ * δ : 0.0) - r * α
    du[3] = 2τ * (isi + α) - 2γ * δ - r * δ
    return nothing
end

# The log-slope of [I] over [t1, t2] from a random seed, (α, δ) = (⟨k⟩, 0).
function keeling_fast_slope(net, τ, γ, dyn; t1, t2)
    q, mk = excess_degree(net), mean_degree(net)
    prob = ODEProblem(keeling_fast!, [0.0, mk, 0.0], (0.0, t2), (q, mk, net.ϕ, τ, γ, dyn === :SIS))
    sol = solve(prob; saveat = [t1, t2], reltol = 1e-11, abstol = 1e-12, maxiters = 10^7)
    return (sol.u[2][1] - sol.u[1][1]) / (t2 - t1)
end

@testset "Keeling closure, strong clustering: the SIS threshold γqϕ/⟨k⟩² and δ = ∞" begin
    γ = 1.0
    sis = CompartmentalModel(sis_model())
    kc(τ, net) = early_growth_rate(net, KeelingClosure(), τ, γ; dynamics = :SIS)
    # bimodal 1/12: q = 66/6.5 = 10.15, ⟨k⟩ = 6.5; with ϕ = 0.6, qϕ(q(1 − ϕ) + ⟨k⟩) > ⟨k⟩², so the
    # linear root of the r = 0 condition has [II]/[I] < 0 and τ_c = γc = γqϕ/⟨k⟩² (0.1 returned
    # Inf here, and +4.24 for r at τ = 0.08 while the ODE decays)
    net = degree_distribution_network([k in (1, 12) ? 0.5 : 0.0 for k in 0:12]; ϕ = 0.6)
    q, mk = excess_degree(net), mean_degree(net)
    c = q * 0.6 / mk^2
    τc = epidemic_threshold(net, KeelingClosure(), γ; dynamics = :SIS)
    @test τc ≈ γ * c
    @test τc ≈ 0.1441966 rtol = 1e-6
    psys = generate_pairwise(sis, net, KeelingClosure(); tspan = (0.0, 100.0))
    @test epidemic_threshold(psys; p = Dict(:γ => γ)) == τc
    @test ode_slope(psys, 0.97τc, γ; t1 = 60.0, t2 = 100.0) < 0 <
          ode_slope(psys, 1.03τc, γ; t1 = 60.0, t2 = 100.0)
    @test abs(kc(τc, net)) < 1e-12
    @test kc(0.97τc, net) < 0 < kc(1.03τc, net)
    # below γc, [II]/[I] → ∞ and [SI]/[I] → α∞ = √(γ/(τc)), so r = √(τγ/c) − γ (the built ODE
    # approaches it at the rate −r, and becomes too stiff to integrate much further)
    for (τ, t1, t2) in ((0.08, 15.0, 20.0), (0.12, 30.0, 40.0))
        r = kc(τ, net)
        @test r ≈ sqrt(τ * γ / c) - γ
        @test ode_slope(psys, τ, γ; t1, t2) ≈ r atol = 2e-3
        @test keeling_fast_slope(net, τ, γ, :SIS; t1 = 200.0, t2 = 400.0) ≈ r atol = 1e-8
    end
    sol = robust_solve(psys, Dict(:τ => 0.08, :γ => γ); maxiters = 10^6,
                       saveat = [15.0, 20.0], reltol = 1e-11, abstol = 1e-28, tspan = (0.0, 20.0),
                         u0 = default_initial_conditions(psys; seed_fraction = 1e-13))
    I = compartment(psys, sol, :I)
    α, δ = sol[psys.pairs[(:S, :I)]] ./ I, sol[psys.pairs[(:I, :I)]] ./ I
    @test α[2] ≈ sqrt(γ / (0.08c)) rtol = 1e-3
    @test log(δ[2] / δ[1]) / 5 ≈ -kc(0.08, net) atol = 0.01
    # supercritical: the fast variables settle at a fixed point (near τ_c, slowly: at τ = 0.15 the
    # log-slope over [70, 100] is 5% above r)
    @test ode_slope(psys, 0.15, γ; t1 = 70.0, t2 = 100.0) > 0 && kc(0.15, net) > 0
    for τ in (0.2, 1.25τc, 2τc)
        r = kc(τ, net)
        T = min(100.0, 11.0 / r)
        @test ode_slope(psys, τ, γ; t1 = 0.7T, t2 = T) ≈ r rtol = 5e-4
    end
    # the subcritical queries that stalled Newton's method: δ = ∞ is the only attractor
    for (pk, ϕ, τ) in (((1, 12), 0.6, 0.02), ((1, 12), 0.6, 0.05), ((1, 12), 0.3, 0.05),
                       ((3, 9), 0.6, 0.05), ((3, 9), 0.6, 0.08), ((3, 9), 0.6, 0.1))
        n2 = degree_distribution_network([k in pk ? 0.5 : 0.0 for k in 0:pk[2]]; ϕ)
        c2 = excess_degree(n2) * ϕ / mean_degree(n2)^2
        r = kc(τ, n2)
        @test r ≈ sqrt(τ * γ / c2) - γ
        t2 = min(800.0, 150 / abs(r))     # [II]/[I] grows like exp(−rt)
        @test keeling_fast_slope(n2, τ, γ, :SIS; t1 = t2 / 2, t2) ≈ r atol = 1e-6
    end
    # A + 1 + B/c ≤ 0: the linear threshold γ(1 − qϕ/⟨k⟩)/(q(1 − ϕ)), on 1/12 with ϕ = 0.3 (just
    # above γc) and on 3/9 with ϕ = 0.6; it is always this one on a homogeneous network
    for (pk, ϕ, τc_ref) in (((1, 12), 0.3, 0.07475857), ((3, 9), 0.6, 0.13461538))
        n2 = degree_distribution_network([k in pk ? 0.5 : 0.0 for k in 0:pk[2]]; ϕ)
        q2, mk2 = excess_degree(n2), mean_degree(n2)
        τc2 = epidemic_threshold(n2, KeelingClosure(), γ; dynamics = :SIS)
        @test τc2 ≈ γ * (1 - q2 * ϕ / mk2) / (q2 * (1 - ϕ))
        @test τc2 ≈ τc_ref rtol = 1e-7
        @test τc2 > γ * q2 * ϕ / mk2^2
        ps2 = generate_pairwise(sis, n2, KeelingClosure(); tspan = (0.0, 100.0))
        @test ode_slope(ps2, 0.97τc2, γ; t1 = 60.0, t2 = 100.0) < 0 <
              ode_slope(ps2, 1.03τc2, γ; t1 = 60.0, t2 = 100.0)
        for f in (1.25, 2.0)
            r = kc(f * τc2, n2)
            T = min(100.0, 11.0 / r)
            @test ode_slope(ps2, f * τc2, γ; t1 = 0.7T, t2 = T) ≈ r rtol = 5e-4
        end
    end
    for n in 3:10, ϕ in (0.3, 0.9)
        @test epidemic_threshold(regular_network(n; ϕ), KeelingClosure(), γ; dynamics = :SIS) ≈
              γ * (1 - (n - 1) * ϕ / n) / ((n - 1) * (1 - ϕ))
    end
    # 3/9, ϕ = 0.6: between γc and τ_c a fixed point is the only attractor (r from the built ODE);
    # at τ = 0.02 a fixed point and δ = ∞ both attract, and the random seed reaches the fixed point
    n39 = degree_distribution_network([k in (3, 9) ? 0.5 : 0.0 for k in 0:9]; ϕ = 0.6)
    ps39 = generate_pairwise(sis, n39, KeelingClosure(); tspan = (0.0, 100.0))
    @test ode_slope(ps39, 0.12, γ; t1 = 60.0, t2 = 100.0) ≈ kc(0.12, n39) atol = 2e-4
    c39 = excess_degree(n39) * 0.6 / mean_degree(n39)^2
    r = kc(0.02, n39)
    @test r ≈ -0.789716 atol = 1e-6
    @test !(r ≈ sqrt(0.02γ / c39) - γ)
    @test keeling_fast_slope(n39, 0.02, γ, :SIS; t1 = 200.0, t2 = 400.0) ≈ r atol = 1e-5
    @test keeling_fast_slope(n39, 0.8 * 0.13461538, γ, :SIS; t1 = 200.0, t2 = 400.0) ≈
          kc(0.8 * 0.13461538, n39) atol = 2e-5
    # SIR on the same networks is unaffected (A = q(1 − ϕ) − 1 > 0): r against the built ODE
    sir = CompartmentalModel(sir_model())
    for n2 in (net, n39)
        ps = generate_pairwise(sir, n2, KeelingClosure(); tspan = (0.0, 100.0))
        τr = epidemic_threshold(n2, KeelingClosure(), γ; dynamics = :SIR)
        for f in (0.8, 1.25, 2.0)
            r = early_growth_rate(n2, KeelingClosure(), f * τr, γ; dynamics = :SIR)
            T = min(100.0, 11.0 / abs(r))
            @test ode_slope(ps, f * τr, γ; t1 = 0.7T, t2 = T) ≈ r atol = 1e-5
        end
    end
end

@testset "Keeling closure, SIR with q(1 − ϕ) < 1 < qϕ/⟨k⟩: bistable, no threshold" begin
    γ = 1.0
    # bimodal 1/20 (70% degree 1): q = 17.0, ⟨k⟩ = 6.7; ϕ = 0.95 gives A = −0.149, B = 1.413. The
    # fast variables either die out (the origin, r = −γ) or settle at a fixed point where the
    # closed-triangle term sustains transmission; the random seed reaches the fixed point at
    # τ = 0.69 (r > 0) but the origin at τ = 2.08, although a fixed point with r = 1.45 attracts
    pk = zeros(21)
    pk[2], pk[21] = 0.7, 0.3
    net = degree_distribution_network(pk; ϕ = 0.95)
    @test_throws ArgumentError epidemic_threshold(net, KeelingClosure(), γ; dynamics = :SIR)
    @test_throws ArgumentError epidemic_threshold(sir_model(), net, KeelingClosure(); p = Dict(:γ => γ))
    r1 = early_growth_rate(net, KeelingClosure(), 0.69192, γ; dynamics = :SIR)
    r2 = early_growth_rate(net, KeelingClosure(), 2.07575, γ; dynamics = :SIR)
    @test r1 ≈ 0.224479 atol = 1e-6
    @test r2 == -γ
    @test keeling_fast_slope(net, 0.69192, γ, :SIR; t1 = 100.0, t2 = 200.0) ≈ r1 atol = 1e-6
    @test keeling_fast_slope(net, 2.07575, γ, :SIR; t1 = 100.0, t2 = 200.0) ≈ r2 atol = 1e-6
    ps = generate_pairwise(CompartmentalModel(sir_model()), net, KeelingClosure(); tspan = (0.0, 100.0))
    @test ode_slope(ps, 0.69192, γ; t1 = 40.0, t2 = 60.0) ≈ r1 atol = 1e-5
    # A < 0 and B ≤ 0 (a homogeneous network with (n − 1)(1 − ϕ) < 1): [SI]/[I] → 0, no epidemic
    @test epidemic_threshold(regular_network(4; ϕ = 0.9), KeelingClosure(), γ) == Inf
    @test early_growth_rate(regular_network(4; ϕ = 0.9), KeelingClosure(), 5.0, γ) == -γ
    @test early_growth_rate(regular_network(5; ϕ = 0.75), KeelingClosure(), 1.0, γ) ≈ -γ   # A = 0
end

# ─── Barnard's improved closure ──────────────────────────────────────────────────
# Barnard's thesis eq. 4.23, written out here independently of closures.jl: the triple [A S I]
# (the S–I link carries the infection, A is the state of the other neighbour of S) is
#   (n − 1)((1 − ϕ)[AS][SI]/(n[S]) + ϕ[AS][SI][IA]/([A] Σ_a [aS][aI]/[a])),
# the clustered term being the probability of state A normalised over A.
sdiv(a, b) = b == 0 ? zero(a) : a / b
function barnard_ASI(A, sing, pair, n, ϕ)
    P(a, b) = get(pair, (a, b), get(pair, (b, a), 0.0))
    den = sum(sdiv(P(a, :S) * P(a, :I), sing[a]) for a in keys(sing))
    return (n - 1) * ((1 - ϕ) * sdiv(P(A, :S) * P(:S, :I), n * sing[:S]) +
                      ϕ * sdiv(P(A, :S) * P(:S, :I) * P(:I, A), sing[A] * den))
end

# The pairwise SIR and SIS equations (ordered pairs, [XX] counted twice) with that closure.
# SIR u = [S, I, R, SS, SI, SR, II, IR, RR]; SIS u = [S, I, SS, SI, II].
function barnard_reference!(du, u, (n, ϕ, τ, γ, dyn), t)
    if dyn === :SIR
        S, I, R, SS, SI, SR, II, IR, RR = u
        sing = Dict(:S => S, :I => I, :R => R)
        pair = Dict((:S, :S) => SS, (:S, :I) => SI, (:S, :R) => SR, (:I, :I) => II,
                    (:I, :R) => IR, (:R, :R) => RR)
    else
        S, I, SS, SI, II = u
        sing = Dict(:S => S, :I => I)
        pair = Dict((:S, :S) => SS, (:S, :I) => SI, (:I, :I) => II)
    end
    SSI, ISI = barnard_ASI(:S, sing, pair, n, ϕ), barnard_ASI(:I, sing, pair, n, ϕ)
    if dyn === :SIR
        RSI = barnard_ASI(:R, sing, pair, n, ϕ)
        du .= (-τ * SI, τ * SI - γ * I, γ * I, -2τ * SSI, τ * (SSI - ISI - SI) - γ * SI,
               -τ * RSI + γ * SI, 2τ * (ISI + SI) - 2γ * II, τ * RSI + γ * II - γ * IR, 2γ * IR)
    else
        du .= (-τ * SI + γ * I, τ * SI - γ * I, -2τ * SSI + 2γ * SI,
               τ * (SSI - ISI - SI) - γ * SI + γ * II, 2τ * (ISI + SI) - 2γ * II)
    end
    return nothing
end

# The log-slope of [I] of the reference ODE over [t1, t2], from a seed ε on the product state.
function barnard_slope(n, ϕ, τ, γ, dyn; t1, t2, ε = 1e-13)
    S, I = 1 - ε, ε
    u0 = dyn === :SIR ? [S, I, 0, n * S^2, n * S * I, 0, n * I^2, 0, 0] :
                        [S, I, n * S^2, n * S * I, n * I^2]
    prob = ODEProblem(barnard_reference!, u0, (0.0, t2), (n, ϕ, τ, γ, dyn))
    sol = solve(prob; saveat = [t1, t2], reltol = 1e-11, abstol = 1e-28, maxiters = 10^6)
    return (log(sol.u[2][2]) - log(sol.u[1][2])) / (t2 - t1)
end

@testset "Barnard's closure: triple_closure is thesis eq. 4.23 with the infector last" begin
    n, ϕ = 6, 0.3
    net = regular_network(n; ϕ)
    # a state with Σ_A [AS] = n[S] (every S node has n neighbours), on which the identity holds
    sing = Dict(:S => 0.7, :I => 0.2, :R => 0.1)
    pair = Dict((:S, :S) => 2.9, (:S, :I) => 0.9, (:S, :R) => 0.4, (:I, :I) => 0.5,
                (:I, :R) => 0.3, (:R, :R) => 0.2)
    @test pair[(:S, :S)] + pair[(:S, :I)] + pair[(:S, :R)] ≈ n * sing[:S]
    for A in (:S, :I, :R)
        @test triple_closure(A, :S, :I, pair, sing, net, BarnardClosure()) ≈
              barnard_ASI(A, sing, pair, n, ϕ)
        @test NodeBasedModels._infection_triple(:I, :S, A, pair, sing, net, BarnardClosure()) ≈
              barnard_ASI(A, sing, pair, n, ϕ)
    end
    # Σ_A [ASI] = (n − 1)[SI] (thesis Proposition 1), which the reversed order does not satisfy
    @test sum(triple_closure(A, :S, :I, pair, sing, net, BarnardClosure()) for A in (:S, :I, :R)) ≈
          (n - 1) * pair[(:S, :I)]
    @test !(sum(triple_closure(:I, :S, A, pair, sing, net, BarnardClosure()) for A in (:S, :I, :R)) ≈
            (n - 1) * pair[(:S, :I)])
end

@testset "Barnard's closure: thresholds and growth rates against the reference ODE" begin
    γ = 1.0
    for (n, ϕ) in ((6, 0.3), (4, 0.2)), dyn in (:SIR, :SIS)
        net = regular_network(n; ϕ)
        τc = epidemic_threshold(net, BarnardClosure(), γ; dynamics = dyn)
        for f in (1.25, 2.0)              # supercritical: an exponential phase with rate r
            r = early_growth_rate(net, BarnardClosure(), f * τc, γ; dynamics = dyn)
            T = min(100.0, 11.0 / r)
            @test barnard_slope(n, ϕ, f * τc, γ, dyn; t1 = 0.7T, t2 = T) ≈ r rtol = 1e-4
        end
        # subcritical: [I] decays at rate r; for SIR the decay also depends on the recovered-block
        # ratio [SR]/[R] the early epidemic left (r is the seeded asymptotic slope)
        r = early_growth_rate(net, BarnardClosure(), 0.8τc, γ; dynamics = dyn)
        @test barnard_slope(n, ϕ, 0.8τc, γ, dyn; t1 = 20.0, t2 = 30.0) ≈ r atol = (dyn === :SIR ? 5e-4 : 1e-5)
        @test barnard_slope(n, ϕ, 0.99τc, γ, dyn; t1 = 60.0, t2 = 100.0) < 0 <
              barnard_slope(n, ϕ, 1.01τc, γ, dyn; t1 = 60.0, t2 = 100.0)
    end
    # SIR, n = 6, ϕ = 0.3: τ_c = 0.299485 (Keeling's closure: 0.319169). Near the threshold the
    # slope converges slowly (the recovered-block ratios relax like 1/t); over t ∈ [200, 400] it
    # has the sign of the fast-variable r at ±0.2% of τ_c.
    τc = epidemic_threshold(regular_network(6; ϕ = 0.3), BarnardClosure(), γ)
    @test τc ≈ 0.2994848055 rtol = 1e-8
    for τ in (0.998τc, 1.002τc)
        r = early_growth_rate(regular_network(6; ϕ = 0.3), BarnardClosure(), τ, γ)
        s = barnard_slope(6, 0.3, τ, γ, :SIR; t1 = 200.0, t2 = 400.0)
        @test sign(s) == sign(r) && abs(s - r) < 1e-4
    end
end

@testset "Barnard's closure, SIR below the threshold: the seeded decay rate" begin
    # Regression (review of WP24): below the threshold the ϕ-continued quasi-equilibrium is not
    # the state the ODE reaches from a seed. Where (n − 1)(1 − ϕ) < 1, [SI]/[I] → 0 and [I] decays
    # at γ (the continued branch gave −0.791 at n = 3, ϕ = 0.9, τ = 1); deep below the threshold
    # the frozen [SR]/[R] matters (continued −0.6538 at n = 6, ϕ = 0.3, τ = 0.1).
    γ = 1.0
    for (n, ϕ, τ, cont, t1, t2, ε, tol) in ((3, 0.9, 1.0, -0.79104, 15.0, 25.0, 1e-13, 1e-4),
                                        (3, 0.6, 1.0, -0.62717, 30.0, 40.0, 1e-6, 5e-3),
                                        (6, 0.9, 0.3, -0.43366, 15.0, 25.0, 1e-13, 5e-4),
                                        (6, 0.3, 0.1, -0.65378, 20.0, 30.0, 1e-13, 5e-4))
        r = early_growth_rate(regular_network(n; ϕ), BarnardClosure(), τ, γ)
        s = barnard_slope(n, ϕ, τ, γ, :SIR; t1, t2, ε)
        @test abs(s - r) < tol
        @test abs(s - r) < abs(s - cont) / 5          # the old continued value is clearly off
        @test -γ <= r < 0
    end
    @test early_growth_rate(regular_network(3; ϕ = 0.9), BarnardClosure(), 1.0, γ) ≈ -γ atol = 1e-8
    # the sign still follows the threshold, and the SIS rate is unchanged (closed-form threshold)
    net = regular_network(6; ϕ = 0.3)
    τc = epidemic_threshold(net, BarnardClosure(), γ)
    @test early_growth_rate(net, BarnardClosure(), 0.999τc, γ) < 0 <
          early_growth_rate(net, BarnardClosure(), 1.001τc, γ)
end

@testset "Barnard's closure in generate_pairwise (triple order, infector last)" begin
    # Regression: generate_pairwise evaluated the infection triple as triple_closure(Z, X, other),
    # the infector first, which is not Barnard's closure (its SIR threshold was 0.3279 instead of
    # 0.2995). The builder now uses NodeBasedModels._infection_triple, [other X Z].
    n, ϕ, τ, γ = 6, 0.3, 0.45, 1.0
    net = regular_network(n; ϕ)
    for (dyn, model) in ((:SIR, CompartmentalModel(sir_model())), (:SIS, CompartmentalModel(sis_model())))
        psys = generate_pairwise(model, net, BarnardClosure())
        f, names, m = pairwise_field(psys)
        keys_ = dyn === :SIR ? (:S, :I, :R, :SS, :SI, :SR, :II, :IR, :RR) : (:S, :I, :SS, :SI, :II)
        x = dyn === :SIR ? [0.7, 0.2, 0.1, 2.3, 0.9, 0.4, 0.5, 0.3, 0.2] : [0.8, 0.2, 3.1, 1.0, 0.7]
        u = zeros(m)
        for (k, v) in zip(keys_, x)
            u[names[k]] = v
        end
        du_ref = similar(x)
        barnard_reference!(du_ref, x, (n, ϕ, τ, γ, dyn), 0.0)
        @test f(u, τ, γ)[[names[k] for k in keys_]] ≈ du_ref
        r = early_growth_rate(net, BarnardClosure(), τ, γ; dynamics = dyn)
        T = min(40.0, 11.0 / r)            # stay in the linear phase from the seed ε = 1e-13
        @test ode_slope(psys, τ, γ; t1 = 0.75T, t2 = T) ≈ r rtol = 1e-3
    end
end

@testset "Golden pairwise/sir_hom6_barnard_phi02 is Barnard's closure (thesis eq. 4.23)" begin
    # Justifies the regenerated golden (triple-order fix): its trajectory is the independent
    # reference ODE above from the same initial condition (n = 6, ϕ = 0.2, τ = 0.2, γ = 0.25).
    file = joinpath(@__DIR__, "..", "golden", "pairwise", "sir_hom6_barnard_phi02.trajectory.csv")
    lines = readlines(file)
    cols = split(lines[1], ',')
    rows = [parse.(Float64, split(l, ',')) for l in lines[2:end]]
    col(name) = [r[findfirst(==(name), cols)] for r in rows]
    t = col("t")
    order = ["[S]", "[I]", "[R]", "[S|S]", "[S|I]", "[S|R]", "[I|I]", "[I|R]", "[R|R]"]
    u0 = [col(c)[1] for c in order]
    sol = solve(ODEProblem(barnard_reference!, u0, (0.0, t[end]), (6, 0.2, 0.2, 0.25, :SIR));
                saveat = t, reltol = 1e-11, abstol = 1e-13)
    for (j, c) in enumerate(order)
        @test maximum(abs.(sol[j, :] .- col(c))) < 1e-8
    end
end

@testset "Model-based numeric analysis" begin
    γ = 0.25
    net = regular_network(4; ϕ = 0.2)
    for (dyn, m) in ((:SIR, sir_model()), (:SIS, sis_model()))
        @test epidemic_threshold(m, net; p = Dict(:γ => γ)) ==                      # default Keeling
              epidemic_threshold(net, KeelingClosure(), γ; dynamics = dyn)
        @test early_growth_rate(m, net, B; p = Dict(:τ => 0.2, :γ => γ)) ==
              early_growth_rate(net, B, 0.2, γ; dynamics = dyn)
        @test epidemic_threshold(CompartmentalModel(m), net, B; p = Dict(:γ => γ)) ==
              epidemic_threshold(net, B, γ; dynamics = dyn)
    end
    cn = ClusteredNetwork(RegularDegree(2), RegularDegree(2))          # k = 6, ϕ = 2/15
    @test epidemic_threshold(sir_model(), cn, KeelingClosure(); p = Dict(:γ => 1.0)) ==
          epidemic_threshold(regular_network(6; ϕ = 2 / 15), KeelingClosure(), 1.0)
    @test_throws ArgumentError epidemic_threshold(seir_model(), net, B; p = Dict(:γ => γ, :σ => 1.0))
    @test_throws ArgumentError epidemic_threshold(sir_model(), net, B; p = Dict(:τ => 0.1))  # no γ
    # a frequency-dependent ContactModel (β = τ⟨k⟩) on a network structure is converted with the
    # mean degree, as on a descriptor (0.2 before this fix: an error that pointed to node_based)
    fd = ContactModel(:sir_fd; contacts = [Contact(:S, :I, :I, :β)],
                      transitions = [NodeTransition(:I, :R, :γ)], convention = FrequencyDependent())
    for nt in (regular_network(4), regular_network(4; ϕ = 0.2),
               degree_distribution_network([0, 0, 0.5, 0, 0, 0, 0, 0, 0.5]))
        cl = default_closure(nt)
        kw = nt isa HeterogeneousNetwork ? (; dynamics = :SIR) : (;)
        @test early_growth_rate(fd, nt, cl; p = Dict(:β => 0.8, :γ => γ)) ≈
              early_growth_rate(nt, cl, 0.8 / mean_degree(nt), γ; kw...)
        @test epidemic_threshold(fd, nt, cl; p = Dict(:γ => γ)) == epidemic_threshold(nt, cl, γ; kw...)
    end
    @test early_growth_rate(fd, ConfigurationNetwork(RegularDegree(4)), B; p = Dict(:β => 0.8, :γ => γ)) ≈
          early_growth_rate(fd, regular_network(4), B; p = Dict(:β => 0.8, :γ => γ))
end

# ─── Exact stochastic simulation (NetworkOutbreaks) ───────────────────────────────
# Fresh random regular graph per run; graphs and runs use NetworkOutbreaks.stable_rng streams
# (DESIGN §J.7); runs are conditioned on major outbreaks where that is the question.

function ssa_runs(model, p, k, N; initial, tspan, nruns, base, observe)
    om = NO.OutbreakModel(model, p)
    out = Vector{Any}(undef, nruns)
    for run in 1:nruns
        g = random_regular_graph(N, k; rng = NO.stable_rng(base + run))
        spec = OutbreakSpec(model = om, network = g, initial = initial, tspan = tspan)
        traj = simulate(spec; seed = base + 2^32 + run, algorithm = NextReaction())
        out[run] = observe(om, traj)
    end
    return out
end

@testset "SIR threshold γ/(k − 2) on 4-regular graphs (NetworkOutbreaks)" begin
    k, N, γ, nruns = 4, 5000, 1.0, 24
    @test epidemic_threshold(regular_network(k), B, γ) == 0.5
    initial = SeedFraction(:I => 0.002)                               # 10 seeds
    fs(om, traj) = NO.final_size(traj)
    below = ssa_runs(sir_model(), Dict(:τ => 0.4, :γ => γ), k, N; initial, tspan = (0.0, 200.0),
                     nruns, base = 11_000, observe = fs)
    above = ssa_runs(sir_model(), Dict(:τ => 0.6, :γ => γ), k, N; initial, tspan = (0.0, 200.0),
                     nruns, base = 12_000, observe = fs)
    major = [f > 0.05 for f in above]
    @info "SIR k = 4, N = $N, $nruns runs each" below_mean = mean(below) below_max = maximum(below) above_major = count(major) above_mean_major = mean(above[major])
    @test maximum(below) < 0.1 && mean(below) < 0.03   # subcritical ((k − 1)T = 0.857): no major outbreak
    @test count(major) >= 0.8nruns                # supercritical ((k − 1)T = 1.125)
    # final size of the major outbreaks against the pairwise model (exact for SIR on regular
    # configuration networks), which starts from the same seed fraction
    psys = generate_pairwise(CompartmentalModel(sir_model()), regular_network(k), B;
                             tspan = (0.0, 200.0), seed_fraction = 0.002)
    Rinf = last(compartment(psys, solve_pairwise(psys, Dict(:τ => 0.6, :γ => γ); saveat = 200.0), :R))
    se = std(above[major]) / sqrt(count(major))
    @test abs(mean(above[major]) - Rinf) < 3se + 0.01
end

@testset "SIS threshold γ/(k − 1) on 3-regular graphs (NetworkOutbreaks)" begin
    k, N, γ, nruns = 3, 2000, 0.25, 12
    τc = epidemic_threshold(regular_network(k), B, γ; dynamics = :SIS)
    @test τc == 0.125
    initial = SeedFraction(:I => 0.05)
    prev(om, traj) = mean(state_at(traj, t)[om.index_of[:I]] for t in 150.0:5.0:200.0) / N
    below = ssa_runs(sis_model(), Dict(:τ => 0.64τc, :γ => γ), k, N; initial, tspan = (0.0, 200.0),
                     nruns, base = 21_000, observe = prev)
    above = ssa_runs(sis_model(), Dict(:τ => 2τc, :γ => γ), k, N; initial, tspan = (0.0, 200.0),
                     nruns, base = 22_000, observe = prev)
    @info "SIS k = 3, N = $N, $nruns runs each" below = maximum(below) above_mean = mean(above) above_min = minimum(above)
    @test all(==(0), below)                       # extinct well before t = 150 below τ_c
    @test minimum(above) > 0.1                    # endemic at 2τ_c
    # the pairwise endemic prevalence is an approximation for SIS; it is within 0.05 here
    psys = generate_pairwise(CompartmentalModel(sis_model()), regular_network(k), B;
                             tspan = (0.0, 400.0), seed_fraction = 0.05)
    Istar = last(compartment(psys, solve_pairwise(psys, Dict(:τ => 2τc, :γ => γ); saveat = 400.0), :I))
    @test abs(mean(above) - Istar) < 0.05
end
