# adopt.jl — NodeBasedModels adopts NetworkEpiCore (WP15; DESIGN_NetworkEpiCore.md §A.4, §A.6,
# §A.7, §E.2, §J.2, §J.6; verified issue B04).
#
# - the NetworkEpiCore bindings are re-exported, not redefined;
# - converters CompartmentalModel ⇄ ContactModel and descriptor → network structure;
# - Transition.via: per-infector rates (B04), `via = []` is the 0.1 behaviour;
# - seeding: the default seeds the entry state (E for SEIR), and the initial condition is the
#   π^PW image of the edge-based one; the replaced SEIR goldens are justified by the
#   edge-based comparison on a regular network (exact: Poisson-type closure, M6/M8);
# - node_based: levels, default closures, descriptors, rate conventions, scenarios;
# - GraphNetwork → ExplicitGraph and B04 (a); deprecations and removals.
#
# The edge-based reference solutions below are written out independently from Miller & Volz
# (2013) / DESIGN §D.4 (per-reaction EB field), not taken from EdgeBasedModels.

using NodeBasedModels
using NetworkEpiCore
using Test
using ModelingToolkit
using OrdinaryDiffEqDefault
using Graphs
using Catalyst
using Symbolics
import NetworkOutbreaks

const TOL = (reltol = 1e-11, abstol = 1e-13)

# Solve an edge-based vector field f!(du, u, p, t) written by hand.
function eb_solve(f!, u0, tspan, saveat)
    prob = ODEProblem(f!, u0, tspan)
    return OrdinaryDiffEqDefault.solve(prob; saveat, TOL...)
end

pw_solve(psys, p, saveat) = solve_pairwise(psys, p; saveat, TOL...)

@testset "NetworkEpiCore bindings are re-exported, not redefined" begin
    for n in names(NodeBasedModels)
        isdefined(NetworkEpiCore, n) || continue
        @test getfield(NodeBasedModels, n) === getfield(NetworkEpiCore, n)
    end
    for n in (:sir_model, :seir_model, :sis_model, :sirs_model, :mean_degree, :compartment,
              :basic_reproduction_number, :solve_epidemic, :model_curves,
              :with_reinfection_counting, :reinfection_totals, :base_compartment_of,
              :infection_count_of, :default_initial_conditions, :contact_model)
        @test n in names(NodeBasedModels)
        @test getfield(NodeBasedModels, n) === getfield(NetworkEpiCore, n)
    end
    # the 0.1 name parsers now come from NetworkEpiCore
    @test base_compartment_of(:S_3) == :S
    @test infection_count_of(:I_2) == 2
    @test sir_model() isa ContactModel
end

@testset "Converters: CompartmentalModel ⇄ ContactModel" begin
    for cm in (sir_model(), seir_model(), sis_model(), sirs_model(), sirv_model(),
               seair_model(), twostrain_model(), with_reinfection_counting(sis_model(), 3))
        m = CompartmentalModel(cm)
        @test m.compartment_names == species_names(cm)
        @test Set(m.infectious_compartments) == Set(infectious_species(cm))
        @test isequivalent(contact_model(m), cm)
    end
    # SEAIR: two infectors at different rates keep their own `via`; branching rates are Exprs
    m = CompartmentalModel(seair_model())
    inf = [t for t in m.transitions if t.type === :infection]
    @test [(t.from, t.to, t.rate, t.via) for t in inf] == [(:S, :E, :τI, [:I]), (:S, :E, :τA, [:A])]
    @test any(t -> t.rate == :(p * σ), m.transitions)
    # one infector per model: via is empty (the 0.1 semantics)
    @test only(CompartmentalModel(sir_model()).transitions[1:1]).via == Symbol[]
    # removal X → ∅ goes to the absorbing compartment :removed (DESIGN §J.2)
    rm = ContactModel(:sird; contacts = [Contact(:S, :I, :I, :τ)],
                      transitions = [NodeTransition(:I, :R, :γ), NodeTransition(:I, nothing, :μ)])
    m = CompartmentalModel(rm)
    @test m.compartment_names == [:S, :I, :R, :removed]
    @test (last(m.transitions).from, last(m.transitions).to, last(m.transitions).rate) ==
          (:I, NodeBasedModels.REMOVED_COMPARTMENT, :μ)
    @test !only(c for c in m.compartments if c.name === :removed).infectious
    # a 0.1 model: one contact per infection transition and infector (empty via = all)
    two = CompartmentalModel(
        [Compartment(:S), Compartment(:I1; infectious = true), Compartment(:I2; infectious = true),
         Compartment(:R)],
        [Transition(:S, :I1, :τ, :infection), Transition(:I1, :I2, :σ, :spontaneous),
         Transition(:I2, :R, :γ, :spontaneous)]; name = :SI1I2R)
    cm2 = contact_model(two)
    @test Set((c.infector, c.rate) for c in contacts(cm2)) == Set([(:I1, :τ), (:I2, :τ)])
    back = CompartmentalModel(cm2)
    @test only(t for t in back.transitions if t.type === :infection).via == Symbol[]
    # the same source, target and infectors twice add their rates (0.1 pairwise semantics)
    dup = CompartmentalModel([Compartment(:S), Compartment(:I; infectious = true)],
                             [Transition(:S, :I, :a, :infection), Transition(:S, :I, :b, :infection),
                              Transition(:I, :S, :γ, :spontaneous)])
    @test only(contacts(contact_model(dup))).rate == :(a + b)
    # validation
    @test_throws ArgumentError Transition(:S, :I, :τ, :spontaneous; via = [:I])
    @test_throws ArgumentError Transition(:S, :I, :τ, :recovery)
    @test_throws ArgumentError Transition(:S, :I, -1.0, :infection)
    @test_throws ArgumentError Transition(:S, :I, :τ, :infection; via = [:I, :I])
    @test_throws ArgumentError CompartmentalModel(
        [Compartment(:S), Compartment(:I; infectious = true), Compartment(:R)],
        [Transition(:S, :I, :τ, :infection; via = [:R])])
    # an infection whose source is one of its infectors is not a network contact
    sup = CompartmentalModel([Compartment(:I1; infectious = true), Compartment(:I2)],
                             [Transition(:I1, :I2, :τ, :infection)])
    @test_throws ArgumentError contact_model(sup)
    @test_throws ArgumentError CompartmentalModel(
        contact_model(@reaction_network begin; β/N, S + I --> E + I; σ, E --> I; end;
                      rates = :frequency, population = :N))
end

@testset "Network structures from descriptors" begin
    h = HomogeneousNetwork(ConfigurationNetwork(RegularDegree(4)))
    @test h.n == 4 && h.ϕ == 0.0
    @test HomogeneousNetwork(ConfigurationNetwork(EmpiricalDegree(Dict(5 => 1.0)))).n == 5
    @test_throws ArgumentError HomogeneousNetwork(ConfigurationNetwork(PoissonDegree(5)))
    het = HeterogeneousNetwork(ConfigurationNetwork(PoissonDegree(5)))
    @test het.mean_degree == 5.0 && het.second_moment == 30.0 && het.excess_degree == 5.0
    @test het.excess_degree / het.mean_degree == closure_constant(PoissonDegree(5))   # K = 1
    @test abs(sum(het.degree_probs) - 1) < 1e-11
    nb = HeterogeneousNetwork(ConfigurationNetwork(NegBinDegree(; mean = 4, var = 8)))
    @test nb.excess_degree / nb.mean_degree ≈ 5 / 4
    bim = HeterogeneousNetwork(ConfigurationNetwork(EmpiricalDegree(Dict(2 => 5/6, 10 => 1/6))))
    @test bim.degree_probs[3] ≈ 5/6 && bim.excess_degree ≈ 5.0
    # clustered: constant degree 2 + 2·2 = 6 with Keeling ϕ = clustering coefficient 2/15
    c = HomogeneousNetwork(ClusteredNetwork(RegularDegree(2), RegularDegree(2)))
    @test c.n == 6 && c.ϕ ≈ 2 / 15
    cp = HeterogeneousNetwork(ClusteredNetwork(PoissonDegree(1), PoissonDegree(2)))
    @test cp.mean_degree ≈ 5.0
    @test cp.ϕ ≈ clustering_coefficient(ClusteredNetwork(PoissonDegree(1), PoissonDegree(2)))
    @test sum(k * cp.degree_probs[k + 1] for k in 0:cp.max_degree) ≈ 5.0 atol = 1e-9
    @test clustering_coefficient(regular_network(6; ϕ = 0.2)) == 0.2 == clustering(regular_network(6; ϕ = 0.2))
    @test network_structure(ConfigurationNetwork(RegularDegree(3))) isa HomogeneousNetwork
    @test network_structure(ConfigurationNetwork(PoissonDegree(2))) isa HeterogeneousNetwork
    @test_throws ArgumentError network_structure(WellMixed(5))
end

# ─── B04: per-infector rates via Transition.via ─────────────────────────────────
@testset "B04: per-infector rates (Miller–Volz multi-stage EBCM, exact on k-regular)" begin
    comps = [Compartment(:S), Compartment(:I1; infectious = true),
             Compartment(:I2; infectious = true), Compartment(:R)]
    cm = CompartmentalModel(comps, [Transition(:S, :I1, :τ1, :infection; via = [:I1]),
                                    Transition(:S, :I1, :τ2, :infection; via = [:I2]),
                                    Transition(:I1, :I2, :γ1, :spontaneous),
                                    Transition(:I2, :R, :γ2, :spontaneous)])
    p = Dict(:τ1 => 0.6, :τ2 => 0.1, :γ1 => 1.0, :γ2 => 1.0)
    ps = generate_pairwise(cm, regular_network(5), BernoulliClosure(); tspan = (0.0, 40.0),
                           seed_fraction = 0.01)
    # d[S]/dt = −τ1[S I1] − τ2[S I2] (0.1 gave −(τ1 + τ2)([S I1] + [S I2]))
    eqS = only(eq for eq in ps.metadata[:equations]
               if isequal(eq.lhs, Differential(ModelingToolkit.get_iv(ps.system))(ps.singles[:S])))
    τ1, τ2 = ps.metadata[:parameters][:τ1], ps.metadata[:parameters][:τ2]
    SI1, SI2 = ps.pairs[(:S, :I1)], ps.pairs[(:S, :I2)]
    @test isequal(Symbolics.value(Symbolics.simplify(eqS.rhs + τ1 * SI1 + τ2 * SI2)), 0)
    sol = pw_solve(ps, p, 0.0:0.05:40.0)
    R = compartment(ps, sol, :R)
    I = compartment(ps, sol, :I1) .+ compartment(ps, sol, :I2)
    @test R[end] ≈ 0.8938 atol = 5e-4          # the 0.1 encoding gave 0.9943
    @test maximum(I) ≈ 0.3562 atol = 5e-4      # the 0.1 encoding gave 0.6066
    # the edge-based model (ψ = x⁵), written out: S = qψ(θ)
    k, q = 5, 0.99
    function eb!(du, u, _, t)
        θ, φ1, φ2, P1, P2, PR = u
        flux = p[:τ1] * φ1 + p[:τ2] * φ2
        du[1] = -flux
        du[2] = -p[:τ1] * φ1 - p[:γ1] * φ1 + flux * q * k * (k - 1) * θ^(k - 2) / k
        du[3] = -p[:τ2] * φ2 + p[:γ1] * φ1 - p[:γ2] * φ2
        du[4] = flux * q * k * θ^(k - 1) - p[:γ1] * P1
        du[5] = p[:γ1] * P1 - p[:γ2] * P2
        du[6] = p[:γ2] * P2
    end
    eb = eb_solve(eb!, [1.0, 0.01, 0.0, 0.01, 0.0, 0.0], (0.0, 40.0), 0.0:0.05:40.0)
    @test maximum(abs.(compartment(ps, sol, :S) .- q .* eb[1, :] .^ k)) < 1e-8
    @test maximum(abs.(R .- eb[6, :])) < 1e-8
    @test maximum(abs.(compartment(ps, sol, :I1) .- eb[4, :])) < 1e-8
    # the same model from Catalyst: the catalysts become the infectors
    rn = @reaction_network begin
        τ1, S + I1 --> 2I1
        τ2, S + I2 --> I1 + I2
        γ1, I1 --> I2
        γ2, I2 --> R
    end
    fromrn = CompartmentalModel(contact_model(rn))
    @test Set((t.rate, t.via) for t in fromrn.transitions if t.type === :infection) ==
          Set([(:τ1, [:I1]), (:τ2, [:I2])])
    nb = node_based(rn, ConfigurationNetwork(RegularDegree(5)); tspan = (0.0, 40.0),
                    seed_fraction = 0.01)
    solnb = solve_epidemic(nb; p, saveat = 0.0:0.05:40.0, TOL...)
    @test maximum(abs.(compartment(nb, solnb, :R) .- R)) < 1e-9
    # 0.1 conversion: warns that the old model summed every rate over every infector
    m = @test_logs((:warn, r"deprecated"),
                   (:warn, r"every infection rate to every infectious compartment"),
                   match_mode = :any, model_from_catalyst(rn))
    @test Set(t.via for t in m.transitions if t.type === :infection) == Set([[:I1], [:I2]])
end

@testset "via = [] is the 0.1 behaviour" begin
    mk(via) = CompartmentalModel(
        [Compartment(:S), Compartment(:I1; infectious = true), Compartment(:I2; infectious = true),
         Compartment(:R)],
        [Transition(:S, :I1, :τ, :infection; via), Transition(:I1, :I2, :σ, :spontaneous),
         Transition(:I2, :R, :γ, :spontaneous)])
    p = Dict(:τ => 0.3, :σ => 0.4, :γ => 0.4)
    a = generate_pairwise(mk(Symbol[]), regular_network(4), BernoulliClosure(); tspan = (0.0, 40.0))
    b = generate_pairwise(mk([:I1, :I2]), regular_network(4), BernoulliClosure(); tspan = (0.0, 40.0))
    sa, sb = pw_solve(a, p, 0:1:40), pw_solve(b, p, 0:1:40)
    for X in (:S, :I1, :I2, :R)
        @test compartment(a, sa, X) == compartment(b, sb, X)
    end
    # the 0.1 semantics, independently: both stages infect at τ, so on a k-regular network
    # (Bernoulli K = (k − 1)/k, exact) the model is the edge-based model with transmission flux
    # τ(φ₁ + φ₂) (Miller & Volz 2013; written out here, ψ = x⁴, seed ε = 1e-3 in I1). The
    # 0.1 numbers themselves are the si1i2r_* goldens (via = [], unchanged).
    k, q = 4, 1 - 1e-3
    function eb!(du, u, _, t)
        θ, φ1, φ2, P1, P2, PR = u
        flux = p[:τ] * (φ1 + φ2)
        du[1] = -flux
        du[2] = -p[:τ] * φ1 - p[:σ] * φ1 + flux * q * k * (k - 1) * θ^(k - 2) / k
        du[3] = -p[:τ] * φ2 + p[:σ] * φ1 - p[:γ] * φ2
        du[4] = flux * q * k * θ^(k - 1) - p[:σ] * P1
        du[5] = p[:σ] * P1 - p[:γ] * P2
        du[6] = p[:γ] * P2
    end
    eb = eb_solve(eb!, [1.0, 1e-3, 0.0, 1e-3, 0.0, 0.0], (0.0, 40.0), 0:1:40)
    @test maximum(abs.(compartment(a, sa, :S) .- q .* eb[1, :] .^ k)) < 1e-8
    @test maximum(abs.(compartment(a, sa, :I2) .- eb[5, :])) < 1e-8
    @test maximum(abs.(compartment(a, sa, :R) .- eb[6, :])) < 1e-8
    # restricting to one infector changes the dynamics
    c = generate_pairwise(mk([:I1]), regular_network(4), BernoulliClosure(); tspan = (0.0, 40.0))
    @test compartment(c, pw_solve(c, p, 0:1:40), :R)[end] < compartment(a, sa, :R)[end] - 0.01
end

# ─── Seeding: the entry state, and the π^PW image of the EB initial condition ────
# π^PW(θ = 1, ξ = 1, φ_X = pop_X = ρ_X) = ([s] = qψ(1), [sX] = qψ'(1)ρ_X, [ss] = q²ψ'(1),
# [X] = ρ_X); non-S pairs [XY] = ⟨k⟩ρ_Xρ_Y; ordered pairs, Σ_{X,Y}[XY] = ⟨k⟩ (DESIGN §E.2).
function pi_pw_check(psys, d::DegreeDistribution, ρ::Dict{Symbol,Float64}, S::Symbol; N = 1.0)
    q = 1 - sum(values(ρ))
    ψ1, dψ1 = pgf(d, 1.0), pgf_derivative(d, 1.0, 1)
    x = merge(Dict(S => q), ρ)
    names = collect(keys(psys.singles))
    u0 = psys.u0
    ok = isapprox(u0[psys.singles[S]], N * q * ψ1; rtol = 1e-14)
    for X in names
        ok &= isapprox(u0[psys.singles[X]], N * get(x, X, 0.0); rtol = 1e-14, atol = 1e-300)
    end
    for ((a, b), v) in psys.pairs
        expected = a === S && b === S ? q^2 * dψ1 :
                   a === S ? q * dψ1 * get(ρ, b, 0.0) : b === S ? q * dψ1 * get(ρ, a, 0.0) :
                   dψ1 * get(x, a, 0.0) * get(x, b, 0.0)
        ok &= isapprox(u0[v], N * expected; rtol = 1e-12, atol = 1e-300)
    end
    total = sum(u0[v] * (a === b ? 1 : 2) for ((a, b), v) in psys.pairs)   # ordered pairs
    ok &= isapprox(total, N * dψ1; rtol = 1e-12)
    return ok
end

@testset "SEIR seeding: the entry state E, initial condition = π^PW(EB)" begin
    ρ = Dict(:E => 0.01)
    for (net, d) in ((regular_network(4), RegularDegree(4)),
                     (regular_network(6; ϕ = 0.2), RegularDegree(6)),
                     (degree_distribution_network([k in (3, 9) ? 0.5 : 0.0 for k in 0:9]),
                      EmpiricalDegree(Dict(3 => 0.5, 9 => 0.5))))
        psys = generate_pairwise(seir_model(), net, BernoulliClosure(); seed_fraction = 0.01)
        @test psys.u0[psys.singles[:I]] == 0.0
        @test pi_pw_check(psys, d, ρ, :S)
        # the 0.1 default is still available
        old = generate_pairwise(seir_model(), net, BernoulliClosure(); seed_fraction = 0.01,
                                seed_state = :first_infectious)
        @test old.u0[old.singles[:I]] == 0.01 && old.u0[old.singles[:E]] == 0.0
        @test pi_pw_check(old, d, Dict(:I => 0.01), :S)
    end
    # the legacy CompartmentalModel path seeds the entry state too
    lm = CompartmentalModel(seir_model())
    ps = generate_pairwise(lm, regular_network(4), BernoulliClosure(); seed_fraction = 0.02, N = 100.0)
    @test pi_pw_check(ps, RegularDegree(4), Dict(:E => 0.02), :S; N = 100.0)
    # a descriptor, several seeded compartments
    nb = node_based(seir_model(), ConfigurationNetwork(PoissonDegree(5));
                    initial = SeedFraction(:E => 0.01, :I => 0.005, :R => 0.1))
    @test pi_pw_check(nb, PoissonDegree(5), Dict(:E => 0.01, :I => 0.005, :R => 0.1), :S)
    # the accumulator counts the infected seeds E and I, not the recovered R (DESIGN §J.8)
    @test nb.u0[nb.metadata[:cumulative]] ≈ 0.015
    # counts on an integer population scale
    ns = node_based(seir_model(), ConfigurationNetwork(RegularDegree(4)); N = 1000,
                    initial = SeedCount(:E => 10))
    @test ns.u0[ns.singles[:E]] ≈ 10.0 && ns.u0[ns.singles[:S]] ≈ 990.0
    @test_throws ArgumentError node_based(seir_model(), ConfigurationNetwork(RegularDegree(4));
                                          initial = SeedCount(:E => 10))    # N = 1 is not a count
    @test_throws ArgumentError node_based(seir_model(), ConfigurationNetwork(RegularDegree(4));
                                          initial = SeedFraction(:X => 0.1))
    # two entry states: an explicit initial is required (DESIGN §E.2)
    @test_throws ArgumentError node_based(twostrain_model(), ConfigurationNetwork(PoissonDegree(5)))
    two = node_based(twostrain_model(), ConfigurationNetwork(PoissonDegree(5));
                     initial = SeedFraction(:I1 => 0.005, :I2 => 0.005))
    @test pi_pw_check(two, PoissonDegree(5), Dict(:I1 => 0.005, :I2 => 0.005), :S)
    # reinfection counting: background S₀, entry I₁ (reached from S₀)
    lifted = with_reinfection_counting(sis_model(), 3)
    rl = node_based(lifted, ConfigurationNetwork(RegularDegree(4)); seed_fraction = 0.01)
    @test rl.metadata[:background] === :S_0
    @test pi_pw_check(rl, RegularDegree(4), Dict(:I_1 => 0.01), :S_0)
    # default_initial_conditions: the stored u0, or a new seeding
    @test default_initial_conditions(nb) === nb.u0
    u = default_initial_conditions(two; initial = SeedFraction(:I1 => 0.02, :I2 => 0.0))
    @test u[two.singles[:I1]] ≈ 0.02 && u[two.singles[:S]] ≈ 0.98
    # … whose default seed state is the one the system was built with
    s1 = node_based(seir_model(), ConfigurationNetwork(RegularDegree(4));
                    seed_state = :first_infectious)
    @test s1.metadata[:seed_state] === :first_infectious
    u1 = default_initial_conditions(s1; seed_fraction = 0.02)
    @test u1[s1.singles[:I]] ≈ 0.02 && u1[s1.singles[:E]] == 0.0
    @test u1[s1.metadata[:cumulative]] ≈ 0.02
    u2 = default_initial_conditions(s1; seed_fraction = 0.02, seed_state = :entry)
    @test u2[s1.singles[:E]] ≈ 0.02 && u2[s1.singles[:I]] == 0.0
end

# ─── The cumulative accumulator: infected seeds and entries into infection (DESIGN §J.8) ─────
# Infection status is structural and decided from the typing, as for NetworkOutbreaks'
# final_size; seeds in R or V are not infections, and a contact recipient that is itself
# infected (superinfection) is not susceptible.

# The accumulator along a solution, and the identity it must satisfy.
function cumulative_residual(psys, sol, expected)
    c = sol[psys.metadata[:cumulative]]
    return maximum(abs.(c .- expected))
end

@testset "cumulative: superinfection seeds count (regression)" begin
    # S + I1 → I1, S + I2 → I2, I1 + I2 → I12 + I2: I1 is a contact recipient but infected. The
    # 0.1-style rule (every source of an infection transition is susceptible) dropped the I1
    # seeds: cumulative(0) was 0.005, not 0.01.
    sup = ContactModel(:superinf;
                       contacts = [Contact(:S, :I1, :I1, :τ1), Contact(:S, :I2, :I2, :τ2),
                                   Contact(:I1, :I2, :I12, :τs)],
                       transitions = [NodeTransition(:I1, :R, :γ), NodeTransition(:I2, :R, :γ),
                                      NodeTransition(:I12, :R, :γ)])
    @test susceptible_species(sup) == [:S]
    @test :I1 in CompartmentalModel(sup).susceptible_compartments   # the lowering's source list
    nb = node_based(sup, ConfigurationNetwork(PoissonDegree(5));
                    initial = SeedFraction(:I1 => 0.005, :I2 => 0.005))
    @test nb.metadata[:infected] == [:I1, :I2]
    @test nb.u0[nb.metadata[:cumulative]] ≈ 0.01
    p = Dict(:τ1 => 0.3, :τ2 => 0.2, :τs => 0.4, :γ => 0.25)
    sol = solve_epidemic(nb; p, tspan = (0.0, 80.0), saveat = 0:1:80, TOL...)
    # every node that leaves S enters I1 or I2, so the fraction ever infected is 1 − S
    @test cumulative_residual(nb, sol, 1 .- compartment(nb, sol, :S)) < 1e-9
    mc = model_curves(nb, sol; t = 0:1:80)
    @test mc[:cumulative][end] ≈ 1 - mc[:S][end] atol = 1e-9
    # a new seeding keeps the rule
    u = default_initial_conditions(nb; initial = SeedFraction(:I1 => 0.02))
    @test u[nb.metadata[:cumulative]] ≈ 0.02
end

@testset "cumulative: seeded R and V are not infected; importation counts" begin
    # SEIR with R seeded: cumulative = E + I seeds + ∫τ[SI] = 1 − S − ρ_R
    nb = node_based(seir_model(), ConfigurationNetwork(PoissonDegree(5));
                    initial = SeedFraction(:E => 0.01, :I => 0.005, :R => 0.1))
    sol = solve_epidemic(nb; p = Dict(:τ => 1 / 6, :σ => 1 / 5, :γ => 1 / 4),
                         tspan = (0.0, 150.0), saveat = 0:1:150, TOL...)
    @test cumulative_residual(nb, sol, 0.9 .- compartment(nb, sol, :S)) < 1e-9
    # SIRV with V pre-vaccinated: V is not infected, and S → V is not an infection, so
    # cumulative = ρ_I + ∫τ[SI] = 1 − S − V
    v = node_based(sirv_model(), ConfigurationNetwork(RegularDegree(4));
                   initial = SeedFraction(:I => 0.01, :V => 0.3))
    @test v.metadata[:infected] == [:I]
    @test v.u0[v.metadata[:cumulative]] ≈ 0.01
    solv = solve_epidemic(v; p = Dict(:τ => 0.4, :γ => 0.2, :ν => 0.02), tspan = (0.0, 60.0),
                          saveat = 0:1:60, TOL...)
    @test cumulative_residual(v, solv, 1 .- compartment(v, solv, :S) .- compartment(v, solv, :V)) <
          1e-9
    # importation S → E (a node transition into infection) is counted: cumulative = 1 − S
    imp = ContactModel(:seir_import; contacts = [Contact(:S, :I, :E, :τ)],
                       transitions = [NodeTransition(:S, :E, :η), NodeTransition(:E, :I, :σ),
                                      NodeTransition(:I, :R, :γ)])
    ni = node_based(imp, ConfigurationNetwork(RegularDegree(4)); seed_fraction = 0.01)
    @test ni.metadata[:infected] == [:E, :I]
    soli = solve_epidemic(ni; p = Dict(:τ => 0.3, :η => 0.01, :σ => 0.5, :γ => 0.25),
                          tspan = (0.0, 60.0), saveat = 0:1:60, TOL...)
    @test cumulative_residual(ni, soli, 1 .- compartment(ni, soli, :S)) < 1e-9
    # tracing S + I → Q + I is not an infection (Q never infects): cumulative = I + R
    trm = ContactModel(:sir_trace; contacts = [Contact(:S, :I, :I, :τ), Contact(:S, :I, :Q, :κ)],
                       transitions = [NodeTransition(:I, :R, :γ), NodeTransition(:Q, :S, :ω)])
    nt = node_based(trm, ConfigurationNetwork(RegularDegree(4));
                    initial = SeedFraction(:I => 0.01, :Q => 0.05))
    @test nt.metadata[:infected] == [:I]
    @test nt.u0[nt.metadata[:cumulative]] ≈ 0.01
    solt = solve_epidemic(nt; p = Dict(:τ => 0.3, :κ => 0.1, :γ => 0.25, :ω => 0.2),
                          tspan = (0.0, 60.0), saveat = 0:1:60, TOL...)
    @test cumulative_residual(nt, solt, compartment(nt, solt, :I) .+ compartment(nt, solt, :R)) <
          1e-9
    # the legacy CompartmentalModel path uses the same rule
    lg = generate_pairwise(CompartmentalModel(seir_model()), regular_network(4),
                           BernoulliClosure(); cumulative = true,
                           initial = SeedFraction(:E => 0.01, :R => 0.2))
    @test lg.metadata[:infected] == [:E, :I]
    @test lg.u0[lg.metadata[:cumulative]] ≈ 0.01
end

@testset "infected compartments: the typing rule, and NetworkOutbreaks' _infected_mask" begin
    sup = ContactModel(:superinf;
                       contacts = [Contact(:S, :I1, :I1, :τ1), Contact(:S, :I2, :I2, :τ2),
                                   Contact(:I1, :I2, :I12, :τs)],
                       transitions = [NodeTransition(:I1, :R, :γ), NodeTransition(:I2, :R, :γ),
                                      NodeTransition(:I12, :R, :γ)])
    imp = ContactModel(:seir_import; contacts = [Contact(:S, :I, :E, :τ)],
                       transitions = [NodeTransition(:S, :E, :η), NodeTransition(:E, :I, :σ),
                                      NodeTransition(:I, :R, :γ)])
    trm = ContactModel(:sir_trace; contacts = [Contact(:S, :I, :I, :τ), Contact(:S, :I, :Q, :κ)],
                       transitions = [NodeTransition(:I, :R, :γ), NodeTransition(:Q, :S, :ω)])
    rem = ContactModel(:sir_removal; contacts = [Contact(:S, :I, :I, :τ)],
                       transitions = [NodeTransition(:I, nothing, :γ), NodeTransition(:S, :V, :ν)])
    # quarantine of latents: E + I → Eq + I, Eq → I (E and Eq are infected; NetworkOutbreaks
    # represents this contact as :contact_trace)
    qm = ContactModel(:seir_q; contacts = [Contact(:S, :I, :E, :τ), Contact(:E, :I, :Eq, :κ)],
                      transitions = [NodeTransition(:E, :I, :σ), NodeTransition(:Eq, :I, :σ),
                                     NodeTransition(:I, :R, :γ)])
    expected = Dict(:sir => [:I], :seir => [:E, :I], :sis => [:I], :sirs => [:I],
                    :sirv => [:I], :seair => [:E, :I, :A], :twostrain => [:I1, :I2],
                    :sis_reinf_L3 => [:I_1, :I_2, :I_3], :superinf => [:I1, :I2],
                    :seir_import => [:E, :I], :sir_trace => [:I], :sir_removal => [:I],
                    :seir_q => [:E, :I, :Eq])
    NO = NetworkOutbreaks
    for cm in (sir_model(), seir_model(), sis_model(), sirs_model(), sirv_model(), seair_model(),
               twostrain_model(), with_reinfection_counting(sis_model(), 3), sup, imp, trm, rem,
               qm)
        m = CompartmentalModel(cm)
        inf = NodeBasedModels._infected_compartments(cm)
        @test inf == expected[nameof(cm)]
        @test inf == NetworkEpiCore.infected_species(cm)            # delegates to the core rule
        @test NodeBasedModels._infected_compartments(m) == inf       # same on the lowering
        trace = nameof(cm) === :seir_q ? [:E] : Symbol[]
        om = NO.OutbreakModel(m.compartment_names, m.infectious_compartments,
                              [NO.OutbreakTransition(t.from, t.to, 1.0,
                                                     t.type === :infection && t.from in trace ?
                                                     :contact_trace : t.type; via = t.via)
                               for t in m.transitions])
        @test m.compartment_names[NO._infected_mask(om)] == inf
    end
end

@testset "SEIR goldens: the entry-seeded pairwise model is the edge-based model (k-regular)" begin
    # Bernoulli closure on a k-regular network: K = (k − 1)/k = K_ψ, so the pairwise node
    # observables equal the Miller–Volz edge-based model (KKR 2023; DESIGN §D.5 M6, M8).
    # Golden seir_hom4_bernoulli: τ = 0.3, σ = 0.5, γ = 0.2, ρ = 0.01, t ∈ [0, 40].
    p = Dict(:τ => 0.3, :σ => 0.5, :γ => 0.2)
    k = 4
    function seir_eb(seedX)
        q = 0.99
        function f!(du, u, _, t)
            θ, φE, φI, PE, PI, PR = u
            du[1] = -p[:τ] * φI
            du[2] = -p[:σ] * φE + p[:τ] * φI * q * (k - 1) * θ^(k - 2)       # qψ''(θ)/ψ'(1)
            du[3] = -p[:τ] * φI + p[:σ] * φE - p[:γ] * φI
            du[4] = p[:τ] * φI * q * k * θ^(k - 1) - p[:σ] * PE
            du[5] = p[:σ] * PE - p[:γ] * PI
            du[6] = p[:γ] * PI
        end
        u0 = seedX === :E ? [1.0, 0.01, 0.0, 0.01, 0.0, 0.0] : [1.0, 0.0, 0.01, 0.0, 0.01, 0.0]
        return q, eb_solve(f!, u0, (0.0, 40.0), 0.0:1.0:40.0)
    end
    for (state, seedX) in ((:entry, :E), (:first_infectious, :I))
        psys = generate_pairwise(seir_model(), regular_network(k), BernoulliClosure();
                                 tspan = (0.0, 40.0), seed_fraction = 0.01, seed_state = state)
        sol = pw_solve(psys, p, 0.0:1.0:40.0)
        q, eb = seir_eb(seedX)
        @test maximum(abs.(compartment(psys, sol, :S) .- q .* eb[1, :] .^ k)) < 1e-8
        @test maximum(abs.(compartment(psys, sol, :E) .- eb[4, :])) < 1e-8
        @test maximum(abs.(compartment(psys, sol, :I) .- eb[5, :])) < 1e-8
        @test maximum(abs.(compartment(psys, sol, :R) .- eb[6, :])) < 1e-8
    end
    # the §E.2 seeding (E) and the 0.1 default (I) are different epidemics
    a = generate_pairwise(seir_model(), regular_network(k), BernoulliClosure(); tspan = (0.0, 40.0),
                          seed_fraction = 0.01)
    b = generate_pairwise(seir_model(), regular_network(k), BernoulliClosure(); tspan = (0.0, 40.0),
                          seed_fraction = 0.01, seed_state = :first_infectious)
    @test maximum(abs.(compartment(a, pw_solve(a, p, 0:1:40), :I) .-
                       compartment(b, pw_solve(b, p, 0:1:40), :I))) > 1e-2
end

@testset "SEAIR on Poisson(5): constant K = 1 is exact (two infectors, branching)" begin
    sc = scenario(:seair_pois5)
    sys = node_based(sc)
    @test sys.network isa HeterogeneousNetwork
    @test sys.network.excess_degree / sys.network.mean_degree == 1.0
    sol = solve_epidemic(sys, sc; TOL...)
    p = sc.params
    μ, q = 5.0, 0.99
    ψ(θ) = exp(μ * (θ - 1))
    function f!(du, u, _, t)
        θ, φE, φI, φA, PE, PI, PA, PR, C = u
        flux = p[:τI] * φI + p[:τA] * φA
        du[1] = -flux
        du[2] = -p[:σ] * φE + flux * q * μ * ψ(θ)                       # qψ''(θ)/ψ'(1) = qμψ(θ)
        du[3] = -p[:τI] * φI + p[:p] * p[:σ] * φE - p[:γ] * φI
        du[4] = -p[:τA] * φA + (1 - p[:p]) * p[:σ] * φE - p[:γ] * φA
        du[5] = flux * q * μ * ψ(θ) - p[:σ] * PE
        du[6] = p[:p] * p[:σ] * PE - p[:γ] * PI
        du[7] = (1 - p[:p]) * p[:σ] * PE - p[:γ] * PA
        du[8] = p[:γ] * (PI + PA)
        du[9] = flux * q * μ * ψ(θ)
    end
    eb = eb_solve(f!, [1.0, 0.01, 0, 0, 0.01, 0, 0, 0, 0.01], sc.tspan, sc.tgrid)
    mc = model_curves(sys, sol; t = sc.tgrid, label = "pairwise (K = 1)")
    @test mc isa ModelCurves && mc.representation === :pairwise
    @test Set(sc.observables) ⊆ Set(keys(mc.values))
    @test maximum(abs.(mc[:S] .- q .* ψ.(eb[1, :]))) < 1e-8
    for (X, row) in ((:E, 5), (:I, 6), (:A, 7), (:R, 8), (:cumulative, 9))
        @test maximum(abs.(mc[X] .- eb[row, :])) < 1e-8
    end
    @test mc[:infectious] ≈ mc[:I] .+ mc[:A]
    # the final size of the scenario (NetworkEpiCore fixed point, seeds included)
    @test abs(mc[:cumulative][end] - sc.expected[:final_size]) < 1e-4
end

@testset "node_based: levels, closures, descriptors" begin
    @test NODE_BASED_LEVELS == (:population, :s_anchored, :individual, :pair, :motif, :neighbourhood)
    @test default_closure(ConfigurationNetwork(PoissonDegree(5))) isa BernoulliClosure
    @test default_closure(ClusteredNetwork(RegularDegree(2), RegularDegree(2))) isa KeelingClosure
    @test default_closure(ExplicitGraph(cycle_graph(5))) isa KirkwoodClosure
    @test default_closure(regular_network(6; ϕ = 0.2)) isa KeelingClosure
    # WP23: a well-mixed population defaults to mass action (MeanFieldClosure)
    @test default_closure(WellMixed(5)) isa MeanFieldClosure
    @test_throws ArgumentError default_closure(DynamicNetwork(RegularDegree(6), NeighbourExchange(1.0)))
    @test_throws ArgumentError node_based(sir_model(), ConfigurationNetwork(PoissonDegree(5));
                                          level = :flat)
    @test_throws ArgumentError node_based(sir_model(), ExplicitGraph(cycle_graph(5)))
    # WP23: the S-anchored level exists (default closure Bernoulli on a configuration network)
    @test node_based(sir_model(), ConfigurationNetwork(PoissonDegree(5));
                     level = :s_anchored) isa SAnchoredSystem
    @test_throws ArgumentError node_based(sir_model(), ConfigurationNetwork(PoissonDegree(5));
                                          level = :s_anchored, closure = KeelingClosure())
    @test_throws ArgumentError node_based(sir_model(), ConfigurationNetwork(PoissonDegree(5));
                                          closure = BarnardClosure())
    @test_throws MethodError node_based(sir_model(), ConfigurationNetwork(PoissonDegree(5));
                                        seedfraction = 0.1)                 # misspelt keyword
    # the factory form equals the legacy network structure
    p = Dict(:τ => 0.3, :γ => 0.2)
    a = generate_pairwise(sir_model(), ConfigurationNetwork(RegularDegree(4)), BernoulliClosure();
                          tspan = (0.0, 40.0), seed_fraction = 0.01)
    b = generate_pairwise(sir_model(), regular_network(4), BernoulliClosure(); tspan = (0.0, 40.0),
                          seed_fraction = 0.01)
    @test a.metadata[:cumulative] === nothing
    @test compartment(a, pw_solve(a, p, 0:1:40), :R) == compartment(b, pw_solve(b, p, 0:1:40), :R)
    @test generate_pairwise(CompartmentalModel(sir_model()), ConfigurationNetwork(RegularDegree(4)),
                            BernoulliClosure()) isa PairwiseSystem
    # node_based adds the cumulative-incidence state; the other states are unchanged
    c = node_based(sir_model(), ConfigurationNetwork(RegularDegree(4)); tspan = (0.0, 40.0),
                   seed_fraction = 0.01)
    sc_ = pw_solve(c, p, 0:1:40)
    @test compartment(c, sc_, :R) ≈ compartment(b, pw_solve(b, p, 0:1:40), :R) rtol = 1e-9
    @test sc_[c.metadata[:cumulative]] ≈ 1 .- compartment(c, sc_, :S) rtol = 1e-8
    # clustered network: homogeneous Keeling with ϕ = 2/15
    cl = node_based(sir_model(), ClusteredNetwork(RegularDegree(2), RegularDegree(2)))
    @test cl.closure isa KeelingClosure && cl.network.ϕ ≈ 2 / 15 && cl.network.n == 6
    # the network N of the Keeling closure is the population scale (no B03 mismatch)
    cN = node_based(sir_model(), ClusteredNetwork(RegularDegree(2), RegularDegree(2)); N = 1000)
    @test cN.network.N == 1000.0
    # frequency-dependent rates become per-contact rates τ = β/⟨k⟩ on the network
    rn = @reaction_network begin
        β / N, S + I --> E + I
        σ, E --> I
        γ, I --> R
    end
    fd = node_based(rn, ConfigurationNetwork(RegularDegree(6)); rates = :frequency,
                    population = :N)
    @test rate_value(first(fd.metadata[:compartmental].transitions).rate, Dict(:β => 0.6)) ≈ 0.1
    pc = node_based(seir_model(), ConfigurationNetwork(RegularDegree(6)))
    pf = Dict(:β => 0.6, :σ => 0.2, :γ => 0.25)
    s1 = solve_epidemic(fd; p = pf, saveat = 0:1:30)
    s2 = solve_epidemic(pc; p = Dict(:τ => 0.1, :σ => 0.2, :γ => 0.25), saveat = 0:1:30)
    @test compartment(fd, s1, :R) ≈ compartment(pc, s2, :R) rtol = 1e-7
    # T_net models are admissible (SIS, SIRS)
    sis = node_based(sis_model(), ConfigurationNetwork(RegularDegree(3)); tspan = (0.0, 80.0))
    ss = solve_epidemic(sis; p = Dict(:τ => 0.5, :γ => 0.25))
    @test population_fraction(sis, ss, :I)[end] > 0.1          # τ/τ_c = 4: endemic
    @test population_fraction(sis, ss, :S) .+ population_fraction(sis, ss, :I) ≈ ones(length(ss.t))
end

@testset "Scenarios: node_based(sc), solve_epidemic(sys, sc), model_curves" begin
    sc = scenario(:sir_reg6)
    sys = node_based(sc)
    @test sys.network isa HomogeneousNetwork && sys.network.n == 6
    @test sys.params == Dict{Any,Float64}(k => v for (k, v) in sc.params)
    sol = solve_epidemic(sys, sc)
    @test sol.t == collect(sc.tgrid)
    mc = model_curves(sys, sol; t = sc.tgrid, label = "pairwise-Bernoulli")
    @test mc.label == "pairwise-Bernoulli"
    @test Set(sc.observables) ⊆ Set(keys(mc.values))
    @test abs(mc[:cumulative][end] - sc.expected[:final_size]) < 1e-4
    @test mc[:S] .+ mc[:I] .+ mc[:R] ≈ ones(length(sc.tgrid))
    # a reinfection-counted model also reports the totals over infection counts
    lifted = with_reinfection_counting(sis_model(), 2)
    rsys = node_based(lifted, ConfigurationNetwork(RegularDegree(3)); tspan = (0.0, 20.0),
                      seed_fraction = 0.01)
    rsol = solve_epidemic(rsys; p = Dict(:τ => 0.5, :γ => 0.25), saveat = 0:1:20)
    rmc = model_curves(rsys, rsol)
    @test rmc[:I] ≈ rmc[:I_1] .+ rmc[:I_2]
    @test population_fraction(rsys, rsol, :I) ≈ rmc[:I]
    @test reinfection_totals(rsys, rsol)[:S] ≈ rmc[:S]
    @test symbolic_ode(rsys) isa SymbolicODE
    @test length(symbolic_ode(rsys).states) == length(rsys.singles) + length(rsys.pairs) + 1
    # parameter names: a misspelt name is an error (it was silently ignored), in `p` and in the
    # values stored at build time
    @test_throws ArgumentError solve_epidemic(sys; p = Dict(:τ => 0.3, :gamma => 0.2))
    bad = node_based(sir_model(), ConfigurationNetwork(RegularDegree(4)); p = Dict(:tau => 0.3))
    @test_throws ArgumentError solve_epidemic(bad; p = Dict(:τ => 0.3, :γ => 0.2))
    # the model's defaults fill in, and a misspelt override does not fall back to them
    dm = ContactModel(:sird; contacts = [Contact(:S, :I, :I, :τ)],
                      transitions = [NodeTransition(:I, :R, :γ)],
                      defaults = Dict(:τ => 0.3, :γ => 0.2))
    dsys = node_based(dm, ConfigurationNetwork(RegularDegree(4)); tspan = (0.0, 10.0))
    dsol = solve_epidemic(dsys; saveat = 0:1:10)
    @test string(dsol.retcode) == "Success"
    @test compartment(dsys, solve_epidemic(dsys; p = Dict(:γ => 0.2), saveat = 0:1:10), :R) ==
          compartment(dsys, dsol, :R)
    err = try
        solve_epidemic(dsys; p = Dict(:γγ => 0.5))
    catch e
        e
    end
    @test err isa ArgumentError && occursin("γγ", sprint(showerror, err))
end

@testset "Graph-level levels on an ExplicitGraph" begin
    g = random_regular_graph(60, 4; seed = 3)
    net = ExplicitGraph(g)
    p = Dict(:τ => 0.3, :γ => 0.2)
    ib = node_based(sir_model(), net; level = :individual, p, tspan = (0.0, 20.0),
                    seed_fraction = 0.05)
    ref = generate_individual_based(CompartmentalModel(sir_model()), GraphNetwork(net);
                                    infection_rate = 0.3, recovery_rate = 0.2,
                                    tspan = (0.0, 20.0), seed_fraction = 0.05)
    @test aggregate(ib, :R) == aggregate(ref, :R)
    @test aggregate(generate_individual_based(sir_model(), net; infection_rate = 0.3,
                                              recovery_rate = 0.2, tspan = (0.0, 20.0),
                                              seed_fraction = 0.05), :R) == aggregate(ref, :R)
    pb = node_based(sir_model(), net; level = :pair, p, tspan = (0.0, 20.0), seed_fraction = 0.05)
    @test pb isa PairBasedResult
    # models the 0.1 graph-level builders would get wrong are refused (B02, B04)
    @test_throws ArgumentError node_based(seir_model(), net; level = :individual,
                                          p = Dict(:τ => 0.3, :σ => 0.5, :γ => 0.2))
    @test_throws ArgumentError node_based(seair_model(), net; level = :individual,
                                          p = Dict(:τI => 0.3, :τA => 0.1, :p => 0.5, :σ => 0.5, :γ => 0.2))
    @test_throws ArgumentError node_based(sir_model(), net; level = :individual)   # no p
    @test_throws ArgumentError node_based(sir_model(), ConfigurationNetwork(RegularDegree(4));
                                          level = :individual, p)
end

@testset "SIS motif and neighbourhood levels on a regular network" begin
    p = Dict(:τ => 0.5, :γ => 0.25)
    net = ConfigurationNetwork(RegularDegree(2))
    ms = node_based(sis_model(), net; level = :motif, closure = MotifClosure(2, 2), p,
                    tspan = (0.0, 10.0), seed_fraction = 0.01)
    ref = motif_based_sis(; β = 0.5, γ = 0.25, k = 2, m = 2, tspan = (0.0, 10.0), ε = 0.01)
    @test ms.u0 == ref.u0
    @test compartment(ms, solve_motif(ms; saveat = 1.0), :I) ≈
          compartment(ref, solve_motif(ref; saveat = 1.0), :I)
    @test_throws ArgumentError node_based(sis_model(), net; level = :motif,
                                          closure = MotifClosure(3, 2), p)
    @test_throws ArgumentError node_based(sir_model(), net; level = :motif,
                                          closure = MotifClosure(2, 2), p = Dict(:τ => 0.5, :γ => 0.25))
    nh = node_based(sis_model(), ConfigurationNetwork(RegularDegree(3)); level = :neighbourhood, p,
                    tspan = (0.0, 10.0), seed_fraction = 0.01)
    nref = generate_neighbourhood(sis_model(), 3, 2; β = 0.5, γ = 0.25, tspan = (0.0, 10.0), ε = 0.01)
    @test nh.u0 == nref.u0
end

# ─── B04 (a): GraphNetwork → ExplicitGraph ───────────────────────────────────────
@testset "B04 (a): GraphNetwork keeps an explicit rate 1.0; ExplicitGraph converter" begin
    g = complete_graph(5)
    net = @test_deprecated GraphNetwork(g; transmission_rate = 1.0)
    @test !isnothing(net.transmission_matrix)                                   # 0.1: nothing
    @test NodeBasedModels._effective_transmission_matrix(net, 0.5)[1, 2] == 1.0 # 0.1: 0.5
    @test GraphNetwork(ExplicitGraph(g); transmission_rate = 1).transmission_matrix[1, 2] == 1.0
    @test isnothing(GraphNetwork(ExplicitGraph(g)).transmission_matrix)
    T = [0.0 0.3; 0.7 0.0]
    @test GraphNetwork(ExplicitGraph(path_graph(2)); transmission_matrix = T).transmission_matrix == T
    @test_throws DimensionMismatch GraphNetwork(ExplicitGraph(path_graph(3)); transmission_matrix = T)
    @test_throws ArgumentError GraphNetwork(ExplicitGraph(g); transmission_rate = -1.0)
    @test_throws ArgumentError GraphNetwork(ExplicitGraph(path_graph(2)); transmission_rate = 1.0,
                                            transmission_matrix = T)
    @test (@test_deprecated GraphNetwork(g, nothing)).transmission_matrix === nothing
    gg = random_regular_graph(200, 4; seed = 1)
    Rend(n; kw...) = aggregate(generate_individual_based(CompartmentalModel(sir_model()), n;
                                                         tspan = (0.0, 50.0), recovery_rate = 1.0,
                                                         seed_fraction = 0.05, kw...), :R)[end]
    @test Rend(GraphNetwork(ExplicitGraph(gg); transmission_rate = 1.0)) ≈
          Rend(GraphNetwork(ExplicitGraph(gg)); infection_rate = 1.0)
end

@testset "Deprecations and removals" begin
    # 0.1 canned models, the 0.1 reinfection lift (bit-identical structure)
    old = @test_deprecated with_reinfection_counting(@test_deprecated(node_sis_model()), 3)
    new = CompartmentalModel(with_reinfection_counting(sis_model(), 3))
    @test sort(old.compartment_names) == sort(new.compartment_names)
    @test Set((t.from, t.to, t.rate, t.type) for t in old.transitions) ==
          Set((t.from, t.to, t.rate, t.type) for t in new.transitions)
    # the 0.1 lift dropped `via`; the deprecated method keeps it
    viam = CompartmentalModel(seair_model())
    lv = @test_deprecated with_reinfection_counting(viam, 1)
    @test any(t -> t.type === :infection && t.via == [:I_1], lv.transitions)
    # the 0.1 graph network constructors
    @test_deprecated GraphNetwork(cycle_graph(4))
    # removed / not defined
    @test_throws ArgumentError generate_pairwise(sir_model(), regular_network(6; ϕ = 0.2),
                                                 EamesClosure())
    err = try
        generate_pairwise(sir_model(), regular_network(4), EamesClosure())
    catch e
        e
    end
    @test occursin("removed in NodeBasedModels 0.2", sprint(showerror, err))
    psys = generate_pairwise(sir_model(), regular_network(4), BernoulliClosure())
    @test_throws ArgumentError mass_action(psys)
    @test psys.model isa ContactModel && isequivalent(psys.model, sir_model())
end
