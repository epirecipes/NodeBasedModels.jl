# 00_legacy_pairwise.jl — legacy NodeBasedModels tests:
# population-level pairwise generation, solving, accessors and SIR dynamics.
#
# Moved verbatim (dedented one level) from the pre-0.2 test/runtests.jl by WP2
# (DESIGN_NetworkEpiCore.md §G.2). Owned by the work package that replaces this
# area; see §G.1. The imports are the original file's, so name resolution is unchanged.
#
# WP15 (NodeBasedModels 0.2): `sir_model()` is a NetworkEpiCore ContactModel; the compartment
# count below is read through `CompartmentalModel(m)`. Everything else is unchanged.

using NodeBasedModels
using Test
using OrdinaryDiffEqDefault
using ModelingToolkit
using Graphs
using Random
using Catalyst
using Symbolics

# ─── Pairwise System Generation (population-level) ────────────────────
@testset "Pairwise System Generation" begin
    @testset "SIR homogeneous Bernoulli" begin
        m = sir_model()
        net = regular_network(6)
        cl = BernoulliClosure()
        psys = generate_pairwise(m, net, cl)

        @test psys isa PairwiseSystem
        @test length(psys.singles) == 3  # S, I, R
        @test length(psys.pairs) == 6    # SS, SI, SR, II, IR, RR
        @test psys.model === m
        @test psys.network === net
    end

    @testset "SIR with Keeling closure" begin
        m = sir_model()
        net = regular_network(6; ϕ=0.2)
        cl = KeelingClosure()
        psys = generate_pairwise(m, net, cl)
        @test psys isa PairwiseSystem
    end

    @testset "SIS homogeneous" begin
        m = sis_model()
        net = regular_network(4)
        psys = generate_pairwise(m, net, BernoulliClosure())
        @test length(psys.singles) == 2  # S, I
        @test length(psys.pairs) == 3    # SS, SI, II
    end

    @testset "SEIR homogeneous" begin
        m = seir_model()
        net = regular_network(6)
        psys = generate_pairwise(m, net, BernoulliClosure())
        @test length(psys.singles) == 4   # S, E, I, R
        @test length(psys.pairs) == 10    # 4·5/2 = 10
    end

    @testset "Equation counts across closures" begin
        m = sir_model()
        net = regular_network(6; ϕ = 0.2)
        K = length(CompartmentalModel(m).compartment_names)
        expected_eqs = K + K * (K + 1) ÷ 2

        for closure in (
            BernoulliClosure(),
            KeelingClosure(),
            BarnardClosure(),
            PowerClosure(1.5),
        )
            psys = generate_pairwise(m, net, closure)
            @test length(ModelingToolkit.equations(psys.system)) == expected_eqs
        end
    end

    @testset "Clustered closure formula regression" begin
        m = sir_model()
        net = regular_network(6; ϕ = 0.2)
        psys = generate_pairwise(m, net, BernoulliClosure())

        keeling_ssi = triple_closure(:S, :S, :I,
            psys.pairs, psys.singles, net, KeelingClosure())
        barnard_isi = triple_closure(:I, :S, :I,
            psys.pairs, psys.singles, net, BarnardClosure())

        # Mathematical sanity rather than brittle string matching:
        # both closures must depend on the relevant pair / single variables.
        for sym in (psys.pairs[(:S, :S)], psys.pairs[(:S, :I)], psys.singles[:S])
            @test sym in Symbolics.get_variables(keeling_ssi)
        end
        for sym in (psys.pairs[(:S, :I)], psys.pairs[(:I, :I)], psys.singles[:S], psys.singles[:I])
            @test sym in Symbolics.get_variables(barnard_isi)
        end
    end

    @testset "SIR Bernoulli equation regression" begin
        psys = generate_pairwise(sir_model(), regular_network(4), BernoulliClosure())
        eqs = ModelingToolkit.equations(psys.system)
        S = psys.singles[:S]; I = psys.singles[:I]; R = psys.singles[:R]
        SI = psys.pairs[(:S, :I)]
        t = ModelingToolkit.get_iv(psys.system)
        D = Differential(t)
        rhs_for(lhs) = begin
            idx = findfirst(eq -> isequal(eq.lhs, lhs), eqs)
            isnothing(idx) && error("no equation with LHS $lhs")
            eqs[idx].rhs
        end
        # Structural checks robust to term reordering across Symbolics versions.
        dS = rhs_for(D(S))
        dI = rhs_for(D(I))
        dR = rhs_for(D(R))
        @test SI in Symbolics.get_variables(dS)
        @test SI in Symbolics.get_variables(dI)
        @test I  in Symbolics.get_variables(dI)
        @test I  in Symbolics.get_variables(dR)
        # Conservation: dS + dI + dR == 0 (verified numerically — robust to
        # symbolic simplification differences across Symbolics versions).
        ps = ModelingToolkit.parameters(psys.system)
        τ_sym = ps[findfirst(p -> nameof(p) === :τ, ps)]
        γ_sym = ps[findfirst(p -> nameof(p) === :γ, ps)]
        total = dS + dI + dR
        subs = Dict(S => 0.6, I => 0.3, R => 0.1, SI => 0.05,
                    τ_sym => 0.2, γ_sym => 0.1)
        @test isapprox(Float64(Symbolics.value(Symbolics.substitute(total, subs))), 0.0; atol = 1e-12)
    end

    @testset "Closure solvability and invariants" begin
        m = sir_model()
        net = regular_network(4; ϕ = 0.2)
        N = 100.0
        k = mean_degree(net)

        for closure in (
            BernoulliClosure(),
            KeelingClosure(),
            BarnardClosure(),
            PowerClosure(1.5),
        )
            psys = generate_pairwise(m, net, closure; N = N, tspan = (0.0, 20.0))
            sol = solve_pairwise(psys, Dict(:τ => 0.2, :γ => 0.1); saveat = 1.0)
            @test sol.retcode == ReturnCode.Success
            # Mixed-convention directed-pair total: cross-pairs (i ≠ j) count as
            # 2× the stored value (each undirected XY edge gives two directed
            # edges X→Y and Y→X), self-pairs (i = j) are already stored as the
            # directed count under the mixed convention.  The total directed-pair
            # count is the network invariant Σ_X k·[X] = k·N.
            directed_pair_total(tidx) = sum(
                sol[psys.pairs[(a, b)]][tidx] * (a == b ? 1 : 2)
                for (a, b) in keys(psys.pairs))

            for tidx in eachindex(sol.t)
                total_single = sum(sol[var][tidx] for var in values(psys.singles))
                @test isapprox(total_single, N; atol = 1e-6)
                @test isapprox(directed_pair_total(tidx), k * N; atol = 1e-4)
                @test all(isfinite(sol[var][tidx]) for var in values(psys.singles))
                if closure isa BernoulliClosure ||
                   closure isa PowerClosure
                    @test all(sol[var][tidx] >= -1e-8 for var in values(psys.singles))
                end
            end
        end
    end

    @testset "Unsupported closures and parameter guards" begin
        m = sir_model()
        @test_throws ArgumentError generate_pairwise(m, regular_network(6; ϕ = 0.2), EamesClosure())
        @test_throws ArgumentError generate_pairwise(m, regular_network(6), KirkwoodClosure())
        @test_throws ArgumentError generate_pairwise(m, erdos_renyi_network(5.0), BarnardClosure())
        @test_throws ArgumentError generate_pairwise(m, erdos_renyi_network(5.0), PowerClosure(1.5))

        psys = generate_pairwise(m, regular_network(4), BernoulliClosure())
        @test_throws ArgumentError solve_pairwise(psys, Dict(:τ => 0.2))
        @test_throws ArgumentError solve_pairwise(psys, Dict(:τ => 0.2, :γ => 0.1, :δ => 1.0))
    end

    @testset "Multi-compartment solvability" begin
        cases = (
            (seir_model(), Dict(:τ => 0.2, :σ => 0.15, :γ => 0.1)),
            (sirs_model(), Dict(:τ => 0.2, :γ => 0.1, :ε => 0.05)),
        )

        for (model, params) in cases
            psys = generate_pairwise(model, regular_network(4), BernoulliClosure();
                N = 100.0, tspan = (0.0, 20.0))
            sol = solve_pairwise(psys, params; saveat = 1.0)
            @test sol.retcode == ReturnCode.Success

            for tidx in eachindex(sol.t)
                total_single = sum(sol[var][tidx] for var in values(psys.singles))
                @test isapprox(total_single, 100.0; atol = 1e-6)
                @test all(sol[var][tidx] >= -1e-8 for var in values(psys.singles))
            end
        end
    end

    @testset "seed_fraction keyword" begin
        m = sir_model()
        net = regular_network(6)
        psys_seed = generate_pairwise(m, net, BernoulliClosure(); seed_fraction = 0.02)
        psys_eps = generate_pairwise(m, net, BernoulliClosure(); ε = 0.02)
        @test psys_seed.u0[psys_seed.singles[:I]] ≈ 0.02
        @test psys_seed.u0[psys_seed.singles[:S]] ≈ 0.98
        @test psys_seed.u0 == psys_eps.u0
    end
end

@testset "solve_epidemic wrapper" begin
    net = regular_network(4)
    psys = generate_pairwise(sir_model(), net, BernoulliClosure(); tspan=(0.0, 20.0))
    params = Dict(:τ => 0.2, :γ => 0.1)
    sol1 = solve_pairwise(psys, params; saveat=1.0)
    sol2 = solve_epidemic(psys, params; saveat=1.0)
    @test isapprox(sol1[psys.singles[:I]][end], sol2[psys.singles[:I]][end]; atol=1e-8)
    @test isapprox(sol1[psys.singles[:S]][end], sol2[psys.singles[:S]][end]; atol=1e-8)
end

@testset "PairwiseSystem accessors" begin
    net = regular_network(4)
    psys = generate_pairwise(sir_model(), net, BernoulliClosure(); tspan=(0.0, 20.0))

    # node_variables / pair_variables
    nodes = node_variables(psys)
    pairs = pair_variables(psys)
    @test nodes === psys.singles
    @test pairs === psys.pairs
    @test :S in keys(nodes) && :I in keys(nodes)

    # default_initial_conditions returns the stored u0
    @test default_initial_conditions(psys) === psys.u0

    # compartment / population_fraction accessors on PairwiseSystem
    sol = solve_pairwise(psys, Dict(:τ => 0.2, :γ => 0.1); saveat=1.0)
    S_ts = compartment(psys, sol, :S)
    @test length(S_ts) > 0
    @test_throws ArgumentError compartment(psys, sol, :NoSuchCompartment)
    # population_fraction without N returns same as compartment
    @test population_fraction(psys, sol, :S) == S_ts
    # with N normalises
    N_total = first(S_ts) + first(compartment(psys, sol, :I)) +
              (haskey(psys.singles, :R) ? first(compartment(psys, sol, :R)) : 0.0)
    frac = population_fraction(psys, sol, :S; N = N_total)
    @test 0.0 <= frac[1] <= 1.0
end

# ─── Pairwise SIR Dynamics regression suite ───────────────────────────
# End-to-end solves that catch convention/sign bugs in the pair-equation
# generator (mixed Keeling/Eames convention).  Ported from the legacy
# PairwiseNetworkModels.jl test suite during the package consolidation.
@testset "Pairwise SIR Dynamics" begin
    # k=6-regular, τ=0.2, γ=0.1 → R₀ = τ(n-2)/γ = 8 (super-critical).
    # Final R(∞) should be near complete attack (>95%).
    @testset "Supercritical SIR final size" begin
        m = sir_model()
        net = regular_network(6)
        psys = generate_pairwise(m, net, BernoulliClosure();
                                  tspan=(0.0, 300.0), N=1.0)
        ic = copy(psys.u0)
        S0, I0, R0 = 0.99, 0.01, 0.0
        k = 6.0
        ic[psys.singles[:S]] = S0
        ic[psys.singles[:I]] = I0
        ic[psys.singles[:R]] = R0
        # Mixed-convention pair init: [XY] = k · N · p_X · p_Y
        ic[psys.pairs[(:S,:S)]] = k * S0 * S0
        ic[psys.pairs[(:S,:I)]] = k * S0 * I0
        ic[psys.pairs[(:S,:R)]] = k * S0 * R0
        ic[psys.pairs[(:I,:I)]] = k * I0 * I0
        ic[psys.pairs[(:I,:R)]] = k * I0 * R0
        ic[psys.pairs[(:R,:R)]] = k * R0 * R0
        p = copy(psys.params)
        p[:τ] = 0.2; p[:γ] = 0.1
        prob = ODEProblem(psys.system, merge(ic, p), psys.tspan)
        sol = solve(prob; reltol=1e-8, abstol=1e-10)
        R_final = sol[psys.singles[:R]][end]
        S_final = sol[psys.singles[:S]][end]
        I_max   = maximum(sol[psys.singles[:I]])
        @test R_final > 0.95           # near-complete attack
        @test S_final < 0.05
        @test I_max > 0.4              # significant prevalence peak
        @test isapprox(S_final + sol[psys.singles[:I]][end] + R_final, 1.0;
                        atol=1e-3)
    end

    @testset "Subcritical SIR no outbreak" begin
        # τ=0.02, γ=0.1, k=6 → R₀ = 0.02·4/0.1 = 0.8 (subcritical).
        m = sir_model()
        net = regular_network(6)
        psys = generate_pairwise(m, net, BernoulliClosure();
                                  tspan=(0.0, 200.0), N=1.0)
        ic = copy(psys.u0)
        S0, I0, R0 = 0.999, 0.001, 0.0
        k = 6.0
        ic[psys.singles[:S]] = S0
        ic[psys.singles[:I]] = I0
        ic[psys.singles[:R]] = R0
        ic[psys.pairs[(:S,:S)]] = k * S0 * S0
        ic[psys.pairs[(:S,:I)]] = k * S0 * I0
        ic[psys.pairs[(:S,:R)]] = k * S0 * R0
        ic[psys.pairs[(:I,:I)]] = k * I0 * I0
        ic[psys.pairs[(:I,:R)]] = k * I0 * R0
        ic[psys.pairs[(:R,:R)]] = k * R0 * R0
        p = copy(psys.params)
        p[:τ] = 0.02; p[:γ] = 0.1
        prob = ODEProblem(psys.system, merge(ic, p), psys.tspan)
        sol = solve(prob; reltol=1e-8, abstol=1e-10)
        @test sol[psys.singles[:R]][end] < 0.05    # no take-off
        @test sol[psys.singles[:S]][end] > 0.95
    end

    @testset "Conservation of singles" begin
        m = sir_model()
        net = regular_network(6)
        psys = generate_pairwise(m, net, BernoulliClosure();
                                  tspan=(0.0, 100.0), N=1.0)
        ic = copy(psys.u0)
        S0, I0, R0 = 0.9, 0.1, 0.0
        k = 6.0
        ic[psys.singles[:S]] = S0
        ic[psys.singles[:I]] = I0
        ic[psys.singles[:R]] = R0
        ic[psys.pairs[(:S,:S)]] = k * S0 * S0
        ic[psys.pairs[(:S,:I)]] = k * S0 * I0
        ic[psys.pairs[(:S,:R)]] = k * S0 * R0
        ic[psys.pairs[(:I,:I)]] = k * I0 * I0
        ic[psys.pairs[(:I,:R)]] = k * I0 * R0
        ic[psys.pairs[(:R,:R)]] = k * R0 * R0
        p = copy(psys.params)
        p[:τ] = 0.15; p[:γ] = 0.1
        prob = ODEProblem(psys.system, merge(ic, p), psys.tspan)
        sol = solve(prob; reltol=1e-8, abstol=1e-10)
        for ti in eachindex(sol.t)
            tot = sol[psys.singles[:S]][ti] +
                   sol[psys.singles[:I]][ti] +
                   sol[psys.singles[:R]][ti]
            @test isapprox(tot, 1.0; atol=1e-6)
        end
    end
end

# ─── Golden: frozen numbers of this area (test/golden/pairwise/) ─────────────────
include(joinpath(@__DIR__, "..", "golden", "GoldenIO.jl"))
GoldenIO.check_area("pairwise")
