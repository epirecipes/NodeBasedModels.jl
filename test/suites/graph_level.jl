# graph_level.jl — WP24: the individual-based (NIMFA) and pair-based models on explicit graphs
# after the 0.2 fixes, and the deprecated Gillespie simulators.
#
# - B02: every spontaneous transition runs at its own rate (0.1 ran them all at recovery_rate);
# - B04: each infection transition has its own force through its own infectors (`via`);
# - seeding: the entry state of the infections (E for SEIR), DESIGN §E.2;
# - the `gillespie_*` functions are deprecated, gillespie_sir through the JumpProcesses extension.
#
# References: the per-node spontaneous dynamics are linear and closure-free (Sharkey 2008,
# Eq. 3/7), so with τ = 0 they have exact solutions; NIMFA is accurate on dense graphs, so on a
# 60-regular graph it is compared with the exact stochastic process (NetworkOutbreaks).

using NodeBasedModels
using Test
using Graphs
using Random
using Statistics
using JumpProcesses
using OrdinaryDiffEqDefault   # EnsembleProblem, EnsembleThreads, remake, solve
using NetworkOutbreaks
import NetworkOutbreaks as NO

@testset "B02: individual-based spontaneous rates are per symbol" begin
    g = random_regular_graph(50, 4; seed = 1)
    net = ExplicitGraph(g)
    N = 50
    # SEIR: node 1 in E, nobody infectious → dE_1/dt = −σ, dI_1/dt = +σ (0.1: ±γ)
    r = generate_individual_based(seir_model(), net; p = Dict(:τ => 0.1, :σ => 1.0, :γ => 0.25),
                                  initial = SeedNodes(:E => [1]), tspan = (0.0, 1.0))
    @test r.state_names == [:S, :E, :I]
    u = zeros(3N)
    for i in 2:N
        u[3(i-1)+1] = 1.0
    end
    u[2] = 1.0
    du = similar(u)
    r.sol.prob.f(du, u, r.sol.prob.p, 0.0)
    @test du[2] ≈ -1.0
    @test du[3] ≈ 1.0

    # τ = 0: exact linear solutions of the node dynamics (Sharkey 2008, Eq. 3/7)
    ts = 0.0:1.0:10.0
    σ, γ = 1.0, 0.25
    r = generate_individual_based(seir_model(), net; p = Dict(:τ => 0.0, :σ => σ, :γ => γ),
                                  initial = SeedNodes(:E => collect(1:N)), tspan = (0.0, 10.0),
                                  reltol = 1e-10, abstol = 1e-12)
    E_exact = exp.(-σ .* ts)
    I_exact = σ / (γ - σ) .* (exp.(-σ .* ts) .- exp.(-γ .* ts))
    @test aggregate(r, :E) ./ N ≈ E_exact rtol = 1e-7
    @test aggregate(r, :I) ./ N ≈ I_exact rtol = 1e-7
    # 0.1 gave E = e^{−γt} (σ silently equal to γ); at t = 2 that is 0.61 against 0.14
    @test abs(aggregate(r, :E)[3] / N - exp(-2γ)) > 0.4

    ε = 0.05
    r = generate_individual_based(sirs_model(), net; p = Dict(:τ => 0.0, :γ => γ, :ε => ε),
                                  initial_infected = collect(1:N), tspan = (0.0, 10.0),
                                  reltol = 1e-10, abstol = 1e-12)
    R_exact = γ / (ε - γ) .* (exp.(-γ .* ts) .- exp.(-ε .* ts))
    @test aggregate(r, :R) ./ N ≈ R_exact rtol = 1e-7

    ν = 0.02
    r = generate_individual_based(sirv_model(), net; p = Dict(:τ => 0.0, :γ => γ, :ν => ν),
                                  seed_fraction = 0.0, tspan = (0.0, 10.0), reltol = 1e-10,
                                  abstol = 1e-12)
    @test aggregate(r, :S) ./ N ≈ exp.(-ν .* ts) rtol = 1e-7

    # a model with several spontaneous rates must not reuse recovery_rate; unknown keys and
    # contradictions are errors
    @test_throws ArgumentError generate_individual_based(seir_model(), net; recovery_rate = 0.25,
                                                         initial_infected = [1], tspan = (0.0, 1.0))
    @test_throws ArgumentError generate_individual_based(seir_model(), net;
        p = Dict(:τ => 0.1, :σ => 1.0, :γ => 0.25, :bogus => 1.0), tspan = (0.0, 1.0))
    @test_throws ArgumentError generate_individual_based(seir_model(), net;
        p = Dict(:τ => 0.1, :σ => 1.0), tspan = (0.0, 1.0))                  # γ missing
    @test_throws ArgumentError generate_individual_based(sir_model(), net;
        p = Dict(:τ => 0.1, :γ => 0.2), recovery_rate = 0.3, tspan = (0.0, 1.0))
    # single-spontaneous-rate models keep the 0.1 keywords (bit-identical to p)
    r1 = generate_individual_based(sir_model(), net; infection_rate = 0.3, recovery_rate = 0.25,
                                   initial_infected = [1], tspan = (0.0, 5.0))
    r2 = generate_individual_based(sir_model(), net; p = Dict(:τ => 0.3, :γ => 0.25),
                                   initial_infected = [1], tspan = (0.0, 5.0))
    @test Array(r1.sol) == Array(r2.sol)
end

@testset "B02: NIMFA SEIR with σ ≠ γ against the exact stochastic process (dense graph)" begin
    # A 60-regular graph, N = 1500, τ = 0.012 (kτ = 0.72), σ = 1, γ = 0.25, 1% seeded in I.
    # NIMFA with uniform seeding on a k-regular graph does not depend on the graph, so the
    # stochastic runs use a fresh graph each (NetworkOutbreaks.stable_rng streams, DESIGN §J.7).
    N, k, nruns = 1500, 60, 20
    p = Dict(:τ => 0.012, :σ => 1.0, :γ => 0.25)
    ρ = 0.01
    ts = 0.0:1.0:40.0
    ib = generate_individual_based(seir_model(), ExplicitGraph(random_regular_graph(N, k; seed = 7)); p,
                                   initial = SeedFraction(:I => ρ), tspan = (0.0, 40.0),
                                   reltol = 1e-8, abstol = 1e-10)
    I_ib = aggregate(ib, :I) ./ N
    ib_bug = generate_individual_based(seir_model(), ExplicitGraph(random_regular_graph(N, k; seed = 7));
                                       p = Dict(:τ => 0.012, :σ => 0.25, :γ => 0.25),
                                       initial = SeedFraction(:I => ρ), tspan = (0.0, 40.0))
    I_bug = aggregate(ib_bug, :I) ./ N                  # what 0.1 computed (σ = γ)
    om = NO.OutbreakModel(seir_model(), p)
    base = 20260926
    I_ssa = zeros(length(ts), nruns)
    for run in 1:nruns
        g = random_regular_graph(N, k; rng = NO.stable_rng(base + run))
        spec = OutbreakSpec(model = om, network = g, initial = SeedFraction(:I => ρ), tspan = (0.0, 40.0))
        traj = simulate(spec; seed = base + 2^32 + run, algorithm = NextReaction())
        iI = om.index_of[:I]
        I_ssa[:, run] = [state_at(traj, t)[iI] for t in ts] ./ N
    end
    # conditioned on major outbreaks (peak prevalence above 5 × the seed fraction); all runs
    # qualify here
    major = [maximum(I_ssa[:, r]) > 5ρ for r in 1:nruns]
    @test count(major) == nruns
    I_mean = vec(mean(I_ssa[:, major]; dims = 2))
    se = vec(std(I_ssa[:, major]; dims = 2)) ./ sqrt(count(major))
    @info "B02 NIMFA vs SSA (N = $N, k = $k, $(count(major)) runs)" peak_ssa = maximum(I_mean) peak_ib = maximum(I_ib) peak_bug = maximum(I_bug) max_se = maximum(se) sup_fixed = maximum(abs.(I_ib .- I_mean)) sup_bug = maximum(abs.(I_bug .- I_mean))
    @test maximum(abs.(I_ib .- I_mean)) < 0.03          # NIMFA error O(1/k) on a dense graph
    @test maximum(abs.(I_bug .- I_mean)) > 0.08         # σ = γ is a different epidemic
end

@testset "B04: per-infector infection transitions (via) in the individual-based model" begin
    comps = [Compartment(:S), Compartment(:I1; infectious = true),
             Compartment(:I2; infectious = true), Compartment(:R)]
    m = CompartmentalModel(comps, [Transition(:S, :I1, :τ1, :infection; via = [:I1]),
                                   Transition(:S, :I1, :τ2, :infection; via = [:I2]),
                                   Transition(:I1, :I2, :γ1, :spontaneous),
                                   Transition(:I2, :R, :γ2, :spontaneous)]; name = :staged)
    g = random_regular_graph(40, 5; seed = 3)
    N = nv(g)
    p = Dict(:τ1 => 0.6, :τ2 => 0.1, :γ1 => 1.0, :γ2 => 1.0)
    r = generate_individual_based(m, ExplicitGraph(g); p, seed_fraction = 0.02, tspan = (0.0, 1.0))
    # an independent per-symbol NIMFA right-hand side at a random state
    rng = MersenneTwister(11)
    x = [let v = rand(rng, 4); v ./ sum(v) end for _ in 1:N]   # (S, I1, I2, R) of every node
    u = reduce(vcat, [xi[1:3] for xi in x])
    du = similar(u)
    r.sol.prob.f(du, u, r.sol.prob.p, 0.0)
    for i in 1:N
        f1 = sum(x[j][2] for j in neighbors(g, i))
        f2 = sum(x[j][3] for j in neighbors(g, i))
        inf = x[i][1] * (p[:τ1] * f1 + p[:τ2] * f2)
        @test du[3(i-1)+1] ≈ -inf
        @test du[3(i-1)+2] ≈ inf - p[:γ1] * x[i][2]
        @test du[3(i-1)+3] ≈ p[:γ1] * x[i][2] - p[:γ2] * x[i][3]
    end
    # 0.1 applied (τ1 + τ2) to (I1 + I2) through both transitions; the ContactModel route gives
    # the same system as the CompartmentalModel
    cm = contact_model(m)
    r2 = generate_individual_based(cm, ExplicitGraph(g); p, seed_fraction = 0.02, tspan = (0.0, 1.0))
    du2 = similar(u)
    r2.sol.prob.f(du2, u, r2.sol.prob.p, 0.0)
    @test du2 ≈ du
    # an explicit transmission matrix replaces a single per-contact rate only
    T = zeros(N, N)
    for e in edges(g)
        T[src(e), dst(e)] = T[dst(e), src(e)] = 0.3
    end
    @test_throws ArgumentError generate_individual_based(m, ExplicitGraph(g); p,
                                                         transmission_matrix = T, tspan = (0.0, 1.0))
end

@testset "Seeding: the entry state of the infections" begin
    g = random_regular_graph(30, 4; seed = 5)
    net = ExplicitGraph(g)
    p = Dict(:τ => 0.2, :σ => 0.5, :γ => 0.25)
    r = generate_individual_based(seir_model(), net; p, initial_infected = [2, 7], tspan = (0.0, 1.0))
    @test node_state(r, 2, :E, 1) == 1.0 && node_state(r, 2, :I, 1) == 0.0
    @test node_state(r, 1, :S, 1) == 1.0
    r = generate_individual_based(seir_model(), net; p, initial_infected = [2, 7],
                                  seed_state = :first_infectious, tspan = (0.0, 1.0))
    @test node_state(r, 2, :I, 1) == 1.0                  # the 0.1 rule
    r = generate_individual_based(seir_model(), net; p, seed_fraction = 0.1, tspan = (0.0, 1.0))
    @test aggregate(r, :E)[1] ≈ 3.0 && aggregate(r, :I)[1] == 0.0
    r = generate_individual_based(seir_model(), net; p, initial = SeedFraction(:E => 0.1, :I => 0.05),
                                  tspan = (0.0, 1.0))
    @test aggregate(r, :E)[1] ≈ 3.0 && aggregate(r, :I)[1] ≈ 1.5 && aggregate(r, :S)[1] ≈ 25.5
    @test_throws ArgumentError generate_individual_based(seir_model(), net; p, initial_infected = [1],
                                                         initial = SeedNodes(:E => [2]))
    @test_throws ArgumentError generate_individual_based(seir_model(), net; p,
                                                         initial = SeedCount(:E => 2))
    @test_throws ArgumentError generate_individual_based(seir_model(), net; p, seed_state = :bogus)
    @test_throws ArgumentError generate_individual_based(seir_model(), net; p, initial_infected = [99])
end

@testset "Pair-based model: parameter values, seeding, tolerances, curves" begin
    g = random_regular_graph(30, 4; seed = 123)
    net = ExplicitGraph(g)
    a = generate_pair_based(sir_model(), net; infection_rate = 0.2, recovery_rate = 0.1,
                            initial_infected = [1, 2], tspan = (0.0, 10.0))
    b = generate_pair_based(sir_model(), net; p = Dict(:τ => 0.2, :γ => 0.1),
                            initial = SeedNodes(:I => [1, 2]), tspan = (0.0, 10.0))
    @test Array(a.sol) == Array(b.sol)
    c = generate_pair_based(sir_model(), net; p = Dict(:τ => 0.2, :γ => 0.1), initial_infected = [1, 2],
                            tspan = (0.0, 10.0), reltol = 1e-10, abstol = 1e-12)
    @test aggregate(c, :R) ≈ aggregate(a, :R) rtol = 1e-4
    # R is derived; any other name is an error (0.1 returned 1 − S − I for a typo such as :E)
    @test node_state(c, 3, :R, 5) ≈ 1 - node_state(c, 3, :S, 5) - node_state(c, 3, :I, 5)
    @test_throws ArgumentError node_state(c, 3, :E, 5)
    @test_throws ArgumentError aggregate(c, :infectious)
    @test_throws ArgumentError compartment(c, :X)
    @test_throws ArgumentError generate_pair_based(sir_model(), net; p = Dict(:τ => 0.2, :β => 0.1))
    @test_throws ArgumentError generate_pair_based(sir_model(), net; p = Dict(:τ => 0.2, :γ => 0.1),
                                                   initial = SeedFraction(:R => 0.5))
    mc = model_curves(c; t = 0:1:10)
    @test mc.representation === :pair
    @test mc[:S] .+ mc[:I] .+ mc[:R] ≈ ones(11)
    @test mc[:I] ≈ aggregate(c, :I) ./ 30
    ib = generate_individual_based(sir_model(), net; p = Dict(:τ => 0.2, :γ => 0.1),
                                   initial_infected = [1, 2], tspan = (0.0, 10.0))
    mi = model_curves(ib; t = 0:1:10)
    @test mi.representation === :individual
    @test mi[:R] ≈ aggregate(ib, :R) ./ 30
    @test mi[:infectious] == mi[:I]
end

# Allocation of the right-hand side, called through a function barrier.
rhs_allocated(f, du, u, p, t) = @allocated f(du, u, p, t)

@testset "Curves and right-hand sides at scale (allocations)" begin
    # model_curves evaluates the solution once per time point (it evaluated it once per node and
    # state: 7 GiB for IB SEIR with N = 1000 and 101 times, 15.5 GiB for PB)
    N = 1000
    g = random_regular_graph(N, 6; seed = 17)
    p = Dict(:τ => 0.1, :σ => 0.5, :γ => 0.25)
    ib = generate_individual_based(seir_model(), ExplicitGraph(g); p, seed_fraction = 0.01,
                                   tspan = (0.0, 40.0))
    t = 0:0.4:40
    mc = model_curves(ib; t)
    @test (@allocated model_curves(ib; t)) < 50 * 2^20
    @test mc[:E] ≈ [sum(ib.sol(ti)[3(i-1)+2] for i in 1:N) / N for ti in t]   # off the saveat grid
    @test mc[:R] ≈ [1 - sum(ib.sol(ti)[3(i-1)+k] for i in 1:N, k in 1:3) / N for ti in t]
    @test model_curves(ib; t = 0:1:40)[:I] ≈ aggregate(ib, :I) ./ N
    pb = generate_pair_based(sir_model(), ExplicitGraph(g); p = Dict(:τ => 0.1, :γ => 0.25),
                             seed_fraction = 0.01, tspan = (0.0, 40.0))
    mp = model_curves(pb; t)
    @test (@allocated model_curves(pb; t)) < 50 * 2^20
    @test mp[:I] ≈ [sum(pb.sol(ti)[2i] for i in 1:N) / N for ti in t]
    @test model_curves(pb; t = 0:1:40)[:S] ≈ aggregate(pb, :S) ./ N
    # the IB right-hand side reuses its buffers (it allocated the rate vector and the neighbour
    # sums on every call)
    f, u = ib.sol.prob.f, copy(ib.sol.u[end])
    du = similar(u)
    rhs_allocated(f, du, u, ib.sol.prob.p, 1.0)
    @test rhs_allocated(f, du, u, ib.sol.prob.p, 1.0) == 0
    f(du, u, ib.sol.prob.p, 1.0)
    du_ref = copy(du)
    f(du, u, ib.sol.prob.p, 1.0)                   # a second call sees clean buffers
    @test du == du_ref
    # the buffers are the problem's parameter object: a copy of it (one per task, as the
    # docstring of generate_individual_based advises for EnsembleThreads) is independent and gives
    # the same field; the documented ensemble recipe reproduces the solution
    prob = ib.sol.prob
    p2 = deepcopy(prob.p)
    @test p2.pressure !== prob.p.pressure && p2.rate_t !== prob.p.rate_t
    du2 = similar(u)
    f(du2, u, p2, 1.0)
    @test du2 == du_ref
    ens = EnsembleProblem(remake(prob; tspan = (0.0, 5.0));
                          prob_func = (prob, i, repeat) -> remake(prob; p = deepcopy(prob.p)))
    esol = solve(ens, EnsembleThreads(); trajectories = 4, saveat = 1.0)
    ref = solve(remake(prob; tspan = (0.0, 5.0)); saveat = 1.0)
    @test all(Array(esol.u[i]) ≈ Array(ref) for i in 1:4)
end

@testset "Deprecated Gillespie simulators" begin
    @test Base.get_extension(NodeBasedModels, :NodeBasedModelsJumpProcessesExt) !== nothing
    g = random_regular_graph(20, 4; seed = 2)
    r = @test_deprecated gillespie_sir(ExplicitGraph(g); infection_rate = 0.3, recovery_rate = 0.2,
                                       initial_infected = [1], tmax = 20.0, seed = 3)
    @test r isa GillespieResult
    ts, S = aggregate(r, :S; saveat = 1.0)
    _, I = aggregate(r, :I; saveat = 1.0)
    _, R = aggregate(r, :R; saveat = 1.0)
    @test all(S .+ I .+ R .== 20)
    s = @test_deprecated gillespie_sis(ExplicitGraph(g); infection_rate = 0.5, recovery_rate = 0.5,
                                       initial_infected = [1, 2], tmax = 5.0, seed = 1)
    @test s isa GillespieSISResult
    avg = @test_deprecated gillespie_sir_average(ExplicitGraph(g); nruns = 3, infection_rate = 0.3,
                                                 recovery_rate = 0.2, tmax_grid = 10.0, dt = 5.0)
    @test length(avg.t_grid) == 3
    @test_throws ArgumentError gillespie_sir(42)
    err = try
        NodeBasedModels._gillespie_sir(:not_a_network)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("GraphNetwork", sprint(showerror, err))
end
