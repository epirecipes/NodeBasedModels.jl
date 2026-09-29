# multitype_pairwise.jl — the population-level multitype pairwise model on MultitypeNetwork
# (WP36e; DESIGN_NetworkEpiCore.md §K, §C.3, §D.5 M6/M7/M8/M10, §D.6 H7/H8, §E.2, §J.2, §J.6, §J.8).
#
# - Equal to the multitype edge-based model: on Poisson-type blocks with the constant closure
#   (:sir_age2, :sir_sbm2, regular, negative binomial and SplitDegrees blocks; SIR, SEAIR with
#   branching and two infectors, vaccination exits, removals), and for every joint degree law with
#   PGFClosure, at the population and S-anchored levels. The reference `eb_multitype` below is the
#   multitype edge-based field (Miller & Volz 2013, §3.3; DESIGN §D.4 on a typed network) written
#   out numerically, independent of NodeBasedModels and EdgeBasedModels; EdgeBasedModels'
#   multitype lift is a second reference when it can be loaded (it is a test-only dependency).
# - Unit laws: one node type is the untyped population pairwise model, pair by pair (SIR, SIS,
#   SEIR); types assigned independently of the network (M10) reproduce the untyped model; a
#   block-diagonal network decouples (H8, structural zeros).
# - Every registered scenario on a MultitypeNetwork; a :pairwise_multitype => :exact_limit
#   declaration needs constant closure factors (the "if" half of the typed M8).
# - The π^PW image, conservation, seeding (fractions of all nodes, §J.6), the API and errors
#   (including networks that break the edge reciprocity).
# - M6 on a typed network as a runtime semiconjugacy (`verify`, with EdgeBasedModels).
# - Against NetworkOutbreaks ensembles (N = 10⁴, 200 runs, a fresh graph per run, the scenarios'
#   stable_rng streams, conditioned on major outbreaks; DESIGN §E.2, §J.7) on :sir_age2 and
#   :sir_sbm2 (their committed summaries when valid, else the same ensembles computed here), and on
#   two typed networks that are not Poisson type.

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
HAVE_EBM || @info "multitype_pairwise: EdgeBasedModels is not loadable here; its cross-checks are skipped"

const TOL = (reltol = 1e-11, abstol = 1e-13)
const AB = strata([:a, :b])
const BIMODAL = EmpiricalDegree(2 => 5 / 6, 10 => 1 / 6)          # K_ψ(1) = 3/2, not Poisson type
const VARIANTS = ((BernoulliClosure(), :population), (PGFClosure(), :population),
                  (BernoulliClosure(), :s_anchored), (PGFClosure(), :s_anchored))

maxdiff(a, b) = maximum(abs.(a .- b))

# Values of the MTK variable v of `sol` on the grid tt.
at(sol, v, tt) = Float64[sol(t; idxs = v) for t in tt]

function solved(sys, sc; label = nothing)
    sol = solve_epidemic(sys, sc; TOL...)
    mc = label === nothing ? model_curves(sys, sol; t = sc.tgrid) :
         model_curves(sys, sol; t = sc.tgrid, label)
    return sol, mc
end

"""
The multitype edge-based model on `net` (Miller & Volz 2013, §3.3; the per-reaction field of
DESIGN §D.4 on a typed network), solved on `tgrid`. Coordinates: θ_{b→a} for the edge classes of
the types with a susceptible class s_a, ξ_a, φ_{Y,e} (an edge from a partner in the node species Y
to a type-e test node) and pop_Y for the node species Y (and one removal sink `removed_<type>`
per type), and the accumulator; S_a = n_a q_a ξ_a ψ_a(θ_{·→a}). Seeds are fractions of all nodes
(§J.6). A species is infected when the base of its label is in `infected_bases`. Returns the
curves by name (species, :infectious, :cumulative) and by coordinate ((:θ, b, a), (:ξ, a),
(:φ, Y, e)), with q, n, the laws and the node types.
"""
function eb_multitype(cm::ContactModel, net::MultitypeNetwork, p, initial::SeedFraction, tspan,
                      tgrid; infected_bases)
    types = net.types
    ti = Dict(a => i for (i, a) in enumerate(types))
    n = Dict(a => net.sizes[ti[a]] for a in types)
    law = Dict(a => net.degrees[ti[a]] for a in types)
    Mc = mean_contacts(net)
    M(a, b) = Mc[ti[a], ti[b]]
    joined(a, b) = M(a, b) > 0
    labels = species_labels(cm)
    tof = Dict{Symbol,Symbol}(X => labels[X].stratum for X in species_names(cm))
    Σ = susceptible_species(cm)
    sus = Dict(tof[s] => s for s in Σ)
    nodes = [X for X in species_names(cm) if !(X in Σ)]
    for t in node_transitions(cm)
        t.to === nothing || continue
        sink = Symbol(:removed_, tof[t.from])
        sink in nodes || (push!(nodes, sink); tof[sink] = tof[t.from])
    end
    infected = Set(X for X in nodes if haskey(labels, X) && labels[X].base in infected_bases)
    idx = Dict{Any,Int}()
    for a in types, b in types
        (haskey(sus, a) && joined(a, b)) && (idx[(:θ, b, a)] = length(idx) + 1)
    end
    for a in types
        haskey(sus, a) && (idx[(:ξ, a)] = length(idx) + 1)
    end
    for Y in nodes, e in types
        joined(e, tof[Y]) && (idx[(:φ, Y, e)] = length(idx) + 1)
    end
    for Y in nodes
        idx[(:pop, Y)] = length(idx) + 1
    end
    icum = length(idx) + 1
    ρ = Dict{Symbol,Float64}(X => v for (X, v) in seed_fractions(initial) if !(X in Σ))
    q = Dict(a => 1 - sum((get(ρ, X, 0.0) for X in nodes if tof[X] === a); init = 0.0) / n[a]
             for a in keys(sus))
    val(r) = Float64(rate_value(r, p))
    cs = [(c.recipient, c.infector, c.product, val(c.rate)) for c in contacts(cm)]
    ts = [(t.from, something(t.to, Symbol(:removed_, tof[t.from])), val(t.rate))
          for t in node_transitions(cm)]
    xarg(u, a) = b -> haskey(idx, (:θ, b, a)) ? u[idx[(:θ, b, a)]] : 1.0
    function f!(du, u, _, t)
        fill!(du, 0.0)
        for (R, J, P, τ) in cs                                 # contact s_a + J → P + J
            a, b = tof[R], tof[J]
            joined(a, b) || continue
            x = xarg(u, a)
            qξ = q[a] * u[idx[(:ξ, a)]]
            h = τ * u[idx[(:φ, J, a)]]
            du[idx[(:θ, b, a)]] -= h
            du[idx[(:φ, J, a)]] -= h
            for e in types
                joined(e, a) || continue
                du[idx[(:φ, P, e)]] += h * qξ * pgf_derivative(law[a], x, b, e) / M(a, e)
            end
            flux = h * n[a] * qξ * pgf_derivative(law[a], x, b)
            du[idx[(:pop, P)]] += flux
            P in infected && (du[icum] += flux)
        end
        for (W, Z, r) in ts
            if W in Σ                                          # exit s_a → Z
                a = tof[W]
                x = xarg(u, a)
                ξ = u[idx[(:ξ, a)]]
                du[idx[(:ξ, a)]] -= r * ξ
                for e in types
                    joined(e, a) || continue
                    du[idx[(:φ, Z, e)]] += r * q[a] * ξ * pgf_derivative(law[a], x, e) / M(a, e)
                end
                flux = r * n[a] * q[a] * ξ * pgf(law[a], x)
                du[idx[(:pop, Z)]] += flux
                Z in infected && (du[icum] += flux)
            else                                               # progression or removal
                for e in types
                    haskey(idx, (:φ, W, e)) || continue
                    v = r * u[idx[(:φ, W, e)]]
                    du[idx[(:φ, W, e)]] -= v
                    du[idx[(:φ, Z, e)]] += v
                end
                v = r * u[idx[(:pop, W)]]
                du[idx[(:pop, W)]] -= v
                du[idx[(:pop, Z)]] += v
                (!(W in infected) && Z in infected) && (du[icum] += v)
            end
        end
        return nothing
    end
    u0 = zeros(icum)
    for (k, i) in idx
        u0[i] = k[1] in (:θ, :ξ) ? 1.0 :
                k[1] === :pop ? get(ρ, k[2], 0.0) : get(ρ, k[2], 0.0) / n[tof[k[2]]]
    end
    u0[icum] = sum((get(ρ, X, 0.0) for X in infected); init = 0.0)
    sol = OrdinaryDiffEqDefault.solve(ODEProblem(f!, u0, Float64.(tspan)); saveat = tgrid, TOL...)
    U = reduce(hcat, sol.u)
    out = Dict{Any,Vector{Float64}}(k => U[i, :] for (k, i) in idx)
    for (a, s) in sus
        θs = Dict(b => out[(:θ, b, a)] for b in types if haskey(idx, (:θ, b, a)))
        out[s] = [n[a] * q[a] * out[(:ξ, a)][j] * pgf(law[a], b -> haskey(θs, b) ? θs[b][j] : 1.0)
                  for j in eachindex(sol.t)]
    end
    for Y in nodes
        out[Y] = out[(:pop, Y)]
    end
    out[:infectious] = reduce(+, (out[J] for J in infectious_species(cm)))
    out[:cumulative] = U[icum, :]
    return (; curves = out, q, n, law, types = tof, M)
end
eb_multitype(sc::Scenario; infected_bases = [:I]) =
    eb_multitype(sc.model, sc.network, sc.params, sc.initial, sc.tspan, sc.tgrid; infected_bases)

# EdgeBasedModels' curves (the multitype lift of WP17), when it can be loaded.
function ebm_curves(sc::Scenario)
    HAVE_EBM || return nothing
    sys = EdgeBasedModels.edge_based(sc.model, sc.network)
    return model_curves(sys, solve_epidemic(sys, sc; TOL...); t = sc.tgrid)
end

# Symbol-keyed curves of the reference that the system also provides (species, sinks, totals).
common_observables(ref, mc) = [X for X in keys(ref) if X isa Symbol && haskey(mc.values, X)]

# A typed network that is not Poisson type: bimodal edges within type a (K_a(a, a)(1) = 3/2, not
# constant), Poisson(1) edges across, 3-regular within type b (K_b(b, b) = 2/3, constant); τ
# calibrated to R₀ = 2 with NetworkEpiCore's NGM, rounded like the canonical scenarios.
function nonpt_scenario()
    net = MultitypeNetwork([:a, :b], [0.5, 0.5],
                           [IndependentDegrees(:a => BIMODAL, :b => PoissonDegree(1.0)),
                            IndependentDegrees(:a => PoissonDegree(1.0), :b => RegularDegree(3))])
    base = scenario(:sir_sbm2)
    τ = round(calibrate(base.model, net, Dict(:τ => 0.1, :γ => 0.25); target = :R0 => 2.0,
                        vary = :τ)[:τ]; sigdigits = 10)
    return derive(base; id = :sir_nonpt2, network = net, params = Dict(:τ => τ),
                  tspan = (0.0, 80.0),
                  backends = Dict(:edge_based => :exact_limit, :pgf_closure => :exact_limit,
                                  :pairwise_multitype => :biased),
                  tags = [:sir, :stratified, :multitype, :derived])
end

# Types assigned independently of a bimodal network (SplitDegrees, M10): the typed :sir_bim.
unstructured_bimodal_scenario() =
    derive(scenario(:sir_unstr2); id = :sir_unstr2_bim,
           network = unstructured(ConfigurationNetwork(BIMODAL), AB),
           backends = Dict(:edge_based => :exact_limit, :pgf_closure => :exact_limit,
                           :pairwise_multitype => :biased),
           tags = [:sir, :stratified, :multitype, :unit_law, :derived])

@testset "Equal to the multitype edge-based model on Poisson SBMs (:sir_age2, :sir_sbm2)" begin
    for id in (:sir_age2, :sir_sbm2)
        sc = scenario(id)
        @test sc.backends[:pairwise_multitype] === :exact_limit
        ref = eb_multitype(sc).curves
        ebm = ebm_curves(sc)
        if ebm !== nothing                                   # the two references agree
            for X in sc.observables
                @test maxdiff(ebm[X], ref[X]) < 1e-8
            end
        end
        for (closure, level) in VARIANTS
            sys = node_based(sc; closure, level)
            @test sys isa MultitypePairwiseSystem
            @test sys.level === level && sys.closure === closure
            # Poisson blocks: every K_a(c, b)(θ) is 1, so the constant closure is exact
            closure isa BernoulliClosure && @test all(v -> v ≈ 1, values(sys.metadata[:K]))
            _, mc = solved(sys, sc)
            @test mc.representation === (level === :s_anchored ? :s_anchored :
                                         closure isa PGFClosure ? :pgf_closure : :pairwise)
            @test Set(sc.observables) ⊆ Set(keys(mc.values))
            for X in sc.observables                           # species, :infectious, :cumulative
                @test maxdiff(mc[X], ref[X]) < 1e-8
                ebm === nothing || @test maxdiff(mc[X], ebm[X]) < 1e-8
            end
            @test maxdiff(mc[:I], mc[:infectious]) < 1e-14      # the total over the types
            @test maxdiff(mc[:S] .+ mc[:I] .+ mc[:R], ones(length(sc.tgrid))) < 1e-9
        end
        # NetworkEpiCore's final size (the ρ-seeded large-outbreak limit), an independent method
        _, mc = solved(node_based(sc), sc)
        @test abs(mc[:cumulative][end] - sc.expected[:final_size]) < 1e-3
    end
    # the default is the constant closure at the population level, with every pair of joined types
    sc = scenario(:sir_age2)
    sys = node_based(sc)
    @test sys.closure isa BernoulliClosure && sys.level === :population
    @test length(sys.singles) == 6 && length(sys.pairs) == 21
    san = node_based(sc; level = :s_anchored)
    @test length(san.pairs) == 11                           # [S_y ·] and [S_o ·], [S_y S_o] once
    @test all(k -> k[1] in (:S_y, :S_o) || k[2] in (:S_y, :S_o), keys(san.pairs))
end

@testset "Every registered scenario on a MultitypeNetwork; the :pairwise_multitype declarations" begin
    # A scenario may declare :pairwise_multitype => :exact_limit only on a MultitypeNetwork whose
    # constant closure factors K_a(c, b)(θ) are constant in θ (the "if" half of the typed M8);
    # then the constant closure, and PGFClosure always, equal the multitype edge-based model.
    typed = [sc for sc in scenarios() if sc.network isa MultitypeNetwork]
    @test Set(sc.id for sc in typed) ⊇ Set([:sir_sbm2, :sir_unstr2, :sir_age2])
    for sc in scenarios()
        v = get(sc.backends, :pairwise_multitype, nothing)
        v === nothing || @test sc.network isa MultitypeNetwork
    end
    for sc in typed
        net = sc.network
        joined = [(a, b) for (i, a) in enumerate(net.types), (j, b) in enumerate(net.types)
                  if !net.structural_zero[i, j]]
        # the edge-class arguments θ_{·→a}: every assignment of {0.3, 0.65, 1} to the types
        grid = [Dict(zip(net.types, v))
                for v in Iterators.product(ntuple(_ -> (0.3, 0.65, 1.0), length(net.types))...)]
        constant_K = all(enumerate(net.types)) do (i, a)
            m = net.degrees[i]
            ends = [b for (a′, b) in joined if a′ === a]
            all(Iterators.product(ends, ends)) do (c, b)
                Ks = [NodeBasedModels._mt_closure_factor(PGFClosure(), m, c, b, e -> θ[e]) for θ in grid]
                maximum(Ks) - minimum(Ks) < 1e-12
            end
        end
        get(sc.backends, :pairwise_multitype, nothing) === :exact_limit && @test constant_K
        # Admissibility: node types are fixed, so every species must be labelled with a node type.
        # The heterogeneous-susceptibility scenarios (WP36f, DESIGN §L.7: :sir_hetsus_bim,
        # :seirv_hetsus_pois5) type only their susceptible (and latent) classes and share I and R,
        # so a node loses its type on infection; they are edge-based only. Such a scenario must
        # not declare :pairwise_multitype, and node_based must refuse it with an ArgumentError.
        labels = species_labels(sc.model)
        typed_species = all(X -> haskey(labels, X) && labels[X].stratum in net.types,
                            species_names(sc.model))
        if !typed_species
            @test !haskey(sc.backends, :pairwise_multitype)
            err = try
                node_based(sc)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError && occursin("has no stratum", sprint(showerror, err))
            continue
        end
        typing(sc.model).theory === :T_EB || continue
        ref = eb_multitype(sc; infected_bases = unique([labels[X].base
                                                        for X in infected_species(sc.model)])).curves
        variants = constant_K ? ((BernoulliClosure(), :population), (PGFClosure(), :population)) :
                   ((PGFClosure(), :population),)
        for (closure, level) in variants
            _, mc = solved(node_based(sc; closure, level), sc)
            for X in sc.observables
                @test maxdiff(mc[X], ref[X]) < 1e-8
            end
        end
    end
end

@testset "Poisson-type blocks: branching, two infectors, exits, removals, SplitDegrees (typed M8)" begin
    regular = sbm_network(AB; mean_contacts = [4.0 2.0; 2.0 3.0], family = m -> RegularDegree(Int(m)))
    negbin = sbm_network(AB; mean_contacts = [4.0 2.0; 2.0 3.0],
                         family = m -> NegBinDegree(mean = m, var = 2m))
    yo = strata([:y, :o]; sizes = [0.4, 0.6])
    poisson = sbm_network(yo; mean_contacts = [6.0 3.0; 2.0 4.0])
    unstr = unstructured(ConfigurationNetwork(RegularDegree(6)), AB)
    removal = ContactModel(:sir_removal; contacts = [Contact(:S, :I, :I, :τ)],
                           transitions = [NodeTransition(:I, nothing, :γ)])
    seair_p = Dict(:τI => 1 / 6, :τA => 1 / 12, :p => 0.6, :σ => 0.2, :γ => 0.25)
    cases = [
        (stratify(seair_model(), AB), regular, seair_p, SeedFraction(:E_a => 0.005, :E_b => 0.005),
         [:E, :A, :I]),
        (stratify(sirv_model(), AB), negbin, Dict(:τ => 1 / 6, :γ => 0.25, :ν => 0.02),
         SeedFraction(:I_a => 0.005, :I_b => 0.005), [:I]),
        (stratify(removal, yo), poisson, Dict(:τ => 0.1, :γ => 0.25),
         SeedFraction(:I_y => 0.004, :I_o => 0.006), [:I]),
        (stratify(sir_model(), AB), unstr, Dict(:τ => 1 / 6, :γ => 0.25),
         SeedFraction(:I_a => 0.004, :I_b => 0.006), [:I])]
    tspan, tgrid = (0.0, 60.0), 0.0:0.5:60.0
    for (cm, net, p, initial, inf) in cases
        ref = eb_multitype(cm, net, p, initial, tspan, tgrid; infected_bases = inf).curves
        for (closure, level) in VARIANTS
            sys = node_based(cm, net; closure, level, p, initial, tspan)
            mc = model_curves(sys, solve_epidemic(sys; saveat = tgrid, TOL...); t = tgrid)
            obs = common_observables(ref, mc)
            @test issubset(vcat(species_names(cm), [:infectious, :cumulative]), obs)
            for X in obs
                @test maxdiff(mc[X], ref[X]) < 1e-8
            end
        end
    end
    # the constant closure factors: (k − 1)/k within a regular block and 1 across independent
    # blocks; K_ψ(1) = 5/6 for every pair of types on the unstructured 6-regular network
    K = node_based(cases[1][1], regular; p = seair_p, initial = cases[1][4]).metadata[:K]
    @test K[(:a, :a, :a)] ≈ 3 / 4 && K[(:b, :b, :b)] ≈ 2 / 3
    @test K[(:a, :a, :b)] ≈ 1 && K[(:a, :b, :b)] ≈ 1 / 2 && K[(:b, :a, :a)] ≈ 1 / 2
    @test all(v -> v ≈ 5 / 6, values(node_based(cases[4][1], unstr; p = cases[4][3],
                                                 initial = cases[4][4]).metadata[:K]))
    # removals go to one absorbing sink per node type, `removed_<type>`, whose total is :removed
    sys = node_based(cases[3][1], poisson; p = cases[3][3], initial = cases[3][4], tspan)
    @test sys.metadata[:sinks] == [:removed_y, :removed_o]
    @test haskey(sys.singles, :removed_y) && haskey(sys.pairs, (:S_y, :removed_o))
    sol = solve_epidemic(sys; saveat = tgrid, TOL...)
    mc = model_curves(sys, sol; t = tgrid)
    @test maxdiff(mc[:removed], mc[:removed_y] .+ mc[:removed_o]) < 1e-14
    @test maxdiff(mc[:S] .+ mc[:I] .+ mc[:removed], ones(length(tgrid))) < 1e-9
    @test compartment(sys, sol, :removed) ≈ sol[sys.singles[:removed_y]] .+ sol[sys.singles[:removed_o]]
end

@testset "PGFClosure is exact for every joint degree law; the constant closure is not" begin
    sc = nonpt_scenario()
    @test abs(basic_reproduction_number(sc.model, sc.network, sc.params) - 2) < 1e-8
    R = eb_multitype(sc)
    ref = R.curves
    ebm = ebm_curves(sc)
    ebm === nothing || @test maxdiff(ebm[:infectious], ref[:infectious]) < 1e-8
    x(a, j) = b -> haskey(ref, (:θ, b, a)) ? ref[(:θ, b, a)][j] : 1.0
    for level in (:population, :s_anchored)
        sys = node_based(sc; closure = PGFClosure(), level)
        @test length(sys.metadata[:thetas]) == 4                  # every edge class is joined
        sol, mc = solved(sys, sc)
        for X in sc.observables
            @test maxdiff(mc[X], ref[X]) < 1e-8
            ebm === nothing || @test maxdiff(mc[X], ebm[X]) < 1e-8
        end
        # θ_{b→a} is the edge-based θ_{b→a}
        for ((b, a), v) in sys.metadata[:thetas]
            @test maxdiff(at(sol, v, sc.tgrid), ref[(:θ, b, a)]) < 1e-8
            @test compartment(sys, sol, Symbol(:θ_, b, :_, a)) == sol[v]
        end
        # the pairs of the susceptible classes are the π^PW image of the edge-based state:
        # [s_a Y] = n_a q_a ξ_a ∂_cψ_a(θ_{·→a}) φ_{Y,a} (Y of type c), with the edge-S
        # φ_{s_c,a} = q_c ξ_c ∂_aψ_c(θ_{·→c})/M_ca
        Σ = sys.metadata[:susceptible]
        for ((X, Y), v) in sys.pairs
            (X in Σ || Y in Σ) || continue
            s, Z = X in Σ ? (X, Y) : (Y, X)
            a, c = R.types[s], R.types[Z]
            image = map(eachindex(sc.tgrid)) do j
                edge = Z in Σ ?
                       R.q[c] * ref[(:ξ, c)][j] * pgf_derivative(R.law[c], x(c, j), a) / R.M(c, a) :
                       ref[(:φ, Z, a)][j]
                R.n[a] * R.q[a] * ref[(:ξ, a)][j] * pgf_derivative(R.law[a], x(a, j), c) * edge
            end
            @test maxdiff(at(sol, v, sc.tgrid), image) < 1e-8
        end
    end
    # the constant closure: K_a(a, a) = 3/2 (bimodal), K_b(b, b) = 2/3 (3-regular), 1 across; it is
    # biased (pinned) because K_a(a, a)(θ) is not constant, and its S-anchored subsystem has the
    # node observables of its population model (M7)
    con = node_based(sc)
    K = con.metadata[:K]
    @test K[(:a, :a, :a)] ≈ 3 / 2 && K[(:b, :b, :b)] ≈ 2 / 3 && K[(:a, :b, :a)] ≈ 1
    Kθ = [NodeBasedModels._mt_closure_factor(PGFClosure(), R.law[:a], :a, :a, b -> θ)
          for θ in 0.3:0.1:1.0]
    @test Kθ[end] ≈ 3 / 2 && maximum(Kθ) - minimum(Kθ) > 0.1
    _, cpop = solved(con, sc)
    _, csan = solved(node_based(sc; level = :s_anchored), sc)
    for c in (cpop, csan)
        ΔR = c[:cumulative][end] - ref[:cumulative][end]
        @test 0.004 < ΔR < 0.008                                 # 0.0057
        @test 0.003 < maxdiff(c[:infectious], ref[:infectious]) < 0.006
    end
    for X in sc.observables
        @test maxdiff(cpop[X], csan[X]) < 1e-8
    end
end

@testset "Unit laws: one node type, types independent of the network (M10), disjoint blocks (H8)" begin
    tt = 0.0:0.5:40.0
    # one node type is the untyped population pairwise model, pair by pair, for the constant closure
    # (PairwiseSystem, K = closure_constant(d); SIS included) and for PGFClosure (SAnchoredSystem)
    for (cm, d) in ((sir_model(), BIMODAL), (sis_model(), BIMODAL),
                    (seir_model(), NegBinDegree(mean = 4, var = 8)))
        one = MultitypeNetwork([:a], [1.0], [IndependentDegrees(:a => d)])
        cm1 = stratify(cm, strata([:a]))
        @test species_names(cm1) == species_names(cm)
        entry = nameof(cm) === :seir ? :E : :I
        p = Dict(:τ => 0.2, :γ => 0.25)
        nameof(cm) === :seir && (p[:σ] = 0.2)
        kw = (; p, initial = SeedFraction(entry => 0.01), tspan = (0.0, 40.0))
        pairs_ = Any[(BernoulliClosure(), :population, node_based(cm, ConfigurationNetwork(d); kw...))]
        if nameof(cm) !== :sis
            for level in (:population, :s_anchored)
                push!(pairs_, (PGFClosure(), level,
                               node_based(cm, ConfigurationNetwork(d); closure = PGFClosure(),
                                          level, kw...)))
            end
        end
        for (closure, level, plain) in pairs_
            typed = node_based(cm1, one; closure, level, kw...)
            @test Set(keys(typed.pairs)) == Set(keys(plain.pairs))
            @test Set(keys(typed.singles)) == Set(keys(plain.singles))
            closure isa BernoulliClosure &&
                @test only(values(typed.metadata[:K])) ≈ closure_constant(d)
            a = solve_epidemic(typed; saveat = tt, TOL...)
            b = solve_epidemic(plain; saveat = tt, TOL...)
            for (k, v) in typed.pairs
                @test maxdiff(at(a, v, tt), at(b, plain.pairs[k], tt)) < 1e-8
            end
            for (X, v) in typed.singles
                @test maxdiff(at(a, v, tt), at(b, plain.singles[X], tt)) < 1e-8
            end
            if closure isa PGFClosure
                @test maxdiff(at(a, typed.metadata[:thetas][(:a, :a)], tt),
                              at(b, plain.metadata[:θ], tt)) < 1e-8
            end
        end
    end

    # M10: stratify on unstructured(net) with proportional seeds reproduces the untyped model; the
    # pairs split as [X_a Y_b] = n_a n_b [XY]. The constant closure keeps the bias of :sir_bim and
    # PGFClosure its exactness; SIS (T_net) with the constant closure too.
    st = strata([:a, :b]; sizes = [0.3, 0.7])
    for (cm, d, closure, τ) in ((sir_model(), BIMODAL, BernoulliClosure(), 1 / 6),
                                (sir_model(), BIMODAL, PGFClosure(), 1 / 6),
                                (sis_model(), RegularDegree(3), BernoulliClosure(), 0.5))
        p = Dict(:τ => τ, :γ => 0.25)
        typed = node_based(stratify(cm, st), unstructured(ConfigurationNetwork(d), st); closure, p,
                           initial = SeedFraction(:I_a => 0.003, :I_b => 0.007), tspan = (0.0, 60.0))
        plain = node_based(cm, ConfigurationNetwork(d); closure, p,
                           initial = SeedFraction(:I => 0.01), tspan = (0.0, 60.0))
        t60 = 0.0:0.5:60.0
        # tstops: compare step values, not the saveat values of OrdinaryDiffEqDefault's composite
        # solver, which after a Vern7 → Rodas5P switch in the last step misplaces the linearly
        # growing SIS accumulator at the endemic equilibrium (typed: 11.357 at t = 59.5 against
        # 11.511 from the dense solution and from tstops; reported to the solve_epidemic owner)
        a = solve_epidemic(typed; saveat = t60, tstops = t60, TOL...)
        b = solve_epidemic(plain; saveat = t60, tstops = t60, TOL...)
        ca, cb = model_curves(typed, a; t = t60), model_curves(plain, b; t = t60)
        for X in vcat(species_names(cm), [:infectious, :cumulative])
            @test maxdiff(ca[X], cb[X]) < 1e-8
        end
        @test maxdiff(at(a, typed.pairs[(:S_a, :I_b)], t60),
                      0.3 * 0.7 .* at(b, plain.pairs[(:S, :I)], t60)) < 1e-8
        @test maxdiff(at(a, typed.pairs[(:S_b, :S_b)], t60),
                      0.7^2 .* at(b, plain.pairs[(:S, :S)], t60)) < 1e-8
    end
    # the typed :sir_bim: the untyped curves of :sir_bim (EB final size 0.4956; constant-K bias 0.034)
    sc = unstructured_bimodal_scenario()
    bim = scenario(:sir_bim)
    for closure in (BernoulliClosure(), PGFClosure())
        _, ct = solved(node_based(sc; closure), sc)
        _, cu = solved(node_based(bim; closure), bim)
        for X in (:S, :I, :R, :infectious, :cumulative)
            @test maxdiff(ct[X], cu[X]) < 1e-8
        end
    end
    _, cpgf = solved(node_based(sc; closure = PGFClosure()), sc)
    _, ccon = solved(node_based(sc), sc)
    @test abs(cpgf[:cumulative][end] - 0.4956) < 5e-4
    @test 0.030 < ccon[:cumulative][end] - cpgf[:cumulative][end] < 0.038

    # H8: no edges between the types (structural zeros), so no cross pairs, the contacts across
    # the zero do nothing, and each type is the untyped model on its own block with the seed
    # fraction ρ_a/n_a within the type
    st = strata([:a, :b]; sizes = [0.4, 0.6])
    net = sbm_network(st; mean_contacts = [5.0 0.0; 0.0 3.0])
    @test Set(structural_zeros(net)) == Set([(:a, :b), (:b, :a)])
    p = Dict(:τ => 0.2, :γ => 0.25)
    initial = SeedFraction(:I_a => 0.004, :I_b => 0.012)
    for model in (stratify(sir_model(), st), disjoint_union(:a => sir_model(), :b => sir_model()))
        for closure in (BernoulliClosure(), PGFClosure())
            sys = node_based(model, net; closure, p, initial, tspan = (0.0, 40.0))
            types = sys.metadata[:types]
            @test all(k -> types[k[1]] === types[k[2]], keys(sys.pairs))
            closure isa PGFClosure && @test Set(keys(sys.metadata[:thetas])) == Set([(:a, :a), (:b, :b)])
            sol = solve_epidemic(sys; saveat = tt, TOL...)
            for (a, d, n, ρ) in ((:a, PoissonDegree(5), 0.4, 0.01), (:b, PoissonDegree(3), 0.6, 0.02))
                plain = node_based(sir_model(), ConfigurationNetwork(d); closure, p,
                                   initial = SeedFraction(:I => ρ), tspan = (0.0, 40.0))
                b = solve_epidemic(plain; saveat = tt, TOL...)
                for X in (:S, :I, :R)
                    @test maxdiff(at(sol, sys.singles[Symbol(X, :_, a)], tt) ./ n,
                                  at(b, plain.singles[X], tt)) < 1e-8
                end
            end
        end
    end
end

@testset "Conservation, and the constant closure on T_net models" begin
    # PGFClosure on the typed non-Poisson-type network in counts (N = 1000): Σ[X] = N; the edge
    # ends of every block, Σ_{t(X) = a, t(Y) = b}[XY] = N n_a M_ab (ordered pairs); the stubs of the
    # susceptible nodes, Σ_{t(Y) = b}[s_a Y] = [s_a] θ_{b→a} ∂_bψ_a(θ)/ψ_a(θ)
    sc = nonpt_scenario()
    net = sc.network
    M = mean_contacts(net)
    law = Dict(a => net.degrees[i] for (i, a) in enumerate(net.types))
    n = Dict(a => net.sizes[i] for (i, a) in enumerate(net.types))
    tt = collect(sc.tgrid)
    for level in (:population, :s_anchored)
        sys = node_based(sc; closure = PGFClosure(), level, N = 1000)
        sol = solve_epidemic(sys, sc; TOL...)
        types = sys.metadata[:types]
        @test maxdiff(sum(at(sol, v, tt) for v in values(sys.singles)), fill(1000.0, length(tt))) < 1e-7
        θ = Dict(k => at(sol, v, tt) for (k, v) in sys.metadata[:thetas])
        # Σ_{t(Y) = b}[s_a Y] over the ordered pairs (u ∈ s_a, v ∈ Y): each stored pair once (the
        # self pair [s_a s_a] already counts both orders)
        for (a, s) in ((:a, :S_a), (:b, :S_b)), b in (:a, :b)
            stubs = sum(at(sol, v, tt) for ((X, Y), v) in sys.pairs
                        if (X === s && types[Y] === b) || (Y === s && types[X] === b))
            S = at(sol, sys.singles[s], tt)
            x = j -> (c -> θ[(c, a)][j])
            expected = [S[j] * θ[(b, a)][j] * pgf_derivative(law[a], x(j), b) / pgf(law[a], x(j))
                        for j in eachindex(tt)]
            @test maxdiff(stubs, expected) < 1e-7
        end
        if level === :population
            # ordered pairs within a type: a cross pair [XY] stands for [XY] and [YX]
            for (i, a) in enumerate(net.types), (j, b) in enumerate(net.types)
                ends = sum(at(sol, v, tt) .* (a === b && X !== Y ? 2 : 1)
                           for ((X, Y), v) in sys.pairs
                           if Set((types[X], types[Y])) == Set((a, b)))
                @test maxdiff(ends, fill(1000 * n[a] * M[i, j], length(tt))) < 1e-6
            end
        end
        # counts on the scale N = 1000 give the fractions of N = 1
        _, c1 = solved(node_based(sc; closure = PGFClosure(), level), sc)
        c1000 = model_curves(sys, sol; t = sc.tgrid)
        for X in sc.observables
            @test maxdiff(c1000[X], c1[X]) < 1e-9
        end
    end
    # SIS and SIRS (arrows back into S) on a Poisson SBM: the constant closure is total on T_net;
    # singles and block edge ends are conserved, and the solution reaches an endemic equilibrium
    # (τ Σ_J [S_a J] = γ[I_a] per type for SIS; γ[I_a] = ε[R_a] for SIRS)
    sbm = scenario(:sir_sbm2).network
    for (cm, p) in ((sis_model(), Dict(:τ => 0.2, :γ => 0.25)),
                    (sirs_model(), Dict(:τ => 0.2, :γ => 0.25, :ε => 0.02)))
        sys = node_based(stratify(cm, AB), sbm; p, initial = SeedFraction(:I_a => 0.005, :I_b => 0.005),
                         tspan = (0.0, 1000.0))
        @test sys.closure isa BernoulliClosure && sys.metadata[:representation] === :pairwise
        t3 = 0.0:5.0:1000.0
        sol = solve_epidemic(sys; saveat = t3, TOL...)
        types = sys.metadata[:types]
        @test maxdiff(sum(at(sol, v, t3) for v in values(sys.singles)), ones(length(t3))) < 1e-8
        ends_ab = sum(at(sol, v, t3) for ((X, Y), v) in sys.pairs if types[X] !== types[Y])
        @test maxdiff(ends_ab, fill(0.5 * 2.0, length(t3))) < 1e-8
        ends_aa = sum(at(sol, v, t3) .* (X === Y ? 1 : 2) for ((X, Y), v) in sys.pairs
                      if types[X] === types[Y] === :a)
        @test maxdiff(ends_aa, fill(0.5 * 6.0, length(t3))) < 1e-8
        mc = model_curves(sys, sol; t = t3)
        @test mc[:infectious][end] > 0.01 && all(v -> v > -1e-12, mc[:S])
        last_(v) = sol(1000.0; idxs = v)
        for a in (:a, :b)
            I = last_(sys.singles[Symbol(:I_, a)])
            if nameof(cm) === :sis
                SJ = last_(sys.pairs[(Symbol(:S_, a), :I_a)]) + last_(sys.pairs[(Symbol(:S_, a), :I_b)])
                @test isapprox(p[:τ] * SJ, p[:γ] * I; rtol = 1e-6)
            else
                @test isapprox(p[:γ] * I, p[:ε] * last_(sys.singles[Symbol(:R_, a)]); rtol = 1e-6)
            end
        end
    end
end

@testset "Seeding (fractions of all nodes, §J.6), parameters and the API" begin
    sc = scenario(:sir_age2)
    cm, net, p = sc.model, sc.network, sc.params
    sys = node_based(cm, net; p)            # default: a fraction ε = 1e-3 of every type, in I
    S_y, S_o, I_y, I_o = (sys.singles[X] for X in (:S_y, :S_o, :I_y, :I_o))
    u = default_initial_conditions(sys)
    @test u === sys.u0
    @test u[I_y] ≈ 1e-3 * 0.4 && u[I_o] ≈ 1e-3 * 0.6 && u[S_y] ≈ 0.999 * 0.4 && u[S_o] ≈ 0.999 * 0.6
    @test u[sys.metadata[:cumulative]] ≈ 1e-3
    @test sys.metadata[:background] == Dict(:y => :S_y, :o => :S_o)
    # the π^PW image of the edge-based initial condition: [XY] = M_ab x_X x_Y / n_b
    @test u[sys.pairs[(:S_y, :I_o)]] ≈ 3.0 * (0.999 * 0.4) * (1e-3 * 0.6) / 0.6
    @test u[sys.pairs[(:S_o, :I_y)]] ≈ 2.0 * (0.999 * 0.6) * (1e-3 * 0.4) / 0.4
    @test u[sys.pairs[(:S_y, :S_y)]] ≈ 6.0 * (0.999 * 0.4)^2 / 0.4
    @test u[sys.pairs[(:S_y, :S_o)]] ≈ 3.0 * (0.999 * 0.4) * (0.999 * 0.6) / 0.6
    # explicit seeds are fractions of all nodes
    v = default_initial_conditions(sys; initial = sc.initial)
    @test v[I_y] ≈ 0.004 && v[I_o] ≈ 0.006 && v[S_y] ≈ 0.396 && v[S_o] ≈ 0.594
    v = default_initial_conditions(sys; initial = SeedFraction(:I_y => 0.004, :R_o => 0.06))
    @test v[S_o] ≈ 0.54 && v[S_y] ≈ 0.396
    @test v[sys.metadata[:cumulative]] ≈ 0.004              # seeded R is not infected (§J.8)
    @test default_initial_conditions(sys; seed_fraction = 0.02)[I_o] ≈ 0.012
    # a susceptible class may be named with the fraction its type leaves, not with another
    @test default_initial_conditions(sys; initial = SeedFraction(:I_y => 0.004, :S_y => 0.396))[S_y] ≈ 0.396
    @test_throws ArgumentError default_initial_conditions(sys; initial = SeedFraction(:I_y => 0.004, :S_y => 0.5))
    @test_throws ArgumentError default_initial_conditions(sys; initial = SeedFraction(:I_y => 0.5))   # > n_y
    @test_throws ArgumentError default_initial_conditions(sys; initial = SeedFraction(:I => 0.01))
    @test_throws ArgumentError default_initial_conditions(sys; initial = SeedFraction(:I_y => 0.1; default = :S_y))
    @test_throws ArgumentError default_initial_conditions(sys; seed_fraction = 1.5)
    # counts on N nodes
    big = node_based(cm, net; p, N = 1000, initial = SeedCount(:I_y => 4, :I_o => 6))
    @test big.u0[big.singles[:I_y]] ≈ 4 && big.u0[big.singles[:S_o]] ≈ 594
    @test big.u0[big.metadata[:cumulative]] ≈ 10
    # SEIR: the entry state E of every type by default, I with seed_state = :first_infectious
    seir = node_based(stratify(seir_model(), AB), scenario(:sir_sbm2).network; seed_fraction = 0.01)
    @test seir.u0[seir.singles[:E_a]] ≈ 0.005 && seir.u0[seir.singles[:E_b]] ≈ 0.005
    @test seir.u0[seir.singles[:I_a]] == 0.0
    w = default_initial_conditions(seir; seed_state = :first_infectious)
    @test w[seir.singles[:I_b]] ≈ 0.005 && w[seir.singles[:E_b]] == 0.0
    # two strains: several entry states per type, so an explicit initial is needed
    two = stratify(twostrain_model(), AB)
    sbm = scenario(:sir_sbm2).network
    @test_throws ArgumentError node_based(two, sbm)
    @test node_based(two, sbm; initial = SeedFraction(:I1_a => 0.005, :I2_b => 0.005)) isa
          MultitypePairwiseSystem

    # solving: keyword, positional and scenario forms; parameter errors
    tt = 0.0:1.0:30.0
    a = solve_epidemic(sys; saveat = tt, TOL...)
    b = solve_epidemic(sys, Dict(:τ => p[:τ], :γ => p[:γ]); saveat = tt, TOL...)
    @test compartment(sys, a, :I_y) == compartment(sys, b, :I_y)
    c = solve_epidemic(sys; p = Dict(:τ => 0.2), saveat = tt, TOL...)
    @test compartment(sys, c, :I_y) != compartment(sys, a, :I_y)
    @test_throws ArgumentError solve_epidemic(sys; p = Dict(:β => 0.3))
    bare = node_based(cm, net)
    @test_throws ArgumentError solve_epidemic(bare)                     # τ and γ have no values
    d = solve_epidemic(sys; initial = SeedFraction(:I_y => 0.02, :I_o => 0.03), saveat = tt, TOL...)
    @test compartment(sys, d, :I)[1] ≈ 0.05
    # observing: compartments, totals over the types, the accumulator, θ
    @test population_fraction(sys, a, :I) ≈ compartment(sys, a, :I_y) .+ compartment(sys, a, :I_o)
    @test compartment(sys, a, :cumulative)[end] ≈ 1 - compartment(sys, a, :S)[end]
    @test Set(keys(compartments(sys, a, [:S_y, :I]))) == Set([:S_y, :I])
    @test_throws ArgumentError compartment(sys, a, :E)
    pgf = node_based(cm, net; p, closure = PGFClosure())
    e = solve_epidemic(pgf; saveat = tt, TOL...)
    @test all(diff(compartment(pgf, e, :θ_o_y)) .<= 1e-14)
    @test_throws ArgumentError population_fraction(pgf, e, :θ_o_y)
    big_sol = solve_epidemic(big; saveat = tt, TOL...)
    @test population_fraction(big, big_sol, :I_y) ≈ compartment(big, big_sol, :I_y) ./ 1000
    mc = model_curves(big, big_sol)
    @test mc[:I_y] ≈ compartment(big, big_sol, :I_y) ./ 1000
    @test Set([:S, :I, :R, :infectious, :cumulative]) ⊆ Set(keys(mc.values))
    @test model_curves(sys, a).label == "multitype pairwise (constant K)"
    @test model_curves(pgf, e).label == "multitype pairwise (PGF closure)"
    @test model_curves(node_based(cm, net; p, level = :s_anchored), a).label ==
          "S-anchored multitype pairwise (constant K)"
    @test model_curves(sys, a; label = "x").label == "x"
    @test node_variables(sys) === sys.singles && pair_variables(sys) === sys.pairs
    @test occursin("2 types", sprint(show, sys)) && occursin("4 θ", sprint(show, pgf))
    @test default_closure(net) isa BernoulliClosure
    # the vector field: θ (PGFClosure), the singles and the tracked pairs, without the accumulator
    @test Set(state_names(symbolic_ode(node_based(cm, net; level = :s_anchored)))) ==
          Set(vcat(species_names(cm),
                   [:S_yS_y, :S_yS_o, :S_yI_y, :S_yI_o, :S_yR_y, :S_yR_o,
                    :S_oS_o, :S_oI_y, :S_oI_o, :S_oR_y, :S_oR_o]))
    @test Set(state_names(symbolic_ode(pgf))) ⊇ Set([:θ_y_y, :θ_o_y, :θ_y_o, :θ_o_o])
    @test !(:cumulative in state_names(symbolic_ode(pgf)))
end

@testset "Errors" begin
    sbm = scenario(:sir_sbm2).network
    sir_ab = scenario(:sir_sbm2).model
    # an unstratified model has no node types
    @test_throws ArgumentError node_based(sir_model(), sbm)
    # SIS: T_net, so neither PGFClosure nor the S-anchored level (an arrow back into S)
    sis_ab = stratify(sis_model(), AB)
    for level in (:population, :s_anchored)
        @test_throws AdmissibilityError node_based(sis_ab, sbm; closure = PGFClosure(), level)
    end
    @test_throws AdmissibilityError node_based(sis_ab, sbm; level = :s_anchored)
    # closures other than the constant and PGF closures
    for closure in (KeelingClosure(), BarnardClosure(), MeanFieldClosure(), KirkwoodClosure(),
                    PowerClosure(2.0), MotifClosure(3, 2))
        @test_throws ArgumentError node_based(sir_ab, sbm; closure)
    end
    @test_throws ArgumentError node_based(sir_ab, sbm; closure = KeelingClosure(), level = :s_anchored)
    # node types are fixed: a contact or a transition that changes the type
    lab(b, a) = SpeciesLabel(b; stratum = a)
    labels = Dict(:S_a => lab(:S, :a), :S_b => lab(:S, :b), :I_a => lab(:I, :a),
                  :I_b => lab(:I, :b), :R_a => lab(:R, :a), :R_b => lab(:R, :b))
    crossc = ContactModel(:crossc; contacts = [Contact(:S_a, :I_b, :I_b, :τ), Contact(:S_b, :I_b, :I_b, :τ),
                                               Contact(:S_a, :I_a, :I_a, :τ)],
                          transitions = [NodeTransition(:I_a, :R_a, :γ), NodeTransition(:I_b, :R_b, :γ)],
                          labels)
    @test_throws ArgumentError node_based(crossc, sbm; initial = SeedFraction(:I_a => 0.01))
    crosst = ContactModel(:crosst; contacts = [Contact(:S_a, :I_a, :I_a, :τ), Contact(:S_b, :I_b, :I_b, :τ)],
                          transitions = [NodeTransition(:I_a, :R_b, :γ), NodeTransition(:I_b, :R_b, :γ)],
                          labels = filter(l -> first(l) !== :R_a, labels))
    @test_throws ArgumentError node_based(crosst, sbm; initial = SeedFraction(:I_a => 0.01))
    # a stratified model must use per-contact rates on a typed network
    sirF = ContactModel(:sirF; contacts = [Contact(:S, :I, :I, :β)],
                        transitions = [NodeTransition(:I, :R, :γ)], convention = FrequencyDependent())
    @test_throws ArgumentError node_based(stratify(sirF, AB), sbm)
    # symbolic degree parameters
    @variables μ
    sym = MultitypeNetwork([:a, :b], [0.5, 0.5],
                           [IndependentDegrees(:a => PoissonDegree(μ), :b => PoissonDegree(1.0)),
                            IndependentDegrees(:a => PoissonDegree(1.0), :b => PoissonDegree(μ))])
    @test_throws ArgumentError node_based(sir_ab, sym)
    # a network that breaks the edge reciprocity (possible only with check_reciprocity = false):
    # unequal edge ends, and a one-sided structural zero
    for Mab in ([5.0 2.0; 1.0 3.0], [5.0 2.0; 0.0 3.0])
        law(i) = IndependentDegrees(:a => PoissonDegree(Mab[i, 1]), :b => PoissonDegree(Mab[i, 2]))
        @test_throws ArgumentError MultitypeNetwork([:a, :b], [0.5, 0.5], [law(1), law(2)])
        nonrec = MultitypeNetwork([:a, :b], [0.5, 0.5], [law(1), law(2)]; check_reciprocity = false)
        err = try
            node_based(sir_ab, nonrec; p = Dict(:τ => 0.1, :γ => 0.25))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("reciprocity", sprint(showerror, err))
    end
    # the closures divide by [s_a]: a type with no susceptible node left
    for closure in (BernoulliClosure(), PGFClosure())
        @test_throws ArgumentError node_based(sir_ab, sbm; closure, initial = SeedFraction(:I_a => 0.5))
    end
    # keywords
    @test_throws ArgumentError node_based(sir_ab, sbm; seed_state = :bogus)
    @test_throws ArgumentError node_based(sir_ab, sbm; N = 0)
    @test_throws MethodError node_based(sir_ab, sbm; seedfraction = 0.1)        # misspelt
    # a removal sink that clashes with a species name
    clash = ContactModel(:clash; contacts = [Contact(:S_a, :I_a, :I_a, :τ), Contact(:S_b, :I_b, :I_b, :τ)],
                         transitions = [NodeTransition(:I_a, nothing, :γ), NodeTransition(:I_b, :removed_a, :γ)],
                         labels = merge(filter(l -> !(first(l) in (:R_a, :R_b)), labels),
                                        Dict(:removed_a => lab(:removed_a, :b))))
    @test_throws ArgumentError node_based(clash, sbm; initial = SeedFraction(:I_a => 0.01))
end

# M6 on a typed network as a runtime morphism: the map π^PW from EdgeBasedModels' multitype field
# onto `symbolic_ode` of the S-anchored multitype system; `verify` checks Dπ·F = G∘π.
function m6_multitype_map(eb, nb)
    co = eb.metadata[:coords]
    net = eb.metadata[:network]
    ti = Dict(a => i for (i, a) in enumerate(net.types))
    law = Dict(a => net.degrees[i] for (a, i) in ti)
    n = Dict(a => net.sizes[i] for (a, i) in ti)
    Mc = mean_contacts(net)
    tof = nb.metadata[:types]
    Σ = nb.metadata[:susceptible]
    q(s) = eb.metadata[:q][s].param
    ξ(s) = get(co, Symbol(:ξ_, s), 1)
    x(a) = b -> get(co, Symbol(:θ_, b, :_, a), 1.0)
    mp = Pair{Any,Any}[]
    for ((b, a), v) in nb.metadata[:thetas]
        push!(mp, v => co[Symbol(:θ_, b, :_, a)])
    end
    for (X, v) in nb.singles
        a = tof[X]
        push!(mp, v => (X in Σ ? n[a] * q(X) * ξ(X) * pgf(law[a], x(a)) : co[Symbol(:pop_, X)]))
    end
    for ((X, Y), v) in nb.pairs
        s, Z = X in Σ ? (X, Y) : (Y, X)
        a, c = tof[s], tof[Z]
        edge = Z in Σ ? q(Z) * ξ(Z) * pgf_derivative(law[c], x(c), a) / Mc[ti[c], ti[a]] :
               co[Symbol(:φ_, Z, :_, a)]
        push!(mp, v => n[a] * q(s) * ξ(s) * pgf_derivative(law[a], x(a), c) * edge)
    end
    return mp
end

@testset "M6 on a typed network as a semiconjugacy: verify(EB → multitype PW^S)" begin
    if HAVE_EBM
        sc = scenario(:sir_age2)
        eb = EdgeBasedModels.edge_based(sc.model, sc.network)
        for closure in (PGFClosure(), BernoulliClosure())        # Poisson blocks: both exact
            nbs = node_based(sc.model, sc.network; closure, level = :s_anchored)
            m = Semiconjugacy(:eb_to_mt_pws, symbolic_ode(eb), symbolic_ode(nbs), m6_multitype_map(eb, nbs))
            @test verify(m).ok
        end
        # a perturbed map ([S_y S_y] without the factor 1/M_yy) is not a semiconjugacy
        nbs = node_based(sc.model, sc.network; closure = PGFClosure(), level = :s_anchored)
        bad = [k => (isequal(k, nbs.pairs[(:S_y, :S_y)]) ? 6 * v : v)
               for (k, v) in m6_multitype_map(eb, nbs)]
        @test !verify(Semiconjugacy(:perturbed, symbolic_ode(eb), symbolic_ode(nbs), bad)).ok
    else
        @test_skip "EdgeBasedModels is not loadable: the typed M6 semiconjugacy is not verified"
    end
end

# The reference ensemble of a scenario (N = 10⁴, 200 runs, a fresh typed graph per run, the
# stable_rng streams, MajorOutbreak(0.05); DESIGN §E.2, §J.7). A registered scenario uses its
# committed summary when a valid one exists (NetworkOutbreaks' data/scenarios, WP30) and otherwise
# computes the same ensemble locally without caching it (policy = :auto; under
# NETEPI_STRICT_CACHE=1, as in CI, a missing committed summary is an error). The derived scenarios
# below are not registered, so their ensembles are always computed here.
reference_summary(sc::Scenario) =
    :derived in sc.tags ? NetworkOutbreaks.summarise(NetworkOutbreaks.scenario_ensemble(sc)) :
    NetworkOutbreaks.scenario_summary(sc; policy = :auto, cache_dir = nothing)

@testset "Against NetworkOutbreaks: exact limit on :sir_age2 and :sir_sbm2; non-Poisson-type typed networks" begin
    for sc in (scenario(:sir_age2), scenario(:sir_sbm2), nonpt_scenario(),
               unstructured_bimodal_scenario())
        ref = reference_summary(sc)
        @test ref.scenario_hash == scenario_hash(sc)
        @test ref.N == 10_000 && ref.nsims == 200
        curves = [last(solved(node_based(sc; closure, level), sc; label))
                  for (closure, level, label) in ((BernoulliClosure(), :population, "const"),
                                                  (PGFClosure(), :population, "pgf"),
                                                  (PGFClosure(), :s_anchored, "s_anchored"))]
        tab = compare(ref, curves)
        obs = vcat(infectious_species(sc.model), [:infectious])
        summary = join(("$(r.label) $(r.observable): D∞ = $(round(r.D∞; sigdigits = 3)), " *
                        "SE∞ = $(round(r.SE∞; sigdigits = 3)), ΔR∞ = $(round(r.ΔR∞; sigdigits = 3)) " *
                        "($(round.(r.ΔR∞_ci; sigdigits = 3)))" for r in tab if r.observable in obs), "\n")
        @info "multitype_pairwise: $(sc.id) against NetworkOutbreaks (N = $(ref.N), " *
              "$(ref.n_major)/$(ref.nsims) major)\n" * summary
        exact = sc.backends[:pairwise_multitype] === :exact_limit ? ("const", "pgf", "s_anchored") :
                ("pgf", "s_anchored")
        for label in exact, X in obs                         # :exact_limit (§E.2 tolerances)
            @test tab[label, X].D∞ < 0.005
            @test abs(tab[label, X].ΔR∞) < 0.005
        end
        if sc.id === :sir_unstr2_bim                          # the pinned bias of constant K
            r = tab["const", :infectious]
            @test r.ΔR∞ > 0.02 && r.ΔR∞_ci[1] > 0
        end
    end
end
