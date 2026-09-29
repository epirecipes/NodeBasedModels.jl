# Golden cases, area "graph_level": the individual-based (order-1, NIMFA) and
# pair-based (order-2, Kirkwood) models on fixed small graphs.
#
# The public calls use the default (loose) solver tolerances, so each case
# re-solves the ODEProblem the call built (`result.sol.prob`: the package's own
# initial condition and right-hand side) at reltol 1e-10 on t = 0:1:T, and also
# freezes that right-hand side at three deterministic probe states. Every
# per-node (and, for PB, per-directed-edge) variable is frozen, plus the
# aggregates [X](t) = Σᵢ⟨Xᵢ⟩ returned by `aggregate`.
#
# WP24 (NodeBasedModels 0.2) replaced the two B02 goldens (verified issue B02: 0.1 ran every
# spontaneous transition of `generate_individual_based` at `recovery_rate`, σ = ε = γ):
# - `ib_seir_karate_node1` and `ib_sirs_karate_node1` run each transition at its own rate
#   (`p`) and seed the entry state of the infections (E for SEIR; DESIGN §E.2). The new numbers
#   are justified by test/suites/graph_level.jl: the exact τ = 0 solutions of the node dynamics
#   (Sharkey 2008, Eq. 3/7) and NIMFA against the exact stochastic process on a dense graph.
# - `ib_seir_karate_node1_sigma_eq_gamma_seedI` and `ib_sirs_karate_node1_eps_eq_gamma` pin the 0.1
#   numbers (σ = γ, resp. ε = γ, and SEIR seeded in I via `seed_state = :first_infectious`); their
#   tables equal the deleted `*_B02` goldens.
# - `ib_staged_karate_via` freezes per-infector infection transitions (verified issue B04).
#
# Owner: WP24 (NodeBasedModels fixes: IB rate dictionary).

include(joinpath(@__DIR__, "..", "common.jl"))

using OrdinaryDiffEqDefault

const GRAPH_T = 30.0

function _ib_layout_names(r)
    # u[K(i-1) + k] is node i in tracked state k (see individual_based.jl)
    return ["$(r.state_names[k])_$(i)" for i in 1:r.N for k in 1:r.K]
end

function _pb_layout_names(r)
    # u[2(i-1)+1] = S_i, u[2(i-1)+2] = I_i, then SS_e, SI_e per directed edge e = (i, j)
    names = String[]
    for i in 1:r.N
        push!(names, "S_$i", "I_$i")
    end
    for (i, j) in r.directed_edges
        push!(names, "SS_$(i)_$(j)", "SI_$(i)_$(j)")
    end
    return names
end

function _resolve(prob, T)
    prob.tspan == (0.0, T) || error("unexpected tspan $(prob.tspan)")
    sol = OrdinaryDiffEqDefault.solve(prob; saveat = 0.0:1.0:T,
                                      reltol = GOLDEN_RELTOL, abstol = GOLDEN_ABSTOL)
    string(sol.retcode) == "Success" || error("graph-level golden solve returned $(sol.retcode)")
    return sol
end

function _layout_trajectory(sol, names)
    cols = Dict{String,Vector{Float64}}()
    for (idx, n) in enumerate(names)
        cols[n] = [u[idx] for u in sol.u]
    end
    return cols
end

function _layout_probes(prob, names; salt)
    states = probe_states(names, 1.0, 3; salt = salt)
    derivs = Dict{String,Float64}[]
    for st in states
        u = [st[n] for n in names]
        du = similar(u)
        prob.f(du, u, prob.p, 0.0)
        push!(derivs, Dict(n => du[idx] for (idx, n) in enumerate(names)))
    end
    return probe_table(states, derivs)
end

function graph_record(name; kind::Symbol, model, graph, net, T = GRAPH_T, description = "",
                      known_defects = String[], notes = String[], kwargs...)
    r = kind === :ib ? generate_individual_based(model, net; tspan = (0.0, T), kwargs...) :
                       generate_pair_based(model, net; tspan = (0.0, T), kwargs...)
    prob = r.sol.prob
    names = kind === :ib ? _ib_layout_names(r) : _pb_layout_names(r)
    length(names) == length(prob.u0) || error("layout has $(length(names)) names for $(length(prob.u0)) states")
    sol = _resolve(prob, T)
    cols = _layout_trajectory(sol, names)
    # aggregates, including the derived (untracked) compartment
    agg_result = kind === :ib ?
        NodeBasedModels.IndividualBasedResult(sol, r.graph, r.N, r.K, r.state_names, r.model) :
        NodeBasedModels.PairBasedResult(sol, r.graph, r.N, r.n_directed_edges, r.directed_edges,
                                        r.edge_index, r.model)
    for X in r.model.compartment_names
        cols["agg:$X"] = aggregate(agg_result, X)
    end
    kwval(v) = v isa AbstractDict ? describe_params(v) : v isa Symbol ? String(v) :
               v isa SeedSpec ? string(v) : v
    kw = Dict{String,Any}(string(k) => kwval(v) for (k, v) in kwargs)
    return GoldenRecord(name;
        description = description,
        known_defects = known_defects,
        atol = 1e-10,
        meta = Dict{String,Any}(
            "kind" => kind === :ib ? "generate_individual_based" : "generate_pair_based",
            "model" => describe_model(model),
            "graph" => String(graph),
            "network" => describe_network(net),
            "keywords" => kw,
            "tspan" => [0.0, T],
            "solver" => GOLDEN_SOLVER * " (re-solve of result.sol.prob)",
            "notes" => notes),
        scalars = Dict{String,Any}("N" => r.N, "n_states" => length(prob.u0)),
        structure = Dict("layout" => names,
                         "state_names" => kind === :ib ? String.(r.state_names) : ["S", "I"]),
        tables = Dict("trajectory" => trajectory_table(sol.t, cols),
                      "rhs_probe" => _layout_probes(prob, names; salt = kind === :ib ? 31 : 37)))
end

const B02_NOTE = "0.1 numbers (B02): σ (resp. ε) set equal to γ, SEIR seeded in I; the tables equal the deleted *_B02 goldens"

# Two infectious stages with stage-specific infectivity: S → I1 at τ1 through I1 and at τ2
# through I2 (verified issue B04: 0.1 applied τ1 + τ2 to I1 + I2).
staged_via_model() = CompartmentalModel(
    [Compartment(:S), Compartment(:I1; infectious = true), Compartment(:I2; infectious = true),
     Compartment(:R)],
    [Transition(:S, :I1, :τ1, :infection; via = [:I1]), Transition(:S, :I1, :τ2, :infection; via = [:I2]),
     Transition(:I1, :I2, :γ1, :spontaneous), Transition(:I2, :R, :γ2, :spontaneous)]; name = :staged_via)

function golden_cases()
    karate, kite, tree, dig = golden_graph(:karate), golden_graph(:kite), golden_graph(:tree),
                              golden_graph(:digraph)
    gc(name; kw...) = GoldenCase(name, () -> graph_record(name; kw...))
    return [
        # ─── individual-based (NIMFA) ───
        gc("ib_sir_karate_node1"; kind = :ib, model = sir_model(), graph = :karate, net = GraphNetwork(ExplicitGraph(karate)),
           infection_rate = 0.3, recovery_rate = 0.25, initial_infected = [1],
           description = "IB SIR on the karate club, node 1 infected"),
        gc("ib_sir_karate_uniform"; kind = :ib, model = sir_model(), graph = :karate, net = GraphNetwork(ExplicitGraph(karate)),
           infection_rate = 0.3, recovery_rate = 0.25, seed_fraction = 0.05,
           description = "IB SIR on the karate club, every node infected with probability 0.05"),
        gc("ib_sis_karate_node1"; kind = :ib, model = sis_model(), graph = :karate, net = GraphNetwork(ExplicitGraph(karate)),
           infection_rate = 0.3, recovery_rate = 0.25, initial_infected = [1],
           description = "IB SIS on the karate club, node 1 infected"),
        gc("ib_seir_karate_node1"; kind = :ib, model = seir_model(), graph = :karate,
           net = GraphNetwork(ExplicitGraph(karate)), p = Dict(:τ => 0.3, :σ => 0.5, :γ => 0.25), initial_infected = [1],
           description = "IB SEIR on the karate club, σ = 0.5 ≠ γ = 0.25; node 1 seeded in E (the entry state)"),
        gc("ib_seir_karate_node1_sigma_eq_gamma_seedI"; kind = :ib, model = seir_model(), graph = :karate,
           net = GraphNetwork(ExplicitGraph(karate)), p = Dict(:τ => 0.3, :σ => 0.25, :γ => 0.25), initial_infected = [1],
           seed_state = :first_infectious, notes = [B02_NOTE],
           description = "IB SEIR on the karate club with the 0.1 behaviour: σ = γ and node 1 seeded in I"),
        gc("ib_sirs_karate_node1"; kind = :ib, model = sirs_model(), graph = :karate,
           net = GraphNetwork(ExplicitGraph(karate)), p = Dict(:τ => 0.3, :γ => 0.25, :ε => 0.05), initial_infected = [1],
           description = "IB SIRS on the karate club, waning ε = 0.05"),
        gc("ib_sirs_karate_node1_eps_eq_gamma"; kind = :ib, model = sirs_model(), graph = :karate,
           net = GraphNetwork(ExplicitGraph(karate)), p = Dict(:τ => 0.3, :γ => 0.25, :ε => 0.25), initial_infected = [1],
           notes = [B02_NOTE],
           description = "IB SIRS on the karate club with the 0.1 behaviour ε = γ"),
        gc("ib_staged_karate_via"; kind = :ib, model = staged_via_model(), graph = :karate,
           net = GraphNetwork(ExplicitGraph(karate)), p = Dict(:τ1 => 0.3, :τ2 => 0.05, :γ1 => 0.5, :γ2 => 0.5),
           initial_infected = [1],
           description = "IB with two infectious stages and stage-specific infectivity (via, B04)"),
        gc("ib_sir_tree_root"; kind = :ib, model = sir_model(), graph = :tree, net = GraphNetwork(ExplicitGraph(tree)),
           infection_rate = 0.5, recovery_rate = 0.25, initial_infected = [1],
           description = "IB SIR on the depth-4 binary tree, root infected"),
        gc("ib_sir_digraph_node1"; kind = :ib, model = sir_model(), graph = :digraph, net = GraphNetwork(dig),   # directed: ExplicitGraph is undirected, so the 0.1 constructor
           infection_rate = 0.6, recovery_rate = 0.25, initial_infected = [1],
           description = "IB SIR on a directed 8-node graph (infection follows edge direction)"),
        gc("ib_sir_kite_hetT"; kind = :ib, model = sir_model(), graph = :kite,
           net = GraphNetwork(ExplicitGraph(kite); transmission_matrix = heterogeneous_transmission(kite)), recovery_rate = 0.25,
           initial_infected = [1],
           description = "IB SIR on Krackhardt's kite with an explicit heterogeneous transmission matrix"),
        # ─── pair-based (Kirkwood) ───
        gc("pb_sir_kite_node1"; kind = :pb, model = sir_model(), graph = :kite, net = GraphNetwork(ExplicitGraph(kite)),
           infection_rate = 0.3, recovery_rate = 0.25, initial_infected = [1],
           description = "PB SIR on Krackhardt's kite, node 1 infected"),
        gc("pb_sir_kite_uniform"; kind = :pb, model = sir_model(), graph = :kite, net = GraphNetwork(ExplicitGraph(kite)),
           infection_rate = 0.3, recovery_rate = 0.25, seed_fraction = 0.05,
           description = "PB SIR on Krackhardt's kite, uniform seeding 0.05"),
        gc("pb_sir_tree_root"; kind = :pb, model = sir_model(), graph = :tree, net = GraphNetwork(ExplicitGraph(tree)),
           infection_rate = 0.5, recovery_rate = 0.25, initial_infected = [1],
           description = "PB SIR on the depth-4 binary tree (PB is exact on trees), root infected"),
        gc("pb_sir_kite_hetT"; kind = :pb, model = sir_model(), graph = :kite,
           net = GraphNetwork(ExplicitGraph(kite); transmission_matrix = heterogeneous_transmission(kite)), recovery_rate = 0.25,
           initial_infected = [1],
           description = "PB SIR on Krackhardt's kite with an explicit heterogeneous transmission matrix"),
    ]
end
