# record.jl — builder of pairwise golden records, shared by the "pairwise" and
# "reinfection" areas. Include `../common.jl` before this file.

using ModelingToolkit
using Symbolics

const PAIRWISE_T = 40.0

# Degrees 3 and 9 with probability 1/2 each: ⟨k⟩ = 6, ⟨k²⟩ = 45.
const HET39 = [k in (3, 9) ? 0.5 : 0.0 for k in 0:9]

_single_col(a) = "[$a]"
_pair_col(a, b) = "[$a|$b]"

function _pairwise_columns(psys)
    cols = Pair{String,Any}[]
    for (a, v) in psys.singles
        push!(cols, _single_col(a) => v)
    end
    for ((a, b), v) in psys.pairs
        push!(cols, _pair_col(a, b) => v)
    end
    return sort!(cols; by = first)
end

function _param_map(psys, params)
    syms = Dict(Symbol(p) => p for p in ModelingToolkit.parameters(psys.system))
    return Dict{Any,Float64}(syms[k] => Float64(v) for (k, v) in params)
end

"""
    pairwise_probes(psys, params, scale) -> GoldenTable

Evaluate the compiled right-hand side of `psys` at three deterministic states.
Singles are drawn in `scale·(0.05, 1)` and pairs in `scale·⟨k⟩·(0.05, 1)`.
"""
function pairwise_probes(psys, params, scale)
    cols = _pairwise_columns(psys)
    prob = ModelingToolkit.ODEProblem(psys.system, merge(psys.u0, _param_map(psys, params)), psys.tspan)
    us = ModelingToolkit.unknowns(psys.system)
    length(us) == length(cols) || error("compiled system has $(length(us)) unknowns, expected $(length(cols))")
    pos = [findfirst(c -> isequal(c[2], u), cols) for u in us]
    any(isnothing, pos) && error("an unknown of the compiled system is not a single or pair variable")
    names = first.(cols)
    kbar = NodeBasedModels.mean_degree(psys.network)
    raw = probe_states(names, 1.0, 3; salt = 17)
    states = [Dict(n => (startswith(n, "[") && occursin('|', n) ? scale * kbar : scale) * x[n] for n in names)
              for x in raw]
    derivs = Dict{String,Float64}[]
    for st in states
        u = [st[names[pos[i]]] for i in eachindex(us)]
        du = similar(u)
        prob.f(du, u, prob.p, 0.0)
        push!(derivs, Dict(names[pos[i]] => du[i] for i in eachindex(us)))
    end
    return probe_table(states, derivs)
end

function pairwise_record(name; model, network, closure, params, N = 1.0, seed_fraction = 0.01,
                         seed_state = :entry, T = PAIRWISE_T, description = "",
                         known_defects = String[], notes = String[], extra = nothing)
    psys = generate_pairwise(model, network, closure; tspan = (0.0, T), N = N,
                             seed_fraction = seed_fraction, seed_state = seed_state)
    sol = solve_pairwise(psys, params; saveat = 0.0:1.0:T,
                         reltol = GOLDEN_RELTOL, abstol = GOLDEN_ABSTOL)
    string(sol.retcode) == "Success" || error("golden $name: solve returned $(sol.retcode)")
    cols = _pairwise_columns(psys)
    traj = trajectory_table(sol.t, Dict(n => sol[v] for (n, v) in cols))
    kbar = NodeBasedModels.mean_degree(network)
    probes = pairwise_probes(psys, params, N)
    rec = GoldenRecord(name;
        description = description,
        known_defects = known_defects,
        atol = 1e-10 * N * max(1.0, kbar),
        meta = Dict{String,Any}(
            "model" => describe_model(model),
            "network" => describe_network(network),
            "closure" => describe_closure(closure),
            "params" => describe_params(params),
            "N" => N, "seed_fraction" => seed_fraction, "seed_state" => String(seed_state),
            "tspan" => [0.0, T],
            "solver" => GOLDEN_SOLVER,
            "notes" => notes),
        scalars = Dict{String,Any}(
            "n_equations" => length(ModelingToolkit.equations(psys.system)),
            "retcode" => string(sol.retcode)),
        structure = Dict(
            "unknowns" => first.(cols),
            "parameters" => [string(Symbol(p)) for p in ModelingToolkit.parameters(psys.system)],
            "u0_keys" => [n for (n, v) in cols if any(k -> isequal(k, v), keys(psys.u0))]),
        tables = Dict("trajectory" => traj, "rhs_probe" => probes))
    # `extra(rec, psys, sol)` may add tables, structure or metadata (used by other areas)
    isnothing(extra) || extra(rec, psys, sol)
    return rec
end
