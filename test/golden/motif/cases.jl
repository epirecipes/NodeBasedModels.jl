# Golden cases, area "motif": SIS motif closures from `motif_based_sis`
# (m = 2 and m = 3 on 2- and 3-regular hosts, plus the generic chain builder at
# m = 4 and a clustered k = 3, m = 3 host).
#
# Each case freezes every motif variable on t = 0:1:T (reltol 1e-10), the
# right-hand side at three deterministic probe states, and the variable-name set.
# Variables are named "<shape>[<state>]", e.g. "P2[IS]" or "singleton[I]".
#
# Owner: WP37 (the motif closures; sis_k2_m3 replaced in the B05 follow-up, see ../README.md).

include(joinpath(@__DIR__, "..", "common.jl"))

const MOTIF_T = 40.0

_motif_names(sys) = Dict(i => "$(shape)[$(join(String.(state)))]" for ((shape, state), i) in sys.index)

function motif_record(name; β, γ, k, m, ε = 0.01, T = MOTIF_T, description = "", kwargs...)
    sys = motif_based_sis(; β = β, γ = γ, k = k, m = m, tspan = (0.0, T), ε = ε, kwargs...)
    sol = solve_motif(sys; saveat = 0.0:1.0:T, reltol = GOLDEN_RELTOL, abstol = GOLDEN_ABSTOL)
    string(sol.retcode) == "Success" || error("motif golden $name: solve returned $(sol.retcode)")
    byidx = _motif_names(sys)
    length(byidx) == length(sys.u0) || error("motif index does not cover the state vector")
    names = [byidx[i] for i in 1:length(sys.u0)]
    cols = Dict(names[i] => [u[i] for u in sol.u] for i in eachindex(names))
    cols["compartment:S"] = compartment(sys, sol, :S)
    cols["compartment:I"] = compartment(sys, sol, :I)
    states = probe_states(names, 1.0, 3; salt = 41)
    derivs = Dict{String,Float64}[]
    for st in states
        u = [st[n] for n in names]
        du = similar(u)
        sys.rhs!(du, u, sys.params, 0.0)
        push!(derivs, Dict(names[i] => du[i] for i in eachindex(names)))
    end
    kw = Dict{String,Any}(string(a) => b for (a, b) in kwargs)
    return GoldenRecord(name;
        description = description, atol = 1e-10,
        meta = Dict{String,Any}("beta" => β, "gamma" => γ, "k" => k, "m" => m, "epsilon" => ε,
                                "tspan" => [0.0, T], "keywords" => kw, "solver" => GOLDEN_SOLVER),
        scalars = Dict{String,Any}("n_variables" => length(names), "n_shapes" => length(sys.shapes)),
        structure = Dict("variables" => names, "shapes" => [String(s.name) for s in sys.shapes]),
        tables = Dict("trajectory" => trajectory_table(sol.t, cols),
                      "rhs_probe" => probe_table(states, derivs)))
end

function golden_cases()
    mc(name; kw...) = GoldenCase(name, () -> motif_record(name; kw...))
    return [
        mc("sis_k2_m2"; β = 1.5, γ = 0.5, k = 2, m = 2, description = "SIS motif closure, ring host, m = 2"),
        mc("sis_k2_m3"; β = 1.5, γ = 0.5, k = 2, m = 3, description = "SIS motif closure, ring host, m = 3"),
        mc("sis_k2_m4_chain"; β = 1.5, γ = 0.5, k = 2, m = 4,
           description = "SIS motif closure, ring host, m = 4 (generic chain builder)"),
        mc("sis_k3_m2"; β = 0.6, γ = 0.5, k = 3, m = 2, description = "SIS motif closure, 3-regular host, m = 2"),
        mc("sis_k3_m3"; β = 0.6, γ = 0.5, k = 3, m = 3,
           description = "SIS motif closure, 3-regular host, m = 3, triangle-free (default n_p3)"),
        mc("sis_k3_m3_nc3"; β = 0.6, γ = 0.5, k = 3, m = 3, n_c3 = 0.2,
           description = "SIS motif closure, 3-regular host, m = 3, with n_c3 = 0.2 triangles per node"),
    ]
end
