# Golden cases, area "neighbourhood": the n = 2 neighbourhood model for SIS
# (Keeling, House, Cooper & Pellis 2016, Approximation 3) from
# `generate_neighbourhood`.
#
# Each case freezes every [S_y], [I_y] on t = 0:1:T (reltol 1e-10), the
# aggregates from `neighbourhood_compartment`, the right-hand side at three
# deterministic probe states, and the variable-name set.
#
# Owner: the WP that owns the neighbourhood model (none scheduled in the design).

include(joinpath(@__DIR__, "..", "common.jl"))

const NBHD_T = 40.0

function neighbourhood_record(name; k, β, γ, ε = 0.01, T = NBHD_T, description = "")
    sys = generate_neighbourhood(sis_model(), k, 2; β = β, γ = γ, tspan = (0.0, T), ε = ε)
    sol = solve_neighbourhood(sys; saveat = 0.0:1.0:T, reltol = GOLDEN_RELTOL, abstol = GOLDEN_ABSTOL)
    string(sol.retcode) == "Success" || error("neighbourhood golden $name: solve returned $(sol.retcode)")
    byidx = Dict(i => "$(X)_$(y)" for ((X, y), i) in sys.index)
    names = [byidx[i] for i in 1:length(sys.u0)]
    cols = Dict(names[i] => [u[i] for u in sol.u] for i in eachindex(names))
    cols["compartment:S"] = neighbourhood_compartment(sys, sol, :S)
    cols["compartment:I"] = neighbourhood_compartment(sys, sol, :I)
    states = probe_states(names, 1.0, 3; salt = 43)
    derivs = Dict{String,Float64}[]
    for st in states
        u = [st[n] for n in names]
        du = similar(u)
        sys.rhs!(du, u, sys.params, 0.0)
        push!(derivs, Dict(names[i] => du[i] for i in eachindex(names)))
    end
    return GoldenRecord(name;
        description = description, atol = 1e-10,
        meta = Dict{String,Any}("beta" => β, "gamma" => γ, "k" => k, "n" => 2, "epsilon" => ε,
                                "tspan" => [0.0, T], "solver" => GOLDEN_SOLVER),
        scalars = Dict{String,Any}("n_variables" => length(names)),
        structure = Dict("variables" => names, "var_names" => String.(sys.var_names)),
        tables = Dict("trajectory" => trajectory_table(sol.t, cols),
                      "rhs_probe" => probe_table(states, derivs)))
end

golden_cases() = [
    GoldenCase("sis_k2_n2", () -> neighbourhood_record("sis_k2_n2"; k = 2, β = 1.5, γ = 0.5,
        description = "SIS neighbourhood model n = 2 on a ring (k = 2)")),
    GoldenCase("sis_k3_n2", () -> neighbourhood_record("sis_k3_n2"; k = 3, β = 1.0, γ = 1.0,
        description = "SIS neighbourhood model n = 2 on a 3-regular host")),
    GoldenCase("sis_k4_n2", () -> neighbourhood_record("sis_k4_n2"; k = 4, β = 0.5, γ = 0.5,
        description = "SIS neighbourhood model n = 2 on a 4-regular host")),
]
