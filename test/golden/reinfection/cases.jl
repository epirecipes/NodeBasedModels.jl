# Golden cases, area "reinfection": pairwise systems of reinfection-counting
# lifts (`with_reinfection_counting`, Keeling et al. 2016 Approximation 1) and
# their base-compartment totals (`reinfection_totals`).
#
# The lifted model is seeded by `generate_pairwise`'s default initial condition: the
# unseeded nodes in the count-0 susceptible class S_0 and the seeds in I_1, the entry state
# reached from S_0 (in 0.1: the first infectious compartment, which is also I_1).
#
# Since 0.2 `sis_model()` is a NetworkEpiCore ContactModel, so the lift is NetworkEpiCore's
# `with_reinfection_counting(cm, L)` and the pairwise system is built from
# `CompartmentalModel(lifted)`; the structure records the compartments and transitions of that
# CompartmentalModel, which equal those of the 0.1 lift.
#
# Owner: the WP that owns NodeBasedModels pairwise generation (WP15 in Phase 2).

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "..", "pairwise", "record.jl"))

function reinfection_record(name; base, L, network, params, T = 40.0, description = "")
    lifted = with_reinfection_counting(base, L)
    cmodel = lifted isa CompartmentalModel ? lifted : CompartmentalModel(lifted)
    function add_totals!(rec, psys, sol)
        totals = reinfection_totals(psys, sol)
        rec.tables["totals"] = trajectory_table(sol.t, Dict(string(k) => v for (k, v) in totals))
        rec.structure["lifted_compartments"] = sort!(String.(cmodel.compartment_names))
        rec.structure["lifted_transitions"] = sort!(["$(t.from) -> $(t.to) [$(t.rate), $(t.type)]"
                                                     for t in cmodel.transitions])
        rec.meta["base_model"] = describe_model(base)
        rec.meta["L"] = L
        return rec
    end
    return pairwise_record(name; model = lifted, network = network, closure = BernoulliClosure(),
                           params = params, T = T, description = description, extra = add_totals!)
end

golden_cases() = [
    GoldenCase("sis_L3_hom4", () -> reinfection_record("sis_L3_hom4"; base = sis_model(), L = 3,
        network = regular_network(4), params = Dict(:τ => 0.5, :γ => 1.0),
        description = "SIS with up to 3 infections counted, 4-regular, Bernoulli closure")),
    GoldenCase("sirs_L2_hom4", () -> reinfection_record("sirs_L2_hom4"; base = sirs_model(), L = 2,
        network = regular_network(4), params = Dict(:τ => 0.4, :γ => 0.25, :ε => 0.1), T = 60.0,
        description = "SIRS with up to 2 infections counted, 4-regular, Bernoulli closure")),
]
