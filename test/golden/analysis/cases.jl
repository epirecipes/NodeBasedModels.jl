# Golden cases, area "analysis": the scalar outputs of analysis.jl —
# `basic_reproduction_number` (the threshold ratio R = 1 + r/γ; numeric and model-based
# symbolic), `epidemic_threshold`, `early_growth_rate` and `disease_free_equilibrium`.
#
# SIR and SIS are separate (WP24, verified issue B01). With q = ⟨k(k−1)⟩/⟨k⟩:
# - Bernoulli: SIR τ_c = γ/(q − 1), r = τ(q − 1) − γ; SIS τ_c = γ/q, r the leading eigenvalue of
#   the [SI], [II] block;
# - Keeling (homogeneous and heterogeneous) and Barnard (homogeneous): the fast-variable
#   quasi-equilibrium of α = [SI]/[I], δ = [II]/[I] (Barnard et al. 2019).
# The 0.1 numbers (B01: heterogeneous networks returned the SIS R₀ and threshold and the SIR growth
# rate, the model-based method ignored the model, the clustered corrections were ad hoc) were
# replaced in WP24. The new numbers are justified by test/suites/analysis_nbm.jl: the thresholds
# and growth rates equal the linearisation (Bernoulli) or the early exponential phase (Keeling) of
# the ODE that generate_pairwise builds, the Keeling threshold equals Barnard's thesis value
# 0.319169 (n = 6, ϕ = 0.3), and the SIR and SIS thresholds separate extinction from outbreaks in
# NetworkOutbreaks simulations. `hom_bernoulli` (SIR on homogeneous networks, the 0.1 default) is
# unchanged.
#
# `clustered_barnard` is Barnard's improved closure as the thesis writes it (eq. 4.23, [ASI] with
# the infector last); its numbers equal the early phase of a reference ODE with that closure
# (analysis_nbm.jl). The Barnard entries of the first WP24 `clustered` golden followed
# generate_pairwise, which evaluates the triple with the infector first (a new verified issue).
#
# A call that throws is frozen as the string "throws <ExceptionType>".
#
# Owner: WP24 (NodeBasedModels fixes: SIR and SIS analysis separated).

include(joinpath(@__DIR__, "..", "common.jl"))

using Symbolics

const TAUS = (0.1, 0.3, 0.42, 0.6)
const GAMMA = 1.0

# Degree distributions as degree_probs vectors (index k+1 ↔ degree k).
const DISTS = Dict(
    "point4" => [0.0, 0, 0, 0, 1.0],                          # 4-regular written as heterogeneous
    "bimodal28" => [k in (2, 8) ? 0.5 : 0.0 for k in 0:8],    # B01's bimodal 2/8 example
    "het39" => [k in (3, 9) ? 0.5 : 0.0 for k in 0:9],
)

_f(x) = x isa Real ? Float64(x) : x

"""Evaluate a model-based symbolic R at numeric (τ, γ) via build_function."""
function eval_symbolic(expr, τname::Symbol, γname::Symbol, τ, γ)
    f = Symbolics.build_function(expr, Symbolics.variable(τname), Symbolics.variable(γname);
                                 expression = Val(false))
    return Float64(f(τ, γ))
end

function numeric_scalars!(sc, label, net, closure; kw...)
    for τ in TAUS
        sc["$label/R0/tau=$τ"] = _f(catch_scalar(() -> basic_reproduction_number(net, closure, τ, GAMMA; kw...)))
        sc["$label/growth/tau=$τ"] = _f(catch_scalar(() -> early_growth_rate(net, closure, τ, GAMMA; kw...)))
    end
    sc["$label/threshold"] = _f(catch_scalar(() -> epidemic_threshold(net, closure, GAMMA; kw...)))
    return sc
end

function record_hom_bernoulli()
    sc = Dict{String,Any}()
    for n in (2, 3, 4, 6)
        numeric_scalars!(sc, "hom$n", regular_network(n), BernoulliClosure())
    end
    return GoldenRecord("hom_bernoulli";
        description = "homogeneous Bernoulli, SIR (the default dynamics): R = τ(n−2)/γ, τc = γ/(n−2), r = τ(n−2)−γ",
        atol = 1e-14, scalars = sc,
        meta = Dict{String,Any}("gamma" => GAMMA, "taus" => collect(TAUS), "dynamics" => "SIR (default)"))
end

function record_hom_bernoulli_sis()
    sc = Dict{String,Any}()
    for n in (2, 3, 4, 6)
        numeric_scalars!(sc, "hom$n", regular_network(n), BernoulliClosure(); dynamics = :SIS)
    end
    return GoldenRecord("hom_bernoulli_sis";
        description = "homogeneous Bernoulli, SIS: τc = γ/(n−1), r = leading eigenvalue of the [SI],[II] block, R = 1 + r/γ",
        atol = 1e-14, scalars = sc,
        meta = Dict{String,Any}("gamma" => GAMMA, "taus" => collect(TAUS), "dynamics" => "SIS"))
end

function record_het_bernoulli()
    sc = Dict{String,Any}()
    nets = Dict{String,Any}(label => degree_distribution_network(p) for (label, p) in DISTS)
    nets["er5"] = erdos_renyi_network(5.0)
    nets["zero_degree"] = degree_distribution_network([1.0])
    for (label, net) in nets, d in (:SIR, :SIS)
        numeric_scalars!(sc, "$label/$d", net, BernoulliClosure(); dynamics = d)
    end
    # the dynamics must be named on a heterogeneous network (0.1 returned the SIS value, B01)
    sc["het39/nodynamics/threshold"] = catch_scalar(() ->
        epidemic_threshold(nets["het39"], BernoulliClosure(), GAMMA))
    return GoldenRecord("het_bernoulli";
        description = "heterogeneous Bernoulli, SIR and SIS: τc = γ/(q−1) and γ/q with q = ⟨k(k−1)⟩/⟨k⟩",
        atol = 1e-14, scalars = sc,
        meta = Dict{String,Any}("gamma" => GAMMA, "taus" => collect(TAUS),
                                "distributions" => DISTS, "er5" => "erdos_renyi_network(5.0)"))
end

function record_clustered()
    sc = Dict{String,Any}()
    for ϕ in (0.1, 0.3), n in (4, 6), d in (:SIR, :SIS)
        net = regular_network(n; ϕ = ϕ)
        numeric_scalars!(sc, "hom$(n)_phi$(ϕ)_keeling/$d", net, KeelingClosure(); dynamics = d)
    end
    for ϕ in (0.1, 0.3), d in (:SIR, :SIS)
        numeric_scalars!(sc, "het39_phi$(ϕ)_keeling/$d", degree_distribution_network(DISTS["het39"]; ϕ = ϕ),
                         KeelingClosure(); dynamics = d)
    end
    # unsupported combinations (frozen as errors)
    numeric_scalars!(sc, "hom4_power15", regular_network(4), PowerClosure(1.5))
    numeric_scalars!(sc, "het39_barnard", degree_distribution_network(DISTS["het39"]), BarnardClosure();
                     dynamics = :SIR)
    return GoldenRecord("clustered";
        description = "Keeling closure (homogeneous, heterogeneous), SIR and SIS: fast-variable quasi-equilibria; unsupported combinations",
        atol = 1e-12, scalars = sc,
        meta = Dict{String,Any}("gamma" => GAMMA, "taus" => collect(TAUS)))
end

function record_clustered_barnard()
    sc = Dict{String,Any}()
    for ϕ in (0.1, 0.3), n in (4, 6), d in (:SIR, :SIS)
        numeric_scalars!(sc, "hom$(n)_phi$(ϕ)_barnard/$d", regular_network(n; ϕ = ϕ), BarnardClosure();
                         dynamics = d)
    end
    return GoldenRecord("clustered_barnard";
        description = "Barnard's improved closure (thesis eq. 4.23, [ASI] with the infector last) on homogeneous networks, SIR and SIS: fast-variable quasi-equilibria",
        atol = 1e-12, scalars = sc,
        meta = Dict{String,Any}("gamma" => GAMMA, "taus" => collect(TAUS)))
end

function record_symbolic()
    sc = Dict{String,Any}()
    nets = Dict("hom4" => regular_network(4), "hom6_phi0.2" => regular_network(6; ϕ = 0.2),
                "het39" => degree_distribution_network(DISTS["het39"]),
                "het39_phi0.1" => degree_distribution_network(DISTS["het39"]; ϕ = 0.1),
                "bimodal28" => degree_distribution_network(DISTS["bimodal28"]))
    models = Dict("sir" => sir_model(), "sis" => sis_model(), "seir" => seir_model(),
                  "sirs_eps_gamma" => sirs_model(; ε = :γ), "sir_ab" => sir_model(; τ = :a, γ = :b))
    closures = Dict("bernoulli" => BernoulliClosure(), "keeling" => KeelingClosure(),
                    "barnard" => BarnardClosure())
    for (mname, m) in models, (nname, net) in nets, (cname, cl) in closures
        key = "$mname/$nname/$cname"
        expr = catch_scalar(() -> basic_reproduction_number(m, net, cl))
        if expr isa AbstractString
            sc[key] = expr
        else
            cmod = CompartmentalModel(m)
            τs = only(unique(t.rate for t in cmod.transitions if t.type == :infection))
            γs = only(unique(t.rate for t in cmod.transitions if t.type == :spontaneous))
            sc[key * "/tau=0.42"] = eval_symbolic(expr, τs, γs, 0.42, GAMMA)
            sc[key * "/tau=0.1"] = eval_symbolic(expr, τs, γs, 0.1, GAMMA)
        end
    end
    # degenerate networks give a symbolic zero
    sc["sir/hom2/bernoulli/string"] = string(basic_reproduction_number(sir_model(), regular_network(2), BernoulliClosure()))
    sc["sir/zero_degree/bernoulli/string"] =
        string(basic_reproduction_number(sir_model(), degree_distribution_network([1.0]), BernoulliClosure()))
    return GoldenRecord("symbolic";
        description = "model-based symbolic R (Bernoulli closure, and the clustered closures when ϕ = 0) evaluated at τ ∈ {0.1, 0.42}, γ = 1; the dynamics are read from the model; clustered closures with ϕ > 0, Barnard on heterogeneous networks and non-SIR/SIS models throw",
        atol = 1e-14, scalars = sc,
        meta = Dict{String,Any}("gamma" => GAMMA,
            "note" => "seir and sirs (even with ε = γ) throw: the strict SIR/SIS classification (B01)"))
end

function record_dfe()
    sc = Dict{String,Any}()
    cases = Dict("sir_hom6_N100" => (sir_model(), regular_network(6), 100.0),
                 "seir_hom4_N1" => (seir_model(), regular_network(4), 1.0),
                 "sis_het39_N1000" => (sis_model(), degree_distribution_network(DISTS["het39"]), 1000.0))
    for (label, (m, net, N)) in cases
        for (k, v) in disease_free_equilibrium(m, net; N = N)
            sc["$label/$k"] = v
        end
    end
    return GoldenRecord("dfe";
        description = "disease_free_equilibrium: [S] = N, [SS] = ⟨k⟩N, everything else 0",
        atol = 1e-14, scalars = sc)
end

golden_cases() = [
    GoldenCase("hom_bernoulli", record_hom_bernoulli),
    GoldenCase("hom_bernoulli_sis", record_hom_bernoulli_sis),
    GoldenCase("het_bernoulli", record_het_bernoulli),
    GoldenCase("clustered", record_clustered),
    GoldenCase("clustered_barnard", record_clustered_barnard),
    GoldenCase("symbolic", record_symbolic),
    GoldenCase("dfe", record_dfe),
]
