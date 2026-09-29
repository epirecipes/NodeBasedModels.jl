# Motif catalogue and validation methodology


- [What this page shows](#what-this-page-shows)
- [The shared first cell](#the-shared-first-cell)
- [1. What a page does with a
  reference](#1-what-a-page-does-with-a-reference)
- [2. The declared verdicts, checked](#2-the-declared-verdicts-checked)
- [3. N-scaling: exact in the limit versus
  biased](#3-n-scaling-exact-in-the-limit-versus-biased)
- [Appendix: motif shapes and state
  classes](#appendix-motif-shapes-and-state-classes)
- [References](#references)

## What this page shows

Every NodeBasedModels page compares a deterministic node-based model
with a NetworkOutbreaks ensemble through the same three calls:
`scenario_summary(sc)`, `comparisonplot(ref, curves...)` and
`compare(ref, curves...)`. The mirrored EdgeBasedModels page E14
explains how the ensembles are built, hashed and cached, and what the
statistics mean. NetworkOutbreaks’ validation vignette
(`NetworkOutbreaks.jl/vignettes/05_validation`) documents the simulation
side. This page gives the node-based view:

1.  what a page may and may not do with a reference, in code;
2.  the verdicts that the scenarios declare for the node-based back
    ends, and every `:exact_limit` and `:biased` declaration checked
    against its committed summary;
3.  the N-scaling protocol for the node-based closures: exact in the
    limit versus biased;
4.  an appendix listing the motif shapes and state classes of the SIS
    motif closures (page N10). It is a table of what the implementation
    enumerates and makes no categorical claims.

## The shared first cell

This cell is the shared cell of DESIGN §E.5, the same as the first cell
of EdgeBasedModels’ page E14 apart from the back-end lines. Quarto shows
only the last value of a cell, the `compare` table; the figure is drawn
in section 1.

``` julia
include(joinpath(@__DIR__, "..", "_shared", "setup.jl"))
require_summaries([:sir_pois5])                          # back end
using NetworkEpiCore, NetworkOutbreaks, Catalyst, Plots
sir = @reaction_network sir begin
    @parameters τ γ
    τ, S + I --> 2I        # contact: per-contact (per-edge) rate τ; S converted, I unchanged
    γ, I --> R             # node-local transition
end
model = contact_model(sir)          # prints the typing report (B.1)
sc    = scenario(:sir_pois5)        # Poisson(5), τ = 1/6, γ = 1/4, 1% seeds in I, t ∈ [0, 60]
@assert isequivalent(model, sc.model)
ref   = scenario_summary(sc)        # committed NO ensemble: N = 10⁴, 200 runs, a fresh G(N, p) graph per run
# --- NBM vignette ---------------------------------------------------------------
using NodeBasedModels
sys   = node_based(model, sc.network)                      # low level
sysF  = generate_pairwise(sir_model(), sc.network, default_closure(sc.network); cumulative = true)  # factory
@assert vector_fields_equal(symbolic_ode(sys), symbolic_ode(sysF))
# --- both -------------------------------------------------------------------------
sol = solve_epidemic(sys, sc)                              # p, initial, tspan, saveat all from the scenario
det = model_curves(sys, sol; t = sc.tgrid, label = "pairwise")   # back end
comparisonplot(ref, det; observables = [:I, :cumulative])  # top: spread ribbon + mean + curve; bottom: residual ± 1.96 SE
compare(ref, det)                                          # D∞, z∞, ΔR∞ (95% CI), Δpeak, Δt_peak
```

    ComparisonTable :sir_pois5  (scenario 34c89792; conditioned mean of 200 runs)
      curve     observable        D∞     t(D∞)       SE∞        z∞       ΔR∞  95% CI                 Δpeak   Δt_peak  coverage
      pairwise  S            0.00503     11.50   0.00222      2.61   0.00012  [-0.00085,  0.00109]   0.00000      0.00     0.788
      pairwise  I            0.00223      7.75   0.00113      2.60   0.00012  [-0.00085,  0.00109]   0.00158      0.00     0.830
      pairwise  R            0.00387     13.25   0.00168      2.39   0.00012  [-0.00085,  0.00109]   0.00012      0.00     0.784
      pairwise  infectious   0.00223      7.75   0.00113      2.60   0.00012  [-0.00085,  0.00109]   0.00158      0.00     0.830
      pairwise  cumulative   0.00503     11.50   0.00222      2.61   0.00012  [-0.00085,  0.00109]   0.00012      3.50     0.788

## 1. What a page does with a reference

A page never simulates and never computes an ensemble. It loads the
committed summary of a registered scenario, and the setup cell of every
page calls `require_summaries`, so that a missing or stale summary is a
render error. A summary is valid only for the scenario hash and the
NetworkOutbreaks algorithm revision it was computed with:

``` julia
(; scenario_hash = first(scenario_hash(sc), 12), in_summary = first(ref.scenario_hash, 12),
   algorithm_revision = NetworkOutbreaks.ALGORITHM_REVISION, summary_revision = ref.algorithm_revision,
   strict_cache = get(ENV, "NETEPI_STRICT_CACHE", "0"))
```

    (scenario_hash = "34c89792c3f7", in_summary = "34c89792c3f7", algorithm_revision = "2", summary_revision = "2", strict_cache = "1")

Changing anything about a scenario, even only the number of runs,
changes its hash. There is then no committed summary, and under the
strict cache the page fails instead of simulating:

``` julia
alt = derive(sc; id = :sir_pois5_demo, nsims = 7)
refused = try
    scenario_summary(alt)
    false
catch e
    msg = sprint(showerror, e)
    e isa ArgumentError && occursin("scenario_summary(:$(alt.id))", msg) && occursin("committed summary", msg) || rethrow()
end
(; hash_of_derived = first(scenario_hash(alt), 8), hash_of_scenario = first(scenario_hash(sc), 8),
   refused_under_strict_cache = refused)
```

    (hash_of_derived = "387c303d", hash_of_scenario = "34c89792", refused_under_strict_cache = true)

Every comparison states the reference (N, runs, conditioning and the
kept fraction), draws the spread band (the pointwise q2.5–q97.5 of the
runs) and the residual with the mean band (±1.96 SE), and tabulates
`compare`:

``` julia
describe_reference(ref);
```

    NetworkOutbreaks reference :sir_pois5 (hash 34c89792): N = 10000 nodes, 200 runs on a fresh graph per run, algorithm :next_reaction; conditioning: major outbreaks only (cumulative incidence excluding seeds ≥ 0.05·N by t_end); 200 of 200 runs kept (major runs), P(major) = 1.000 (95% CI 0.981–1.000); no time alignment.

``` julia
tab = compare(ref, det)
comparisonplot(ref, det; observables = [:I, :cumulative])
```

![The pairwise model of `:sir_pois5` against its reference (spread band
q2.5–q97.5 of the major runs; residual panel: mean band ±1.96
SE).](index_files/figure-commonmark/cell-6-output-1.svg)

``` julia
tab
```

    ComparisonTable :sir_pois5  (scenario 34c89792; conditioned mean of 200 runs)
      curve     observable        D∞     t(D∞)       SE∞        z∞       ΔR∞  95% CI                 Δpeak   Δt_peak  coverage
      pairwise  S            0.00503     11.50   0.00222      2.61   0.00012  [-0.00085,  0.00109]   0.00000      0.00     0.788
      pairwise  I            0.00223      7.75   0.00113      2.60   0.00012  [-0.00085,  0.00109]   0.00158      0.00     0.830
      pairwise  R            0.00387     13.25   0.00168      2.39   0.00012  [-0.00085,  0.00109]   0.00012      0.00     0.784
      pairwise  infectious   0.00223      7.75   0.00113      2.60   0.00012  [-0.00085,  0.00109]   0.00158      0.00     0.830
      pairwise  cumulative   0.00503     11.50   0.00222      2.61   0.00012  [-0.00085,  0.00109]   0.00012      3.50     0.788

The two bands measure different things. At the simulated peak the spread
band is much wider than the mean band:

``` julia
st = ref.cond[:I]
i = argmax(st.mean)
(; t_peak = ref.t[i], spread_width = st.q975[i] - st.q025[i], mean_band_width = 2 * 1.96 * st.se[i],
   ratio = (st.q975[i] - st.q025[i]) / (2 * 1.96 * st.se[i]), sqrt_runs = sqrt(ref.n_major))
```

    (t_peak = 11.25, spread_width = 0.034019999999999995, mean_band_width = 0.00253232, ratio = 13.434321096859795, sqrt_runs = 14.142135623730951)

## 2. The declared verdicts, checked

Each scenario declares a verdict for each back end it names:
`:exact_limit`, `:biased`, `:approximate` or `:inadmissible`. The
node-based back ends and the number of scenarios that declare each
verdict:

``` julia
nbm_keys = (:pairwise_const, :pairwise_bernoulli, :pairwise_keeling, :pairwise_multitype, :pgf_closure,
            :s_anchored, :mean_field, :reinfection, :motif, :neighbourhood, :individual, :pair)
verdicts = (:exact_limit, :biased, :approximate, :inadmissible)
all_sc = scenarios()
md_table(vcat(["back end"], collect(string.(verdicts))),
         [vcat([string("`:", k, "`")], [count(s -> get(s.backends, k, nothing) === v, all_sc) for v in verdicts])
          for k in nbm_keys])
```

| back end              | exact_limit | biased | approximate | inadmissible |
|-----------------------|------------:|-------:|------------:|-------------:|
| `:pairwise_const`     |          11 |      6 |           2 |            0 |
| `:pairwise_bernoulli` |           1 |      0 |           1 |            0 |
| `:pairwise_keeling`   |           0 |      0 |           5 |            0 |
| `:pairwise_multitype` |           2 |      0 |           0 |            0 |
| `:pgf_closure`        |          18 |      0 |           0 |            1 |
| `:s_anchored`         |           1 |      0 |           0 |            0 |
| `:mean_field`         |           1 |      0 |           0 |            0 |
| `:reinfection`        |           0 |      0 |           1 |            0 |
| `:motif`              |           0 |      0 |           1 |            0 |
| `:neighbourhood`      |           0 |      0 |           1 |            0 |
| `:individual`         |           1 |      0 |           3 |            0 |
| `:pair`               |           0 |      0 |           1 |            0 |

The `:exact_limit` declarations of the population-level node-based back
ends are checked here against their committed summaries. Each back end
is built from the scenario:

- `:pairwise_const`, `:pairwise_multitype` and `:mean_field` with
  `node_based(sc)` (the default closure of the descriptor);
- `:pgf_closure` with `node_based(sc; closure = PGFClosure())`;
- `:s_anchored` with `node_based(sc; level = :s_anchored)`.

A back end passes when D∞ and \|ΔR∞\| are both below 0.005 for the
prevalence of the infectious compartments (`:infectious`), as for the
edge-based table of E14:

``` julia
builders = Dict(:pairwise_const => sc -> node_based(sc), :pairwise_multitype => sc -> node_based(sc),
                :mean_field => sc -> node_based(sc), :pgf_closure => sc -> node_based(sc; closure = PGFClosure()),
                :s_anchored => sc -> node_based(sc; level = :s_anchored))
function check(s, key)
    r = scenario_summary(s)
    y = builders[key](s)
    c = model_curves(y, solve_epidemic(y, s); t = s.tgrid, label = string(key))
    row = compare(r, c; observables = [:infectious])[string(key), :infectious]
    return r, row
end
exact_rows = []
for s in all_sc, key in keys(builders)
    get(s.backends, key, nothing) === :exact_limit || continue
    r, row = check(s, key)
    push!(exact_rows, (string("`:", s.id, "`"), string("`:", key, "`"), r.N, r.nsims, row.D∞, row.z∞, row.ΔR∞,
                       row.D∞ < 0.005 && abs(row.ΔR∞) < 0.005 ? "yes" : "**no**"))
end
sort!(exact_rows; by = r -> (r[2], r[1]))
md_table(["scenario", "back end", "N", "runs", "D∞(infectious)", "z∞", "ΔR∞", "passes"], exact_rows)
```

| scenario | back end | N | runs | D∞(infectious) | z∞ | ΔR∞ | passes |
|----|----|---:|---:|---:|---:|---:|----|
| `:sir_wm5` | `:mean_field` | 10000 | 200 | 0.001433 | 2.217 | -0.0004265 | yes |
| `:seair_pois5` | `:pairwise_const` | 10000 | 200 | 0.0006032 | 2.895 | -0.001017 | yes |
| `:seir_pois5` | `:pairwise_const` | 10000 | 200 | 0.0003383 | 1.101 | 3.48e-05 | yes |
| `:sir_erl2_pois5` | `:pairwise_const` | 10000 | 200 | 0.001866 | 1.853 | -5.536e-05 | yes |
| `:sir_erl3_pois5` | `:pairwise_const` | 10000 | 200 | 0.002276 | 2.259 | 0.0002948 | yes |
| `:sir_erl5_pois5` | `:pairwise_const` | 10000 | 200 | 0.001384 | 1.502 | -0.0002584 | yes |
| `:sir_nb4` | `:pairwise_const` | 10000 | 200 | 0.001145 | 1.736 | 0.0003432 | yes |
| `:sir_pois5_N100000` | `:pairwise_const` | 100000 | 20 | 0.0009282 | 1.539 | 0.0001942 | yes |
| `:sir_pois5_N1000` | `:pairwise_const` | 1000 | 2000 | 0.0136 | 14.85 | 0.0007155 | **no** |
| `:sir_pois5` | `:pairwise_const` | 10000 | 200 | 0.002228 | 2.596 | 0.0001227 | yes |
| `:sir_reg6` | `:pairwise_const` | 10000 | 200 | 0.001709 | 2.736 | 0.0002649 | yes |
| `:twostrain_pois5` | `:pairwise_const` | 10000 | 200 | 0.001277 | 1.926 | -0.0005572 | yes |
| `:sir_age2` | `:pairwise_multitype` | 10000 | 200 | 0.001985 | 3.122 | 0.0004932 | yes |
| `:sir_sbm2` | `:pairwise_multitype` | 10000 | 200 | 0.001717 | 3.296 | 0.001119 | yes |
| `:seair_pois5` | `:pgf_closure` | 10000 | 200 | 0.0006032 | 2.895 | -0.001017 | yes |
| `:seir_pois5` | `:pgf_closure` | 10000 | 200 | 0.0003383 | 1.101 | 3.48e-05 | yes |
| `:sir_bim_N100000` | `:pgf_closure` | 100000 | 20 | 0.0009376 | 1.943 | -0.0003395 | yes |
| `:sir_bim_N1000` | `:pgf_closure` | 1000 | 2000 | 0.01765 | 21.31 | 0.004346 | **no** |
| `:sir_bim` | `:pgf_closure` | 10000 | 200 | 0.00232 | 3.57 | -0.000345 | yes |
| `:sir_erl2_pois5` | `:pgf_closure` | 10000 | 200 | 0.001866 | 1.853 | -5.536e-05 | yes |
| `:sir_erl3_pois5` | `:pgf_closure` | 10000 | 200 | 0.002276 | 2.259 | 0.0002948 | yes |
| `:sir_erl5_pois5` | `:pgf_closure` | 10000 | 200 | 0.001384 | 1.502 | -0.0002584 | yes |
| `:sir_nb4` | `:pgf_closure` | 10000 | 200 | 0.001145 | 1.736 | 0.0003432 | yes |
| `:sir_pl_N100000` | `:pgf_closure` | 100000 | 20 | 0.001396 | 2.969 | -0.0002673 | yes |
| `:sir_pl_N1000` | `:pgf_closure` | 1000 | 2000 | 0.0118 | 20.49 | 0.0148 | **no** |
| `:sir_pl` | `:pgf_closure` | 10000 | 200 | 0.002645 | 3.943 | 0.001731 | yes |
| `:sir_pois5_N100000` | `:pgf_closure` | 100000 | 20 | 0.0009282 | 1.539 | 0.0001942 | yes |
| `:sir_pois5_N1000` | `:pgf_closure` | 1000 | 2000 | 0.0136 | 14.85 | 0.0007155 | **no** |
| `:sir_pois5` | `:pgf_closure` | 10000 | 200 | 0.002228 | 2.596 | 0.0001227 | yes |
| `:sir_reg6` | `:pgf_closure` | 10000 | 200 | 0.001709 | 2.736 | 0.0002649 | yes |
| `:sir_vax_pois5` | `:pgf_closure` | 10000 | 200 | 0.001909 | 2.257 | 0.00213 | yes |
| `:twostrain_pois5` | `:pgf_closure` | 10000 | 200 | 0.001277 | 1.926 | -0.0005572 | yes |
| `:sir_vax_pois5` | `:s_anchored` | 10000 | 200 | 0.001909 | 2.257 | 0.00213 | yes |

``` julia
npass = count(r -> r[end] == "yes", exact_rows)
println(npass, " of ", length(exact_rows), " exact-limit declarations pass D∞ < 0.005 and |ΔR∞| < 0.005.")
failing = [string(r[1], " ", r[2]) for r in exact_rows if r[end] != "yes"]
isempty(failing) || println("not passing: ", join(failing, ", "))
```

    29 of 33 exact-limit declarations pass D∞ < 0.005 and |ΔR∞| < 0.005.
    not passing: `:sir_pois5_N1000` `:pairwise_const`, `:sir_bim_N1000` `:pgf_closure`, `:sir_pl_N1000` `:pgf_closure`, `:sir_pois5_N1000` `:pgf_closure`

The rule is calibrated for N = 10⁴. At N = 10³ the O(1/N) finite-size
bias of the simulation is larger than the tolerance, so a miss at N =
10³ is not by itself evidence of a wrong model. The N-scaling protocol
in section 3 separates the two cases.

The `:biased` declarations say that the gap does not vanish as N grows.
The tests pin it as \|ΔR∞\| \> 0.02 for constant-K pairwise models on
non-Poisson networks. Here are the same declarations, with the PGF
closure, which is exact on the same networks, next to them:

``` julia
biased_rows = []
for s in all_sc, key in (:pairwise_const, :pairwise_multitype)
    get(s.backends, key, nothing) === :biased || continue
    r, row = check(s, key)
    pgf = haskey(s.backends, :pgf_closure) ? last(check(s, :pgf_closure)) : nothing
    push!(biased_rows, (string("`:", s.id, "`"), string("`:", key, "`"), r.N, row.ΔR∞,
                        "[$(fmt(row.ΔR∞_ci[1])), $(fmt(row.ΔR∞_ci[2]))]", abs(row.ΔR∞) > 0.02 ? "yes" : "**no**",
                        pgf === nothing ? NaN : pgf.ΔR∞))
end
sort!(biased_rows; by = r -> r[1])
md_table(["scenario", "back end", "N", "ΔR∞", "95% CI", "abs(ΔR∞) > 0.02", "ΔR∞ of the PGF closure"], biased_rows)
```

| scenario | back end | N | ΔR∞ | 95% CI | abs(ΔR∞) \> 0.02 | ΔR∞ of the PGF closure |
|----|----|---:|---:|----|----|---:|
| `:sir_bim_N100000` | `:pairwise_const` | 100000 | 0.03344 | \[0.03177, 0.03512\] | yes | -0.0003395 |
| `:sir_bim_N1000` | `:pairwise_const` | 1000 | 0.03813 | \[0.03638, 0.03988\] | yes | 0.004346 |
| `:sir_bim` | `:pairwise_const` | 10000 | 0.03344 | \[0.03175, 0.03512\] | yes | -0.000345 |
| `:sir_pl_N100000` | `:pairwise_const` | 100000 | 0.06814 | \[0.06606, 0.07021\] | yes | -0.0002673 |
| `:sir_pl_N1000` | `:pairwise_const` | 1000 | 0.0832 | \[0.08016, 0.08624\] | yes | 0.0148 |
| `:sir_pl` | `:pairwise_const` | 10000 | 0.07013 | \[0.06786, 0.07241\] | yes | 0.001731 |

The `:approximate` verdicts (the Keeling clustered closure, the SIS
approximations, the graph-level models) carry no pass rule. Their errors
are printed on pages N05, N09–N12 and N04.

## 3. N-scaling: exact in the limit versus biased

Three scenarios are also run at N ∈ {10³, 10⁴, 10⁵} with {2000, 200, 20}
runs, so N·runs is constant. A representation that is exact in the limit
approaches the Monte Carlo floor as N grows, with a bias of order 1/N. A
structurally biased one levels off at its bias. The pairwise model with
the constant closure K = ⟨k(k − 1)⟩/⟨k⟩² is exact on the Poisson network
of `:sir_pois5` and biased on the bimodal and power-law networks of
`:sir_bim` and `:sir_pl`. The PGF closure is exact on all three:

``` julia
bases = [:sir_pois5, :sir_bim, :sir_pl]
variant(id, N) = N == 10_000 ? id : Symbol(id, "_N", N)
Ns = [1_000, 10_000, 100_000]
scaling = Dict{Tuple{Symbol,String},Vector{Float64}}()
floors = Dict{Symbol,Vector{Float64}}()
srows = []
for id in bases, N in Ns
    s = scenario(variant(id, N))
    r = scenario_summary(s)
    push!(get!(floors, id, Float64[]), maximum(r.cond[:I].se))
    for (label, y) in (("pairwise (constant K)", node_based(s)), ("PGF closure", node_based(s; closure = PGFClosure())))
        c = model_curves(y, solve_epidemic(y, s); t = s.tgrid, label = label)
        row = compare(r, c; observables = [:I])[label, :I]
        push!(get!(scaling, (id, label), Float64[]), row.D∞)
        push!(srows, (string("`:", id, "`"), N, r.nsims, label, row.D∞, maximum(r.cond[:I].se), row.ΔR∞))
    end
end
md_table(["scenario", "N", "runs", "model", "D∞(I)", "SE∞ (MC floor)", "ΔR∞"], srows)
```

| scenario | N | runs | model | D∞(I) | SE∞ (MC floor) | ΔR∞ |
|----|---:|---:|----|---:|---:|---:|
| `:sir_pois5` | 1000 | 2000 | pairwise (constant K) | 0.0136 | 0.001153 | 0.0007155 |
| `:sir_pois5` | 1000 | 2000 | PGF closure | 0.0136 | 0.001153 | 0.0007155 |
| `:sir_pois5` | 10000 | 200 | pairwise (constant K) | 0.002228 | 0.001135 | 0.0001227 |
| `:sir_pois5` | 10000 | 200 | PGF closure | 0.002228 | 0.001135 | 0.0001227 |
| `:sir_pois5` | 100000 | 20 | pairwise (constant K) | 0.0009282 | 0.001157 | 0.0001942 |
| `:sir_pois5` | 100000 | 20 | PGF closure | 0.0009282 | 0.001157 | 0.0001942 |
| `:sir_bim` | 1000 | 2000 | pairwise (constant K) | 0.02504 | 0.00092 | 0.03813 |
| `:sir_bim` | 1000 | 2000 | PGF closure | 0.01765 | 0.00092 | 0.004346 |
| `:sir_bim` | 10000 | 200 | pairwise (constant K) | 0.01006 | 0.001038 | 0.03344 |
| `:sir_bim` | 10000 | 200 | PGF closure | 0.00232 | 0.001038 | -0.000345 |
| `:sir_bim` | 100000 | 20 | pairwise (constant K) | 0.00894 | 0.000756 | 0.03344 |
| `:sir_bim` | 100000 | 20 | PGF closure | 0.0009376 | 0.000756 | -0.0003395 |
| `:sir_pl` | 1000 | 2000 | pairwise (constant K) | 0.03114 | 0.000614 | 0.0832 |
| `:sir_pl` | 1000 | 2000 | PGF closure | 0.0118 | 0.000614 | 0.0148 |
| `:sir_pl` | 10000 | 200 | pairwise (constant K) | 0.02173 | 0.000743 | 0.07013 |
| `:sir_pl` | 10000 | 200 | PGF closure | 0.002645 | 0.000743 | 0.001731 |
| `:sir_pl` | 100000 | 20 | pairwise (constant K) | 0.01997 | 0.000637 | 0.06814 |
| `:sir_pl` | 100000 | 20 | PGF closure | 0.001396 | 0.000637 | -0.0002673 |

``` julia
p = plot(; xscale = :log10, yscale = :log10, xlabel = "N", ylabel = "D∞(I)", legend = :outerright,
         title = "N-scaling of D∞(I)", size = (820, 440))
for id in bases
    plot!(p, Ns, scaling[(id, "PGF closure")]; marker = :circle, label = "PGF :$(id)")
    plot!(p, Ns, scaling[(id, "pairwise (constant K)")]; marker = :square, linestyle = :dash, label = "const K :$(id)")
    plot!(p, Ns, floors[id]; linestyle = :dot, color = :gray, label = "SE∞ :$(id)")
end
p
```

![N-scaling of the largest prevalence gap D∞(I) (log–log). Solid: PGF
closure; dashed: constant K; dotted: the largest standard error of the
simulated mean (Monte Carlo floor) of each
scenario.](index_files/figure-commonmark/cell-14-output-1.svg)

``` julia
fmtv(v) = join(fmt.(v), " → ")
for id in bases
    println("- `:", id, "`: PGF closure D∞(I) ", fmtv(scaling[(id, "PGF closure")]), "; constant K ",
            fmtv(scaling[(id, "pairwise (constant K)")]), " (N = 10³ → 10⁴ → 10⁵).")
end
```

- `:sir_pois5`: PGF closure D∞(I) 0.0136 → 0.002228 → 0.0009282;
  constant K 0.0136 → 0.002228 → 0.0009282 (N = 10³ → 10⁴ → 10⁵).
- `:sir_bim`: PGF closure D∞(I) 0.01765 → 0.00232 → 0.0009376; constant
  K 0.02504 → 0.01006 → 0.00894 (N = 10³ → 10⁴ → 10⁵).
- `:sir_pl`: PGF closure D∞(I) 0.0118 → 0.002645 → 0.001396; constant K
  0.03114 → 0.02173 → 0.01997 (N = 10³ → 10⁴ → 10⁵).

On the Poisson network the two closures give the same model, since K = 1
there. On the other two networks the constant-K gap stays above the
PGF-closure gap at N = 10⁵. The PGF closure keeps shrinking with N.

## Appendix: motif shapes and state classes

The SIS motif closures of page N10 track, for every connected induced
shape H with up to m vertices and every assignment of S/I to its
vertices, the number of induced copies of H in that state, counted as
unordered embeddings up to the automorphisms of H. The tables are
generated from the package’s own enumeration (`enumerate_shapes`,
`enumerate_state_classes`), so they list what the implementation tracks:
k = 2 (a ring) with 2 ≤ m ≤ 6, and k = 3 with m ∈ {2, 3, 4}.

``` julia
orders = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 2), (3, 3), (3, 4)]
crow = []
for (k, m) in orders
    shapes = NodeBasedModels.enumerate_shapes(MotifClosure(k, m))
    nclass = [length(NodeBasedModels.enumerate_state_classes(sh, [:S, :I])) for sh in shapes]
    push!(crow, ("($(k), $(m))", join(("`$(sh.name)`" for sh in shapes), ", "), sum(nclass)))
end
md_table(["(k, m)", "shapes tracked", "variables"], crow)
```

| (k, m) | shapes tracked | variables |
|----|----|---:|
| (2, 2) | `singleton`, `P2` | 5 |
| (2, 3) | `singleton`, `P2`, `P3` | 11 |
| (2, 4) | `singleton`, `P2`, `P4` | 15 |
| (2, 5) | `singleton`, `P2`, `P5` | 25 |
| (2, 6) | `singleton`, `P2`, `P6` | 41 |
| (3, 2) | `singleton`, `P2` | 5 |
| (3, 3) | `singleton`, `P2`, `P3`, `C3` | 15 |
| (3, 4) | `singleton`, `P2`, `P3`, `C3`, `P4`, `K13`, `paw`, `C4`, `K4me`, `K4` | 65 |

``` julia
shape_rows = []
seen = Set{Symbol}()
for (k, m) in orders, sh in NodeBasedModels.enumerate_shapes(MotifClosure(k, m))
    sh.name in seen && continue
    push!(seen, sh.name)
    classes = NodeBasedModels.enumerate_state_classes(sh, [:S, :I])
    push!(shape_rows, ("`$(sh.name)`", sh.n_nodes, isempty(sh.edges) ? "–" : join(("$(a)–$(b)" for (a, b) in sh.edges), ", "),
                       length(sh.automorphisms), length(classes),
                       join(("[" * join(string.(st), "") * "]×$(o)" for (st, o) in sort(classes; by = first)), " ")))
end
md_table(["shape", "vertices", "edges", "order of Aut", "state classes", "classes × orbit size"], shape_rows)
```

| shape | vertices | edges | order of Aut | state classes | classes × orbit size |
|----|---:|----|---:|---:|----|
| `singleton` | 1 | – | 1 | 2 | \[I\]×1 \[S\]×1 |
| `P2` | 2 | 1–2 | 2 | 3 | \[II\]×1 \[IS\]×2 \[SS\]×1 |
| `P3` | 3 | 1–2, 2–3 | 2 | 6 | \[III\]×1 \[IIS\]×2 \[ISI\]×1 \[ISS\]×2 \[SIS\]×1 \[SSS\]×1 |
| `P4` | 4 | 1–2, 2–3, 3–4 | 2 | 10 | \[IIII\]×1 \[IIIS\]×2 \[IISI\]×2 \[IISS\]×2 \[ISIS\]×2 \[ISSI\]×1 \[ISSS\]×2 \[SIIS\]×1 \[SISS\]×2 \[SSSS\]×1 |
| `P5` | 5 | 1–2, 2–3, 3–4, 4–5 | 2 | 20 | \[IIIII\]×1 \[IIIIS\]×2 \[IIISI\]×2 \[IIISS\]×2 \[IISII\]×1 \[IISIS\]×2 \[IISSI\]×2 \[IISSS\]×2 \[ISIIS\]×2 \[ISISI\]×1 \[ISISS\]×2 \[ISSIS\]×2 \[ISSSI\]×1 \[ISSSS\]×2 \[SIIIS\]×1 \[SIISS\]×2 \[SISIS\]×1 \[SISSS\]×2 \[SSISS\]×1 \[SSSSS\]×1 |
| `P6` | 6 | 1–2, 2–3, 3–4, 4–5, 5–6 | 2 | 36 | \[IIIIII\]×1 \[IIIIIS\]×2 \[IIIISI\]×2 \[IIIISS\]×2 \[IIISII\]×2 \[IIISIS\]×2 \[IIISSI\]×2 \[IIISSS\]×2 \[IISIIS\]×2 \[IISISI\]×2 \[IISISS\]×2 \[IISSII\]×1 \[IISSIS\]×2 \[IISSSI\]×2 \[IISSSS\]×2 \[ISIIIS\]×2 \[ISIISI\]×1 \[ISIISS\]×2 \[ISISIS\]×2 \[ISISSI\]×2 \[ISISSS\]×2 \[ISSIIS\]×2 \[ISSISS\]×2 \[ISSSIS\]×2 \[ISSSSI\]×1 \[ISSSSS\]×2 \[SIIIIS\]×1 \[SIIISS\]×2 \[SIISIS\]×2 \[SIISSS\]×2 \[SISISS\]×2 \[SISSIS\]×1 \[SISSSS\]×2 \[SSIISS\]×1 \[SSISSS\]×2 \[SSSSSS\]×1 |
| `C3` | 3 | 1–2, 2–3, 1–3 | 6 | 4 | \[III\]×1 \[IIS\]×3 \[ISS\]×3 \[SSS\]×1 |
| `K13` | 4 | 1–2, 1–3, 1–4 | 6 | 8 | \[IIII\]×1 \[IIIS\]×3 \[IISS\]×3 \[ISSS\]×1 \[SIII\]×1 \[SIIS\]×3 \[SISS\]×3 \[SSSS\]×1 |
| `paw` | 4 | 1–2, 1–3, 2–3, 1–4 | 2 | 12 | \[IIII\]×1 \[IIIS\]×1 \[IISI\]×2 \[IISS\]×2 \[ISSI\]×1 \[ISSS\]×1 \[SIII\]×1 \[SIIS\]×1 \[SISI\]×2 \[SISS\]×2 \[SSSI\]×1 \[SSSS\]×1 |
| `C4` | 4 | 1–2, 2–3, 3–4, 4–1 | 8 | 6 | \[IIII\]×1 \[IIIS\]×4 \[IISS\]×4 \[ISIS\]×2 \[ISSS\]×4 \[SSSS\]×1 |
| `K4me` | 4 | 1–2, 2–3, 3–4, 4–1, 1–3 | 4 | 9 | \[IIII\]×1 \[IIIS\]×2 \[IISI\]×2 \[IISS\]×4 \[ISIS\]×1 \[ISSS\]×2 \[SISI\]×1 \[SISS\]×2 \[SSSS\]×1 |
| `K4` | 4 | 1–2, 1–3, 1–4, 2–3, 2–4, 3–4 | 24 | 5 | \[IIII\]×1 \[IIIS\]×4 \[IISS\]×6 \[ISSS\]×4 \[SSSS\]×1 |

The orbit size \|orb(σ)\| = \|Aut(H)\|/\|Stab(σ)\| is the number of
labelled state assignments that the automorphisms of H identify with the
canonical class σ. For the pair shape P₂ the classes and their orbit
sizes are below. The pairwise model of NodeBasedModels counts ordered
pairs for \[SS\] and \[II\] and each SI edge once for \[SI\], so \[SS\]
= 2E_SS, \[SI\] = E_IS and \[II\] = 2E_II (the counting convention of
`motif_based.jl`):

``` julia
p2 = only(sh for sh in NodeBasedModels.enumerate_shapes(MotifClosure(3, 2)) if sh.name === :P2)
NodeBasedModels.enumerate_state_classes(p2, [:S, :I])
```

    3-element Vector{Tuple{Vector{Symbol}, Int64}}:
     ([:I, :I], 1)
     ([:I, :S], 2)
     ([:S, :S], 1)

No Lean statement is involved on this page.

## References

<div id="refs">

</div>
