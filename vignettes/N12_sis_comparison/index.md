# All SIS approximations against simulation


- [What this page shows](#what-this-page-shows)
- [The shared first cell](#the-shared-first-cell)
- [Every approximation on the scenario
  grid](#every-approximation-on-the-scenario-grid)
- [The transient](#the-transient)
- [The endpoint](#the-endpoint)
- [The take-off](#the-take-off)
- [References](#references)

## What this page shows

The edge-based model stops at SIS (page E13): an arrow back into the
susceptible class is outside T_EB. NodeBasedModels has four families of
deterministic approximations for SIS, introduced on the previous pages:

- the population pairwise model with the Bernoulli closure (page N09);
- the pairwise model with reinfection counting, capped at L (page N09);
- the motif closures of order m = 2, 3, 4 (page N10);
- the neighbourhood model with n = 2 (page N11).

This page compares all of them with the one committed NetworkOutbreaks
ensemble of `:sis_reg3`. It gives the transient and the endpoint in two
tables, both computed from the summary. The mirrored EdgeBasedModels
page E13 shows the same comparison for the pairwise, motif and
neighbourhood models.

## The shared first cell

This cell is the same as the first cell of EdgeBasedModels’ page E13,
apart from the back-end lines. On the EBM side, the back end refuses the
model. Here it builds the pairwise model twice.

``` julia
include(joinpath(@__DIR__, "..", "_shared", "setup.jl"))
require_summaries([:sis_reg3])                           # back end
using Statistics
using NetworkEpiCore, NetworkOutbreaks, Catalyst, Plots
sis = @reaction_network sis begin
    @parameters τ γ
    τ, S + I --> 2I        # contact: per-contact (per-edge) rate τ
    γ, I --> S             # recovery back into the susceptible class (type resus)
end
model = contact_model(sis)          # prints the typing report: T_net, not T_EB
sc    = scenario(:sis_reg3)         # 3-regular, τ = 1/2, γ = 1/4, 1% seeds in I, t ∈ [0, 80]
@assert isequivalent(model, sc.model)
ref   = scenario_summary(sc)        # committed NetworkOutbreaks ensemble, conditioned on survival
# --- NBM vignette ---------------------------------------------------------------
using NodeBasedModels
sys  = node_based(model, sc.network)                       # low level: pairwise, Bernoulli closure
sysF = generate_pairwise(sis_model(), sc.network, default_closure(sc.network); cumulative = true)  # factory
@assert vector_fields_equal(symbolic_ode(sys), symbolic_ode(sysF))
model
```

    ContactModel :sis  (source: Catalyst.ReactionSystem; method: stoichiometry; rates: PerContact)
      species       S (Sus)   I
      contacts      [1] S + I → I + I    τ    contact    infector I, entry I
      transitions   [2] I → S            γ    resus
      typing        T_net  ⇒  edge_based ✗  s_anchored ✗  pairwise ✓  individual ✓  pair ✓  stochastic ✓  mass_action ✓
      violations    `γ, I --> S` (type resus: node → sus) produces the susceptible species S.
      assumptions   Sus inferred as recipients \ contact products = {S}

The scenario and the reference:

``` julia
a = anchors(sc);
describe_reference(ref);
```

    :sis_reg3: γ = 0.25, τ = 0.5; seeds I 0.01; t = 0:0.25:80; edge-based R₀ not defined (model outside T_EB); pairwise threshold τ_c = 0.125, τ/τ_c = 4 (differs from the canonical anchors: no edge-based R₀, τ ≠ 1/6)
    NetworkOutbreaks reference :sis_reg3 (hash 485d890e): N = 10000 nodes, 200 runs on a fresh graph per run, algorithm :next_reaction; conditioning: surviving runs only (prevalence at t_end > 0); 200 of 200 runs kept (surviving runs), P(survival) = 1.000 (95% CI 0.981–1.000); no time alignment.

The scenario lists the back ends it expects to accept SIS, each with its
verdict. None of them is exact. The error of each is reported here, not
asserted:

``` julia
sort(collect(sc.backends); by = first)
```

    7-element Vector{Pair{Symbol, Symbol}}:
             :edge_based => :inadmissible
                  :motif => :approximate
          :neighbourhood => :approximate
     :pairwise_bernoulli => :approximate
         :pairwise_const => :approximate
            :pgf_closure => :inadmissible
            :reinfection => :approximate

## Every approximation on the scenario grid

Every model starts from the scenario’s 1% infected at random. The
reinfection-counting models seed the count-one class, and they are
integrated with Tsit5 (page N09 shows why). The motif and neighbourhood
systems are numeric ODEs with their own solvers, and their prevalence is
wrapped as `ModelCurves` on the scenario grid.

``` julia
using OrdinaryDiffEqDefault: Tsit5
ρ = last(only(sc.initial.fractions))
cpw = model_curves(sys, solve_epidemic(sys, sc); t = sc.tgrid, label = "pairwise")
function reinfection_curves(L)
    m = node_based(with_reinfection_counting(model, L), sc.network)
    sol = solve_epidemic(m, sc; initial = SeedFraction(:I_1 => ρ), solver = Tsit5())   # see page N09
    tot = reinfection_totals(m, sol)
    ModelCurves(sc.tgrid, Dict(:I => tot[:I], :cumulative => model_curves(m, sol; t = sc.tgrid)[:cumulative]);
                label = "reinfection L = $(L)", representation = :reinfection,
                metadata = Dict(:variables => length(m.singles) + length(m.pairs)))
end
function motif_curves(m)
    s = node_based(model, sc.network; level = :motif, closure = MotifClosure(3, m), p = sc.params,
                   initial = sc.initial, tspan = sc.tspan)
    ModelCurves(sc.tgrid, Dict(:I => compartment(s, solve_motif(s; saveat = sc.tgrid), :I));
                label = "motif m = $(m)", representation = :motif,
                metadata = Dict(:variables => length(s.variables)))
end
nb  = node_based(model, sc.network; level = :neighbourhood, n = 2, p = sc.params, initial = sc.initial,
                 tspan = sc.tspan)
cnb = ModelCurves(sc.tgrid, Dict(:I => neighbourhood_compartment(nb, solve_neighbourhood(nb; saveat = sc.tgrid), :I));
                  label = "neighbourhood n = 2", representation = :neighbourhood,
                  metadata = Dict(:variables => length(nb.var_names)))
approx = [cpw, reinfection_curves(1), reinfection_curves(2), motif_curves(2), motif_curves(3), motif_curves(4), cnb]
nvars = Dict(c.label => get(c.metadata, :variables, length(sys.singles) + length(sys.pairs)) for c in approx)
tab = compare(ref, approx; observables = [:I])
```

    ComparisonTable :sis_reg3  (scenario 485d890e; conditioned mean of 200 runs)
      curve                observable        D∞     t(D∞)       SE∞        z∞       ΔR∞  95% CI                 Δpeak   Δt_peak  coverage
      pairwise             I            0.14556      9.50   0.00160     93.69  14.70400  [14.70400, 14.70400]  -0.00023      2.00     0.153
      reinfection L = 1    I            0.00289     14.25   0.00160      6.71  14.48108  [14.48108, 14.48108]  -0.00023      8.75     0.287
      reinfection L = 2    I            0.00382      9.75   0.00160      5.33  14.47579  [14.47579, 14.47579]  -0.00023     41.00     0.243
      motif m = 2          I            0.14556      9.50   0.00160     93.69       NaN  [     NaN,      NaN]  -0.00023      4.00     0.153
      motif m = 3          I            0.06009      9.75   0.00160     39.65       NaN  [     NaN,      NaN]  -0.00025    -17.50     0.202
      motif m = 4          I            0.00888     10.75   0.00160     11.27       NaN  [     NaN,      NaN]  -0.00016    -15.50     0.688
      neighbourhood n = 2  I            0.02525     10.75   0.00160     21.41       NaN  [     NaN,      NaN]  -0.00106     42.25     0.769

``` julia
distinct_styles!(comparisonplot(ref, approx...; observables = [:I]))
```

![Every SIS approximation against the committed NetworkOutbreaks
ensemble of `:sis_reg3`, conditioned on survival (spread band q2.5–q97.5
of the runs; residual panel: mean band ±1.96
SE).](index_files/figure-commonmark/cell-6-output-1.svg)

The ΔR∞ column of `compare` is not meaningful for SIS. It subtracts the
simulated fraction of nodes ever infected from the model’s cumulative
incidence at t = 80, which counts reinfections, and only the
pairwise-type curves carry cumulative incidence at all (page N09 prints
both numbers). The peak columns are not meaningful on an endemic plateau
either. The two tables below are the comparison.

## The transient

The transient is summarised by the largest prevalence gap D∞ and its
time, by the standardised gap z∞, by the fraction of the grid on which
the curve lies within the mean band (coverage), and by the time t½ at
which prevalence first reaches half of its final simulated value:

``` julia
st = ref.cond[:I]
half = st.mean[end] / 2
t_half(x) = sc.tgrid[findfirst(≥(half), x)]
trows = Any[(c.label, nvars[c.label], tab[c.label, :I].D∞, tab[c.label, :I].t_D∞, tab[c.label, :I].z∞,
          tab[c.label, :I].coverage, t_half(c[:I]), t_half(c[:I]) - t_half(st.mean)) for c in approx]
push!(trows, ("NetworkOutbreaks (mean)", "–", 0.0, NaN, NaN, NaN, t_half(st.mean), 0.0))
md_table(["model", "variables", "D∞(I)", "t at D∞", "z∞", "coverage", "t½", "t½ − NO"], trows)
```

| model                   | variables |    D∞(I) | t at D∞ |    z∞ | coverage |   t½ | t½ − NO |
|-------------------------|----------:|---------:|--------:|------:|---------:|-----:|--------:|
| pairwise                |         5 |   0.1456 |     9.5 | 93.69 |   0.1526 |  7.5 |   -1.25 |
| reinfection L = 1       |         9 | 0.002891 |   14.25 | 6.713 |   0.2866 | 8.75 |       0 |
| reinfection L = 2       |        20 | 0.003824 |    9.75 | 5.333 |    0.243 | 8.75 |       0 |
| motif m = 2             |         5 |   0.1456 |     9.5 | 93.69 |   0.1526 |  7.5 |   -1.25 |
| motif m = 3             |        15 |  0.06009 |    9.75 | 39.65 |   0.2025 | 8.25 |    -0.5 |
| motif m = 4             |        65 | 0.008878 |   10.75 | 11.27 |   0.6885 |  8.5 |   -0.25 |
| neighbourhood n = 2     |         8 |  0.02525 |   10.75 | 21.41 |   0.7695 |  8.5 |   -0.25 |
| NetworkOutbreaks (mean) |         – |        0 |     NaN |   NaN |      NaN | 8.75 |       0 |

## The endpoint

The endpoint is summarised by the prevalence at t = 80 and its mean over
the last 20 time units, each against the simulated mean with its
standard error. For the models that carry it, it also includes the mean
number of infections per node:

``` julia
late = findall(≥(last(sc.tgrid) - 20), sc.tgrid)
cum = ref.cond[:cumulative]
erows = Any[]
for c in approx
    cumv = haskey(c.values, :cumulative) ? c[:cumulative][end] : NaN
    push!(erows, (c.label, c[:I][end], c[:I][end] - st.mean[end], mean(c[:I][late]),
                  mean(c[:I][late]) - mean(st.mean[late]), cumv, cumv - cum.mean[end]))
end
push!(erows, ("NetworkOutbreaks (mean)", st.mean[end], st.se[end], mean(st.mean[late]), NaN, cum.mean[end], cum.se[end]))
md_table(["model", "I(80)", "I(80) − NO", "mean I, t ∈ [60, 80]", "difference", "cumulative(80)", "difference"], erows)
```

| model | I(80) | I(80) − NO | mean I, t ∈ \[60, 80\] | difference | cumulative(80) | difference |
|----|---:|---:|---:|---:|---:|---:|
| pairwise | 0.8182 | 0.001126 | 0.8182 | 0.001057 | 15.7 | 0.2381 |
| reinfection L = 1 | 0.8182 | 0.001126 | 0.8182 | 0.001057 | 15.48 | 0.01513 |
| reinfection L = 2 | 0.8182 | 0.001126 | 0.8182 | 0.001057 | 15.48 | 0.00984 |
| motif m = 2 | 0.8182 | 0.001126 | 0.8182 | 0.001057 | NaN | NaN |
| motif m = 3 | 0.8163 | -0.0007423 | 0.8163 | -0.0008111 | NaN | NaN |
| motif m = 4 | 0.8174 | 0.0003262 | 0.8174 | 0.0002625 | NaN | NaN |
| neighbourhood n = 2 | 0.8173 | 0.0002934 | 0.8173 | 0.0002246 | NaN | NaN |
| NetworkOutbreaks (mean) | 0.8171 | 0.0003 | 0.8171 | NaN | 15.47 | 0.003598 |

The last row gives the simulated values, with the standard error of the
mean in the difference columns where it is defined.

At t = 80 every approximation is within 0.001126 of the simulated
endemic prevalence 0.8171, whose standard error is 0.0003. The endemic
level is therefore not what separates them. The transient does. Over the
whole time course the closest curve is reinfection L = 1 (D∞(I) =
0.002891), and the furthest is pairwise (D∞(I) = 0.1456), together with
the motif closure m = 2. The motif closure with m = 2 is Keeling’s
pairwise closure: its curve differs from the pairwise one by at most
1.753e-09.

## The take-off

``` julia
p = plot(ref, :I; median = false, legend = :bottomright, xlabel = "t", ylabel = "I (fraction of nodes)",
         title = "SIS on a 3-regular network: take-off")
for c in approx
    plot!(p, c, :I)
end
distinct_styles!(p)
xlims!(p, 0, 30)
```

![The first 30 time units of the comparison (prevalence; band: spread
q2.5–q97.5 of the surviving
runs).](index_files/figure-commonmark/cell-10-output-1.svg)

``` julia
p2 = plot(; xscale = :log10, yscale = :log10, xlabel = "number of variables", ylabel = "D∞(I)",
          legend = :outerright, title = "Accuracy and size", size = (820, 460),
          left_margin = 5Plots.mm, bottom_margin = 6Plots.mm)
shapes = [:circle, :rect, :diamond, :utriangle, :dtriangle, :star5, :hexagon]
for (i, c) in enumerate(approx)       # decreasing sizes and distinct shapes keep coinciding points visible
    scatter!(p2, [nvars[c.label]], [tab[c.label, :I].D∞]; label = c.label, markershape = shapes[i],
             markersize = 10 - i, markerstrokewidth = 1.5, markeralpha = 0.7)
end
hline!(p2, [maximum(st.se)]; linestyle = :dot, color = :gray, label = "SE∞")
p2
```

![Largest prevalence gap D∞(I) against the number of variables of each
approximation (log scales); the dotted line is the largest standard
error of the simulated mean. The pairwise and motif m = 2 points
coincide, so they are drawn with different shapes and
sizes.](index_files/figure-commonmark/cell-11-output-1.svg)

None of these is exact. They are moment closures of an SIS process on a
random 3-regular graph, and the simulation is the reference. No Lean
statement is involved on this page.

## References

<div id="refs">

</div>
