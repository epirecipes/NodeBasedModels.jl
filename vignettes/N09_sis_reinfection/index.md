# SIS: pairwise and reinfection counting


- [What this page shows](#what-this-page-shows)
- [The SIS pairwise model, twice](#the-sis-pairwise-model-twice)
- [Reinfection counting as a syntax
  transform](#reinfection-counting-as-a-syntax-transform)
- [Against simulation](#against-simulation)
  - [A numerical caveat](#a-numerical-caveat)
- [Reinfection histograms](#reinfection-histograms)
- [Summary](#summary)
- [References](#references)

## What this page shows

In SIS a recovered node is susceptible again (I → S, typing `:resus`).
The model is outside the edge-based fragment T_EB (page E13), but the
pairwise model accepts it. On a 3-regular network the pairwise model
reaches the right endemic level but gets the take-off wrong. Keeling,
House, Cooper and Pellis (2016, *PLoS Comput. Biol.* 12:e1005296) refine
the pairwise model by counting infections: a node that has been infected
p times is kept in its own class S_p or I_p, up to a cap L. This page:

1.  builds the SIS pairwise model of `:sis_reg3` from a Catalyst
    network, then with the factory, and asserts that the two are the
    same;
2.  applies the syntax transform `with_reinfection_counting(model, L)`
    for L = 0, …, 4 and lifts each result, again in both forms;
3.  compares prevalence and cumulative incidence with the committed
    NetworkOutbreaks ensemble, which is conditioned on survival;
4.  compares the fractions of nodes infected p times at the end of the
    run with the reinfection histogram stored in the committed summary.

``` julia
include(joinpath(@__DIR__, "..", "_shared", "setup.jl"))
require_summaries([:sis_reg3])
```

## The SIS pairwise model, twice

``` julia
using Catalyst
sis = @reaction_network sis begin
    @parameters τ γ
    τ, S + I --> 2I        # contact: per-contact (per-edge) rate τ
    γ, I --> S             # recovery back into the susceptible class (type resus)
end
model = contact_model(sis)          # the typing report: T_net, not T_EB
```

    ContactModel :sis  (source: Catalyst.ReactionSystem; method: stoichiometry; rates: PerContact)
      species       S (Sus)   I
      contacts      [1] S + I → I + I    τ    contact    infector I, entry I
      transitions   [2] I → S            γ    resus
      typing        T_net  ⇒  edge_based ✗  s_anchored ✗  pairwise ✓  individual ✓  pair ✓  stochastic ✓  mass_action ✓
      violations    `γ, I --> S` (type resus: node → sus) produces the susceptible species S.
      assumptions   Sus inferred as recipients \ contact products = {S}

``` julia
sc = scenario(:sis_reg3)            # 3-regular, τ = 1/2, γ = 1/4, 1% seeds in I, t ∈ [0, 80]
@assert isequivalent(model, sc.model)
sys  = node_based(model, sc.network)                                                           # low level
sysF = generate_pairwise(sis_model(), sc.network, default_closure(sc.network); cumulative = true)  # factory
@assert vector_fields_equal(symbolic_ode(sys), symbolic_ode(sysF))
(; closure = default_closure(sc.network), K = closure_constant(sc.network.degrees))
```

    (closure = BernoulliClosure(), K = 0.6666666666666666)

On a k-regular network the default closure is the Bernoulli (Keeling)
closure \[ABC\] ≈ K\[AB\]\[BC\]/\[B\] with K = (k − 1)/k. The equations,
with the pairs counted as ordered pairs (\[SS\] counts each SS edge
twice):

``` julia
symbolic_ode(sys)
```

    SymbolicODE :pairwise_sis (6 states)
      dS/dt = I(t)*γ - SI(t)*τ
      dI/dt = -I(t)*γ + SI(t)*τ
      dSS/dt = 2SI(t)*γ - 1.3333333333333333ifelse(S(t) == 0, 0, (SI(t)*SS(t)) / S(t))*τ
      dSI/dt = II(t)*γ - SI(t)*γ - SI(t)*τ + 0.6666666666666666ifelse(S(t) == 0, 0, (SI(t)*SS(t)) / S(t))*τ - 0.6666666666666666ifelse(S(t) == 0, 0, (SI(t)^2) / S(t))*τ
      dII/dt = -2II(t)*γ + 2SI(t)*τ + 1.3333333333333333ifelse(S(t) == 0, 0, (SI(t)^2) / S(t))*τ
      dcumulative/dt = SI(t)*τ
      parameters  γ, τ

The scenario is above the pairwise threshold τ_c = γ/(k − 1):

``` julia
a = anchors(sc);
```

    :sis_reg3: γ = 0.25, τ = 0.5; seeds I 0.01; t = 0:0.25:80; edge-based R₀ not defined (model outside T_EB); pairwise threshold τ_c = 0.125, τ/τ_c = 4 (differs from the canonical anchors: no edge-based R₀, τ ≠ 1/6)

Here τ/τ_c = 4, so the pairwise model has an endemic equilibrium.

## Reinfection counting as a syntax transform

`with_reinfection_counting(model, L)` refines every compartment X into
X_p, where p is the number of infections the node has had, saturating at
L. An infection takes S_p to I\_{min(p+1, L)} whatever the count of the
infector; every other reaction keeps the count. For L = 2:

``` julia
m2 = with_reinfection_counting(model, 2)
```

    ContactModel :sis_reinf_L2  (source: transform; method: explicit; rates: PerContact)
      species       S_0 (Sus)   S_1 (Sus)   S_2 (Sus)   I_1   I_2
      contacts      [1] S_0 + I_1 → I_1 + I_1    τ    contact    infector I_1, entry I_1
                    [2] S_0 + I_2 → I_1 + I_2    τ    contact    infector I_2, entry I_1
                    [3] S_1 + I_1 → I_2 + I_1    τ    contact    infector I_1, entry I_2
                    [4] S_1 + I_2 → I_2 + I_2    τ    contact    infector I_2, entry I_2
                    [5] S_2 + I_1 → I_2 + I_1    τ    contact    infector I_1, entry I_2
                    [6] S_2 + I_2 → I_2 + I_2    τ    contact    infector I_2, entry I_2
      transitions   [7] I_1 → S_1                γ    resus
                    [8] I_2 → S_2                γ    resus
      typing        T_net  ⇒  edge_based ✗  s_anchored ✗  pairwise ✓  individual ✓  pair ✓  stochastic ✓  mass_action ✓
      violations    `γ, I_1 --> S_1` (type resus: node → sus) produces the susceptible species S_1.
                    `γ, I_2 --> S_2` (type resus: node → sus) produces the susceptible species S_2.
                    `τ, S_1 + I_1 --> I_2 + I_1`: S_1 is a second susceptible class (multiple_sus).
                    `τ, S_2 + I_1 --> I_2 + I_1`: S_2 is a third susceptible class (multiple_sus).
      assumptions   Sus inferred as recipients \ contact products = {S}
                    transform of a catalyst model (method stoichiometry)
                    with_reinfection_counting(L = 2): X ↦ X_p by infection count p ≤ 2 (saturating); infections: S_I_to_I

``` julia
md_table(["reaction", "type", "from", "to", "rate"],
         vcat([(string(c.name), "contact", "$(c.recipient) + $(c.infector)", "$(c.product) + $(c.infector)",
                string(c.rate)) for c in contacts(m2)],
              [(string(t.name), "transition", string(t.from), string(t.to), string(t.rate))
               for t in m2.transitions]))
```

| reaction       | type       | from      | to        | rate |
|----------------|------------|-----------|-----------|------|
| S_0_I_1_to_I_1 | contact    | S_0 + I_1 | I_1 + I_1 | τ    |
| S_0_I_2_to_I_1 | contact    | S_0 + I_2 | I_1 + I_2 | τ    |
| S_1_I_1_to_I_2 | contact    | S_1 + I_1 | I_2 + I_1 | τ    |
| S_1_I_2_to_I_2 | contact    | S_1 + I_2 | I_2 + I_2 | τ    |
| S_2_I_1_to_I_2 | contact    | S_2 + I_1 | I_2 + I_1 | τ    |
| S_2_I_2_to_I_2 | contact    | S_2 + I_2 | I_2 + I_2 | τ    |
| I_1_to_S_1     | transition | I_1       | S_1       | γ    |
| I_2_to_S_2     | transition | I_2       | S_2       | γ    |

The refined model has three susceptible classes S₀, S₁ and S₂ and arrows
back into two of them, so it is again a T_net model for the node-based
and stochastic back ends only. Nodes that have never been infected start
in S₀. The seeds are counted as infected once, so they start in I₁, as
in NetworkOutbreaks, whose reinfection counter also counts a seed once.
The seeding is therefore the scenario’s 1% placed in the count-one
class:

``` julia
ρ = last(only(sc.initial.fractions))
seed(L) = SeedFraction(Symbol(:I_, min(1, L)) => ρ)       # L = 0: the unrefined model, I₀ = I
(; ρ, seeds = [seed(L) for L in 0:4])
```

    (ρ = 0.01, seeds = SeedFraction[SeedFraction([:I_0 => 0.01], nothing), SeedFraction([:I_1 => 0.01], nothing), SeedFraction([:I_1 => 0.01], nothing), SeedFraction([:I_1 => 0.01], nothing), SeedFraction([:I_1 => 0.01], nothing)])

Each refined model is lifted in both forms, and the two vector fields
are compared:

``` julia
Ls = 0:4
refined = Dict(L => with_reinfection_counting(model, L) for L in Ls)
pw = Dict(L => node_based(refined[L], sc.network) for L in Ls)
rows = map(Ls) do L
    f = generate_pairwise(with_reinfection_counting(sis_model(), L), sc.network, BernoulliClosure(); cumulative = true)
    (L, length(pw[L].singles), length(pw[L].pairs), vector_fields_equal(symbolic_ode(pw[L]), symbolic_ode(f)))
end
md_table(["L", "singles", "pairs", "node_based = generate_pairwise"], rows)
```

|   L | singles | pairs | node_based = generate_pairwise |
|----:|--------:|------:|-------------------------------:|
|   0 |       2 |     3 |                           true |
|   1 |       3 |     6 |                           true |
|   2 |       5 |    15 |                           true |
|   3 |       7 |    28 |                           true |
|   4 |       9 |    45 |                           true |

For L = 1 the system is small enough to read. S₀ holds the nodes that
have never been infected and S₁ those that have recovered at least once:

``` julia
symbolic_ode(pw[1])
```

    SymbolicODE :pairwise_sis_reinf_L1 (10 states)
      dS_0/dt = -S_0I_1(t)*τ
      dS_1/dt = I_1(t)*γ - S_1I_1(t)*τ
      dI_1/dt = -I_1(t)*γ + S_0I_1(t)*τ + S_1I_1(t)*τ
      dS_0S_0/dt = -1.3333333333333333ifelse(S_0(t) == 0, 0, (S_0I_1(t)*S_0S_0(t)) / S_0(t))*τ
      dS_0S_1/dt = S_0I_1(t)*γ - 0.6666666666666666(ifelse(S_0(t) == 0, 0, (S_0I_1(t)*S_0S_1(t)) / S_0(t)) + ifelse(S_1(t) == 0, 0, (S_0S_1(t)*S_1I_1(t)) / S_1(t)))*τ
      dS_0I_1/dt = -S_0I_1(t)*(γ + τ) + 0.6666666666666666ifelse(S_0(t) == 0, 0, (S_0I_1(t)*S_0S_0(t)) / S_0(t))*τ + 0.6666666666666666ifelse(S_1(t) == 0, 0, (S_0S_1(t)*S_1I_1(t)) / S_1(t))*τ - 0.6666666666666666ifelse(S_0(t) == 0, 0, (S_0I_1(t)^2) / S_0(t))*τ
      dS_1S_1/dt = 2S_1I_1(t)*γ - 1.3333333333333333ifelse(S_1(t) == 0, 0, (S_1I_1(t)*S_1S_1(t)) / S_1(t))*τ
      dS_1I_1/dt = I_1I_1(t)*γ - S_1I_1(t)*(γ + τ) - 0.6666666666666666ifelse(S_1(t) == 0, 0, (S_1I_1(t)^2) / S_1(t))*τ + 0.6666666666666666(ifelse(S_0(t) == 0, 0, (S_0I_1(t)*S_0S_1(t)) / S_0(t)) + ifelse(S_1(t) == 0, 0, (S_1I_1(t)*S_1S_1(t)) / S_1(t)))*τ
      dI_1I_1/dt = -2I_1I_1(t)*γ + 2S_0I_1(t)*τ + 2S_1I_1(t)*τ + 1.3333333333333333ifelse(S_0(t) == 0, 0, (S_0I_1(t)^2) / S_0(t))*τ + 1.3333333333333333ifelse(S_1(t) == 0, 0, (S_1I_1(t)^2) / S_1(t))*τ
      dcumulative/dt = S_0I_1(t)*τ + S_1I_1(t)*τ
      parameters  γ, τ

The closure acts on the refined pairs, for example \[S₀ S₀ I₁\] ≈
K\[S₀S₀\]\[S₀I₁\]/\[S₀\]. Summed over p, the refined singles are not the
solution of the unrefined pairwise model, because the triples are closed
at the refined level. The refinement therefore changes the prevalence
curve, as the comparison below shows.

## Against simulation

The pairwise systems are solved on the scenario grid. The refined models
are labelled with the representation `:reinfection`, so that the plots
give each L its own colour; L = 0 keeps the pairwise style. Their
prevalence is the sum over counts, I = Σ_p I_p (`reinfection_totals`),
and their `:cumulative` accumulator counts the seeds and every
infection, reinfections included. NetworkOutbreaks’ `:cumulative`
observable counts the same events, so for SIS it is the mean number of
infections per node (it exceeds 1).

The systems are integrated with the explicit Runge–Kutta method Tsit5 at
the package’s default tolerances (reltol 10⁻⁸, abstol 10⁻¹⁰); the
numerical caveat below says why the default auto-switching solver is not
used here.

``` julia
using OrdinaryDiffEqDefault: Tsit5
sols = Dict(L => solve_epidemic(pw[L], sc; initial = seed(L), solver = Tsit5()) for L in Ls)
function lumped_curves(L)
    tot = reinfection_totals(pw[L], sols[L])
    cum = model_curves(pw[L], sols[L]; t = sc.tgrid)[:cumulative]
    ModelCurves(sc.tgrid, Dict(:I => tot[:I], :S => tot[:S], :cumulative => cum);
                label = "pairwise, L = $(L)", representation = L == 0 ? :pairwise : :reinfection)
end
dets = [lumped_curves(L) for L in Ls]
ref = scenario_summary(sc)
tab = compare(ref, dets; observables = [:I, :cumulative])
```

    ComparisonTable :sis_reg3  (scenario 485d890e; conditioned mean of 200 runs)
      curve            observable        D∞     t(D∞)       SE∞        z∞       ΔR∞  95% CI                 Δpeak   Δt_peak  coverage
      pairwise, L = 0  I            0.14556      9.50   0.00160     93.69  14.70400  [14.70400, 14.70400]  -0.00023      5.00     0.153
      pairwise, L = 0  cumulative   0.27836     10.75   0.00362     79.82  14.70400  [14.70400, 14.70400]   0.23805      0.00     0.025
      pairwise, L = 1  I            0.00289     14.25   0.00160      6.71  14.48108  [14.48108, 14.48108]  -0.00023      8.75     0.287
      pairwise, L = 1  cumulative   0.01559     78.25   0.00362      4.32  14.48108  [14.48108, 14.48108]   0.01513      0.00     0.408
      pairwise, L = 2  I            0.00382      9.75   0.00160      5.33  14.47579  [14.47579, 14.47579]  -0.00023     41.00     0.243
      pairwise, L = 2  cumulative   0.01031     78.25   0.00362      2.85  14.47579  [14.47579, 14.47579]   0.00984      0.00     0.667
      pairwise, L = 3  I            0.00392      9.75   0.00160      5.29  14.47558  [14.47558, 14.47558]  -0.00023     43.00     0.246
      pairwise, L = 3  cumulative   0.01009     78.25   0.00362      2.79  14.47558  [14.47558, 14.47558]   0.00962      0.00     0.698
      pairwise, L = 4  I            0.00393      9.75   0.00160      5.28  14.47555  [14.47555, 14.47555]  -0.00023     43.00     0.246
      pairwise, L = 4  cumulative   0.01007     78.25   0.00362      2.79  14.47555  [14.47555, 14.47555]   0.00960      0.00     0.698

``` julia
describe_reference(ref);
```

    NetworkOutbreaks reference :sis_reg3 (hash 485d890e): N = 10000 nodes, 200 runs on a fresh graph per run, algorithm :next_reaction; conditioning: surviving runs only (prevalence at t_end > 0); 200 of 200 runs kept (surviving runs), P(survival) = 1.000 (95% CI 0.981–1.000); no time alignment.

``` julia
distinct_styles!(comparisonplot(ref, dets...; observables = [:I, :cumulative]))
```

![Pairwise SIS models with reinfection counting (L = 0 is the plain
pairwise model) against the committed NetworkOutbreaks ensemble of
`:sis_reg3`, conditioned on survival (spread band q2.5–q97.5 of the
runs; residual panel: mean band ±1.96
SE).](index_files/figure-commonmark/cell-14-output-1.svg)

The table above is `compare` over the whole time course. Its ΔR∞ column
is not a comparison of like with like for SIS. `compare` defines ΔR∞ as
the model’s `:cumulative` at t = 80 minus the mean of the ensemble’s
per-run final size, and for SIS that final size is the fraction of nodes
*ever* infected:

``` julia
fs = ref.final_size[ref.major]
(; mean_fraction_ever_infected = sum(fs) / length(fs),
   pairwise_cumulative_80 = dets[1][:cumulative][end],
   ΔR∞_pairwise = tab[dets[1].label, :cumulative].ΔR∞)
```

    (mean_fraction_ever_infected = 1.0, pairwise_cumulative_80 = 15.704003220875496, ΔR∞_pairwise = 14.704003220875496)

The model’s `:cumulative` counts every infection, reinfections included,
so ΔR∞ is the mean number of infections per node minus the fraction ever
infected, and it is large for every L. The like for like comparison of
`:cumulative` is the `cumulative(80) − NO` column below, against the
simulated mean of `:cumulative`. The comparisonplot’s peak statistics
are not meaningful on an endemic plateau either. The quantities that
matter are the largest gap and the endemic level:

``` julia
st, sc_cum = ref.cond[:I], ref.cond[:cumulative]
md_table(["model", "D∞(I)", "t at D∞(I)", "I(80)", "I(80) − NO", "cumulative(80)", "cumulative(80) − NO"],
         vcat([(c.label, tab[c.label, :I].D∞, tab[c.label, :I].t_D∞, c[:I][end], c[:I][end] - st.mean[end],
                c[:cumulative][end], c[:cumulative][end] - sc_cum.mean[end]) for c in dets],
              [("NetworkOutbreaks (mean ± SE)", NaN, NaN, st.mean[end], st.se[end], sc_cum.mean[end], sc_cum.se[end])]))
```

| model | D∞(I) | t at D∞(I) | I(80) | I(80) − NO | cumulative(80) | cumulative(80) − NO |
|----|---:|---:|---:|---:|---:|---:|
| pairwise, L = 0 | 0.1456 | 9.5 | 0.8182 | 0.001126 | 15.7 | 0.2381 |
| pairwise, L = 1 | 0.002891 | 14.25 | 0.8182 | 0.001126 | 15.48 | 0.01513 |
| pairwise, L = 2 | 0.003824 | 9.75 | 0.8182 | 0.001126 | 15.48 | 0.00984 |
| pairwise, L = 3 | 0.003921 | 9.75 | 0.8182 | 0.001126 | 15.48 | 0.009624 |
| pairwise, L = 4 | 0.003925 | 9.75 | 0.8182 | 0.001126 | 15.48 | 0.009599 |
| NetworkOutbreaks (mean ± SE) | NaN | NaN | 0.8171 | 0.0003 | 15.47 | 0.003598 |

The plain pairwise model (L = 0) misses the take-off by up to D∞(I) =
0.1456. Counting a single infection (L = 1) brings this to 0.002891, and
L = 2, 3, 4 give 0.003824, 0.003921, 0.003925. The lumped L = 1
prevalence differs from the plain pairwise prevalence by up to 0.1464,
which is the effect of closing the triples at the refined level. The
first refinement does almost all the work. It separates the nodes that
have never been infected, whose neighbourhoods are still close to those
of an SIR epidemic, from the nodes that have recovered next to an
infected neighbour. The endemic levels at t = 80 are the same for every
L, 0.8182: the refinement changes the approach to the endemic
equilibrium, not the equilibrium. They are 0.001126 above the simulated
level, which is 3.753 standard errors of the simulated mean: a small
bias of the pairwise closure at equilibrium. Counting infections also
corrects most of the plain pairwise model’s excess in the mean number of
infections per node (the `cumulative(80)` column).

The early phase, where the models differ:

``` julia
p = plot(ref, :I; median = false, legend = :bottomright, xlabel = "t", ylabel = "I (fraction of nodes)",
         title = "SIS on a 3-regular network: take-off")
for c in dets
    plot!(p, c, :I)
end
distinct_styles!(p)
xlims!(p, 0, 30)
```

![The first 30 time units of the figure above (prevalence; band: spread
q2.5–q97.5 of the surviving
runs).](index_files/figure-commonmark/cell-18-output-1.svg)

### A numerical caveat

With the default solver of `solve_epidemic` (an automatic switch between
non-stiff and stiff methods), for L = 4, the prevalence agrees with the
Tsit5 solution but the `:cumulative` accumulator does not. The check
integrates the accumulator’s own right-hand side, τ Σ\_{p,q}\[S_p I_q\],
by the trapezoidal rule on the saved Tsit5 solution:

``` julia
default4 = solve_epidemic(pw[4], sc; initial = seed(4))
isSI(X, Y) = Set(base_compartment_of(refined[4], Z) for Z in (X, Y)) == Set([:S, :I])
SIpairs = [v for ((X, Y), v) in pw[4].pairs if isSI(X, Y)]
cumrate(sol) = [sc.params[:τ] * sum(sol[v][j] for v in SIpairs) for j in eachindex(sol.t)]
r4 = cumrate(sols[4]); dt = step(sc.tgrid)
quadrature = ρ + sum((r4[1:end-1] .+ r4[2:end]) ./ 2) * dt
(; tsit5 = dets[5][:cumulative][end], default_solver = model_curves(pw[4], default4; t = sc.tgrid)[:cumulative][end],
   trapezoid_on_tsit5 = quadrature,
   max_prevalence_gap = maximum(abs.(reinfection_totals(pw[4], default4)[:I] .- dets[5][:I])))
```

    (tsit5 = 15.475551085249329, default_solver = 15.318376250741608, trapezoid_on_tsit5 = 15.475533100027446, max_prevalence_gap = 1.1356475759782825e-9)

The Tsit5 value agrees with the quadrature to within the trapezoidal
error of the 0.25 grid, and the default solver’s value does not. This is
a defect of the default solver path for this system, reported to the
package owners; the page does not depend on it.

## Reinfection histograms

The committed summary stores, for the end of the run (t = 80), the mean
fraction of nodes that have been infected p times, over the surviving
runs:

``` julia
h = ref.extras[:reinfection_histogram]["cond"]
(; t_end = last(sc.tgrid), buckets = length(h), total = sum(h), mode = argmax(h) - 1,
   mean_infections = sum((p - 1) * h[p] for p in eachindex(h)), cumulative_mean = sc_cum.mean[end])
```

    (t_end = 80.0, buckets = 35, total = 1.0, mode = 15, mean_infections = 15.465947000000002, cumulative_mean = 15.465952)

The mean of the histogram is the simulated `:cumulative` at t = 80, as
it should be. By t = 80 the typical node has been infected 15 times:

``` julia
bar(0:length(h)-1, h; xlabel = "number of infections p", ylabel = "fraction of nodes", label = "NetworkOutbreaks",
    title = "Reinfection histogram of :sis_reg3 at t = 80", legend = :topright)
vline!([1, 2, 3, 4]; linestyle = :dash, color = :gray, label = "caps L = 1…4")
```

![Fraction of nodes infected p times by t = 80 (NetworkOutbreaks, mean
over the surviving runs), and where the caps L = 1, …, 4 of the pairwise
models fall.](index_files/figure-commonmark/cell-21-output-1.svg)

A model with cap L puts every node infected L or more times into the top
class, so the right comparison is with the simulated histogram saturated
at L. The model’s fraction infected p times is \[S_p\] + \[I_p\]:

``` julia
model_hist(L) = [sols[L][pw[L].singles[Symbol(:S_, q)]][end] +
                 (haskey(pw[L].singles, Symbol(:I_, q)) ? sols[L][pw[L].singles[Symbol(:I_, q)]][end] : 0.0)
                 for q in 0:L]
saturated(L) = [q < L ? h[q+1] : sum(h[L+1:end]) for q in 0:L]
rows = [(L, q, model_hist(L)[q+1], saturated(L)[q+1]) for L in 1:4 for q in 0:L]
md_table(["L", "p (top class: p ≥ L)", "pairwise fraction", "NetworkOutbreaks fraction"], rows)
```

|   L | p (top class: p ≥ L) | pairwise fraction | NetworkOutbreaks fraction |
|----:|---------------------:|------------------:|--------------------------:|
|   1 |                    0 |         1.365e-23 |                         0 |
|   1 |                    1 |                 1 |                         1 |
|   2 |                    0 |         6.126e-28 |                         0 |
|   2 |                    1 |         2.969e-08 |                         0 |
|   2 |                    2 |                 1 |                         1 |
|   3 |                    0 |         5.238e-29 |                         0 |
|   3 |                    1 |          2.97e-08 |                         0 |
|   3 |                    2 |         6.504e-07 |                         0 |
|   3 |                    3 |                 1 |                         1 |
|   4 |                    0 |          4.31e-29 |                         0 |
|   4 |                    1 |          2.97e-08 |                         0 |
|   4 |                    2 |         6.504e-07 |                         0 |
|   4 |                    3 |         6.894e-06 |                     1e-05 |
|   4 |                    4 |                 1 |                         1 |

With L ≤ 4 this comparison can only test the lowest buckets. Almost
every node has been infected at least four times, and the fraction
infected exactly three times is 6.894e-06 in the L = 4 model and 1e-05
in the simulation. The smallest nonzero fraction the ensemble can
resolve is 1/(N · runs) = 5e-07. The mean of the whole histogram is
compared through `:cumulative` in the table above.

## Summary

``` julia
md_table(["quantity", "value"],
         [("pairwise threshold τ_c (τ/τ_c)", "$(fmt(a.τc)) ($(fmt(sc.params[:τ] / a.τc)))"),
          ("D∞(I), plain pairwise (L = 0)", D[1]),
          ("D∞(I), reinfection counting L = 1 / 4", "$(fmt(D[2])) / $(fmt(D[5]))"),
          ("simulated I(80) ± SE", "$(fmt(st.mean[end])) ± $(fmt(st.se[end]))"),
          ("simulated mean number of infections per node at t = 80", sc_cum.mean[end]),
          ("pairwise L = 0 / L = 4 cumulative(80)", "$(fmt(dets[1][:cumulative][end])) / $(fmt(dets[5][:cumulative][end]))")])
```

| quantity | value |
|----|----|
| pairwise threshold τ_c (τ/τ_c) | 0.125 (4) |
| D∞(I), plain pairwise (L = 0) | 0.1456 |
| D∞(I), reinfection counting L = 1 / 4 | 0.002891 / 0.003925 |
| simulated I(80) ± SE | 0.8171 ± 0.0003 |
| simulated mean number of infections per node at t = 80 | 15.47 |
| pairwise L = 0 / L = 4 cumulative(80) | 15.7 / 15.48 |

No Lean statement is involved on this page. Page
[N12](../N12_sis_comparison/index.md) puts the reinfection-counting
models next to the motif and neighbourhood approximations.

## References

<div id="refs">

</div>
