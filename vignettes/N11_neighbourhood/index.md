# Neighbourhood model (n = 2)


- [What this page shows](#what-this-page-shows)
- [The model, twice](#the-model-twice)
- [The equations](#the-equations)
- [Conservation laws](#conservation-laws)
- [Against simulation](#against-simulation)
- [References](#references)

## What this page shows

The neighbourhood model \[Keeling, House, Cooper and Pellis 2016, *PLoS
Comput. Biol.* 12:e1005296, approximation 3; compare the neighbourhood
configuration model of Hadjichrysanthou et al.
([2012](#ref-hadjichrysanthou2012))\] classifies every node of a
k-regular network by its own state and the number y of its k neighbours
that are infected: \[S_y\] and \[I_y\], y = 0, …, k. A node’s own events
are exact in these variables. The events of its neighbours need a
closure, which at order n = 2 uses the states of pairs. This page:

1.  builds the model from the Catalyst SIS model of `:sis_reg3` and with
    the direct constructor, and checks that the two are the same system;
2.  writes out the equations and checks the numeric right-hand side
    against the independent symbolic builder;
3.  checks the conservation laws along a solution;
4.  compares the prevalence with the committed NetworkOutbreaks ensemble
    and with the pairwise model.

``` julia
include(joinpath(@__DIR__, "..", "_shared", "setup.jl"))
require_summaries([:sis_reg3])
```

## The model, twice

``` julia
using Catalyst, StableRNGs
sis = @reaction_network sis begin
    @parameters τ γ
    τ, S + I --> 2I        # contact: per-contact (per-edge) rate τ
    γ, I --> S             # recovery back into the susceptible class
end
model = contact_model(sis)
sc = scenario(:sis_reg3)            # 3-regular, τ = 1/2, γ = 1/4, 1% seeds in I, t ∈ [0, 80]
@assert isequivalent(model, sc.model)
ρ = last(only(sc.initial.fractions))
k = 3
nb  = node_based(model, sc.network; level = :neighbourhood, n = 2, p = sc.params,
                 initial = sc.initial, tspan = sc.tspan)                                      # low level
nbD = generate_neighbourhood(CompartmentalModel(model), k, 2; β = sc.params[:τ], γ = sc.params[:γ],
                             tspan = sc.tspan, ε = ρ)                                         # direct constructor
function max_rhs_gap(a, rhs_b; n = 25, seed = 1)
    rng = StableRNG(seed)
    d1, d2 = similar(a.u0), similar(a.u0)
    gap = 0.0
    for _ in 1:n
        u = a.u0 .* (0.2 .+ 1.6 .* rand(rng, length(a.u0))) .+ 1e-3 .* rand(rng, length(a.u0))
        a.rhs!(d1, u, a.params, 0.0)
        rhs_b(d2, u)
        gap = max(gap, maximum(abs.(d1 .- d2)))
    end
    return gap
end
(; variables = nb.var_names, u0_equal = nb.u0 == nbD.u0,
   max_rhs_gap = max_rhs_gap(nb, (du, u) -> nbD.rhs!(du, u, nbD.params, 0.0)))
```

    (variables = [:S_0, :S_1, :S_2, :S_3, :I_0, :I_1, :I_2, :I_3], u0_equal = true, max_rhs_gap = 0.0)

The initial condition is random mixing: a fraction ρ = 0.01 of the nodes
is infected independently, so the number of infected neighbours is
Binomial(k, ρ) for every node.

## The equations

With per-contact rate τ and recovery rate γ, a node in \[S_y\] is
infected at rate τy and a node in \[I_y\] recovers at rate γ. These
events are the node’s own and need no closure. A neighbour of the node
recovers at rate γ, which moves the node from y to y − 1. A susceptible
neighbour of the node is infected at a rate that depends on that
neighbour’s other neighbours. The n = 2 closure replaces it by its
average over the pairs of the same type:

$$\omega_S = \tau\,\frac{\sum_y y\,(k-y)\,[S_y]}{\sum_y (k-y)\,[S_y]}, \qquad
\omega_I = \tau\,\frac{\sum_y y^2\,[S_y]}{\sum_y y\,[S_y]}.$$

Here ω_S is for a susceptible neighbour of an S node and ω_I for a
susceptible neighbour of an I node. The model is

$$\begin{aligned}
\frac{d[S_y]}{dt} &= \gamma [I_y] - \tau y [S_y] + \gamma\big((y+1)[S_{y+1}] - y[S_y]\big)
  + \omega_S\big((k-y+1)[S_{y-1}] - (k-y)[S_y]\big),\\
\frac{d[I_y]}{dt} &= \tau y [S_y] - \gamma [I_y] + \gamma\big((y+1)[I_{y+1}] - y[I_y]\big)
  + \omega_I\big((k-y+1)[I_{y-1}] - (k-y)[I_y]\big),
\end{aligned}$$

with \[S\_{−1}\] = \[S\_{k+1}\] = 0 and the same for I.
`build_neighbourhood_symbolic_rhs(k)` writes these equations with
Symbolics from the paper’s equations, without calling the numeric
builder. For k = 3, with the variables renamed:

``` julia
rhs_sym!, keys_sym, exprs = build_neighbourhood_symbolic_rhs(k)
@assert [Symbol(X, :_, y) for (X, y) in keys_sym] == nb.var_names
pretty = Dict{Any,Any}(Symbolics.variable(Symbol("u_", i)) => Symbolics.variable(nb.var_names[i]) for i in eachindex(nb.var_names))
pretty[Symbolics.variable(:β_sym)] = Symbolics.variable(:τ)       # the builder names the per-contact rate β
pretty[Symbolics.variable(:γ_sym)] = Symbolics.variable(:γ)
for (i, e) in enumerate(exprs)
    println("d", nb.var_names[i], "/dt = ", Symbolics.substitute(e, pretty))
end
```

    dS_0/dt = I_0*γ + S_1*γ - 3S_0*ifelse((3S_0 + 2S_1 + S_2) < 1.0e-12, 0, (2S_1 + 2S_2) / (3S_0 + 2S_1 + S_2))*τ
    dS_1/dt = I_1*γ + (-S_1 + 2S_2)*γ - S_1*τ + (3S_0 - 2S_1)*ifelse((3S_0 + 2S_1 + S_2) < 1.0e-12, 0, (2S_1 + 2S_2) / (3S_0 + 2S_1 + S_2))*τ
    dS_2/dt = I_2*γ + (-2S_2 + 3S_3)*γ - 2S_2*τ + (2S_1 - S_2)*ifelse((3S_0 + 2S_1 + S_2) < 1.0e-12, 0, (2S_1 + 2S_2) / (3S_0 + 2S_1 + S_2))*τ
    dS_3/dt = I_3*γ - 3S_3*γ - 3S_3*τ + S_2*ifelse((3S_0 + 2S_1 + S_2) < 1.0e-12, 0, (2S_1 + 2S_2) / (3S_0 + 2S_1 + S_2))*τ
    dI_0/dt = -I_0*γ + I_1*γ - 3I_0*ifelse((S_1 + 2S_2 + 3S_3) < 1.0e-12, 0, (S_1 + 4S_2 + 9S_3) / (S_1 + 2S_2 + 3S_3))*τ
    dI_1/dt = (-I_1 + 2I_2)*γ - I_1*γ + S_1*τ + (3I_0 - 2I_1)*ifelse((S_1 + 2S_2 + 3S_3) < 1.0e-12, 0, (S_1 + 4S_2 + 9S_3) / (S_1 + 2S_2 + 3S_3))*τ
    dI_2/dt = (-2I_2 + 3I_3)*γ - I_2*γ + 2S_2*τ + (2I_1 - I_2)*ifelse((S_1 + 2S_2 + 3S_3) < 1.0e-12, 0, (S_1 + 4S_2 + 9S_3) / (S_1 + 2S_2 + 3S_3))*τ
    dI_3/dt = -4I_3*γ + 3S_3*τ + I_2*ifelse((S_1 + 2S_2 + 3S_3) < 1.0e-12, 0, (S_1 + 4S_2 + 9S_3) / (S_1 + 2S_2 + 3S_3))*τ

The `ifelse` guards the ratios where their denominators vanish. The
numeric right-hand side agrees with the symbolic one:

``` julia
(; max_gap_numeric_vs_symbolic = max_rhs_gap(nb, (du, u) -> rhs_sym!(du, u, (sc.params[:τ], sc.params[:γ]), 0.0)))
```

    (max_gap_numeric_vs_symbolic = 6.938893903907228e-18,)

## Conservation laws

The node count is conserved, Σ_y(\[S_y\] + \[I_y\]) = 1. Every SI edge
is counted once from its S end and once from its I end, so Σ_y y\[S_y\]
= Σ_y (k − y)\[I_y\] should hold as well. The closure must preserve
this. Along the solution on the scenario grid:

``` julia
sol = solve_neighbourhood(nb; saveat = sc.tgrid)
S(y) = [u[nb.index[(:S, y)]] for u in sol.u]
I(y) = [u[nb.index[(:I, y)]] for u in sol.u]
nodes  = sum(S(y) .+ I(y) for y in 0:k)
SI_S   = sum(y .* S(y) for y in 0:k)
SI_I   = sum((k - y) .* I(y) for y in 0:k)
(; max_node_defect = maximum(abs.(nodes .- 1)), max_SI_defect = maximum(abs.(SI_S .- SI_I)))
```

    (max_node_defect = 1.1102230246251565e-15, max_SI_defect = 4.4797499043625066e-14)

## Against simulation

``` julia
cnb = ModelCurves(sc.tgrid, Dict(:I => neighbourhood_compartment(nb, sol, :I), :S => neighbourhood_compartment(nb, sol, :S));
                  label = "neighbourhood n = 2", representation = :neighbourhood)
pw  = node_based(model, sc.network)                  # population pairwise, Bernoulli closure K = (k − 1)/k
cpw = model_curves(pw, solve_epidemic(pw, sc); t = sc.tgrid, label = "pairwise")
ref = scenario_summary(sc)
tab = compare(ref, cnb, cpw; observables = [:I])
```

    ComparisonTable :sis_reg3  (scenario 485d890e; conditioned mean of 200 runs)
      curve                observable        D∞     t(D∞)       SE∞        z∞       ΔR∞  95% CI                 Δpeak   Δt_peak  coverage
      neighbourhood n = 2  I            0.02525     10.75   0.00160     21.41       NaN  [     NaN,      NaN]  -0.00106     42.25     0.769
      pairwise             I            0.14556      9.50   0.00160     93.69  14.70400  [14.70400, 14.70400]  -0.00023      2.00     0.153

``` julia
describe_reference(ref);
```

    NetworkOutbreaks reference :sis_reg3 (hash 485d890e): N = 10000 nodes, 200 runs on a fresh graph per run, algorithm :next_reaction; conditioning: surviving runs only (prevalence at t_end > 0); 200 of 200 runs kept (surviving runs), P(survival) = 1.000 (95% CI 0.981–1.000); no time alignment.

``` julia
distinct_styles!(comparisonplot(ref, cnb, cpw; observables = [:I]))
```

![Neighbourhood model (n = 2) and pairwise model against the committed
NetworkOutbreaks ensemble of `:sis_reg3`, conditioned on survival
(spread band q2.5–q97.5 of the runs; residual panel: mean band ±1.96
SE).](index_files/figure-commonmark/cell-9-output-1.svg)

The neighbourhood model does not track cumulative incidence (ΔR∞ is
NaN). The pairwise row’s ΔR∞ is its `:cumulative` at t = 80, which
counts reinfections, minus the simulated fraction of nodes ever
infected, so it is not a final-size error (page N09 explains this):

``` julia
fs = ref.final_size[ref.major]
(; pairwise_cumulative_80 = cpw[:cumulative][end], mean_fraction_ever_infected = sum(fs) / length(fs),
   ΔR∞_pairwise = tab[cpw.label, :I].ΔR∞)
```

    (pairwise_cumulative_80 = 15.704003220388449, mean_fraction_ever_infected = 1.0, ΔR∞_pairwise = 14.704003220388449)

The peak columns are not meaningful on an endemic plateau either. The
largest gap and the endemic level:

``` julia
st = ref.cond[:I]
md_table(["model", "variables", "D∞(I)", "t at D∞", "z∞", "I(80)", "I(80) − NO"],
         [(cnb.label, length(nb.var_names), tab[cnb.label, :I].D∞, tab[cnb.label, :I].t_D∞, tab[cnb.label, :I].z∞,
           cnb[:I][end], cnb[:I][end] - st.mean[end]),
          (cpw.label, length(pw.singles) + length(pw.pairs), tab[cpw.label, :I].D∞, tab[cpw.label, :I].t_D∞,
           tab[cpw.label, :I].z∞, cpw[:I][end], cpw[:I][end] - st.mean[end]),
          ("NetworkOutbreaks (mean ± SE)", "–", NaN, NaN, NaN, st.mean[end], st.se[end])])
```

| model                        | variables |   D∞(I) | t at D∞ |    z∞ |  I(80) | I(80) − NO |
|------------------------------|----------:|--------:|--------:|------:|-------:|-----------:|
| neighbourhood n = 2          |         8 | 0.02525 |   10.75 | 21.41 | 0.8173 |  0.0002934 |
| pairwise                     |         5 |  0.1456 |     9.5 | 93.69 | 0.8182 |   0.001126 |
| NetworkOutbreaks (mean ± SE) |         – |     NaN |     NaN |   NaN | 0.8171 |     0.0003 |

With 8 variables the neighbourhood model reduces the largest prevalence
gap of the pairwise model from 0.1456 to 0.02525. Both approach the
simulated endemic level: the endemic gaps are 0.0002934 and 0.001126,
against a standard error of 0.0003. The model’s own output also includes
the distribution of the number of infected neighbours. For example, at t
= 80 a susceptible node has on average 2.237 infected neighbours:

``` julia
bar(0:k, [S(y)[end] for y in 0:k]; bar_width = 0.35, label = "S_y", xlabel = "infected neighbours y",
    ylabel = "fraction of nodes", title = "Neighbourhood model at t = 80")
bar!((0:k) .+ 0.35, [I(y)[end] for y in 0:k]; bar_width = 0.35, label = "I_y")
```

![The neighbourhood distribution at t = 80: the fraction of all nodes in
\[S_y\] and
\[I_y\].](index_files/figure-commonmark/cell-13-output-1.svg)

The neighbourhood distribution is a prediction of the model only. The
committed summary does not store it, so it is not compared with the
simulation here.

No Lean statement is involved on this page. Page
[N12](../N12_sis_comparison/index.md) compares every SIS approximation.

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-hadjichrysanthou2012" class="csl-entry">

Hadjichrysanthou, Christoforos, Mark Broom, and István Z. Kiss. 2012.
“Approximating Evolutionary Dynamics on Networks Using a Neighbourhood
Configuration Model.” *Journal of Theoretical Biology* 312: 13–21.
<https://doi.org/10.1016/j.jtbi.2012.07.015>.

</div>

</div>
