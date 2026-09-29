# Multitype pairwise models


- [What this page shows](#what-this-page-shows)
- [The model: Catalyst, stratification,
  lift](#the-model-catalyst-stratification-lift)
  - [The equations](#the-equations)
- [The morphism to the multitype edge-based
  model](#the-morphism-to-the-multitype-edge-based-model)
- [Where the constant closure is
  biased](#where-the-constant-closure-is-biased)
- [Against simulation](#against-simulation)
- [Summary](#summary)
- [References](#references)

## What this page shows

A typed configuration network (`MultitypeNetwork`) gives every node a
type a, a fraction n_a of the nodes, and a joint degree law for its
edges to each type. A stratified model has one copy of every compartment
per type. NodeBasedModels lifts it to a population-level multitype
pairwise model with singles \[X\] and ordered pairs \[XY\] for every
pair of types that are joined by edges. Every triple is centred at the
recipient R of a contact and closed as

$$[Y\,R\,J] \approx K_a\big(t(Y), t(J)\big)\,\frac{[YR]\,[RJ]}{[R]}, \qquad a = t(R),$$

where t(X) is the node type of compartment X. With the constant closure,
K_a(c, b) = ∂\_c∂\_bψ_a(1)/(∂\_cψ_a(1)∂\_bψ_a(1)). With `PGFClosure`,
K_a is evaluated at the edge-based θ’s. This page:

1.  builds the models of `:sir_sbm2` and `:sir_age2` from a Catalyst
    network, a stratification and `node_based`;
2.  checks them against the multitype edge-based model of
    EdgeBasedModels, both along solutions and as a morphism (the typed
    π^PW map, checked with `verify`);
3.  shows, on a typed network that is not of Poisson type, that the
    constant closure is biased while `PGFClosure` is not;
4.  compares all three variants with the committed NetworkOutbreaks
    ensembles.

``` julia
include(joinpath(@__DIR__, "..", "_shared", "setup.jl"))
require_summaries([:sir_sbm2, :sir_age2])
```

## The model: Catalyst, stratification, lift

``` julia
using Catalyst
import EdgeBasedModels
sir = @reaction_network sir begin
    @parameters τ γ
    τ, S + I --> 2I        # contact: per-contact (per-edge) rate τ
    γ, I --> R
end
base = contact_model(sir)
sc_age = scenario(:sir_age2)       # sizes 0.4/0.6, mean contacts [6 3; 2 4], between-group rate τ/2
age = strata([:y, :o]; sizes = [0.4, 0.6])
model_age = stratify(base, age; contact_rates = (a, b) -> a == b ? :τ : :(τ / 2))
@assert isequivalent(model_age, sc_age.model)
sc_sbm = scenario(:sir_sbm2)       # sizes 0.5/0.5, mean contacts [6 2; 2 4], one rate τ
model_sbm = stratify(base, strata([:a, :b]; sizes = [0.5, 0.5]))
@assert isequivalent(model_sbm, sc_sbm.model)
sys_age = node_based(model_age, sc_age.network)     # low level: constant closure (BernoulliClosure)
sys_sbm = node_based(model_sbm, sc_sbm.network)
(; age = sys_age, sbm = sys_sbm)
```

    (age = MultitypePairwiseSystem(sir_strat; level = :population, 2 types, 6 singles, 21 pairs, closure = BernoulliClosure()), sbm = MultitypePairwiseSystem(sir_strat; level = :population, 2 types, 6 singles, 21 pairs, closure = BernoulliClosure()))

The stratified model keeps one contact per ordered pair of types, with
the per-contact rate of that pair:

``` julia
model_age
```

    ContactModel :sir_strat  (source: transform; method: explicit; rates: PerContact)
      species       S_y (Sus)   S_o (Sus)   I_y   I_o   R_y   R_o
      contacts      [1] S_y + I_y → I_y + I_y    τ      contact     infector I_y, entry I_y
                    [2] S_y + I_o → I_y + I_o    τ/2    contact     infector I_o, entry I_y
                    [3] S_o + I_y → I_o + I_y    τ/2    contact     infector I_y, entry I_o
                    [4] S_o + I_o → I_o + I_o    τ      contact     infector I_o, entry I_o
      transitions   [5] I_y → R_y                γ      progress
                    [6] I_o → R_o                γ      progress
      typing        T_EB  ⇒  edge_based ✓  s_anchored ✓  pairwise ✓  individual ✓  pair ✓  stochastic ✓  mass_action ✓
      assumptions   Sus inferred as recipients \ contact products = {S}
                    transform of a catalyst model (method stoichiometry)
                    stratify: typed product over T_net with the strata y, o (sizes 0.4, 0.6); contacts (s,a) + (J,b) → (X,a) + (J,b), transitions (X,a) → (Y,a), no transitions between strata; contact_rates = a function, transition_rates = :same

The two networks, their mean-contact matrices M_ab = E\[k\_{a→b}\], and
the reciprocity n_a M_ab = n_b M_ba that makes them consistent:

``` julia
rows = []
for sc in (sc_sbm, sc_age)
    net = sc.network
    M = mean_contacts(net)
    push!(rows, (string("`:", sc.id, "`"), join(net.types, ", "), join(fmt.(net.sizes), ", "),
                 string(round.(M; digits = 4)), maximum(abs.(net.sizes .* M .- (net.sizes .* M)'))))
end
md_table(["scenario", "types", "sizes n_a", "M", "max abs(n_a M_ab − n_b M_ba)"], rows)
```

| scenario | types | sizes n_a | M | max abs(n_a M_ab − n_b M_ba) |
|----|----|----|----|---:|
| `:sir_sbm2` | a, b | 0.5, 0.5 | \[6.0 2.0; 2.0 4.0\] | 0 |
| `:sir_age2` | y, o | 0.4, 0.6 | \[6.0 3.0; 2.0 4.0\] | 2.22e-16 |

Both scenarios have their rate τ calibrated to R₀ = 2 with the multitype
next-generation matrix:

``` julia
anchors(sc_sbm);
anchors(sc_age);
```

    :sir_sbm2: γ = 0.25, τ = 0.0954915; seeds I_a 0.005, I_b 0.005; t = 0:0.25:60; R₀ = 2; no NodeBasedModels pairwise threshold for a MultitypeNetwork (differs from the canonical anchors: τ ≠ 1/6)
    :sir_age2: γ = 0.25, τ = 0.105235; seeds I_y 0.004, I_o 0.006; t = 0:0.25:60; R₀ = 2; no NodeBasedModels pairwise threshold for a MultitypeNetwork (differs from the canonical anchors: τ ≠ 1/6)

### The equations

The model of `:sir_age2` has 6 singles and 21 pairs, plus the
cumulative-incidence accumulator. Here are the equations of the young
susceptibles and of their pairs with infected nodes:

``` julia
so_age = symbolic_ode(sys_age)
for (x, r) in zip(so_age.states, so_age.rhs)
    n = string(x)
    (n in ("S_y(t)", "I_y(t)") || (startswith(n, "S_y") && occursin("I_", n))) && println("d", n, "/dt = ", r)
end
```

    dS_y(t)/dt = -(1//2)*S_yI_o(t)*τ - S_yI_y(t)*τ
    dI_y(t)/dt = -I_y(t)*γ + (1//2)*S_yI_o(t)*τ + S_yI_y(t)*τ
    dS_yI_y(t)/dt = (-S_yI_o(t)*S_yI_y(t)*τ) / (2S_y(t)) + (S_yI_o(t)*S_yS_y(t)*τ) / (2S_y(t)) + (S_yI_y(t)*S_yS_y(t)*τ) / S_y(t) - S_yI_y(t)*γ - (S_yI_y(t) + (S_yI_y(t)^2) / S_y(t))*τ
    dS_yI_o(t)/dt = (S_oI_o(t)*S_yS_o(t)*τ) / S_o(t) + (S_oI_y(t)*S_yS_o(t)*τ) / (2S_o(t)) + (-S_yI_o(t)*S_yI_y(t)*τ) / S_y(t) - S_yI_o(t)*γ - (1//2)*(S_yI_o(t) + (S_yI_o(t)^2) / S_y(t))*τ

On independent Poisson blocks every closure constant is 1:

``` julia
md_table(["recipient type a", "type of Y", "type of J", "K_a(t(Y), t(J))"],
         [(string(a), string(c), string(b), K) for ((a, c, b), K) in sort!(collect(sys_age.metadata[:K]); by = first)])
```

| recipient type a | type of Y | type of J | K_a(t(Y), t(J)) |
|------------------|-----------|-----------|----------------:|
| o                | o         | o         |               1 |
| o                | o         | y         |               1 |
| o                | y         | o         |               1 |
| o                | y         | y         |               1 |
| y                | o         | o         |               1 |
| y                | o         | y         |               1 |
| y                | y         | o         |               1 |
| y                | y         | y         |               1 |

## The morphism to the multitype edge-based model

The multitype edge-based model of Miller and Volz ([Miller and Volz
2013](#ref-millervolz2013struct)) has one θ\_{b→a} per edge class, the
survival factor ξ_a of each type (1 here, as there are no exits), and
per-edge φ’s. Its image under π^PW is

$$[s_a] = N n_a q_a \xi_a \psi_a(\theta_{\cdot\to a}),\quad
[s_a Y] = N n_a q_a \xi_a\, \partial_{t(Y)}\psi_a(\theta_{\cdot\to a})\, \varphi_{Y,a},$$

and \[X\] = N pop_X for the other compartments. On this image \[Y s_a
J\] = K_a(θ)\[Y s_a\]\[s_a J\]/\[s_a\] holds exactly for every joint
PGF. So the S-anchored pairwise model with `PGFClosure` is always the
image of the edge-based model, and so is the model with the constant
closure when every K_a is constant (Poisson-type blocks). As a runtime
check, the map is built from the edge-based coordinates and checked with
`verify`, which tests Dπ·F = G∘π symbolically at random probe points:

``` julia
function m6_multitype_map(eb, nb)
    co = eb.metadata[:coords]
    net = eb.metadata[:network]
    ti = Dict(a => i for (i, a) in enumerate(net.types))
    law = Dict(a => net.degrees[i] for (a, i) in ti)
    n = Dict(a => net.sizes[i] for (a, i) in ti)
    Mc = mean_contacts(net)
    tof = nb.metadata[:types]
    Σ = nb.metadata[:susceptible]
    q(s) = eb.metadata[:q][s].param
    ξ(s) = get(co, Symbol(:ξ_, s), 1)
    x(a) = b -> get(co, Symbol(:θ_, b, :_, a), 1.0)
    mp = Pair{Any,Any}[]
    for ((b, a), v) in nb.metadata[:thetas]               # PGFClosure: the auxiliary θ's are the EB θ's
        push!(mp, v => co[Symbol(:θ_, b, :_, a)])
    end
    for (X, v) in nb.singles
        a = tof[X]
        push!(mp, v => (X in Σ ? n[a] * q(X) * ξ(X) * pgf(law[a], x(a)) : co[Symbol(:pop_, X)]))
    end
    for ((X, Y), v) in nb.pairs
        s, Z = X in Σ ? (X, Y) : (Y, X)
        a, c = tof[s], tof[Z]
        edge = Z in Σ ? q(Z) * ξ(Z) * pgf_derivative(law[c], x(c), a) / Mc[ti[c], ti[a]] :
               co[Symbol(:φ_, Z, :_, a)]
        push!(mp, v => n[a] * q(s) * ξ(s) * pgf_derivative(law[a], x(a), c) * edge)
    end
    return mp
end
eb_age = EdgeBasedModels.edge_based(model_age, sc_age.network)
m6 = map((BernoulliClosure(), PGFClosure())) do closure
    nbs = node_based(model_age, sc_age.network; closure, level = :s_anchored)
    verify(Semiconjugacy(:eb_to_mt_pws, symbolic_ode(eb_age), symbolic_ode(nbs), m6_multitype_map(eb_age, nbs))).ok
end
# a wrong map: [S_y S_y] without the factor 1/M_yy
nbs = node_based(model_age, sc_age.network; closure = PGFClosure(), level = :s_anchored)
bad = [k => (isequal(k, nbs.pairs[(:S_y, :S_y)]) ? 6 * v : v) for (k, v) in m6_multitype_map(eb_age, nbs)]
m6_bad = verify(Semiconjugacy(:perturbed, symbolic_ode(eb_age), symbolic_ode(nbs), bad)).ok
(; constant_closure = m6[1], pgf_closure = m6[2], perturbed_map = m6_bad)
```

    (constant_closure = true, pgf_closure = true, perturbed_map = false)

The morphism verifies for both closures, and a map with one wrong factor
fails, so the check is not vacuous. Along solutions, the full multitype
pairwise model (all pairs, not only the S-anchored ones) and the
edge-based model agree on every observable they share, to the
integration tolerance:

``` julia
TOL = (reltol = 1e-10, abstol = 1e-12)
gaps = map([sc_sbm, sc_age]) do sc
    nb = node_based(sc)
    eb = EdgeBasedModels.edge_based(sc)
    cn = model_curves(nb, solve_epidemic(nb, sc; TOL...); t = sc.tgrid)
    ce = model_curves(eb, solve_epidemic(eb, sc; TOL...); t = sc.tgrid)
    common = [X for X in keys(ce.values) if haskey(cn.values, X)]
    (string("`:", sc.id, "`"), length(common), maximum(maximum(abs.(cn[X] .- ce[X])) for X in common))
end
md_table(["scenario", "shared observables", "max abs(pairwise − edge-based)"], gaps)
```

| scenario    | shared observables | max abs(pairwise − edge-based) |
|-------------|-------------------:|-------------------------------:|
| `:sir_sbm2` |                  8 |                      2.238e-13 |
| `:sir_age2` |                  8 |                      1.731e-13 |

## Where the constant closure is biased

The constant closure is exact only when every K_a(c, b)(θ) is constant
in θ. A typed network with bimodal degrees {2, 10} within type a,
Poisson(1) edges across the types and a 3-regular graph within type b is
not of that kind: K_a(a, a)(1) = 3/2 and it varies with θ. The scenario
below is derived from `:sir_sbm2` with τ recalibrated to R₀ = 2. It is
not registered and has no reference ensemble, so here the edge-based
model, which is exact in the large-N limit for any joint degree law, is
the reference:

``` julia
bim = EmpiricalDegree(2 => 5 / 6, 10 => 1 / 6)
nonpt_net = MultitypeNetwork([:a, :b], [0.5, 0.5],
                             [IndependentDegrees(:a => bim, :b => PoissonDegree(1.0)),
                              IndependentDegrees(:a => PoissonDegree(1.0), :b => RegularDegree(3))])
τ_np = round(calibrate(sc_sbm.model, nonpt_net, Dict(:τ => 0.1, :γ => 0.25); target = :R0 => 2.0,
                       vary = :τ)[:τ]; sigdigits = 10)
sc_np = derive(sc_sbm; id = :sir_nonpt2, network = nonpt_net, params = Dict(:τ => τ_np), tspan = (0.0, 80.0),
               backends = Dict(:edge_based => :exact_limit, :pgf_closure => :exact_limit,
                               :pairwise_multitype => :biased),
               tags = [:sir, :stratified, :multitype, :derived])
eb_np = EdgeBasedModels.edge_based(sc_np)
ref_np = model_curves(eb_np, solve_epidemic(eb_np, sc_np; TOL...); t = sc_np.tgrid, label = "edge-based")
np_rows = map(((BernoulliClosure(), :population, "constant K"), (PGFClosure(), :population, "PGF closure"),
               (PGFClosure(), :s_anchored, "S-anchored, PGF closure"))) do (closure, level, label)
    nb = node_based(sc_np; closure, level)
    c = model_curves(nb, solve_epidemic(nb, sc_np; TOL...); t = sc_np.tgrid, label = label)
    (label, maximum(abs.(c[:infectious] .- ref_np[:infectious])), c[:cumulative][end] - ref_np[:cumulative][end])
end
K_np = node_based(sc_np).metadata[:K]
(; τ = τ_np, K_a_aa = K_np[(:a, :a, :a)], K_b_bb = K_np[(:b, :b, :b)])
```

    (τ = 0.1538455062, K_a_aa = 1.5000000000000004, K_b_bb = 0.6666666666666666)

``` julia
md_table(["pairwise model", "max abs(infectious − EB)", "final size − EB"], np_rows)
```

| pairwise model          | max abs(infectious − EB) | final size − EB |
|-------------------------|-------------------------:|----------------:|
| constant K              |                 0.004399 |        0.005664 |
| PGF closure             |                3.689e-14 |       3.408e-14 |
| S-anchored, PGF closure |                1.955e-14 |       4.985e-14 |

The constant closure overestimates the final size by 0.005664 of the
population, while the PGF closure agrees with the edge-based model to
4.985e-14. This is the typed version of the bias of constant-K pairwise
models on non-Poisson networks (page N02).

## Against simulation

On the Poisson blocks of `:sir_sbm2` and `:sir_age2` the three variants
are the same model, and they are all exact in the large-N limit (the
scenarios declare `:pairwise_multitype => :exact_limit`):

``` julia
variants = ((BernoulliClosure(), :population, "multitype pairwise (constant K)"),
            (PGFClosure(), :population, "multitype pairwise (PGF closure)"),
            (PGFClosure(), :s_anchored, "S-anchored (PGF closure)"))
function curves_for(sc)
    return [model_curves(nb, solve_epidemic(nb, sc); t = sc.tgrid, label = label)
            for (nb, label) in ((node_based(sc; closure, level), label) for (closure, level, label) in variants)]
end
ref_sbm, ref_age = scenario_summary(sc_sbm), scenario_summary(sc_age)
det_sbm, det_age = curves_for(sc_sbm), curves_for(sc_age)
(; sbm = sc_sbm.backends[:pairwise_multitype], age = sc_age.backends[:pairwise_multitype])
```

    (sbm = :exact_limit, age = :exact_limit)

``` julia
describe_reference(ref_sbm);
describe_reference(ref_age);
```

    NetworkOutbreaks reference :sir_sbm2 (hash f8e96436): N = 10000 nodes, 200 runs on a fresh graph per run, algorithm :next_reaction; conditioning: major outbreaks only (cumulative incidence excluding seeds ≥ 0.05·N by t_end); 200 of 200 runs kept (major runs), P(major) = 1.000 (95% CI 0.981–1.000); no time alignment.
    NetworkOutbreaks reference :sir_age2 (hash 7ccd6a9b): N = 10000 nodes, 200 runs on a fresh graph per run, algorithm :next_reaction; conditioning: major outbreaks only (cumulative incidence excluding seeds ≥ 0.05·N by t_end); 200 of 200 runs kept (major runs), P(major) = 1.000 (95% CI 0.981–1.000); no time alignment.

``` julia
distinct_styles!(comparisonplot(ref_sbm, det_sbm...; observables = [:I_a, :I_b, :cumulative],
               left_margin = 6Plots.mm, bottom_margin = 6Plots.mm))
```

![`:sir_sbm2`: multitype pairwise models against the committed
NetworkOutbreaks ensemble, by type and in total (spread band q2.5–q97.5
of the major runs; residual panel: mean band ±1.96 SE). The three
variants coincide.](index_files/figure-commonmark/cell-15-output-1.svg)

``` julia
compare(ref_sbm, det_sbm; observables = [:I_a, :I_b, :infectious, :cumulative])
```

    ComparisonTable :sir_sbm2  (scenario f8e96436; conditioned mean of 200 runs)
      curve                             observable        D∞     t(D∞)       SE∞        z∞       ΔR∞  95% CI                 Δpeak   Δt_peak  coverage
      multitype pairwise (constant K)   I_a          0.00113     16.50   0.00060      3.31   0.00112  [-0.00008,  0.00232]   0.00078      0.25     0.905
      multitype pairwise (constant K)   I_b          0.00070     15.00   0.00048      2.38   0.00112  [-0.00008,  0.00232]   0.00033      0.00     0.971
      multitype pairwise (constant K)   infectious   0.00172     15.00   0.00100      3.30   0.00112  [-0.00008,  0.00232]   0.00112      0.25     0.917
      multitype pairwise (constant K)   cumulative   0.00147     19.25   0.00223      2.21   0.00112  [-0.00008,  0.00232]   0.00112      0.00     0.846
      multitype pairwise (PGF closure)  I_a          0.00113     16.50   0.00060      3.31   0.00112  [-0.00008,  0.00232]   0.00078      0.25     0.905
      multitype pairwise (PGF closure)  I_b          0.00070     15.00   0.00048      2.38   0.00112  [-0.00008,  0.00232]   0.00033      0.00     0.971
      multitype pairwise (PGF closure)  infectious   0.00172     15.00   0.00100      3.30   0.00112  [-0.00008,  0.00232]   0.00112      0.25     0.917
      multitype pairwise (PGF closure)  cumulative   0.00147     19.25   0.00223      2.21   0.00112  [-0.00008,  0.00232]   0.00112      0.00     0.846
      S-anchored (PGF closure)          I_a          0.00113     16.50   0.00060      3.31   0.00112  [-0.00008,  0.00232]   0.00078      0.25     0.905
      S-anchored (PGF closure)          I_b          0.00070     15.00   0.00048      2.38   0.00112  [-0.00008,  0.00232]   0.00033      0.00     0.971
      S-anchored (PGF closure)          infectious   0.00172     15.00   0.00100      3.30   0.00112  [-0.00008,  0.00232]   0.00112      0.25     0.917
      S-anchored (PGF closure)          cumulative   0.00147     19.25   0.00223      2.21   0.00112  [-0.00008,  0.00232]   0.00112      0.00     0.846

``` julia
distinct_styles!(comparisonplot(ref_age, det_age...; observables = [:I_y, :I_o, :cumulative],
               left_margin = 6Plots.mm, bottom_margin = 6Plots.mm))
```

![`:sir_age2`: as above, for the age-stratified model (young y, old
o).](index_files/figure-commonmark/cell-17-output-1.svg)

``` julia
tab_age = compare(ref_age, det_age; observables = [:I_y, :I_o, :infectious, :cumulative])
```

    ComparisonTable :sir_age2  (scenario 7ccd6a9b; conditioned mean of 200 runs)
      curve                             observable        D∞     t(D∞)       SE∞        z∞       ΔR∞  95% CI                 Δpeak   Δt_peak  coverage
      multitype pairwise (constant K)   I_y          0.00120     12.00   0.00054      3.46   0.00049  [-0.00081,  0.00179]   0.00111      0.00     0.925
      multitype pairwise (constant K)   I_o          0.00089     15.00   0.00049      2.76   0.00049  [-0.00081,  0.00179]   0.00080      0.25     0.867
      multitype pairwise (constant K)   infectious   0.00199     13.25   0.00093      3.12   0.00049  [-0.00081,  0.00179]   0.00165     -0.25     0.793
      multitype pairwise (constant K)   cumulative   0.00353     14.00   0.00207      2.04   0.00049  [-0.00081,  0.00179]   0.00049      0.00     0.954
      multitype pairwise (PGF closure)  I_y          0.00120     12.00   0.00054      3.46   0.00049  [-0.00081,  0.00179]   0.00111      0.00     0.925
      multitype pairwise (PGF closure)  I_o          0.00089     15.00   0.00049      2.76   0.00049  [-0.00081,  0.00179]   0.00080      0.25     0.867
      multitype pairwise (PGF closure)  infectious   0.00199     13.25   0.00093      3.12   0.00049  [-0.00081,  0.00179]   0.00165     -0.25     0.793
      multitype pairwise (PGF closure)  cumulative   0.00353     14.00   0.00207      2.04   0.00049  [-0.00081,  0.00179]   0.00049      0.00     0.954
      S-anchored (PGF closure)          I_y          0.00120     12.00   0.00054      3.46   0.00049  [-0.00081,  0.00179]   0.00111      0.00     0.925
      S-anchored (PGF closure)          I_o          0.00089     15.00   0.00049      2.76   0.00049  [-0.00081,  0.00179]   0.00080      0.25     0.867
      S-anchored (PGF closure)          infectious   0.00199     13.25   0.00093      3.12   0.00049  [-0.00081,  0.00179]   0.00165     -0.25     0.793
      S-anchored (PGF closure)          cumulative   0.00353     14.00   0.00207      2.04   0.00049  [-0.00081,  0.00179]   0.00049      0.00     0.954

``` julia
acc = map([(sc_sbm, ref_sbm, det_sbm), (sc_age, ref_age, det_age)]) do (sc, ref, dets)
    tab = compare(ref, dets; observables = [:infectious])
    r = tab[dets[1].label, :infectious]
    (string("`:", sc.id, "`"), r.D∞, r.z∞, r.ΔR∞, "[$(fmt(r.ΔR∞_ci[1])), $(fmt(r.ΔR∞_ci[2]))]",
     r.D∞ < 0.005 && abs(r.ΔR∞) < 0.005 ? "yes" : "**no**")
end
md_table(["scenario", "D∞(infectious)", "z∞", "ΔR∞", "95% CI of ΔR∞", "D∞ < 0.005 and abs(ΔR∞) < 0.005"], acc)
```

| scenario | D∞(infectious) | z∞ | ΔR∞ | 95% CI of ΔR∞ | D∞ \< 0.005 and abs(ΔR∞) \< 0.005 |
|----|---:|---:|---:|----|----|
| `:sir_sbm2` | 0.001717 | 3.296 | 0.001119 | \[-7.989e-05, 0.002317\] | yes |
| `:sir_age2` | 0.001985 | 3.122 | 0.0004932 | \[-0.0008071, 0.001794\] | yes |

``` julia
ci0 = [r[5] for r in acc]
contains0(t) = (v = parse.(Float64, split(strip(t, ['[', ']']), ", ")); v[1] <= 0 <= v[2])
println("The exact-limit rule (D∞(infectious) < 0.005 and |ΔR∞| < 0.005 at N = 10⁴ and 200 runs) holds on ",
        count(r -> r[end] == "yes", acc), " of the 2 scenarios. The largest gaps are ", fmt(acc[1][2]), " and ",
        fmt(acc[2][2]), ", and the 95% interval of ΔR∞ contains 0 on ", count(contains0, ci0), " of them.")
```

The exact-limit rule (D∞(infectious) \< 0.005 and \|ΔR∞\| \< 0.005 at N
= 10⁴ and 200 runs) holds on 2 of the 2 scenarios. The largest gaps are
0.001717 and 0.001985, and the 95% interval of ΔR∞ contains 0 on 2 of
them.

## Summary

``` julia
md_table(["statement", "holds here", "checked by"],
         [("Catalyst → stratify → node_based gives the scenario models", true, "`isequivalent` (asserted above)"),
          ("typed π^PW: EB → PW^S (constant K, Poisson blocks)", m6[1], "`verify`"),
          ("typed π^PW: EB → PW^S (PGF closure)", m6[2], "`verify`"),
          ("a perturbed map is rejected", !m6_bad, "`verify`"),
          ("pairwise = EB along solutions on `:sir_sbm2` and `:sir_age2`", all(r -> r[3] < 1e-7, gaps), "max gap < 10⁻⁷"),
          ("constant K biased on a non-Poisson typed network", abs(np_rows[1][3]) > 1e-3, "final size vs EB (gap > 10⁻³)"),
          ("PGF closure = EB on the same network", abs(np_rows[2][3]) < 1e-6, "final size vs EB (gap < 10⁻⁶)"),
          ("exact-limit rule against NetworkOutbreaks", all(r -> r[end] == "yes", acc), "`compare`")])
```

| statement | holds here | checked by |
|----|---:|----|
| Catalyst → stratify → node_based gives the scenario models | true | `isequivalent` (asserted above) |
| typed π^PW: EB → PW^S (constant K, Poisson blocks) | true | `verify` |
| typed π^PW: EB → PW^S (PGF closure) | true | `verify` |
| a perturbed map is rejected | true | `verify` |
| pairwise = EB along solutions on `:sir_sbm2` and `:sir_age2` | true | max gap \< 10⁻⁷ |
| constant K biased on a non-Poisson typed network | true | final size vs EB (gap \> 10⁻³) |
| PGF closure = EB on the same network | true | final size vs EB (gap \< 10⁻⁶) |
| exact-limit rule against NetworkOutbreaks | true | `compare` |

No Lean statement is cited on this page. The typed π^PW map is checked
at run time by `verify`, not in Lean.

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-millervolz2013struct" class="csl-entry">

Miller, Joel C., and Erik M. Volz. 2013. “Incorporating Disease and
Population Structure into Models of SIR Disease in Contact Networks.”
*PLoS ONE* 8 (8): e69162.
<https://doi.org/10.1371/journal.pone.0069162>.

</div>

</div>
