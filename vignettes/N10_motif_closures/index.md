# Motif closures


- [What this page shows](#what-this-page-shows)
- [The motif systems, twice](#the-motif-systems-twice)
- [The variables](#the-variables)
- [m = 2 is the pairwise model](#m--2-is-the-pairwise-model)
- [Against simulation](#against-simulation)
- [References](#references)

## What this page shows

The pairwise model tracks nodes and edges and closes triples. A motif
closure of order m \[Keeling, House, Cooper and Pellis 2016, *PLoS
Comput. Biol.* 12:e1005296, approximation 2; see also House and Keeling
([2011](#ref-house2011))\] tracks every connected induced subgraph with
up to m vertices, by the states of its vertices, and closes the (m +
1)-vertex quantities. On a k-regular network with k = 3 the package
implements m = 2 (Keeling’s pairwise model), m = 3 (paths and triangles)
and m = 4 (the six connected four-vertex shapes). This page:

1.  builds the motif systems from the Catalyst SIS model of `:sis_reg3`
    and from the direct constructor `motif_based_sis`, and checks that
    they are the same system;
2.  lists the variables of each order and checks the numeric right-hand
    sides against the independent symbolic builder;
3.  checks that m = 2 is the pairwise model;
4.  compares m = 2, 3, 4 with the committed NetworkOutbreaks ensemble.

``` julia
include(joinpath(@__DIR__, "..", "_shared", "setup.jl"))
require_summaries([:sis_reg3])
```

## The motif systems, twice

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
ms = 2:4
low = Dict(m => node_based(model, sc.network; level = :motif, closure = MotifClosure(3, m),
                           p = sc.params, initial = sc.initial, tspan = sc.tspan) for m in ms)    # low level
direct = Dict(m => motif_based_sis(; β = sc.params[:τ], γ = sc.params[:γ], k = 3, m, tspan = sc.tspan, ε = ρ)
              for m in ms)                                                                   # direct constructor
# a summary of the m = 3 system: its shapes and its variables, written shape[states]
(; m = 3, shapes = [s.name for s in low[3].shapes], n_variables = length(low[3].variables),
   variables = join((string(v.shape.name, "[", join(v.state, " "), "]") for v in low[3].variables), ", "))
```

    (m = 3, shapes = [:singleton, :P2, :P3, :C3], n_variables = 15, variables = "singleton[I], singleton[S], P2[I I], P2[I S], P2[S S], P3[I I I], P3[I I S], P3[I S I], P3[I S S], P3[S I S], P3[S S S], C3[I I I], C3[I I S], C3[I S S], C3[S S S]")

`motif_based_sis` names the per-contact rate β, the keyword name it had
in version 0.1; it is the τ of the rest of the package. A motif system
is a numeric ODE (`rhs!`, `u0`), not a ModelingToolkit system, so the
two forms are compared by their initial conditions and by their
right-hand sides at random states:

``` julia
function max_rhs_gap(a, b, rhs_b = (du, u) -> b.rhs!(du, u, b.params, 0.0); n = 25, seed = 1)
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
md_table(["m", "variables", "u0 equal", "max rhs gap at 25 random states"],
         [(m, length(low[m].variables), low[m].u0 == direct[m].u0, max_rhs_gap(low[m], direct[m])) for m in ms])
```

|   m | variables | u0 equal | max rhs gap at 25 random states |
|----:|----------:|---------:|--------------------------------:|
|   2 |         5 |     true |                               0 |
|   3 |        15 |     true |                               0 |
|   4 |        65 |     true |                               0 |

## The variables

A variable counts the induced copies of a shape H whose vertices are in
a given state, up to the automorphisms of H, as unordered embeddings.
For the pair shape P₂ this gives E_SS, E_IS and E_II. The Keeling
pairwise counts are \[SS\] = 2E_SS, \[SI\] = E_IS and \[II\] = 2E_II. On
a large random 3-regular graph the default host counts are those of a
tree, with no triangles or other cycles. Per node they are:

``` julia
k = 3
(; edges = k / 2, P3 = binomial(k, 2), P4 = (k / 2) * (k - 1)^2, K13 = binomial(k, 3), C3 = 0)
```

    (edges = 1.5, P3 = 3, P4 = 6.0, K13 = 1, C3 = 0)

``` julia
rows = []
for m in ms
    byshape = Dict{Symbol,Int}()
    for v in low[m].variables
        byshape[v.shape.name] = get(byshape, v.shape.name, 0) + 1
    end
    for sh in low[m].shapes
        push!(rows, (m, string(sh.name), sh.n_nodes, length(sh.edges), length(sh.automorphisms),
                     get(byshape, sh.name, 0)))
    end
end
md_table(["m", "shape", "vertices", "edges", "order of Aut", "state classes"], rows)
```

|   m | shape     | vertices | edges | order of Aut | state classes |
|----:|-----------|---------:|------:|-------------:|--------------:|
|   2 | singleton |        1 |     0 |            1 |             2 |
|   2 | P2        |        2 |     1 |            2 |             3 |
|   3 | singleton |        1 |     0 |            1 |             2 |
|   3 | P2        |        2 |     1 |            2 |             3 |
|   3 | P3        |        3 |     2 |            2 |             6 |
|   3 | C3        |        3 |     3 |            6 |             4 |
|   4 | singleton |        1 |     0 |            1 |             2 |
|   4 | P2        |        2 |     1 |            2 |             3 |
|   4 | P3        |        3 |     2 |            2 |             6 |
|   4 | C3        |        3 |     3 |            6 |             4 |
|   4 | P4        |        4 |     3 |            2 |            10 |
|   4 | K13       |        4 |     3 |            6 |             8 |
|   4 | paw       |        4 |     4 |            2 |            12 |
|   4 | C4        |        4 |     4 |            8 |             6 |
|   4 | K4me      |        4 |     5 |            4 |             9 |
|   4 | K4        |        4 |     6 |           24 |             5 |

The realised graphs of the ensemble are locally tree-like, as the
defaults assume. The mean clustering coefficient of the 200 sampled
graphs is 0.00016.

The closure at order m replaces each (m + 1)-vertex quantity with a
ratio of tracked ones. An S vertex i of a motif has n_ext(i) = k −
deg_motif(i) neighbours outside the motif. For m = 3 the expected number
of infected ones is estimated from a single-vertex anchor,

$$n_{\mathrm{ext}}(i)\,\frac{L(\sigma_i, I)}{k\,\langle \sigma_i \rangle},$$

where L(σᵢ, I) is the labelled count of pairs (σᵢ, I) and ⟨σᵢ⟩ the
number of nodes in state σᵢ. For m = 2 this is Keeling’s factor (k −
1)/k. For m = 4 the external neighbour of a four-vertex motif is closed
through a three-vertex anchor that contains the infected vertex (a
per-shape Kirkwood closure), and the P₃ and C₃ equations are the
marginals of the four-vertex equations.

The numeric right-hand sides are checked against
`build_motif_symbolic_rhs`, a symbolic builder that assembles the
equations from the shape and state-class enumeration and does not call
the numeric builders. At m = 4 it uses the same table of closure rules
as the numeric code, so there it checks the assembly of the equations,
not the choice of closure:

``` julia
function oracle_gap(m)
    rhs_sym!, keys_sym, _ = build_motif_symbolic_rhs(MotifClosure(3, m))
    order_ok = keys_sym == [(v.shape.name, v.state) for v in low[m].variables]
    return (order_ok, max_rhs_gap(low[m], low[m], (du, u) -> rhs_sym!(du, u, (sc.params[:τ], sc.params[:γ]), 0.0)))
end
md_table(["m", "same variable order", "max abs(numeric − symbolic) at 25 random states"],
         [(m, oracle_gap(m)...) for m in ms])
```

|   m | same variable order | max abs(numeric − symbolic) at 25 random states |
|----:|--------------------:|------------------------------------------------:|
|   2 |                true |                                       2.776e-17 |
|   3 |                true |                                        2.22e-16 |
|   4 |                true |                                       4.441e-16 |

## m = 2 is the pairwise model

``` julia
pw  = node_based(model, sc.network)                  # population pairwise, Bernoulli closure K = (k − 1)/k
cpw = model_curves(pw, solve_epidemic(pw, sc); t = sc.tgrid, label = "pairwise")
function motif_curves(m)
    s = low[m]
    sol = solve_motif(s; saveat = sc.tgrid)
    ModelCurves(sc.tgrid, Dict(:I => compartment(s, sol, :I), :S => compartment(s, sol, :S));
                label = "motif m = $(m)", representation = :motif)
end
cms = [motif_curves(m) for m in ms]
(; max_gap_I = maximum(abs.(cms[1][:I] .- cpw[:I])))
```

    (max_gap_I = 1.7526654572108669e-9,)

## Against simulation

``` julia
ref = scenario_summary(sc)
tab = compare(ref, cms; observables = [:I])
```

    ComparisonTable :sis_reg3  (scenario 485d890e; conditioned mean of 200 runs)
      curve        observable        D∞     t(D∞)       SE∞        z∞       ΔR∞  95% CI                 Δpeak   Δt_peak  coverage
      motif m = 2  I            0.14556      9.50   0.00160     93.69       NaN  [     NaN,      NaN]  -0.00023      4.00     0.153
      motif m = 3  I            0.06009      9.75   0.00160     39.65       NaN  [     NaN,      NaN]  -0.00025    -17.50     0.202
      motif m = 4  I            0.00888     10.75   0.00160     11.27       NaN  [     NaN,      NaN]  -0.00016    -15.50     0.688

``` julia
describe_reference(ref);
```

    NetworkOutbreaks reference :sis_reg3 (hash 485d890e): N = 10000 nodes, 200 runs on a fresh graph per run, algorithm :next_reaction; conditioning: surviving runs only (prevalence at t_end > 0); 200 of 200 runs kept (surviving runs), P(survival) = 1.000 (95% CI 0.981–1.000); no time alignment.

``` julia
distinct_styles!(comparisonplot(ref, cms...; observables = [:I]))
```

![Motif closures m = 2, 3, 4 against the committed NetworkOutbreaks
ensemble of `:sis_reg3`, conditioned on survival (spread band q2.5–q97.5
of the runs; residual panel: mean band ±1.96
SE).](index_files/figure-commonmark/cell-11-output-1.svg)

The motif systems do not track cumulative incidence, so ΔR∞ is not
defined for them (NaN in the table), and the peak columns are not
meaningful on an endemic plateau. The largest gap and the endemic level:

``` julia
st = ref.cond[:I]
md_table(["model", "variables", "D∞(I)", "t at D∞", "z∞", "I(80)", "I(80) − NO"],
         vcat([(c.label, length(low[m].variables), tab[c.label, :I].D∞, tab[c.label, :I].t_D∞, tab[c.label, :I].z∞,
                c[:I][end], c[:I][end] - st.mean[end]) for (m, c) in zip(ms, cms)],
              [("NetworkOutbreaks (mean ± SE)", "–", NaN, NaN, NaN, st.mean[end], st.se[end])]))
```

| model | variables | D∞(I) | t at D∞ | z∞ | I(80) | I(80) − NO |
|----|---:|---:|---:|---:|---:|---:|
| motif m = 2 | 5 | 0.1456 | 9.5 | 93.69 | 0.8182 | 0.001126 |
| motif m = 3 | 15 | 0.06009 | 9.75 | 39.65 | 0.8163 | -0.0007423 |
| motif m = 4 | 65 | 0.008878 | 10.75 | 11.27 | 0.8174 | 0.0003262 |
| NetworkOutbreaks (mean ± SE) | – | NaN | NaN | NaN | 0.8171 | 0.0003 |

The error of the transient decreases with the order: D∞(I) = 0.1456,
0.06009, 0.008878 for m = 2, 3, 4, at a cost of 5, 15, 65 variables.
None of the closures is exact on a random 3-regular graph, and at m = 4
the gap is still z∞ = 11.27 standard errors at its largest. The error is
largest during the take-off:

``` julia
p = plot(ref, :I; median = false, legend = :bottomright, xlabel = "t", ylabel = "I (fraction of nodes)",
         title = "Motif closures: take-off")
for c in cms
    plot!(p, c, :I)
end
distinct_styles!(p)
xlims!(p, 0, 30)
```

![The first 30 time units (prevalence; band: spread q2.5–q97.5 of the
surviving runs).](index_files/figure-commonmark/cell-14-output-1.svg)

The design of this page planned to show m = 4 as an open implementation
issue, because an earlier order-4 builder gave a larger error than m =
3. With the current builder, whose right-hand side agrees with the
symbolic lift above, the error decreases strictly with the order:

``` julia
(; D∞ = Dict(zip(ms, D)), strictly_decreasing_in_m = all(diff(D) .< 0))
```

    (D∞ = Dict(4 => 0.008877952628343344, 2 => 0.1455586067337321, 3 => 0.06009337362097189), strictly_decreasing_in_m = true)

No Lean statement is involved on this page. Page
[N11](../N11_neighbourhood/index.md) gives the neighbourhood model, and
page [N12](../N12_sis_comparison/index.md) compares every SIS
approximation.

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-house2011" class="csl-entry">

House, Thomas, and Matt J. Keeling. 2011. “Insights from Unifying Modern
Approximations to Infections on Networks.” *Journal of the Royal Society
Interface* 8 (54): 67–73. <https://doi.org/10.1098/rsif.2010.0179>.

</div>

</div>
