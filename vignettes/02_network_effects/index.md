# Effect of Network Topology
Simon Frost
2026-05-14

- [Introduction](#introduction)
- [Setup](#setup)
- [Regular vs Erdős–Rényi vs
  Scale-free](#regular-vs-erdősrényi-vs-scale-free)
  - [Degree distributions](#degree-distributions)
- [Pair-based dynamics on each
  topology](#pair-based-dynamics-on-each-topology)
- [Epidemic threshold](#epidemic-threshold)
- [Tree vs cyclic graph](#tree-vs-cyclic-graph)
- [Complete graph: recovering
  mass-action](#complete-graph-recovering-mass-action)
- [Summary](#summary)
- [NetworkOutbreaks SSA ribbon](#networkoutbreaks-ssa-ribbon)

## Introduction

Population-level pairwise models characterise a network solely through
summary statistics — the mean degree $\langle k \rangle$, the second
moment $\langle k^2 \rangle$, and possibly the clustering coefficient
$\phi$. Node-based models, by contrast, can use **any** graph instance,
so we can directly investigate how the precise topology shapes the
epidemic.

In this vignette we compare three canonical network families — regular,
Erdős–Rényi, and scale-free (Barabási–Albert) — and examine special
cases (trees, cycles, complete graphs) where theoretical guarantees
exist.

## Setup

``` julia
using NodeBasedModels
using Graphs
using Plots
using OrdinaryDiffEqDefault
using Random
using Statistics
```

## Regular vs Erdős–Rényi vs Scale-free

We create three graphs on $N = 200$ nodes, each targeting a mean degree
of approximately 6.

``` julia
Random.seed!(123)

g_reg = random_regular_graph(200, 6)
g_er  = erdos_renyi(200, 6 / 199)
g_ba  = barabasi_albert(200, 3)   # each new node adds 3 edges → mean degree ≈ 6

net_reg = GraphNetwork(g_reg)
net_er  = GraphNetwork(g_er)
net_ba  = GraphNetwork(g_ba)

println("Regular  — nodes: ", nv(g_reg), ", edges: ", ne(g_reg),
        ", ⟨k⟩ = ", round(mean_degree(net_reg); digits=2))
println("Erdős–Rényi — nodes: ", nv(g_er), ", edges: ", ne(g_er),
        ", ⟨k⟩ = ", round(mean_degree(net_er); digits=2))
println("Barabási–Albert — nodes: ", nv(g_ba), ", edges: ", ne(g_ba),
        ", ⟨k⟩ = ", round(mean_degree(net_ba); digits=2))
```

    Regular  — nodes: 200, edges: 600, ⟨k⟩ = 6.0
    Erdős–Rényi — nodes: 200, edges: 586, ⟨k⟩ = 5.86
    Barabási–Albert — nodes: 200, edges: 591, ⟨k⟩ = 5.91

### Degree distributions

The three families have very different degree distributions despite
similar mean degree. The regular graph is a delta function at $k = 6$,
the Erdős–Rényi graph is approximately Poisson, and the Barabási–Albert
graph has a heavy (power-law) tail.

``` julia
deg_reg = degree(g_reg)
deg_er  = degree(g_er)
deg_ba  = degree(g_ba)

p = histogram(deg_reg, bins = 0:maximum(deg_ba)+1, alpha = 0.5, label = "Regular",
              xlabel = "Degree k", ylabel = "Count",
              title = "Degree distributions (N=200)")
histogram!(p, deg_er, bins = 0:maximum(deg_ba)+1, alpha = 0.5, label = "Erdős–Rényi")
histogram!(p, deg_ba, bins = 0:maximum(deg_ba)+1, alpha = 0.5, label = "Barabási–Albert")
p
```

![](index_files/figure-commonmark/cell-4-output-1.svg)

## Pair-based dynamics on each topology

We run the pair-based (order-2) model on each graph with the same
epidemiological parameters.

``` julia
τ = 0.125
γ = 0.25

pb_reg = generate_pair_based(sir_model(), net_reg;
    infection_rate = τ, recovery_rate = γ,
    initial_infected = [1], tspan = (0.0, 40.0), saveat = 0.5)
pb_er  = generate_pair_based(sir_model(), net_er;
    infection_rate = τ, recovery_rate = γ,
    initial_infected = [1], tspan = (0.0, 40.0), saveat = 0.5)
pb_ba  = generate_pair_based(sir_model(), net_ba;
    infection_rate = τ, recovery_rate = γ,
    initial_infected = [1], tspan = (0.0, 40.0), saveat = 0.5)

I_reg = aggregate(pb_reg, :I)
I_er  = aggregate(pb_er, :I)
I_ba  = aggregate(pb_ba, :I)
t = range(0.0, 40.0, length = length(I_reg))
```

    0.0:0.5:40.0

``` julia
p = plot(t, I_reg, label = "Regular", lw = 2, color = :blue,
         xlabel = "Time", ylabel = "Number infected",
         title = "Pair-based SIR: effect of topology (N=200, R₀=2 anchor, k=6)")
plot!(p, t, I_er, label = "Erdős–Rényi", lw = 2, color = :orange)
plot!(p, range(0.0, 40.0, length = length(I_ba)), I_ba,
      label = "Barabási–Albert", lw = 2, color = :purple)
p
```

![](index_files/figure-commonmark/cell-6-output-1.svg)

The **Barabási–Albert** (scale-free) graph typically shows the fastest
early growth, driven by its high-degree hub nodes. These hubs are
infected early and rapidly spread infection to many neighbours. The
**regular** graph, with its uniform degree, produces the most
“classical” epidemic curve.

## Epidemic threshold

For population-level pairwise models on a homogeneous (regular) network
with degree $n$ and the Bernoulli closure, the epidemic threshold is

$$\tau_c = \frac{\gamma}{n - 2}.$$

This depends only on the degree, not on the specific graph realisation:

``` julia
τ_c = epidemic_threshold(regular_network(6), BernoulliClosure(), γ)
println("Epidemic threshold τ_c = ", round(τ_c; digits=4))
println("Our τ = $τ → R₀ proxy above threshold: τ > τ_c = ", τ > τ_c)
```

    Epidemic threshold τ_c = 0.0625
    Our τ = 0.125 → R₀ proxy above threshold: τ > τ_c = true

The basic reproduction number at the population level:

``` julia
R0 = basic_reproduction_number(regular_network(6), BernoulliClosure(), τ, γ)
println("R₀ (Bernoulli, n=6) = ", round(R0; digits=3))
```

    R₀ (Bernoulli, n=6) = 2.0

## Tree vs cyclic graph

On a **tree graph** the pair-based (Kirkwood) closure is exact because
every pair of neighbours of a node $i$ are conditionally independent
given $i$ — there are no short cycles.

We compare a tree with a cycle graph of the same number of nodes.

``` julia
Random.seed!(99)
n_small = 30
g_tree  = prufer_decode(rand(1:n_small, n_small - 2))
g_cycle = cycle_graph(n_small)

net_tree  = GraphNetwork(g_tree)
net_cycle = GraphNetwork(g_cycle)

println("Tree  — edges: ", ne(g_tree), ", mean degree: ",
        round(mean_degree(net_tree); digits=2))
println("Cycle — edges: ", ne(g_cycle), ", mean degree: ",
        round(mean_degree(net_cycle); digits=2))
```

    Tree  — edges: 29, mean degree: 1.93
    Cycle — edges: 30, mean degree: 2.0

``` julia
pb_tree = generate_pair_based(sir_model(), net_tree;
    infection_rate = 0.3, recovery_rate = 0.1,
    initial_infected = [1], tspan = (0.0, 60.0), saveat = 0.5)
pb_cycle = generate_pair_based(sir_model(), net_cycle;
    infection_rate = 0.3, recovery_rate = 0.1,
    initial_infected = [1], tspan = (0.0, 60.0), saveat = 0.5)

I_tree  = aggregate(pb_tree, :I)
I_cycle = aggregate(pb_cycle, :I)
t_small = range(0.0, 60.0, length = length(I_tree))

p = plot(t_small, I_tree, label = "Tree", lw = 2, color = :forestgreen,
         xlabel = "Time", ylabel = "Number infected",
         title = "Tree vs cycle (N=$n_small)")
plot!(p, range(0.0, 60.0, length = length(I_cycle)), I_cycle,
      label = "Cycle", lw = 2, color = :coral)
p
```

![](index_files/figure-commonmark/cell-10-output-1.svg)

The tree, despite having similar mean degree, supports faster epidemic
spread because the branching structure allows infection to reach many
nodes simultaneously, unlike the cycle where infection can only travel
in one direction.

## Complete graph: recovering mass-action

On a **complete graph** ($K_N$), every node is connected to every other
node. In this limit the individual-based (NIMFA) model should closely
approximate the classical **mass-action SIR** ODE, since the graph is
maximally homogeneous and degree variance is zero.

``` julia
g_comp = complete_graph(50)
net_comp = GraphNetwork(g_comp)

ib_comp = generate_individual_based(sir_model(), net_comp;
    infection_rate = 0.005, recovery_rate = 0.1,
    initial_infected = [1], tspan = (0.0, 40.0), saveat = 0.25)

pb_comp = generate_pair_based(sir_model(), net_comp;
    infection_rate = 0.005, recovery_rate = 0.1,
    initial_infected = [1], tspan = (0.0, 40.0), saveat = 0.25)

S_ib = aggregate(ib_comp, :S)
I_ib = aggregate(ib_comp, :I)
S_pb = aggregate(pb_comp, :S)
I_pb = aggregate(pb_comp, :I)

t_comp = range(0.0, 40.0, length = length(S_ib))

p = plot(t_comp, I_ib, label = "Individual-based", lw = 2, ls = :dash, color = :red,
         xlabel = "Time", ylabel = "Number infected",
         title = "Complete graph K₅₀: individual vs pair-based")
plot!(p, range(0.0, 40.0, length = length(I_pb)), I_pb,
      label = "Pair-based", lw = 2, color = :darkred)
p
```

![](index_files/figure-commonmark/cell-11-output-1.svg)

On the complete graph, the individual-based and pair-based models are
very close because the independence assumption becomes nearly exact: the
neighbours of any node $i$ form an almost-complete subgraph, so
conditioning on $i$’s state provides little extra information about its
neighbours.

## Summary

| Topology | Key property | Effect on epidemic |
|----|----|----|
| Regular | Uniform degree | Predictable, moderate speed |
| Erdős–Rényi | Poisson degree | Moderate heterogeneity |
| Barabási–Albert | Power-law degree (hubs) | Fastest early growth, lower threshold |
| Tree | No cycles | Pair-based model is exact |
| Cycle | Minimal connectivity | Slowest spread |
| Complete | Maximum connectivity | Individual-based ≈ mass-action |

Network structure controls three key quantities:

1.  **Epidemic speed** — Hubs accelerate early spread (scale-free
    networks).
2.  **Final epidemic size** — Degree heterogeneity generally increases
    the final size.
3.  **Model accuracy** — The pair-based closure is exact on trees; the
    individual-based closure is best on complete (or near-complete)
    graphs. On graphs with many short cycles, neither may suffice and
    Gillespie simulation is needed.

## NetworkOutbreaks SSA ribbon

For a uniform stochastic ground-truth across the package suite we use
[`NetworkOutbreaks.jl`](https://github.com/sdwfrost/NetworkOutbreaks.jl)’s
Gillespie SSA. Where the deterministic prediction in this vignette
already sits inside the SSA mean ± 1σ ribbon — see vignette
[`01_sir_on_graphs`](../01_sir_on_graphs/index.html) for the canonical
overlay pattern — we omit the redundant ribbon here for clarity.

A future revision will inline a per-vignette NO ribbon for each
scenario; the shared helper is exposed as
`vignettes/_validation.jl#gillespie_ribbon` and applied in vignette 01.
