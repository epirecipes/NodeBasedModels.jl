# SIR Epidemic on a Graph
Simon Frost
2026-03-29

- [Introduction](#introduction)
- [Setup](#setup)
- [Create a graph](#create-a-graph)
- [Individual-based model (order-1)](#individual-based-model-order-1)
- [Pair-based model (order-2)](#pair-based-model-order-2)
- [Gillespie validation](#gillespie-validation)
- [Discussion](#discussion)

## Introduction

In contrast to the edge-based approach of
[EdgeBasedModels.jl](https://github.com/sdwfrost/edgebasedmodels), which
works with degree distributions and probability generating functions,
**NodeBasedModels.jl** operates directly on specific graph instances.
This means every node and every edge is tracked explicitly, allowing us
to capture the full heterogeneity of a network — not just its degree
distribution, but the precise adjacency structure.

[Sharkey (2011)](https://doi.org/10.1007/s00285-010-0340-1) showed that
node-level epidemic models can be organised into a hierarchy of **moment
closures**:

- **Order-1 (individual-based / NIMFA):** Tracks the marginal
  probability $\langle S_i \rangle$, $\langle I_i \rangle$ for each node
  $i$. Pairs are approximated by independence:
  $$\langle S_i I_j \rangle \approx \langle S_i \rangle \langle I_j \rangle.$$
- **Order-2 (pair-based):** Tracks both node marginals and pair
  probabilities $\langle S_i I_j \rangle$ for each edge $(i,j)$. Triples
  are closed via the Kirkwood superposition:
  $$\langle A_k B_i C_j \rangle \approx \frac{\langle A_k B_i \rangle \langle B_i C_j \rangle}{\langle B_i \rangle}.$$
- **Exact (Gillespie):** Continuous-time Markov chain simulation using
  the Gillespie algorithm. No approximation — the “gold standard.”

Higher-order closures are more accurate but more expensive. In this
vignette we run all three levels on the same graph and compare their
predictions for an SIR epidemic.

## Setup

``` julia
using NodeBasedModels
using Graphs
using Plots
using OrdinaryDiffEqDefault
using Random
```

## Create a graph

We build a random regular graph with $N = 100$ nodes and degree $k = 6$.
Every node has exactly 6 neighbours.

``` julia
Random.seed!(42)
g = random_regular_graph(100, 6)
net = GraphNetwork(g)

println("Nodes:       ", nv(g))
println("Edges:       ", ne(g))
println("Mean degree: ", mean_degree(net))
```

    Nodes:       100
    Edges:       300
    Mean degree: 6.0

## Individual-based model (order-1)

The individual-based (NIMFA) approximation tracks the probability that
each node is in a given state. For the SIR model on a graph with
adjacency matrix $A$, the equations for node $i$ are

$$\frac{d \langle S_i \rangle}{dt} = -\tau \sum_{j} A_{ij} \langle S_i \rangle \langle I_j \rangle, \qquad
\frac{d \langle I_i \rangle}{dt} = \tau \sum_{j} A_{ij} \langle S_i \rangle \langle I_j \rangle - \gamma \langle I_i \rangle.$$

Because pairs are factorised
($\langle S_i I_j \rangle \approx \langle S_i \rangle \langle I_j \rangle$),
the system has $2N$ ODEs.

``` julia
ib_result = generate_individual_based(
    sir_model(), net;
    infection_rate = 0.15,
    recovery_rate  = 0.1,
    initial_infected = [1],
    tspan  = (0.0, 80.0),
    saveat = 0.5
)

S_ib = aggregate(ib_result, :S)
I_ib = aggregate(ib_result, :I)
R_ib = 100.0 .- S_ib .- I_ib
t_ib = range(0.0, 80.0, length = length(S_ib))

p = plot(t_ib, S_ib, label = "S (individual)", lw = 2, color = :blue,
         xlabel = "Time", ylabel = "Number of individuals",
         title = "SIR on a 6-regular graph (N=100)")
plot!(p, t_ib, I_ib, label = "I (individual)", lw = 2, color = :red)
plot!(p, t_ib, R_ib, label = "R (individual)", lw = 2, color = :green)
p
```

![](index_files/figure-commonmark/cell-4-output-1.svg)

## Pair-based model (order-2)

The pair-based model additionally tracks the joint probability
$\langle S_i I_j \rangle$ for every directed edge $(i,j)$. Triples like
$\langle S_k S_i I_j \rangle$ that appear in the pair equations are
closed using the **Kirkwood closure**:

$$\langle A_k B_i C_j \rangle \approx \frac{\langle A_k B_i \rangle \langle B_i C_j \rangle}{\langle B_i \rangle}.$$

This closure is **exact on tree graphs** (no cycles), because in that
case nodes $k$ and $j$ are conditionally independent given $i$.

``` julia
pb_result = generate_pair_based(
    sir_model(), net;
    infection_rate = 0.15,
    recovery_rate  = 0.1,
    initial_infected = [1],
    tspan  = (0.0, 80.0),
    saveat = 0.5
)

S_pb = aggregate(pb_result, :S)
I_pb = aggregate(pb_result, :I)
R_pb = 100.0 .- S_pb .- I_pb
t_pb = range(0.0, 80.0, length = length(S_pb))
```

    0.0:0.5:80.0

``` julia
p = plot(t_ib, I_ib, label = "I (individual-based)", lw = 2, ls = :dash, color = :red,
         xlabel = "Time", ylabel = "Number infected",
         title = "Individual vs pair-based")
plot!(p, t_pb, I_pb, label = "I (pair-based)", lw = 2, color = :darkred)
p
```

![](index_files/figure-commonmark/cell-6-output-1.svg)

## Gillespie validation

The Gillespie algorithm simulates the exact continuous-time Markov
chain. We run 50 realisations and compute the mean trajectory together
with a 90% confidence interval (5th–95th percentile envelope).

``` julia
gill_avg = gillespie_sir_average(
    net;
    nruns          = 50,
    dt             = 0.5,
    tmax_grid      = 80.0,
    infection_rate = 0.15,
    recovery_rate  = 0.1,
    initial_infected = [1]
)
```

    (t_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5  …  75.5, 76.0, 76.5, 77.0, 77.5, 78.0, 78.5, 79.0, 79.5, 80.0], S_mean = [99.0, 98.56, 98.06, 97.34, 96.2, 95.24, 93.94, 91.76, 89.2, 86.82  …  20.2, 20.2, 20.2, 20.2, 20.2, 20.2, 20.2, 20.2, 20.2, 20.2], I_mean = [1.0, 1.36, 1.78, 2.44, 3.38, 4.24, 5.3, 7.14, 9.4, 11.08  …  0.12, 0.12, 0.12, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.08], R_mean = [0.0, 0.08, 0.16, 0.22, 0.42, 0.52, 0.76, 1.1, 1.4, 2.1  …  79.68, 79.68, 79.68, 79.68, 79.68, 79.7, 79.7, 79.7, 79.7, 79.72], S_q05 = [99.0, 97.0, 95.0, 93.0, 90.0, 85.0, 80.0, 71.0, 66.0, 59.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], S_q95 = [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0  …  99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0], I_q05 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], I_q95 = [1.0, 3.0, 4.0, 7.0, 9.0, 13.0, 17.0, 24.0, 28.0, 32.0  …  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], final_sizes = [98, 1, 100, 100, 100, 99, 100, 1, 99, 100  …  100, 99, 100, 100, 98, 1, 100, 100, 100, 100])

``` julia
p = plot(t_ib, I_ib, label = "Individual-based (order-1)", lw = 2, ls = :dash, color = :red,
         xlabel = "Time", ylabel = "Number infected",
         title = "Moment closure hierarchy — SIR on 6-regular graph")
plot!(p, t_pb, I_pb, label = "Pair-based (order-2)", lw = 2, color = :darkred)
plot!(p, gill_avg.t_grid, gill_avg.I_mean, label = "Gillespie mean (n=50)",
      lw = 2, color = :black)
plot!(p, gill_avg.t_grid, gill_avg.I_mean,
      ribbon = (gill_avg.I_mean .- gill_avg.I_q05, gill_avg.I_q95 .- gill_avg.I_mean),
      fillalpha = 0.2, color = :grey, label = "Gillespie 90% CI")
p
```

![](index_files/figure-commonmark/cell-8-output-1.svg)

The plot illustrates the characteristic ordering of the hierarchy:

$$\text{Individual-based} \;\ge\; \text{Pair-based} \;\ge\; \text{Gillespie mean}.$$

The individual-based model over-estimates the epidemic because the
pairwise independence assumption ignores **dynamical correlations**: if
node $i$ infected node $j$, then $j$ is less likely to re-infect $i$
because $i$ is probably still infected. This “2-cycle” effect is
captured by the pair-based model.

## Discussion

The three levels of the moment closure hierarchy trade off accuracy
against computational cost:

| Level | Variables | Closure | Exact when? |
|----|----|----|----|
| Order-1 (individual) | $O(N)$ | Independence | Complete graph (mean-field limit) |
| Order-2 (pair-based) | $O(N + M)$ | Kirkwood | Tree graphs (no cycles) |
| Exact (Gillespie) | $O(N)$ state | None | Always (stochastic) |

The order-1 approximation ignores correlations introduced by
**2-cycles** (anomalous terms of the form
$\langle S_i I_j \rangle - \langle S_i \rangle \langle I_j \rangle$).
These terms are always positive during an epidemic, so order-1
systematically over-estimates the force of infection.

The order-2 approximation captures pair correlations but neglects
**triangles** (3-cycles). On graphs with low clustering (like random
regular graphs), the pair-based model is already very accurate. On
highly clustered graphs, higher-order closures or the exact Gillespie
simulation may be needed.

In the next vignette, we explore how different network topologies affect
both the epidemic dynamics and the accuracy of these approximations.
