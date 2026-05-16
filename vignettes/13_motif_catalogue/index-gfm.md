

- [Motif Shapes and State-Class
  Catalogue](#motif-shapes-and-state-class-catalogue)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Counting convention](#counting-convention)
  - [Catalogue generation](#catalogue-generation)
  - [Variable-count summary](#variable-count-summary)
  - [Shape diagrams](#shape-diagrams)
  - [Notes on the Lean-certified obstruction at (k=3,
    m=4)](#notes-on-the-lean-certified-obstruction-at-k3-m4)
  - [References](#references)
  - [NetworkOutbreaks SSA ribbon](#networkoutbreaks-ssa-ribbon)

# Motif Shapes and State-Class Catalogue

Simon Frost 2026-05-14

- [Introduction](#introduction)
- [Setup](#setup)
- [Counting convention](#counting-convention)
- [Catalogue generation](#catalogue-generation)
- [Variable-count summary](#variable-count-summary)
- [Shape diagrams](#shape-diagrams)
- [Notes on the Lean-certified obstruction at (k=3,
  m=4)](#notes-on-the-lean-certified-obstruction-at-k3-m4)
- [References](#references)
- [NetworkOutbreaks SSA ribbon](#networkoutbreaks-ssa-ribbon)

## Introduction

The motif/subgraph-closure framework of [Keeling, House, Cooper & Pellis
(2016)](https://doi.org/10.1371/journal.pcbi.1005296) tracks, for each
connected $m$-vertex induced subgraph *shape* $H$ of the host network
and each canonical state assignment $\sigma$ of $H$’s vertices, the
count $E_{H,\sigma}(t)$ of such induced copies in the current state. Two
embeddings are identified if one is sent to the other by a graph
automorphism of $H$; this reduces the state space from
$|Q|^m \times |\text{Emb}(H,G)|$ to a manageable set of orbit
representatives.

Closure enters at the $(m\!+\!1)$-th order: every $(m\!+\!1)$-vertex
induced amplitude is expressed as a Kirkwood-type ratio of tracked
$m$-vertex and lower-order quantities. For the SIS model on a
$k$-regular host this gives an autonomous ODE system whose variable
count grows with $(k,m)$ but remains dramatically smaller than the full
$2^N$ master equation.

This vignette enumerates every shape and every SIS state class for all
supported $(k,m)$ pairs: $k=2$ with $2 \le m \le 6$, and $k=3$ with
$m \in \{2,3,4\}$. It is generated programmatically from the package’s
own `enumerate_shapes` and `enumerate_state_classes` functions so the
tables are guaranteed to match the running implementation.

## Setup

``` julia
using NodeBasedModels
```

## Counting convention

Variables count **unordered induced embeddings** of a shape $H$
partitioned by canonical state class under $\mathrm{Aut}(H)$. Conversion
to the “directed-pair” or “labelled” counts used in the standard Keeling
pairwise convention multiplies by the orbit size
$|\mathrm{orb}(\sigma)|$:

$$L_\sigma = E_{\mathrm{canon}(\sigma)} \cdot |\mathrm{orb}(\sigma)|,$$

where
$|\mathrm{orb}(\sigma)| = |\mathrm{Aut}(H)| / |\mathrm{Stab}(\sigma)|$.

For the $P_2$ shape ($|\mathrm{Aut}|=2$):

| Canonical class | Orbit size | Relation to Keeling pairwise    |
|-----------------|:----------:|---------------------------------|
| $[SS]$          |     1      | $[SS]_{\mathrm{pw}} = 2 E_{SS}$ |
| $[IS]$          |     2      | $[SI]_{\mathrm{pw}} = E_{IS}$   |
| $[II]$          |     1      | $[II]_{\mathrm{pw}} = 2 E_{II}$ |

## Catalogue generation

``` julia
function fmt_state(st)
    join(string.(st), ",")
end

function shape_ascii(name::Symbol)
    d = Dict(
        :singleton => "●",
        :P2        => "●─●",
        :P3        => "●─●─●",
        :C3        => "●─●\n └─●",
        :P4        => "●─●─●─●",
        :P5        => "●─●─●─●─●",
        :P6        => "●─●─●─●─●─●",
        :K13       => "  ●\n●─●─●\n  ●",
        :paw       => "●─●\n │╲\n ●─●",
        :C4        => "●─●\n │ │\n ●─●",
        :K4me      => "●─●\n │╲│\n ●─●",
        :K4        => "●─●\n │╳│\n ●─●",
    )
    get(d, name, string(name))
end

for (k, m) in [(2,2),(2,3),(2,4),(2,5),(2,6),(3,2),(3,3),(3,4)]
    closure = MotifClosure(k, m)
    shapes  = NodeBasedModels.enumerate_shapes(closure)
    println("=" ^ 60)
    println("(k=$k, m=$m)  —  $(length(shapes)) shape(s)")
    println("=" ^ 60)
    total_vars = 0
    for sh in shapes
        classes = NodeBasedModels.enumerate_state_classes(sh, [:S, :I])
        n_vars  = length(classes)
        total_vars += n_vars
        println()
        println("  Shape :$(sh.name)  |V|=$(sh.n_nodes)  |E|=$(length(sh.edges))",
                "  |Aut|=$(length(sh.automorphisms))  state-classes=$n_vars")
        if !isempty(sh.edges)
            println("  Edges: ", join(["($a,$b)" for (a,b) in sh.edges], " "))
        end
        println("  ┌─────────────────────┬────────────┐")
        println("  │ Canonical state      │ Orbit size │")
        println("  ├─────────────────────┼────────────┤")
        for (st, osz) in sort(classes; by=x->x[1])
            println("  │ [$(rpad(fmt_state(st),19))] │ $(lpad(osz,10)) │")
        end
        println("  └─────────────────────┴────────────┘")
    end
    println()
    println("  Total variables for (k=$k, m=$m): $total_vars")
end
```

    ============================================================
    (k=2, m=2)  —  2 shape(s)
    ============================================================

      Shape :singleton  |V|=1  |E|=0  |Aut|=1  state-classes=2
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I                  ] │          1 │
      │ [S                  ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P2  |V|=2  |E|=1  |Aut|=2  state-classes=3
      Edges: (1,2)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I                ] │          1 │
      │ [I,S                ] │          2 │
      │ [S,S                ] │          1 │
      └─────────────────────┴────────────┘

      Total variables for (k=2, m=2): 5
    ============================================================
    (k=2, m=3)  —  3 shape(s)
    ============================================================

      Shape :singleton  |V|=1  |E|=0  |Aut|=1  state-classes=2
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I                  ] │          1 │
      │ [S                  ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P2  |V|=2  |E|=1  |Aut|=2  state-classes=3
      Edges: (1,2)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I                ] │          1 │
      │ [I,S                ] │          2 │
      │ [S,S                ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P3  |V|=3  |E|=2  |Aut|=2  state-classes=6
      Edges: (1,2) (2,3)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I              ] │          1 │
      │ [I,I,S              ] │          2 │
      │ [I,S,I              ] │          1 │
      │ [I,S,S              ] │          2 │
      │ [S,I,S              ] │          1 │
      │ [S,S,S              ] │          1 │
      └─────────────────────┴────────────┘

      Total variables for (k=2, m=3): 11
    ============================================================
    (k=2, m=4)  —  3 shape(s)
    ============================================================

      Shape :singleton  |V|=1  |E|=0  |Aut|=1  state-classes=2
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I                  ] │          1 │
      │ [S                  ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P2  |V|=2  |E|=1  |Aut|=2  state-classes=3
      Edges: (1,2)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I                ] │          1 │
      │ [I,S                ] │          2 │
      │ [S,S                ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P4  |V|=4  |E|=3  |Aut|=2  state-classes=10
      Edges: (1,2) (2,3) (3,4)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I,I            ] │          1 │
      │ [I,I,I,S            ] │          2 │
      │ [I,I,S,I            ] │          2 │
      │ [I,I,S,S            ] │          2 │
      │ [I,S,I,S            ] │          2 │
      │ [I,S,S,I            ] │          1 │
      │ [I,S,S,S            ] │          2 │
      │ [S,I,I,S            ] │          1 │
      │ [S,I,S,S            ] │          2 │
      │ [S,S,S,S            ] │          1 │
      └─────────────────────┴────────────┘

      Total variables for (k=2, m=4): 15
    ============================================================
    (k=2, m=5)  —  3 shape(s)
    ============================================================

      Shape :singleton  |V|=1  |E|=0  |Aut|=1  state-classes=2
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I                  ] │          1 │
      │ [S                  ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P2  |V|=2  |E|=1  |Aut|=2  state-classes=3
      Edges: (1,2)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I                ] │          1 │
      │ [I,S                ] │          2 │
      │ [S,S                ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P5  |V|=5  |E|=4  |Aut|=2  state-classes=20
      Edges: (1,2) (2,3) (3,4) (4,5)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I,I,I          ] │          1 │
      │ [I,I,I,I,S          ] │          2 │
      │ [I,I,I,S,I          ] │          2 │
      │ [I,I,I,S,S          ] │          2 │
      │ [I,I,S,I,I          ] │          1 │
      │ [I,I,S,I,S          ] │          2 │
      │ [I,I,S,S,I          ] │          2 │
      │ [I,I,S,S,S          ] │          2 │
      │ [I,S,I,I,S          ] │          2 │
      │ [I,S,I,S,I          ] │          1 │
      │ [I,S,I,S,S          ] │          2 │
      │ [I,S,S,I,S          ] │          2 │
      │ [I,S,S,S,I          ] │          1 │
      │ [I,S,S,S,S          ] │          2 │
      │ [S,I,I,I,S          ] │          1 │
      │ [S,I,I,S,S          ] │          2 │
      │ [S,I,S,I,S          ] │          1 │
      │ [S,I,S,S,S          ] │          2 │
      │ [S,S,I,S,S          ] │          1 │
      │ [S,S,S,S,S          ] │          1 │
      └─────────────────────┴────────────┘

      Total variables for (k=2, m=5): 25
    ============================================================
    (k=2, m=6)  —  3 shape(s)
    ============================================================

      Shape :singleton  |V|=1  |E|=0  |Aut|=1  state-classes=2
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I                  ] │          1 │
      │ [S                  ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P2  |V|=2  |E|=1  |Aut|=2  state-classes=3
      Edges: (1,2)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I                ] │          1 │
      │ [I,S                ] │          2 │
      │ [S,S                ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P6  |V|=6  |E|=5  |Aut|=2  state-classes=36
      Edges: (1,2) (2,3) (3,4) (4,5) (5,6)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I,I,I,I        ] │          1 │
      │ [I,I,I,I,I,S        ] │          2 │
      │ [I,I,I,I,S,I        ] │          2 │
      │ [I,I,I,I,S,S        ] │          2 │
      │ [I,I,I,S,I,I        ] │          2 │
      │ [I,I,I,S,I,S        ] │          2 │
      │ [I,I,I,S,S,I        ] │          2 │
      │ [I,I,I,S,S,S        ] │          2 │
      │ [I,I,S,I,I,S        ] │          2 │
      │ [I,I,S,I,S,I        ] │          2 │
      │ [I,I,S,I,S,S        ] │          2 │
      │ [I,I,S,S,I,I        ] │          1 │
      │ [I,I,S,S,I,S        ] │          2 │
      │ [I,I,S,S,S,I        ] │          2 │
      │ [I,I,S,S,S,S        ] │          2 │
      │ [I,S,I,I,I,S        ] │          2 │
      │ [I,S,I,I,S,I        ] │          1 │
      │ [I,S,I,I,S,S        ] │          2 │
      │ [I,S,I,S,I,S        ] │          2 │
      │ [I,S,I,S,S,I        ] │          2 │
      │ [I,S,I,S,S,S        ] │          2 │
      │ [I,S,S,I,I,S        ] │          2 │
      │ [I,S,S,I,S,S        ] │          2 │
      │ [I,S,S,S,I,S        ] │          2 │
      │ [I,S,S,S,S,I        ] │          1 │
      │ [I,S,S,S,S,S        ] │          2 │
      │ [S,I,I,I,I,S        ] │          1 │
      │ [S,I,I,I,S,S        ] │          2 │
      │ [S,I,I,S,I,S        ] │          2 │
      │ [S,I,I,S,S,S        ] │          2 │
      │ [S,I,S,I,S,S        ] │          2 │
      │ [S,I,S,S,I,S        ] │          1 │
      │ [S,I,S,S,S,S        ] │          2 │
      │ [S,S,I,I,S,S        ] │          1 │
      │ [S,S,I,S,S,S        ] │          2 │
      │ [S,S,S,S,S,S        ] │          1 │
      └─────────────────────┴────────────┘

      Total variables for (k=2, m=6): 41
    ============================================================
    (k=3, m=2)  —  2 shape(s)
    ============================================================

      Shape :singleton  |V|=1  |E|=0  |Aut|=1  state-classes=2
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I                  ] │          1 │
      │ [S                  ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P2  |V|=2  |E|=1  |Aut|=2  state-classes=3
      Edges: (1,2)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I                ] │          1 │
      │ [I,S                ] │          2 │
      │ [S,S                ] │          1 │
      └─────────────────────┴────────────┘

      Total variables for (k=3, m=2): 5
    ============================================================
    (k=3, m=3)  —  4 shape(s)
    ============================================================

      Shape :singleton  |V|=1  |E|=0  |Aut|=1  state-classes=2
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I                  ] │          1 │
      │ [S                  ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P2  |V|=2  |E|=1  |Aut|=2  state-classes=3
      Edges: (1,2)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I                ] │          1 │
      │ [I,S                ] │          2 │
      │ [S,S                ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P3  |V|=3  |E|=2  |Aut|=2  state-classes=6
      Edges: (1,2) (2,3)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I              ] │          1 │
      │ [I,I,S              ] │          2 │
      │ [I,S,I              ] │          1 │
      │ [I,S,S              ] │          2 │
      │ [S,I,S              ] │          1 │
      │ [S,S,S              ] │          1 │
      └─────────────────────┴────────────┘

      Shape :C3  |V|=3  |E|=3  |Aut|=6  state-classes=4
      Edges: (1,2) (2,3) (1,3)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I              ] │          1 │
      │ [I,I,S              ] │          3 │
      │ [I,S,S              ] │          3 │
      │ [S,S,S              ] │          1 │
      └─────────────────────┴────────────┘

      Total variables for (k=3, m=3): 15
    ============================================================
    (k=3, m=4)  —  10 shape(s)
    ============================================================

      Shape :singleton  |V|=1  |E|=0  |Aut|=1  state-classes=2
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I                  ] │          1 │
      │ [S                  ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P2  |V|=2  |E|=1  |Aut|=2  state-classes=3
      Edges: (1,2)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I                ] │          1 │
      │ [I,S                ] │          2 │
      │ [S,S                ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P3  |V|=3  |E|=2  |Aut|=2  state-classes=6
      Edges: (1,2) (2,3)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I              ] │          1 │
      │ [I,I,S              ] │          2 │
      │ [I,S,I              ] │          1 │
      │ [I,S,S              ] │          2 │
      │ [S,I,S              ] │          1 │
      │ [S,S,S              ] │          1 │
      └─────────────────────┴────────────┘

      Shape :C3  |V|=3  |E|=3  |Aut|=6  state-classes=4
      Edges: (1,2) (2,3) (1,3)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I              ] │          1 │
      │ [I,I,S              ] │          3 │
      │ [I,S,S              ] │          3 │
      │ [S,S,S              ] │          1 │
      └─────────────────────┴────────────┘

      Shape :P4  |V|=4  |E|=3  |Aut|=2  state-classes=10
      Edges: (1,2) (2,3) (3,4)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I,I            ] │          1 │
      │ [I,I,I,S            ] │          2 │
      │ [I,I,S,I            ] │          2 │
      │ [I,I,S,S            ] │          2 │
      │ [I,S,I,S            ] │          2 │
      │ [I,S,S,I            ] │          1 │
      │ [I,S,S,S            ] │          2 │
      │ [S,I,I,S            ] │          1 │
      │ [S,I,S,S            ] │          2 │
      │ [S,S,S,S            ] │          1 │
      └─────────────────────┴────────────┘

      Shape :K13  |V|=4  |E|=3  |Aut|=6  state-classes=8
      Edges: (1,2) (1,3) (1,4)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I,I            ] │          1 │
      │ [I,I,I,S            ] │          3 │
      │ [I,I,S,S            ] │          3 │
      │ [I,S,S,S            ] │          1 │
      │ [S,I,I,I            ] │          1 │
      │ [S,I,I,S            ] │          3 │
      │ [S,I,S,S            ] │          3 │
      │ [S,S,S,S            ] │          1 │
      └─────────────────────┴────────────┘

      Shape :paw  |V|=4  |E|=4  |Aut|=2  state-classes=12
      Edges: (1,2) (1,3) (2,3) (1,4)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I,I            ] │          1 │
      │ [I,I,I,S            ] │          1 │
      │ [I,I,S,I            ] │          2 │
      │ [I,I,S,S            ] │          2 │
      │ [I,S,S,I            ] │          1 │
      │ [I,S,S,S            ] │          1 │
      │ [S,I,I,I            ] │          1 │
      │ [S,I,I,S            ] │          1 │
      │ [S,I,S,I            ] │          2 │
      │ [S,I,S,S            ] │          2 │
      │ [S,S,S,I            ] │          1 │
      │ [S,S,S,S            ] │          1 │
      └─────────────────────┴────────────┘

      Shape :C4  |V|=4  |E|=4  |Aut|=8  state-classes=6
      Edges: (1,2) (2,3) (3,4) (4,1)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I,I            ] │          1 │
      │ [I,I,I,S            ] │          4 │
      │ [I,I,S,S            ] │          4 │
      │ [I,S,I,S            ] │          2 │
      │ [I,S,S,S            ] │          4 │
      │ [S,S,S,S            ] │          1 │
      └─────────────────────┴────────────┘

      Shape :K4me  |V|=4  |E|=5  |Aut|=4  state-classes=9
      Edges: (1,2) (2,3) (3,4) (4,1) (1,3)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I,I            ] │          1 │
      │ [I,I,I,S            ] │          2 │
      │ [I,I,S,I            ] │          2 │
      │ [I,I,S,S            ] │          4 │
      │ [I,S,I,S            ] │          1 │
      │ [I,S,S,S            ] │          2 │
      │ [S,I,S,I            ] │          1 │
      │ [S,I,S,S            ] │          2 │
      │ [S,S,S,S            ] │          1 │
      └─────────────────────┴────────────┘

      Shape :K4  |V|=4  |E|=6  |Aut|=24  state-classes=5
      Edges: (1,2) (1,3) (1,4) (2,3) (2,4) (3,4)
      ┌─────────────────────┬────────────┐
      │ Canonical state      │ Orbit size │
      ├─────────────────────┼────────────┤
      │ [I,I,I,I            ] │          1 │
      │ [I,I,I,S            ] │          4 │
      │ [I,I,S,S            ] │          6 │
      │ [I,S,S,S            ] │          4 │
      │ [S,S,S,S            ] │          1 │
      └─────────────────────┴────────────┘

      Total variables for (k=3, m=4): 65

## Variable-count summary

``` julia
println("| (k,m) | Shapes tracked                          | State classes | Variables |")
println("|-------|----------------------------------------|--------------|-----------|")
for (k, m) in [(2,2),(2,3),(2,4),(2,5),(2,6),(3,2),(3,3),(3,4)]
    closure = MotifClosure(k, m)
    shapes  = NodeBasedModels.enumerate_shapes(closure)
    names   = join([":$(sh.name)" for sh in shapes], " + ")
    nvars   = sum(length(NodeBasedModels.enumerate_state_classes(sh, [:S,:I]))
                  for sh in shapes)
    nclass  = nvars
    println("| (k=$k,m=$m) | $(rpad(names,38)) | $(lpad(nclass,12)) | $(lpad(nvars,9)) |")
end
```

    | (k,m) | Shapes tracked                          | State classes | Variables |
    |-------|----------------------------------------|--------------|-----------|
    | (k=2,m=2) | :singleton + :P2                       |            5 |         5 |
    | (k=2,m=3) | :singleton + :P2 + :P3                 |           11 |        11 |
    | (k=2,m=4) | :singleton + :P2 + :P4                 |           15 |        15 |
    | (k=2,m=5) | :singleton + :P2 + :P5                 |           25 |        25 |
    | (k=2,m=6) | :singleton + :P2 + :P6                 |           41 |        41 |
    | (k=3,m=2) | :singleton + :P2                       |            5 |         5 |
    | (k=3,m=3) | :singleton + :P2 + :P3 + :C3           |           15 |        15 |
    | (k=3,m=4) | :singleton + :P2 + :P3 + :C3 + :P4 + :K13 + :paw + :C4 + :K4me + :K4 |           65 |        65 |

## Shape diagrams

``` julia
all_names = [:singleton, :P2, :P3, :C3, :P4, :K13, :paw, :C4, :K4me, :K4]
for nm in all_names
    println("$(rpad(string(nm),10))  $(shape_ascii(nm))")
    println()
end
```

    singleton   ●

    P2          ●─●

    P3          ●─●─●

    C3          ●─●
     └─●

    P4          ●─●─●─●

    K13           ●
    ●─●─●
      ●

    paw         ●─●
     │╲
     ●─●

    C4          ●─●
     │ │
     ●─●

    K4me        ●─●
     │╲│
     ●─●

    K4          ●─●
     │╳│
     ●─●

## Notes on the Lean-certified obstruction at (k=3, m=4)

The `(k=3, m=4)` implementation is intentionally documented as a
boundary case, not as a guaranteed monotone refinement of `m=3`. Lean
theorem **T3b** in
`EdgeBasedModels.jl/proofs/EBCMCategory/MarginalisationCharacterization.lean`
certifies that the current order-4 Kirkwood RHS need not marginalise to
a better order-3 RHS; theorem **T7** gives the corresponding small-time
trajectory lower bound. In practical terms, simply adding the 4-vertex
variables does not force the projected dynamics to improve the 3-vertex
closure on the random 3-regular benchmark.

This does **not** prove that every possible `(k=3, m=5)` or higher
closure is impossible. It says that further `k=3` extensions require new
analysis (for example, a non-Kirkwood or constrained-consistency
closure) rather than a mechanical “more motifs is always better”
interpretation.

The obstruction is a categorical fact about the shape of the
*marginalisation functor* on the poset of induced subgraphs, not a
numerical accident. See Vignette 11 for the live numeric witness and the
formal statement.

## References

- Keeling M.J., House T., Cooper A.J., Pellis L. (2016). Systematic
  Approximations to Susceptible-Infectious-Susceptible Dynamics on
  Networks. *PLoS Comput. Biol.* 12(9): e1005296.
  <https://doi.org/10.1371/journal.pcbi.1005296>
- Sharkey K.J. (2011). Deterministic epidemic models on contact
  networks: Correlations and unbiological terms. *Theor. Pop. Biol.*
  79(4):115–129.
- Kirkwood J.G. (1935). Statistical mechanics of fluid mixtures. *J.
  Chem. Phys.* 3:300–313.

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
