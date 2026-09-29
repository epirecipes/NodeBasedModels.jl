# API reference

NodeBasedModels re-exports the NetworkEpiCore names it uses (`ContactModel`, `sir_model`,
`ConfigurationNetwork`, `SeedFraction`, `solve_epidemic`, `model_curves`, …), which are
documented in NetworkEpiCore. The docstrings below are NodeBasedModels' own, including its
methods of the NetworkEpiCore generics.

## `node_based` and the population-level systems

```@autodocs
Modules = [NodeBasedModels]
Pages   = ["NodeBasedModels.jl", "lift.jl", "compat.jl", "population_pairwise.jl", "closures.jl",
           "closures_pgf.jl", "s_anchored.jl", "wellmixed.jl", "multitype_pairwise.jl"]
```

## Graph-level models

```@autodocs
Modules = [NodeBasedModels]
Pages   = ["individual_based.jl", "pair_based.jl"]
```

## SIS closures: reinfection counting, motifs, neighbourhoods

```@autodocs
Modules = [NodeBasedModels]
Pages   = ["reinfection_counting.jl", "motif_based.jl", "motif_symbolic.jl",
           "neighbourhood_based.jl", "neighbourhood_symbolic.jl"]
```

## Analysis

```@autodocs
Modules = [NodeBasedModels]
Pages   = ["analysis.jl"]
```

## The 0.1 model and network types

```@autodocs
Modules = [NodeBasedModels]
Pages   = ["compartments.jl", "networks.jl"]
```

## Deprecated

```@autodocs
Modules = [NodeBasedModels]
Pages   = ["gillespie.jl", "deprecated.jl"]
```

## Index

```@index
Modules = [NodeBasedModels]
```
