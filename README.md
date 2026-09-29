# NodeBasedModels.jl

Node-level deterministic models of epidemics on networks. The package builds population pairwise
models with a choice of triple closure, the exact PGF closure of the S-anchored pairwise model,
mean-field (mass-action) models, individual- and pair-based models on an explicit graph,
reinfection-counting lifts, and the SIS motif and neighbourhood closures. Every system is a
ModelingToolkit system.

Version 0.2 is built on [NetworkEpiCore.jl](../NetworkEpiCore.jl) (NEC), whose names it
re-exports. A model is a NEC `ContactModel`, a network is a NEC `NetworkDescriptor`, and the
verb is `node_based(model, network; closure, level)`. The same objects go to
[EdgeBasedModels.jl](../EdgeBasedModels.jl) (`edge_based`) and to
[NetworkOutbreaks.jl](../NetworkOutbreaks.jl) (`simulate`), and the three packages load
together without name clashes. Unlike the edge-based lift, `node_based` is total on the
node-level type theory T_net: SIS and SIRS are accepted. Changes from 0.1 are in
[MIGRATION.md](MIGRATION.md).

## Quick start: the low-level API

```julia
using NetworkEpiCore, NodeBasedModels, Catalyst

seir = @reaction_network seir begin
    @parameters τ σ γ
    τ, S + I --> E + I     # contact: per-contact (per-edge) rate τ
    σ, E --> I
    γ, I --> R
end
model = contact_model(seir)                     # the same ContactModel as seir_model()
net   = ConfigurationNetwork(PoissonDegree(5))
p     = Dict(:τ => 1/6, :σ => 1/5, :γ => 1/4)

sys = node_based(model, net)                    # PairwiseSystem, closure constant K = ⟨k(k−1)⟩/⟨k⟩² = 1
sol = solve_epidemic(sys; p, initial = SeedFraction(:E => 0.01), tspan = (0.0, 150.0))
mc  = model_curves(sys, sol; t = 0:1:150)       # S, E, I, R, :infectious, :cumulative
mc.values[:cumulative][end]                     # 0.8002: the edge-based final size, since Poisson is PT
```

Seeding defaults to the model's **unique entry state** (E for SEIR, I for SIR). The initial pairs
are the random arrangement, which is the image of the edge-based initial condition. The 0.1
entry points are still available, and they also accept `ContactModel`s and descriptors.
`generate_pairwise(model, network, closure)` keeps its signature. With `cumulative = true`
it gives the same vector field as `node_based`:
`vector_fields_equal(symbolic_ode(sys), symbolic_ode(generate_pairwise(seir_model(), net, default_closure(net); cumulative = true)))`
is `true`.

### Levels and closures

| call | model | notes |
|---|---|---|
| `node_based(m, ConfigurationNetwork(d))` | population pairwise, `BernoulliClosure` (constant K) | total on T_net; exact **iff** d is Poisson-type (M8) |
| `node_based(m, net; closure = KeelingClosure())`, `BarnardClosure()`, `PowerClosure()` | pairwise with clustering closures | `ClusteredNetwork` defaults to Keeling |
| `node_based(m, net; closure = PGFClosure())` | S-anchored pairwise with K_ψ = ψψ''/ψ'² | exact for every degree PGF on T_EB models (M6) |
| `node_based(m, net; level = :s_anchored)` | the S-anchored subsystem PW^S (an `SAnchoredSystem`) | T_EB models only |
| `node_based(m, WellMixed(κ))` | `MeanFieldSystem` (`MeanFieldClosure`) | equals `mass_action(m; κ)` |
| `node_based(m, MultitypeNetwork(…))` | multitype pairwise (`MultitypePairwiseSystem`) | `:sir_sbm2`, `:sir_age2` |
| `node_based(m, ExplicitGraph(g); level = :individual \| :pair, p)` | individual-based (NIMFA) and pair-based (Kirkwood) models on one graph | `:pair` is SIR only; exact on trees |
| `node_based(m, net; level = :motif, closure = MotifClosure(k, m))` | SIS motif closures (Keeling, House, Cooper & Pellis 2016) | regular networks |
| `node_based(m, net; level = :neighbourhood, n = 2)` | SIS neighbourhood (ego-network) model | regular networks |
| `with_reinfection_counting(sis_model(), L)`, then `node_based` | reinfection-counting pairwise lift | `reinfection_totals`, `reinfection_histogram` |

`EamesClosure` and `mass_action(::PairwiseSystem)` raise errors with migration messages: the first
was never implemented, and the second was not a mass-action model. `gillespie_sir` and
`gillespie_sis` are deprecated, and will be removed in 0.3; use NetworkOutbreaks' `simulate`.

## Morphisms and exactness

- **EB → S-anchored pairwise (M6), exact.** For every C² degree PGF ψ and every T_EB model,
  EdgeBasedModels' `pairwise_image(edge_based(m, net))` is a semiconjugacy onto PW^S with the
  closure K_ψ. The `PGFClosure` system has exactly that vector field. On `:sir_bim`,
  `vector_fields_equal(pairwise_image(eb).ode, symbolic_ode(node_based(m, net; level = :s_anchored, closure = PGFClosure())))`
  is `true`.
- **PT ⇔ constant closure (M8).** The constant closure K = ⟨k(k−1)⟩/⟨k⟩² is exact **iff**
  ψ' = αψ^κ (Poisson, binomial, regular, negative binomial). NEC provides `closure_constant(d)`
  and `is_poisson_type(d)`. For example, `NegBinDegree(mean = 4, var = 8)` gives K = 1.25 and
  `(α = 4.0, κ = 1.25)`. On any other degree distribution the constant closure is **biased**,
  and the N-scaling protocol below shows the bias.
- **Well-mixed.** `vector_fields_equal(symbolic_ode(node_based(m, WellMixed(5))), mass_action(m; κ = 5))`
  is `true` for SIR.
- **Composition is lax.** Pairwise models do not commute with gluing. A contact drains [Zs] for
  every Z, so the right order is to compose the syntax first and then lift it (vignette N08,
  which checks the failure numerically at a witness state; no Lean theorem is cited for it).
- **Motif m = 4.** The k = 3, m = 4 motif closure had a bookkeeping bug (verified issue B05),
  which is now fixed; it now improves on m = 3. The 0.1 docs blamed a Lean "marginalisation
  obstruction" for the old error. That was wrong, and the claim is withdrawn.

## Validation

The references are the committed NetworkOutbreaks ensembles of the shared NEC scenarios
(`NetworkOutbreaks.jl/data/scenarios`). Each has N = 10⁴, 200 runs and a fresh graph per run,
and is hash-keyed and loaded in strict-cache mode. The comparison code is the same as on the
EdgeBasedModels pages:

```julia
using NetworkOutbreaks
sc  = scenario(:sir_bim); ref = scenario_summary(sc)
for (label, kw) in (("constant K", (;)), ("PGF closure", (; closure = PGFClosure())))
    sys = node_based(sc; kw...); sol = solve_epidemic(sys, sc)
    display(compare(ref, model_curves(sys, sol; t = sc.tgrid, label)))
end
```

The table below shows values printed by this code. D∞ is max_t |I_det − Ī| over the prevalence
of infectious nodes, and ΔR∞ = R_det − mean final size. The rule for an exact limit is
D∞ < 0.005 and |ΔR∞| < 0.005 at N = 10⁴. The **N-scaling protocol** repeats a scenario at
N = 10³ (2000 runs) and N = 10⁵ (20 runs). An exact limit's error keeps falling as N grows,
while a structural bias levels off.

| scenario | closure | N = 10³: D∞ / ΔR∞ | N = 10⁴: D∞ / ΔR∞ (95% CI) | N = 10⁵: D∞ / ΔR∞ |
|---|---|---|---|---|
| `:sir_pois5` (Poisson, PT) | constant K | – | 0.00223 / +0.00012 (−0.00085, 0.00109) | – |
| `:sir_bim` (bimodal) | constant K | 0.02504 / +0.03813 | 0.01006 / **+0.03344** (0.03175, 0.03512) | 0.00894 / **+0.03344** |
| `:sir_bim` | PGF closure | 0.01765 / +0.00435 | 0.00232 / −0.00034 (−0.00203, 0.00134) | 0.00094 / −0.00034 |
| `:sir_pl` (power law) | constant K | – | 0.02173 / **+0.07013** (0.06786, 0.07241) | 0.01997 / **+0.06814** |
| `:sir_pl` | PGF closure | – | 0.00265 / +0.00173 (−0.00054, 0.00400) | 0.00140 / −0.00027 |

For constant K the bias in the final size, ΔR∞ ≈ +0.033 on `:sir_bim` and ≈ +0.07 on `:sir_pl`,
does not shrink as N grows. The PGF closure's error keeps shrinking toward the Monte Carlo
floor. Vignette N13 runs every scenario that declares a node-based back end; the `:pgf_closure`
back end is declared exact on 18 of them.

SIS has no exact finite closure. On `:sis_reg3` (3-regular, τ = 1/2, γ = 1/4), vignette N12 prints
the following D∞(I) against the ensemble:

| model | pairwise | reinfection L = 1 | motif m = 2 | motif m = 3 | motif m = 4 | neighbourhood n = 2 |
|---|---:|---:|---:|---:|---:|---:|
| D∞(I) | 0.1456 | 0.0029 | 0.1456 | 0.0601 | 0.0089 | 0.0253 |

## Vignettes

The Quarto pages are in [`vignettes/`](vignettes/). They are rendered with the strict cache,
every number is printed by code on the page, and Lean is cited only from NetworkEpiCore's
`CITABLE.txt`. N01's shared first cell is the same as EdgeBasedModels' E01, apart from the
back-end lines.

| page | topic |
|---|---|
| [N01](vignettes/N01_reaction_network_to_pairwise/index.md) | from a reaction network to a pairwise model |
| [N02](vignettes/N02_closures_degree_heterogeneity/index.md) | closures and degree heterogeneity (constant K vs PGF closure) |
| [N03](vignettes/N03_natural_history/index.md) | natural history in pairwise models |
| [N04](vignettes/N04_graph_level/index.md) | graph-level models: individual- and pair-based |
| [N05](vignettes/N05_clustering/index.md) | clustering |
| [N06](vignettes/N06_mass_action/index.md) | back to mass action |
| [N07](vignettes/N07_thresholds/index.md) | thresholds and R₀: SIR versus SIS |
| [N08](vignettes/N08_pairwise_composition/index.md) | why pairwise composition is lax |
| [N09](vignettes/N09_sis_reinfection/index.md) | SIS: pairwise and reinfection counting |
| [N10](vignettes/N10_motif_closures/index.md) | motif closures |
| [N11](vignettes/N11_neighbourhood/index.md) | neighbourhood model (n = 2) |
| [N12](vignettes/N12_sis_comparison/index.md) | all SIS approximations against simulation |
| [N13](vignettes/N13_motif_catalogue_validation/index.md) | motif catalogue and validation methodology |
| [N14](vignettes/N14_multitype_pairwise/index.md) | multitype pairwise models |

## Lean proofs

The citable mathematics is in **`NetworkEpiCore.jl/proofs`** (namespace `NEP`), gated by its
axiom gate and listed in its `CITABLE.txt`; the vignettes cite only names whose alignment
claims pass SA-PASS and were rated strong in the spot check. The results relevant here are `NEP.eb_to_pws` (M6),
`NEP.pt_iff_const_closure` (M8), `NEP.eb_to_pws_pt_any` and `NEP.eb_to_pws_pt_closure_one`
(the constant closure is exact on Poisson-type networks). That pairwise gluing is not strict is
checked numerically (vignette N08), not cited from Lean.
The status of the alignment audit is in the NetworkEpiCore README.

`proofs/` in this repository (`PairwiseProofs`: closure conditions and invariant regions of the
pair equations) is a standalone Lake project. It has no `sorry` and no `axiom`. It is not part
of the NetworkEpiCore trusted library, the SA-PASS alignment audit has not been run on it, and
the vignettes do not cite it.

## Installation

The packages are developed side by side and are not registered yet:

```julia
using Pkg
Pkg.develop([PackageSpec(path = "NetworkEpiCore.jl"), PackageSpec(path = "NodeBasedModels.jl")])
```

Catalyst is needed only for `contact_model(::ReactionSystem)`, and NetworkOutbreaks only for
`scenario_summary`.

## Pair-counting convention

A cross pair [XY] (X ≠ Y) counts each XY edge once, and a self pair [XX] counts each XX edge
twice, so 2Σ_{X≠Y}[XY] + Σ_X[XX] = ⟨k⟩N.

## License

MIT
