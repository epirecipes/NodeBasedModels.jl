# Migrating to NodeBasedModels 0.2

NodeBasedModels 0.2 is built on **NetworkEpiCore** (NEC), the core that EdgeBasedModels and
NetworkOutbreaks share. The model you pass is a NEC `ContactModel`, the network is a NEC
network descriptor, and the verb is `node_based`. Most 0.1 code keeps running: the old names
forward with a deprecation warning. The exceptions are old functions that gave wrong numbers,
which now throw an error with a migration message. The behaviour changes are listed below. Run
Julia with `--depwarn=yes` (the default under `Pkg.test`) to see every deprecated call.

```julia
using NetworkEpiCore, NodeBasedModels            # NodeBasedModels re-exports the NEC names it uses

cm  = seir_model()                               # a ContactModel: S + I → E + I at τ, E → I, I → R
net = ConfigurationNetwork(PoissonDegree(5))
sys = node_based(cm, net)                        # heterogeneous pairwise, K = ⟨k(k−1)⟩/⟨k⟩² = 1
sol = solve_epidemic(sys; p = Dict(:τ => 1/6, :σ => 1/5, :γ => 1/4), tspan = (0.0, 150.0))
model_curves(sys, sol; t = 0:1:150)              # NEC ModelCurves: S, E, I, R, :infectious, :cumulative

sc  = scenario(:seir_pois5)                      # or everything from a shared scenario
sys = node_based(sc); sol = solve_epidemic(sys, sc)
```

## Names

| 0.1 | 0.2 | Behaviour in 0.2 |
|---|---|---|
| `sir_model()`, `sis_model()`, `seir_model()`, `sirs_model()` returned a `CompartmentalModel` | the same names now return a NEC `ContactModel` (the NEC binding, re-exported) | **changed type**. Every builder accepts it. `CompartmentalModel(sir_model())` gives the old model, named `:sir` rather than `:SIR` |
| `node_sir_model` etc. (aliases of the above) | `sir_model` etc. | depwarn; returns the 0.1 `CompartmentalModel` (named `:SIR`) |
| `generate_pairwise(model, network, closure)` | `node_based(model, net; closure)` | kept. It also accepts a `ContactModel` and a network descriptor. `node_based` adds the `cumulative` state and `generate_pairwise` does not |
| `model_from_catalyst(rn; infectious)` | `contact_model(rn)` (NEC Catalyst extension; `using Catalyst`) | depwarn; returns `CompartmentalModel(contact_model(rn))`. It warns if 0.1 would have built a different model (see B04 below). If `infectious` names a set other than the catalysts of the contacts, it throws an error |
| `GraphNetwork(g)`, `GraphNetwork(g; transmission_rate, transmission_matrix)`, `GraphNetwork(g, T)` | `ExplicitGraph(g)` (NEC): `node_based(model, ExplicitGraph(g); level = :individual \| :pair, p)`, `generate_individual_based(model, ExplicitGraph(g); …)` | depwarn. A per-edge rate matrix is `GraphNetwork(ExplicitGraph(g); transmission_matrix = T)` |
| `with_reinfection_counting(::CompartmentalModel, L)` | `with_reinfection_counting(sis_model(), L)` (NEC, on a `ContactModel`), then `node_based` | depwarn; returns the 0.1 lift. A model with per-infector transitions is lifted through NEC, which keeps `via` |
| `base_compartment_of(:S_3)`, `infection_count_of(:S_3)` | the same (now NEC bindings); prefer `base_compartment_of(cm, X)` / `infection_count_of(cm, X)`, which read labels, not names | unchanged |
| `reinfection_totals(psys, sol)` | the same | for a system built from a `ContactModel`, the lumping reads the NEC labels. For a 0.1 lift it parses names, as before |
| `EamesClosure()` | `KeelingClosure()` or `BarnardClosure()` | **error** when used ("removed in NodeBasedModels 0.2"): the closure was never implemented or validated |
| `mass_action(::PairwiseSystem)` | `node_based(cm, WellMixed(κ); closure = MeanFieldClosure())`, `NetworkEpiCore.mass_action(cm; κ)` | **error** with that message |
| `mean_degree`, `excess_degree`, `compartment`, `compartments`, `population_fraction`, `basic_reproduction_number`, `epidemic_threshold`, `early_growth_rate`, `disease_free_equilibrium`, `default_initial_conditions`, `solve_epidemic`, `reinfection_histogram` | the same names, now NEC generics with NodeBasedModels methods | `using EdgeBasedModels, NodeBasedModels` no longer reports clashes for these names |

Catalyst is no longer a dependency of NodeBasedModels. Load it yourself (`using Catalyst`),
which also loads the NEC Catalyst front end `contact_model(rn)`.

## Behaviour changes

- **Seeding (DESIGN §E.2).** The default initial condition seeds the **entry state** of the
  infections: the target of the infection transitions out of the background compartment. That
  is I for SIR, **E for SEIR** (0.1 seeded I, the first infectious compartment) and I₁ for a
  reinfection-counted or staged model. `seed_state = :first_infectious` restores the 0.1 rule.
  A model with several entry states (two strains) needs an explicit `initial`:
  `generate_pairwise(…; initial = SeedFraction(:I1 => 0.005, :I2 => 0.005))`. The SEIR goldens
  were replaced for this change, and `seed_state = :first_infectious` reproduces the 0.1
  numbers exactly.
- **Initial pairs.** They are the random arrangement [XY] = ⟨k⟩N x_X x_Y. This is the image under
  π^PW of the edge-based initial condition θ = 1, φ_X = pop_X = ρ_X: [s] = q, [sX] = ⟨k⟩qρ_X,
  [ss] = ⟨k⟩q². The pair convention is unchanged: a cross pair [XY] counts each XY edge once,
  and a self pair [XX] counts each XX edge twice.
- **Per-infector rates (verified issue B04).** `Transition` has a new field
  `via::Vector{Symbol}`, the infectors of an infection transition:
  `Transition(:S, :I1, :τ1, :infection; via = [:I1])`. Empty `via` means every infectious
  compartment, which is the 0.1 behaviour. 0.1 applied every infection rate to every infectious
  compartment, so two infection transitions at τ1 and τ2 gave
  −(τ1 + τ2)([S I1] + [S I2]) in d[S]/dt instead of −τ1[S I1] − τ2[S I2]. `contact_model(rn)`
  and `CompartmentalModel(cm)` set `via` from the catalysts of the contacts.
- **Rates.** `Transition.rate` may be a parameter name, a non-negative number, an `Expr`
  (NEC `RATE_OPS`, with `t` for time) or a symbolic expression. 0.1 accepted only a `Symbol`, and
  `model_from_catalyst` turned expressions into invented parameter names. Contact rates follow the
  model's NEC rate convention: `node_based` converts frequency- or density-dependent rates to
  per-contact rates τ with `per_contact_rates(cm, net)`, using the mean degree of the network.
- **Validation.** A `Transition` must have type `:infection` or `:spontaneous`, and `via` only
  on infections. A transition `X → X` is rejected (0.1 silently drained X). Rates must be ≥ 0.
  A `CompartmentalModel` needs unique compartment names and infectious `via` compartments.
  `GraphNetwork` rate matrices must be N × N, finite and ≥ 0.
- **`GraphNetwork(g; transmission_rate = 1.0)` (verified issue B04).** In 0.1 an explicit rate
  of 1.0 was taken as "not given", so the solver's `infection_rate` (default 0.5) was used. It
  is now kept, and `Int` rates are accepted.
- **`population_fraction(psys, sol, X)`** divides by the population scale the system was built
  with. The default is `N = 1`, where the result is unchanged. 0.1 returned counts unless `N`
  was passed. It also sums the refinements of X (reinfection counts, stages, strata) when X is
  a base compartment.
- **`PairwiseSystem`** has a new field `metadata` (the lowered `CompartmentalModel`, the
  per-contact `ContactModel`, the descriptor, the seeding, the uncompiled equations and so on).
  `psys.model` is the model as passed, which may be a `ContactModel`. There are new methods
  `solve_epidemic(psys; p, initial, tspan, saveat)`, `solve_epidemic(psys, sc::Scenario)`,
  `default_initial_conditions(psys; initial, seed_fraction, seed_state)`, `model_curves`,
  `symbolic_ode` and `compartments`. `solve_pairwise` takes `u0` and `tspan` overrides.
  `default_initial_conditions` defaults to the seed fraction and `seed_state` the system was
  built with. `solve_epidemic(psys; p)`, like `solve_pairwise`, throws an error for a parameter
  name that is not a parameter of the model.
- **The `cumulative` accumulator** (new; `node_based` adds it, `generate_pairwise(…; cumulative =
  true)` on request) uses the infection status of NetworkOutbreaks' `final_size`, defined
  structurally (DESIGN §J.8). It starts at the seeds in *infected* compartments and counts
  every entry into an infected compartment from a non-infected one. E and I of SEIR are
  infected. S, R, a vaccinated V and a traced Q are not, so `SeedFraction(:I => 0.01, :V => 0.3)`
  gives `cumulative(0) = 0.01`. In a superinfection model (`I1 + I2 → I12 + I2`) the recipient
  I1 is infected, not susceptible.
- **Graph-level builders.** `generate_individual_based`, `generate_pair_based` and
  `generate_neighbourhood` accept a `ContactModel` and an `ExplicitGraph`.
  `node_based(…; level = :individual)` refuses models that the 0.1 individual-based builder gets
  wrong: several spontaneous rates (verified issue B02), per-infector contacts, or an entry state
  other than the first infectious compartment.
- **Reinfection counting on a `ContactModel`** (NEC) drops refined compartments that no node
  can reach, so SIR gets no `S_1`, …. For SIS and SIRS the compartments and dynamics are those
  of 0.1.

## Threshold analysis (`basic_reproduction_number`, `epidemic_threshold`, `early_growth_rate`)

- **SIR and SIS are separate (verified issue B01).** On a `HeterogeneousNetwork` the keyword
  `dynamics = :SIR` or `:SIS` is required (0.1 returned the SIS R₀ and threshold but the SIR
  growth rate); on a `HomogeneousNetwork` it defaults to `:SIR`. The model-based methods read
  the dynamics from the model.
- **Keeling's closure takes N = Σ_X [X] from the state (verified issue B03)**, not the keyword N,
  so a system built on a network structure whose N differs from the keyword is consistent.
- **Keeling's SIS threshold on strongly clustered heterogeneous networks** is now γqϕ/⟨k⟩² when
  qϕ(q(1 − ϕ) + ⟨k⟩) > ⟨k⟩² (0.1 returned `Inf` there, and a wrong-branch growth rate).
- **Keeling's SIR `epidemic_threshold` throws an `ArgumentError` when q(1 − ϕ) < 1 < qϕ/⟨k⟩**:
  the fast variables are bistable and the seeded growth is not monotone in τ, so there is no
  threshold (0.1 returned a number, e.g. 6.919, while the ODE already grows at τ = 0.69).
  `early_growth_rate` still gives the rate from a random seed.
- **Barnard's closure in `generate_pairwise`** evaluates the infection triple with the infector
  last ([A S I], thesis eq. 4.23). 0.2 builds before this fix used the reversed order, which is
  not Barnard's closure (SIR threshold 0.3279 instead of 0.2995 at n = 6, ϕ = 0.3); Barnard
  systems now agree with `epidemic_threshold` / `early_growth_rate` on them.
- **Barnard's SIR growth rate below the threshold** is the asymptotic decay rate of [I] from
  the product-state seed (the recovered block keeps the history of the early epidemic). Where
  (n − 1)(1 − ϕ) < 1 this is −γ; 0.1 returned the continued supercritical branch, O(γ) off
  there (−0.791 instead of −1 at n = 3, ϕ = 0.9, τ = γ = 1). Supercritical rates and the
  thresholds are unchanged.

## Graph-level and stochastic simulators

- **`gillespie_sir`, `gillespie_sir_average` and `gillespie_sis` are deprecated** (removed in
  0.3); use NetworkOutbreaks' `simulate`. JumpProcesses is now a weak dependency: the
  `gillespie_sir` kernel lives in the extension `NodeBasedModelsJumpProcessesExt`, which loads
  with JumpProcesses (normally always, since ModelingToolkit loads it; otherwise
  `using JumpProcesses`, or the call throws an `ArgumentError` saying so).
- **`PairBasedResult`**: `node_state` and `aggregate` throw an `ArgumentError` for an unknown
  compartment (0.1 returned 1 − S − I).
- **`IndividualBasedResult`**: `sol.prob.p` now holds the right-hand side's work buffers (0.1
  had `NullParameters`). To solve copies concurrently, use `remake(prob; p = deepcopy(prob.p))`.

## Motif closures (`motif_based_sis`)

- **k = 3, m = 4 is fixed (verified issue B05)**: the per-shape Kirkwood flow no longer carries
  the spurious slot factor n_ext/k, it sums every completion of the anchor by the external
  neighbour, and the 3-from-4 multiplicities come from the host counts (they were the constants
  5 and 3). The numbers change and depend on the supplied host counts; m = 4 now improves on
  m = 3 against SSA (0.1: sup error 0.35; now about 0.012 on the vignette host). The m = 4
  warning is removed. The 0.1 attribution of the m = 4 error to a Lean "Kirkwood
  marginalisation obstruction" was wrong.
- **k = 2, m = 3 changes too**: the single-vertex anchor now has the slot factor 1/2 at each end
  (0.1 doubled external infection; on a ring, N = 1000, the sup error against SSA drops from
  0.318 to 0.153).
- **`build_motif_symbolic_rhs`** has a keyword `multiplicities = (P3 = 5.0, C3 = 3.0)` (the
  3-from-4 divisors at k = 3, m = 4); pass `NodeBasedModels._mat_3from4_multiplicities(; counts...)`
  to match a `motif_based_sis` system built with host counts.
- **`induced_subgraph_counts_4vertex`** throws an `ArgumentError` on a directed graph.
- **`motif_based_sis(k = 3, m = 4; n_c3 > 0)` without `n_paw`, `n_k4me` or `n_k4`** throws an
  `ArgumentError` (0.1 silently froze the triangle variables); pass the host counts, e.g. from
  `induced_subgraph_counts_4vertex(g)`.

## Versions

NodeBasedModels 0.2, EdgeBasedModels 0.2 and NetworkOutbreaks 0.2 are released together with
NetworkEpiCore 0.1. NodeBasedModels accepts NetworkOutbreaks 0.1 and 0.2 (a test dependency).
