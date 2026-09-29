# Exactness and validation

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
  tested numerically; the Lean witness `NEP.pwS_glue_not_strict_witness` awaits its SA-PASS
  re-check and is not cited as aligned).
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

