# Why pairwise composition is lax


- [What this page shows](#what-this-page-shows)
- [The shared first cell](#the-shared-first-cell)
- [Compose, then lift: the glued
  SEIR](#compose-then-lift-the-glued-seir)
- [Lift, then compose: the parts cannot be
  lifted](#lift-then-compose-the-parts-cannot-be-lifted)
- [The F2 counterexample: two strains glued along S and
  R](#the-f2-counterexample-two-strains-glued-along-s-and-r)
- [The S-anchored pairwise model](#the-s-anchored-pairwise-model)
- [Stratification commutes with gluing, then
  lift](#stratification-commutes-with-gluing-then-lift)
- [Summary](#summary)
- [References](#references)

## What this page shows

NetworkEpiCore builds models from parts: `open_model` exposes some
species of a reaction network as legs, `glue` identifies legs and
concatenates the reaction lists (a pushout), and `stratify` copies a
model over node types. For the edge-based lift the order does not
matter: the lift of a glued model is the sum of the lifts of the parts
(law H1 of DESIGN §D.6, shown on the mirrored page E12). This page shows
the pairwise side:

1.  **compose, then lift** works. The pairwise model of the glued SEIR
    of `:seir_pois5` is the pairwise model of the canned SEIR, and
    stratifying a glued SIR on the typed network of `:sir_sbm2` gives
    the pairwise model of that scenario;
2.  **lift, then compose** does not. The parts of SEIR cannot be lifted
    on their own, and when they can (two SIR strains), the pairwise
    field of the glued model has terms that neither part has. In the
    glued model a contact of one part drains pairs that involve the
    species of the other part, through closed triples. This is failure
    **F2** of DESIGN §D.6: pairwise models are *lax* under gluing, not
    strict;
3.  the same failure for the S-anchored pairwise model PW^S, evaluated
    at a witness state.

The page compares with simulation only once, for the glued SEIR. It
needs no new summaries.

## The shared first cell

This cell is the same as the first cell of EdgeBasedModels’ page E12,
apart from the back-end section.

``` julia
include(joinpath(@__DIR__, "..", "_shared", "setup.jl"))
using NetworkEpiCore, NetworkOutbreaks, Catalyst, Plots
transmission = @reaction_network transmission begin
    @parameters τ
    τ, S + I --> E + I     # contact: S converted to E, I unchanged
end
history = @reaction_network history begin
    @parameters σ γ
    σ, E --> I             # latency
    γ, I --> R             # recovery
end
tr = open_model(contact_model(transmission); legs = [[:S], [:E, :I]])
pr = open_model(contact_model(history); legs = [[:E, :I, :R]])
seir = glue(tr, pr; on = [:E, :I])     # pushout along E and I, reactions concatenated
sc   = scenario(:seir_pois5)           # SEIR on Poisson(5), τ = 1/6, σ = 1/5, γ = 1/4, 1% seeds in E
@assert isequivalent(seir, sc.model)
ref  = scenario_summary(sc)
# --- NBM vignette ---------------------------------------------------------------
using NodeBasedModels
sys  = node_based(seir, sc.network)                        # low level: the glued open model
sysF = generate_pairwise(seir_model(), sc.network, default_closure(sc.network); cumulative = true)  # factory
@assert vector_fields_equal(symbolic_ode(sys), symbolic_ode(sysF))
seir
```

    OpenContactModel :transmission_history  (source: transform; method: explicit; rates: PerContact)
      species       S (Sus)   E   I   R
      contacts      [1] S + I → E + I    τ    contact     infector I, entry E
      transitions   [2] E → I            σ    progress
                    [3] I → R            γ    progress
      typing        T_EB  ⇒  edge_based ✓  s_anchored ✓  pairwise ✓  individual ✓  pair ✓  stochastic ✓  mass_action ✓
      assumptions   glue of :transmission, :history along E, I: pushout of the species; reactions concatenated, the rates of identical reactions added
                    Sus = the union of the parts' susceptible classes = {S}
                    transmission: Sus inferred as recipients \ contact products = {S}
                    history: Sus inferred as recipients \ contact products = {}
      legs          [[S], [E, I], [E, I, R]]

## Compose, then lift: the glued SEIR

The pairwise model of the glued SEIR has the singles \[S\], \[E\],
\[I\], \[R\] and one pair per unordered pair of compartments. Every
triple is closed with the constant K = ⟨k(k − 1)⟩/⟨k⟩², which is 1 on
the Poisson(5) network:

``` julia
(; closure = default_closure(sc.network), K = closure_constant(sc.network.degrees))
```

    (closure = BernoulliClosure(), K = 1.0)

``` julia
symbolic_ode(sys)
```

    SymbolicODE :pairwise_transmission_history (15 states)
      dS/dt = -SI(t)*τ
      dE/dt = -E(t)*σ + SI(t)*τ
      dI/dt = E(t)*σ - I(t)*γ
      dR/dt = I(t)*γ
      dSS/dt = -10.0ifelse((5.0S(t)) == 0, 0, (SI(t)*SS(t)) / (5.0S(t)))*τ
      dSE/dt = -SE(t)*σ + 5.0ifelse((5.0S(t)) == 0, 0, (SI(t)*SS(t)) / (5.0S(t)))*τ - 5.0ifelse((5.0S(t)) == 0, 0, (SE(t)*SI(t)) / (5.0S(t)))*τ
      dSI/dt = SE(t)*σ - SI(t)*γ - SI(t)*τ - 5.0ifelse((5.0S(t)) == 0, 0, (SI(t)^2) / (5.0S(t)))*τ
      dSR/dt = SI(t)*γ - 5.0ifelse((5.0S(t)) == 0, 0, (SI(t)*SR(t)) / (5.0S(t)))*τ
      dEE/dt = -2EE(t)*σ + 10.0ifelse((5.0S(t)) == 0, 0, (SE(t)*SI(t)) / (5.0S(t)))*τ
      dEI/dt = EE(t)*σ - EI(t)*(γ + σ) + SI(t)*τ + 5.0ifelse((5.0S(t)) == 0, 0, (SI(t)^2) / (5.0S(t)))*τ
      dER/dt = EI(t)*γ - ER(t)*σ + 5.0ifelse((5.0S(t)) == 0, 0, (SI(t)*SR(t)) / (5.0S(t)))*τ
      dII/dt = 2EI(t)*σ - 2II(t)*γ
      dIR/dt = ER(t)*σ + II(t)*γ - IR(t)*γ
      dRR/dt = 2IR(t)*γ
      dcumulative/dt = SI(t)*τ
      parameters  γ, σ, τ

The contact S + I → E + I converts the S node of an \[S I\] pair. It
therefore changes every pair \[Z S\] with Z the other neighbour of that
S node: the closed triples \[Z S I\] ≈ K\[Z S\]\[S I\]/\[S\] appear in
d\[SS\]/dt, d\[SE\]/dt, d\[SI\]/dt and d\[SR\]/dt. This is the whole
story of the page. A contact acts on all pairs of its recipient,
including pairs with compartments that belong to other parts of a glued
model.

The model is exact on a Poisson network (constant closure, PT degree
distribution), so it matches the simulation of `:seir_pois5`:

``` julia
sol = solve_epidemic(sys, sc)
det = model_curves(sys, sol; t = sc.tgrid, label = "pairwise (glued SEIR)")
tab = compare(ref, det; observables = [:E, :I, :cumulative])
comparisonplot(ref, det; observables = [:E, :I, :cumulative])
```

![Pairwise model of the glued SEIR against the committed
NetworkOutbreaks ensemble (spread band q2.5–q97.5 of the runs; residual
panel: mean band ±1.96
SE).](index_files/figure-commonmark/cell-5-output-1.svg)

``` julia
anchors(sc);
describe_reference(ref);
```

    :seir_pois5: γ = 0.25, σ = 0.2, τ = 0.166667; seeds E 0.01; t = 0:0.5:150; R₀ = 2 (canonical anchors)
    NetworkOutbreaks reference :seir_pois5 (hash b3d403ab): N = 10000 nodes, 200 runs on a fresh graph per run, algorithm :next_reaction; conditioning: major outbreaks only (cumulative incidence excluding seeds ≥ 0.05·N by t_end); 200 of 200 runs kept (major runs), P(major) = 1.000 (95% CI 0.981–1.000); no time alignment.

``` julia
tab
```

    ComparisonTable :seir_pois5  (scenario b3d403ab; conditioned mean of 200 runs)
      curve                  observable        D∞     t(D∞)       SE∞        z∞       ΔR∞  95% CI                 Δpeak   Δt_peak  coverage
      pairwise (glued SEIR)  E            0.00067     32.00   0.00063      1.86   0.00003  [-0.00095,  0.00102]   0.00050      0.00     1.000
      pairwise (glued SEIR)  I            0.00034     57.00   0.00051      1.10   0.00003  [-0.00095,  0.00102]  -0.00003      0.00     1.000
      pairwise (glued SEIR)  cumulative   0.00087     38.50   0.00271      0.80   0.00003  [-0.00095,  0.00102]   0.00003      0.50     1.000

The largest prevalence gap is D∞(I) = 0.0003383, and the final-size
difference is ΔR∞ = 3.48e-05.

## Lift, then compose: the parts cannot be lifted

The edge-based lift is defined reaction by reaction, so each part has a
lift even when it is not a model on its own (E12 adds the parts’
`lift_contributions`). The pairwise model is not built reaction by
reaction. It needs the complete list of compartments to know which pairs
exist, a susceptible class to hold the unseeded nodes, and an infection
to seed. Neither part of SEIR has all three:

``` julia
parts_lifted = map((("transmission", tr), ("history", pr))) do (name, part)
    msg = try
        node_based(part, sc.network)
        "lifted"
    catch e
        sprint(showerror, e)
    end
    println(name, ": ", msg)
    msg == "lifted"
end;
```

    transmission: ArgumentError: model :transmission has no infection (a contact from a susceptible class into the infection chain), so it has no entry state to seed by default; pass an explicit `initial`, e.g. SeedFraction(:X => ρ)
    history: ArgumentError: Model must have at least one infectious compartment

## The F2 counterexample: two strains glued along S and R

Two SIR strains that compete for the same susceptibles and share the
recovered class can each be lifted on their own. Glued along S and R,
they give the two-strain model of `:twostrain_pois5`:

``` julia
A = open_model(ContactModel(:a; contacts = [Contact(:S, :I1, :I1, :τ1)], transitions = [NodeTransition(:I1, :R, :γ)]);
               legs = [[:S, :R]])
B = open_model(ContactModel(:b; contacts = [Contact(:S, :I2, :I2, :τ2)], transitions = [NodeTransition(:I2, :R, :γ)]);
               legs = [[:S, :R]])
AB = glue(A, B; on = [:S, :R])
@assert isequivalent(AB, scenario(:twostrain_pois5).model)
pois5 = scenario(:twostrain_pois5).network
pwA  = node_based(A, pois5)
pwB  = node_based(B, pois5)
pwAB = node_based(AB, pois5; initial = scenario(:twostrain_pois5).initial)   # two entry states: explicit seeds
(; A = pwA, B = pwB, glued = pwAB)
```

    (A = PairwiseSystem(a; 3 singles, 6 pairs, closure = BernoulliClosure()), B = PairwiseSystem(b; 3 singles, 6 pairs, closure = BernoulliClosure()), glued = PairwiseSystem(a_b; 4 singles, 10 pairs, closure = BernoulliClosure()))

A strict gluing law would say that the field of the glued model is the
sum of the parts’ fields, pushed forward along the inclusions A → A + B
← B. To compare them, a pair is named by its two compartments as an
unordered pair, so that \[I₂R\] in B and \[RI₂\] in the glued model are
the same coordinate. The helper below does this, and then subtracts the
parts from the glued model, coordinate by coordinate:

``` julia
# the right-hand sides and state variables of a pairwise system, keyed by compartment (singles) or by the
# sorted pair of compartments (pairs)
function field_by_coordinate(pw)
    so = symbolic_ode(pw)
    rhs = Dict(string(x) => r for (x, r) in zip(so.states, so.rhs))
    field, var = Dict{Any,Any}(), Dict{Any,Any}()
    for (X, v) in pw.singles
        field[X], var[X] = rhs[string(v)], v
    end
    for ((X, Y), v) in pw.pairs
        k = Tuple(sort([X, Y]))
        field[k], var[k] = rhs[string(v)], v
    end
    return field, var
end
coord(k) = k isa Symbol ? "[$(k)]" : "[" * join(string.(k), " ") * "]"
(fA, vA), (fB, vB), (fAB, vAB) = field_by_coordinate(pwA), field_by_coordinate(pwB), field_by_coordinate(pwAB)
# push a part's field forward: rename its variables to the glued model's variables of the same coordinate
push_forward(f, v, k) = haskey(f, k) ? Symbolics.substitute(f[k], Dict(v[c] => vAB[c] for c in keys(v))) : 0
rows = []
for k in sort!(collect(keys(fAB)); by = coord)
    Δ = Symbolics.simplify(fAB[k] - push_forward(fA, vA, k) - push_forward(fB, vB, k))
    push!(rows, (coord(k), haskey(fA, k) ? "yes" : "no", haskey(fB, k) ? "yes" : "no",
                 string(Δ) == "0" ? "0" : "`" * string(Δ) * "`"))
end
md_table(["coordinate", "in A", "in B", "glued − (A + B)"], rows)
```

    | coordinate | in A | in B | glued − (A + B) |
    |---|---|---|---|
    | [I1 I1] | yes | no | 0 |
    | [I1 I2] | no | no | `-2I1I2(t)*γ + 5.0ifelse((5.0S(t)) == 0, 0, (SI1(t)*SI2(t)) / (5.0S(t)))*τ1 + 5.0ifelse((5.0S(t)) == 0, 0, (SI1(t)*SI2(t)) / (5.0S(t)))*τ2` |
    | [I1 R] | yes | no | `(I1I1(t) + I1I2(t))*γ - I1I1(t)*γ` |
    | [I1 S] | yes | no | `-5.0ifelse((5.0S(t)) == 0, 0, (SI1(t)*SI2(t)) / (5.0S(t)))*τ2` |
    | [I1] | yes | no | 0 |
    | [I2 I2] | no | yes | 0 |
    | [I2 R] | no | yes | `(I1I2(t) + I2I2(t))*γ - I2I2(t)*γ` |
    | [I2 S] | no | yes | `-5.0ifelse((5.0S(t)) == 0, 0, (SI1(t)*SI2(t)) / (5.0S(t)))*τ1` |
    | [I2] | no | yes | 0 |
    | [R R] | yes | yes | 0 |
    | [R S] | yes | yes | 0 |
    | [R] | yes | yes | 0 |
    | [S S] | yes | yes | 0 |
    | [S] | yes | yes | 0 |

``` julia
nnew = count(r -> r[2] == "no" && r[3] == "no", rows)
nbad = count(r -> (r[2] == "yes" || r[3] == "yes") && r[4] != "0", rows)
println("The glued pairwise model has ", length(rows), " coordinates. Of these, ", nnew, " (the pairs of an ",
        "A-only and a B-only compartment) are in neither part. Of the ", length(rows) - nnew, " coordinates that are ",
        "in a part, ", nbad, " have a field that differs from the sum of the parts' fields.")
```

The glued pairwise model has 14 coordinates. Of these, 1 (the pairs of
an A-only and a B-only compartment) are in neither part. Of the 13
coordinates that are in a part, 4 have a field that differs from the sum
of the parts’ fields.

The differences are of three kinds:

- **\[S I₁\] and \[S I₂\]: the closed triple \[I₁ S I₂\].** When the S
  node of an \[S I₁\] pair is infected by strain 2 through another
  neighbour, the pair is lost. The rate is τ₂\[I₁ S I₂\] ≈ τ₂ K \[S
  I₁\]\[S I₂\]/\[S\] (the `ifelse` guards \[S\] = 0). Part A does not
  contain I₂, so its field cannot have this term. This is the F2 term;
- **\[I₁ R\] and \[I₂ R\]: recovery inside the new pairs.** When the I₂
  node of an \[I₁ I₂\] pair recovers, the pair becomes \[I₁ R\]. \[I₁
  I₂\] is a new coordinate, so neither part has the flow;
- **the new coordinate** \[I₁ I₂\] has its own equation, which is not
  the image of anything in A or B.

The edge-based lift of the same gluing is strict. In the edge-based
field there is no pair of compartments, only one θ per edge and per-edge
φ’s, and each reaction’s term depends only on its own compartments. The
check with EdgeBasedModels’ lift:

``` julia
import EdgeBasedModels
eb_glued = symbolic_ode(EdgeBasedModels.edge_based(AB, pois5))
eb_parts = symbolic_ode(EdgeBasedModels.sum_contributions(EdgeBasedModels.lift_contributions(A, pois5),
                                                          EdgeBasedModels.lift_contributions(B, pois5)))
(; edge_based_strict = vector_fields_equal(eb_glued, eb_parts), pairwise_strict = nbad == 0 && nnew == 0)
```

    (edge_based_strict = true, pairwise_strict = false)

So the right order for pairwise and pair-based back ends is to compose
the syntax first and lift the result. The glued model then is the canned
one, and the lift is well defined.

## The S-anchored pairwise model

The S-anchored pairwise model PW^S keeps the singles and only the pairs
\[S Z\] with a susceptible end. It is the image of the edge-based model
(morphism M6), and the smallest pairwise system for which the failure
can be stated. Glue SIR with a vaccination exit S → V along S. The
result is the canned SIR-with-vaccination model:

``` julia
vax  = open_model(ContactModel(:vax; transitions = [NodeTransition(:S, :V, :ν)]); legs = [[:S]])
sirv = glue(open_model(sir_model(); legs = [[:S, :I, :R]]), vax; on = [:S])
@assert isequivalent(sirv, sirv_model())
pws = node_based(sirv, pois5; level = :s_anchored)
symbolic_ode(pws)
```

    SymbolicODE :s_anchored_sir_vax (8 states)
      dS/dt = -S(t)*ν - SI(t)*τ
      dI/dt = -I(t)*γ + SI(t)*τ
      dR/dt = I(t)*γ
      dV/dt = S(t)*ν
      dSS/dt = 2((-SI(t)*SS(t)*τ) / S(t)) - 2SS(t)*ν
      dSI/dt = (SI(t)*SS(t)*τ) / S(t) - SI(t)*γ - SI(t)*ν - (SI(t) + (SI(t)^2) / S(t))*τ
      dSR/dt = (-SI(t)*SR(t)*τ) / S(t) + SI(t)*γ - SR(t)*ν
      dSV/dt = (-SI(t)*SV(t)*τ) / S(t) + SS(t)*ν - SV(t)*ν
      parameters  γ, ν, τ

The contact of the SIR part drains \[S V\] through the triple \[V S I\]
≈ K\[S V\]\[S I\]/\[S\]: the term −τ\[SI\]\[SV\]/\[S\] of d\[SV\]/dt (K
= 1 on Poisson(5)). The SIR part has no V, so its PW^S system has no \[S
V\] coordinate. The vaccination part has no contact, and on its own it
cannot be lifted at all:

``` julia
try
    node_based(vax, pois5; level = :s_anchored)
catch e
    print(sprint(showerror, e))
end
```

    AdmissibilityError: node_based(:vax, ConfigurationNetwork(PoissonDegree(5.0)); level = :s_anchored):
      The model has no susceptible class (no contact converts a susceptible node).
      The S-anchored pairwise model needs exactly one susceptible class per node type
      (an untyped network has one node type). Back ends that accept this model:
      node_based (pairwise, individual, pair, motif, neighbourhood), simulate,
      mass_action.
      Try: add a contact s + J → X + J whose recipient s is susceptible.

Its contribution to the glued field is the part without contacts (τ =
0). At the witness state \[S\] = \[SS\] = \[SI\] = \[SV\] = 1 and \[I\]
= \[V\] = 0 (with \[R\] = 0), with τ = ν = 1, the \[S V\] component of
the glued field and of the sum of the parts are:

``` julia
so = symbolic_ode(pws)
snames = Dict(string(x) => x for x in so.states)
pars = Dict(string(p) => p for p in so.parameters)
state = Dict(snames[n] => v for (n, v) in ("S(t)" => 1.0, "SS(t)" => 1.0, "SI(t)" => 1.0, "SV(t)" => 1.0,
                                           "SR(t)" => 0.0, "I(t)" => 0.0, "R(t)" => 0.0, "V(t)" => 0.0))
at(expr, τ) = Symbolics.value(Symbolics.substitute(expr, merge(state, Dict(pars["τ"] => τ, pars["ν"] => 1.0,
                                                                           pars["γ"] => 1.0))))
iSV = findfirst(x -> string(x) == "SV(t)", so.states)
glued_SV = at(so.rhs[iSV], 1.0)       # the glued model
parts_SV = at(so.rhs[iSV], 0.0)       # SIR part: no [SV] coordinate (0); vaccination part: the τ = 0 terms
@assert glued_SV != parts_SV "the witness unexpectedly satisfies the strict gluing law"
(; glued = glued_SV, sum_of_parts = parts_SV)
```

    (glued = -1.0, sum_of_parts = 0.0)

``` julia
println("The glued value is ", fmt(glued_SV), " and the sum of the parts is ", fmt(parts_SV), ", so the strict gluing law ",
        "fails for PW^S. No Lean theorem is cited for this failure: it is checked numerically by this evaluation, which the ",
        "page asserts (it does not render if the two values agree). The lax comparison map is not formalised either.")
```

The glued value is -1 and the sum of the parts is 0, so the strict
gluing law fails for PW^S. No Lean theorem is cited for this failure: it
is checked numerically by this evaluation, which the page asserts (it
does not render if the two values agree). The lax comparison map is not
formalised either.

## Stratification commutes with gluing, then lift

On the two-block network of `:sir_sbm2`, gluing an infection part and a
recovery part along I and then stratifying over the blocks gives the
scenario’s model. Stratification also commutes with gluing at the level
of syntax. The multitype pairwise model of the result is the one of the
scenario:

``` julia
st      = strata([:a, :b]; sizes = [0.5, 0.5])
infect  = open_model(ContactModel(:infect; contacts = [Contact(:S, :I, :I, :τ)]); legs = [[:S, :I]])
recover = open_model(ContactModel(:recover; transitions = [NodeTransition(:I, :R, :γ)]); legs = [[:I, :R]])
sir_glued = glue(infect, recover; on = [:I])
sb = scenario(:sir_sbm2)
pw_strat = node_based(stratify(sir_glued, st), sb.network)
strat = (; syntax = isequivalent(stratify(sir_glued, st), sb.model),
           commutes = isequivalent(stratify(sir_glued, st),
                                   glue(stratify(infect, st), stratify(recover, st); on = [:I_a, :I_b])),
           lift = vector_fields_equal(symbolic_ode(pw_strat), symbolic_ode(node_based(sb.model, sb.network))))
```

    (syntax = true, commutes = true, lift = true)

The stratified recovery part again has no susceptible class, so it
cannot be lifted on its own:

``` julia
try
    node_based(stratify(recover, st), sb.network)
catch e
    print(sprint(showerror, e))
end
```

    ArgumentError: node_based(:recover_strat, MultitypeNetwork; level = :population, closure = NodeBasedModels.BernoulliClosure()): the node type a has no unique susceptible class to hold its unseeded nodes; pass an explicit `initial` naming the fraction of all nodes in every compartment of that type

## Summary

``` julia
md_table(["statement", "holds here", "checked by"],
         [("PW(glue(transmission, history)) = PW(canned SEIR)", vector_fields_equal(symbolic_ode(sys), symbolic_ode(sysF)), "`vector_fields_equal`"),
          ("the parts of SEIR can be lifted on their own", all(parts_lifted), "`node_based` errors above"),
          ("PW(glue(A, B)) = PW(A) + PW(B) (two strains)", nbad == 0 && nnew == 0, "symbolic difference above"),
          ("EB(glue(A, B)) = EB(A) + EB(B) (two strains)", vector_fields_equal(eb_glued, eb_parts), "`vector_fields_equal` (H1, page E12)"),
          ("PW^S(glue(SIR, vax)) = sum of the parts at the witness state", glued_SV == parts_SV, "evaluation above (numerical)"),
          ("stratify commutes with glue, then lift (`:sir_sbm2`)", all(values(strat)), "`isequivalent`, `vector_fields_equal`")])
```

| statement | holds here | checked by |
|----|---:|----|
| PW(glue(transmission, history)) = PW(canned SEIR) | true | `vector_fields_equal` |
| the parts of SEIR can be lifted on their own | false | `node_based` errors above |
| PW(glue(A, B)) = PW(A) + PW(B) (two strains) | false | symbolic difference above |
| EB(glue(A, B)) = EB(A) + EB(B) (two strains) | true | `vector_fields_equal` (H1, page E12) |
| PW^S(glue(SIR, vax)) = sum of the parts at the witness state | false | evaluation above (numerical) |
| stratify commutes with glue, then lift (`:sir_sbm2`) | true | `isequivalent`, `vector_fields_equal` |

Pairwise composition is lax: there is a comparison between the glued
model and the parts, but it is not an equality, because a contact drains
every pair of its recipient. Compose the syntax, then lift.

## References

<div id="refs">

</div>
