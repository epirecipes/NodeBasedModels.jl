# multitype_pairwise.jl — population-level multitype pairwise models on `MultitypeNetwork`
# descriptors (typed stratified models), `node_based(model, net::MultitypeNetwork; closure, level)`.
#
# Owner: WP36e (DESIGN_NetworkEpiCore.md §K; §C.2, §C.3, §D.4, §D.5 M6/M7/M8/M10, §E.2, §J.2,
# §J.6, §J.8). The `_node_based` hook conventions are those of lift.jl; the solve/observe methods
# are those of `_NodeClosureSystem` (s_anchored.jl).
#
# The mathematics
# ---------------
# A typed configuration network (`MultitypeNetwork`): a fraction n_a of the nodes has type a, and
# a type-a node has k_{a→b} edges to type-b nodes, with the joint PGF ψ_a(x) = E[Π_b x_b^{k_{a→b}}]
# and the mean contacts M_ab = E[k_{a→b}] = ∂_bψ_a(1) (reciprocity n_a M_ab = n_b M_ba). Every
# species X carries its node type t(X) (the stratum of its label: `stratify`, `disjoint_union`);
# contacts s + J → P + J keep the recipient's type (t(P) = t(s)), node transitions stay within a
# type, and a removal X → ∅ goes to the absorbing sink `removed_<t(X)>` (one per node type, as in
# EdgeBasedModels' multitype lift; DESIGN §J.2).
#
# [X] is the number of nodes in X and [XY] the ordered pair count (the number of ordered adjacent
# pairs (u, v) with u ∈ X, v ∈ Y; a cross pair counts each XY edge once, a self pair [XX] twice:
# the convention of population_pairwise.jl and s_anchored.jl). A pair is tracked only when the
# types are joined by edges (M_{t(X)t(Y)} > 0); a structural zero has no pairs, and a contact
# across it has no partners, so it contributes nothing. With A(X, Y) the rate of change of [XY]
# through events at the first node u,
#
#     d[XY]/dt = A(X, Y) + A(Y, X),
#
#     contact  R + J → P + J (τ):  A(P, Y) += τ(δ_{YJ}[RY] + [Y R J]),   A(R, Y) −= the same
#     move     W → Z (a):          A(Z, Y) += a[WY],                     A(W, Y) −= a[WY]
#
# with the singles d[R] −= τ[RJ], d[P] += τ[RJ], d[W] −= a[W], d[Z] += a[W]. Every triple is
# centred at the recipient R of a contact, and is closed by
#
#     [Y R J] = K_a(c, b) [YR][RJ]/[R],      a = t(R), c = t(Y), b = t(J),
#
# - `BernoulliClosure` (the default): the constant K_a(c, b) = ∂_c∂_bψ_a(1)/(∂_cψ_a(1)∂_bψ_a(1))
#   = E[k_{a→c}(k_{a→b} − δ_{cb})]/(M_ac M_ab), the exact triple count of a random arrangement
#   (one type: K = ψ''(1)/ψ'(1)² = `closure_constant(d)`). It is total on T_net models (SIS and
#   SIRS included).
# - `PGFClosure`: K_a(c, b)(θ) = ψ_a(θ)∂_c∂_bψ_a(θ)/(∂_cψ_a(θ)∂_bψ_a(θ)) at θ_{·→a}, with an
#   auxiliary θ_{b→a} (the probability that an edge from a type-b partner has not transmitted to a
#   type-a test node) for every edge class, θ(0) = 1 and
#       θ̇_{b→a} = −ψ_a(θ)/([s_a]∂_bψ_a(θ)) · Σ_{contacts s_a + J → P + J, t(J) = b} τ[s_a J],
#   for T_EB models (one susceptible class s_a per type, no arrow back into it).
#
# The typed π^PW map (DESIGN §D.5 M6 on a typed network): with the multitype edge-based
# coordinates of EdgeBasedModels (Miller & Volz 2013, §3.3; θ_{b→a}, ξ_a, φ_{Y,a}, pop_Y, q_a),
#
#     [s_a]     = N n_a q_a ξ_a ψ_a(θ_{·→a}),            [X] = N pop_X,
#     [s_a Y]   = N n_a q_a ξ_a ∂_{t(Y)}ψ_a(θ_{·→a}) φ_{Y,a},
#     [Y s_a J] = N n_a q_a ξ_a ∂_{t(Y)}∂_{t(J)}ψ_a(θ_{·→a}) φ_{Y,a} φ_{J,a},
#
# so [Y s_a J] = K_a(θ)[Y s_a][s_a J]/[s_a] holds on the image for every joint PGF, and the edge-
# based θ̇_{b→a} = −Σ τ φ_{J,a} is the θ equation above. PGFClosure is therefore exact against the
# multitype edge-based model for every `MultivariateDegree`. The constant closure is exact when
# every K_a(c, b)(θ) is constant (the "if" half of the typed analogue of M8; a non-constant K is
# biased in general, as the tests pin on two such networks): independent Poisson-type blocks
# (`sbm_network` with Poisson, binomial, regular or negative binomial families; K_a(c, b) = 1 for
# c ≠ b), and `SplitDegrees` with a Poisson-type total (`unstructured`, where K_a = K_ψ(Σ_b n_b θ_b)).
# On independent Poisson blocks, ψ_a(x) = Π_b exp(M_ab(x_b − 1)), every K_a(c, b)(θ) = 1.
#
# Seeding (DESIGN §E.2, §J.6): seed fractions are fractions of ALL N nodes, SeedFraction(:I_a => ρ)
# puts ρN nodes of type a in I_a; the unseeded nodes of type a are in its background class (its
# susceptible class). By default a fraction `seed_fraction` of the nodes of every type is seeded
# in that type's unique entry state. The initial pairs are the π^PW image of the edge-based
# initial condition (a random arrangement): [XY] = N M_{t(X)t(Y)} x_X x_Y / n_{t(Y)} for the
# fractions x of all nodes, so [s_a Y] = N n_a q_a M_ab ρ_Y/n_b; θ = 1.

export MultitypePairwiseSystem

"""
    MultitypePairwiseSystem

The population-level pairwise model of a stratified model on a typed configuration network
(`MultitypeNetwork`: `sbm_network`, `unstructured`, or any `MultivariateDegree` per type), returned
by [`node_based`](@ref)`(model, net::MultitypeNetwork; closure, level)`. Every species is labelled
with its node type (`stratify(model, strata(...))` or `disjoint_union`); node types are fixed, and
removals `X → ∅` go to one absorbing sink `removed_<type>` per node type. A model with a species
that carries no node type is refused with an `ArgumentError`: in particular the
heterogeneous-susceptibility models (WP36f, DESIGN §L.7; the scenarios `:sir_hetsus_bim` and
`:seirv_hetsus_pois5`), whose susceptibility classes are the strata of `unstructured(net, st)` but
whose infected and removed classes are shared, so a node loses its type on infection. They are
edge-based only (DESIGN §H H22 defers heterogeneous susceptibility; WP36f and §L.7 add it to
EdgeBasedModels only), and their scenarios declare no `:pairwise_multitype` backend. The singles [X] are the
numbers of nodes in X and the pairs [XY] the ordered pair counts (a cross pair counts each XY edge
once, a self pair [XX] twice), tracked only between types joined by edges (structural zeros of
the network have no pairs). Every triple is centred at the recipient R (of type a) of a contact
and closed by [Y R J] = K_a(t(Y), t(J)) [YR][RJ]/[R]:

- `closure = BernoulliClosure()` (the default, `level = :population`): the constant
  K_a(c, b) = ∂_c∂_bψ_a(1)/(∂_cψ_a(1)∂_bψ_a(1)) = E[k_{a→c}(k_{a→b} − δ_{cb})]/(E[k_{a→c}]E[k_{a→b}]),
  for every model (SIS and SIRS included). For T_EB models it equals the multitype edge-based
  model exactly when every K_a(c, b)(θ) below is constant: Poisson-type blocks (`sbm_network`
  with the Poisson, binomial, regular or negative binomial family, where K_a(c, b) = 1 for
  c ≠ b; on Poisson blocks every K is 1), or `SplitDegrees` with a Poisson-type total
  (`unstructured`). Otherwise it is biased in general. Representation `:pairwise`;
- `closure = PGFClosure()`: K_a(c, b)(θ) = ψ_a(θ)∂_c∂_bψ_a(θ)/(∂_cψ_a(θ)∂_bψ_a(θ)) with an auxiliary
  state `θ_<b>_<a>` per edge class (the edge-based θ_{b→a}), exact against the multitype
  edge-based model for every joint degree law (T_EB models only). Representation `:pgf_closure`;
- `level = :s_anchored` (with either closure): the S-anchored subsystem, the singles and the pairs
  [s_a Y] of the susceptible classes only (T_EB models). Representation `:s_anchored`.

Contact rates are per-contact rates τ: a stratified model on a `MultitypeNetwork` must use the
`PerContact` convention (`per_contact_rates(model, net)` refuses the others, an
`ArgumentError`). States are counts on the population scale `N` (fractions of all nodes for
`N = 1`). Seed fractions are fractions of **all** nodes (DESIGN §J.6):
`SeedFraction(:I_y => 0.004, :I_o => 0.006)` seeds 1% of the network, and the unseeded nodes of a
type start in its susceptible class; by default a fraction `seed_fraction` of the nodes of every
type starts in the unique entry state of that type. The initial pairs are the π^PW image of the
edge-based initial condition, [XY] = N M_{t(X)t(Y)} x_X x_Y/n_{t(Y)} (x: fractions of all
nodes), and θ = 1.

Fields: `system` (the compiled ModelingToolkit system), `u0`, `tspan`, `params` (values given when
the system was built), `model` (as passed), `network`, `closure`, `level`, `singles`
(compartment => [X]) and `pairs` ((X, Y) => [XY], keys in compartment order), `metadata`: `:θ`
(always `nothing`), `:thetas` ((b, a) => θ_{b→a}, empty for the constant closure), `:K`
((a, c, b) => K_a(c, b), numbers or expressions in θ), `:types` (species => node type), `:sinks`,
`:susceptible`, `:background` (node type => its background class), `:contact_model` (the
per-contact `ContactModel`), `:N`, `:seed`, `:seed_state`, `:default_fraction`, `:cumulative`,
`:infected` (DESIGN §J.8), `:infectious`, `:parameters`, `:states` and `:rhs` (the uncompiled
field without the accumulator), `:representation`, `:label`, `:name`, `:level`.

It is solved and read like an [`SAnchoredSystem`](@ref): [`solve_epidemic`](@ref) (also
`solve_epidemic(sys, sc::Scenario)`), [`model_curves`](@ref), [`compartment`](@ref),
[`population_fraction`](@ref), `default_initial_conditions` and `symbolic_ode(sys)`.

```julia
sc  = scenario(:sir_age2)                  # stratified SIR on a two-type Poisson SBM
sys = node_based(sc)                       # constant K = 1 on Poisson blocks: exact
sol = solve_epidemic(sys, sc)
model_curves(sys, sol; t = sc.tgrid)       # :S_y, …, :infectious, :cumulative, and :S, :I, :R
```
"""
struct MultitypePairwiseSystem <: _NodeClosureSystem
    system::Any
    u0::Dict{Any,Float64}
    tspan::Tuple{Float64,Float64}
    params::Dict{Symbol,Float64}
    model::Any
    network::MultitypeNetwork
    closure::ClosureMethod
    level::Symbol
    singles::Dict{Symbol,Any}
    pairs::Dict{Tuple{Symbol,Symbol},Any}
    metadata::Dict{Symbol,Any}
end

function Base.show(io::IO, sys::MultitypePairwiseSystem)
    θ = isempty(sys.metadata[:thetas]) ? "" : ", $(length(sys.metadata[:thetas])) θ"
    print(io, "MultitypePairwiseSystem(", nameof(sys.metadata[:contact_model]), "; level = :",
          sys.level, ", ", length(sys.network.types), " types, ", length(sys.singles),
          " singles, ", length(sys.pairs), " pairs", θ, ", closure = ", sys.closure, ")")
end

"""
    node_variables(sys::MultitypePairwiseSystem) -> Dict{Symbol,Any}
    pair_variables(sys::MultitypePairwiseSystem) -> Dict{Tuple{Symbol,Symbol},Any}

The single variables [X] (by compartment) and the tracked pair variables [XY] (by `(X, Y)`, in
compartment order) of a [`MultitypePairwiseSystem`](@ref); the θ_{b→a} of `PGFClosure` are
`sys.metadata[:thetas]`.
"""
node_variables(sys::MultitypePairwiseSystem) = sys.singles
pair_variables(sys::MultitypePairwiseSystem) = sys.pairs

"""
    default_closure(::MultitypeNetwork) -> BernoulliClosure

A typed configuration network is closed by the constant multitype closure
K_a(c, b) = ∂_c∂_bψ_a(1)/(∂_cψ_a(1)∂_bψ_a(1)) (see [`MultitypePairwiseSystem`](@ref)), exact
against the multitype edge-based model on Poisson-type blocks; `PGFClosure()` is exact for every
joint degree law.
"""
default_closure(::MultitypeNetwork) = BernoulliClosure()

# ─── node_based methods ──────────────────────────────────────────────────────

function _node_based(::Val{:population}, closure::Union{BernoulliClosure,PGFClosure},
                     cm::ContactModel, net::MultitypeNetwork; kw...)
    return _multitype_pairwise(cm, net, closure; level = :population, kw...)
end

function _node_based(::Val{:s_anchored}, closure::Union{BernoulliClosure,PGFClosure},
                     cm::ContactModel, net::MultitypeNetwork; kw...)
    return _multitype_pairwise(cm, net, closure; level = :s_anchored, kw...)
end

const _MT_CLOSURES =
    "the multitype pairwise model closes its triples with BernoulliClosure() (the constant " *
    "K_a(c, b) = ∂c∂bψ_a(1)/(∂cψ_a(1)∂bψ_a(1)), exact on Poisson-type blocks) or PGFClosure() " *
    "(K_a(θ) with the auxiliary θ_{b→a}, exact for every joint degree law)"

function _node_based(::Val{:population}, closure::ClosureMethod, cm::ContactModel,
                     net::MultitypeNetwork; kw...)
    hint = closure isa MeanFieldClosure ?
           "; a typed network has persistent partnerships, so the mean-field closure is not a " *
           "closure of it" :
           closure isa Union{KeelingClosure,BarnardClosure} ?
           "; a MultitypeNetwork is a configuration model without triangles (no clustering ϕ)" : ""
    throw(ArgumentError("node_based: $(_MT_CLOSURES); got $(closure)$(hint)"))
end

_s_anchored_unsupported(closure, ::MultitypeNetwork) =
    "on a MultitypeNetwork the S-anchored subsystem closes the S-centred triples with " *
    "BernoulliClosure() (constant K_a(c, b)) or PGFClosure() (K_a(θ)); got $(closure)"

# ─── Closure factors ─────────────────────────────────────────────────────────

# K_a(c, b) of [Y R J] = K [YR][RJ]/[R] for the joint degree law m of the centre's type a, the end
# types c (of Y) and b (of J), and the edge-class argument x (a function b -> x_b; x ≡ 1 for the
# constant closure).
_mt_closure_factor(::BernoulliClosure, m::MultivariateDegree, c::Symbol, b::Symbol, x) =
    Float64(pgf_derivative(m, _ -> 1.0, c, b)) /
    (Float64(mean_degree(m, c)) * Float64(mean_degree(m, b)))
_mt_closure_factor(::PGFClosure, m::MultivariateDegree, c::Symbol, b::Symbol, x) =
    pgf(m, x) * pgf_derivative(m, x, c, b) / (pgf_derivative(m, x, c) * pgf_derivative(m, x, b))

_mt_closure_label(::BernoulliClosure) = "constant K"
_mt_closure_label(::PGFClosure) = "PGF closure"

# The mean contact matrix M[a, b] = E[k_{a→b}] as numbers, or an ArgumentError. The pairwise
# model stores a cross pair [X_a Y_b] = [Y_b X_a] once, so the edge ends must balance: the
# reciprocity n_a M_ab = n_b M_ba, with a structural zero on both sides or neither (a network built
# with `check_reciprocity = false` may break it; no graph realises it).
function _mt_numeric_means(net::MultitypeNetwork, where)
    M = mean_contacts(net)
    for v in M
        (v isa Real && !(v isa Symbolics.Num) && isfinite(v)) || throw(ArgumentError(
            "$where: the mean contacts of the MultitypeNetwork are not all numeric " *
            "($(M)); the multitype pairwise model needs numeric degree parameters"))
    end
    M = Float64.(M)
    types, n, Z = net.types, net.sizes, net.structural_zero
    bad = String[]
    for i in eachindex(types), j in (i + 1):length(types)
        x, y = n[i] * M[i, j], n[j] * M[j, i]
        (Z[i, j] == Z[j, i] && isapprox(x, y; rtol = 1e-8, atol = 1e-14)) && continue
        a, b = types[i], types[j]
        push!(bad, "n_$(a)·E[k_{$(a)→$(b)}] = $(x) vs n_$(b)·E[k_{$(b)→$(a)}] = $(y)")
    end
    isempty(bad) || throw(ArgumentError(
        "$where: the MultitypeNetwork breaks the edge reciprocity n_a·E[k_{a→b}] = " *
        "n_b·E[k_{b→a}] ($(join(bad, "; "))), so no network realises it and the pair counts " *
        "[X_a Y_b] = [Y_b X_a] are undefined; build it with check_reciprocity = true"))
    return M
end

# ─── The builder ─────────────────────────────────────────────────────────────

"""
    _multitype_pairwise(cm::ContactModel, net::MultitypeNetwork, closure; level, source, p, name,
                        tspan, N, ε, seed_fraction, seed_state, initial, cumulative)
        -> MultitypePairwiseSystem

Build the multitype pairwise model of `cm` on `net` (the header of src/multitype_pairwise.jl
gives the equations): every tracked pair for `level = :population`, the pairs of the
susceptible classes only for `level = :s_anchored`. The keywords are those of
[`node_based`](@ref) at the population level: `tspan = (0.0, 100.0)`, the population scale
`N = 1.0`, the seeding (`initial`, or a fraction `seed_fraction = ε = 1e-3` of every node type
in `seed_state = :entry`, the type's unique entry state, or `:first_infectious`) and
`cumulative = true` (the cumulative-incidence accumulator). `require_admissible(cm, :pairwise)`
(constant closure, population level) or `require_admissible(cm, :s_anchored)` (PGFClosure or the
S-anchored level: T_EB, one susceptible class per node type) is checked first.
"""
function _multitype_pairwise(cm::ContactModel, net::MultitypeNetwork, closure::ClosureMethod;
                             level::Symbol, source = cm, p = nothing,
                             name::Union{Nothing,Symbol} = nothing,
                             tspan::Tuple{Real,Real} = (0.0, 100.0), N::Real = 1.0,
                             ε::Real = 1e-3, seed_fraction::Real = ε,
                             seed_state::Symbol = :entry,
                             initial::Union{Nothing,SeedSpec} = nothing,
                             cumulative::Bool = true)
    level in (:s_anchored, :population) || throw(ArgumentError(
        "_multitype_pairwise: level must be :s_anchored or :population; got :$(level)"))
    where = "node_based(:$(nameof(cm)), MultitypeNetwork; level = :$(level), " *
            "closure = $(closure))"
    (isfinite(N) && N > 0) || throw(ArgumentError(
        "$where: the population scale N must be finite and > 0; got $(N)"))
    seed_state in (:entry, :first_infectious) || throw(ArgumentError(
        "$where: seed_state must be :entry or :first_infectious; got $(repr(seed_state))"))
    M = _mt_numeric_means(net, where)
    s_centred = level === :s_anchored || _closure_theta(closure)
    cmτ = _per_contact_model(cm, net)
    require_admissible(cmτ, s_centred ? :s_anchored : :pairwise; network = net)

    types = net.types
    tindex = Dict(a => i for (i, a) in enumerate(types))
    n = Dict(a => net.sizes[i] for (a, i) in tindex)
    law = Dict(a => net.degrees[i] for (a, i) in tindex)
    joined(a::Symbol, b::Symbol) = !net.structural_zero[tindex[a], tindex[b]]

    # Node types of the species (their strata) and of the removal sinks.
    species = species_names(cmτ)
    labels = species_labels(cmτ)
    type_of = Dict{Symbol,Symbol}()
    for X in species
        a = haskey(labels, X) ? labels[X].stratum : :all
        a in types || throw(ArgumentError(
            "$where: the species $(X) " *
            (a === :all ? "has no stratum" : "belongs to the stratum $(a), which is not a node type") *
            "; on a MultitypeNetwork (types $(join(types, ", "))) every species must be labelled " *
            "with its node type (use stratify(model, strata(...)) or disjoint_union)" *
            (a === :all ? "; heterogeneous-susceptibility models with shared infected classes " *
                          "are supported by EdgeBasedModels only" : "")))
        type_of[X] = a
    end
    sinks = Symbol[]
    for t in node_transitions(cmτ)
        t.to === nothing || continue
        sink = Symbol(:removed_, type_of[t.from])
        sink in species && throw(ArgumentError(
            "$where: the model has a removal $(t.from) → ∅ and a species named :$(sink), the " *
            "absorbing compartment that removals of type $(type_of[t.from]) go to; rename it"))
        if !(sink in sinks)
            push!(sinks, sink)
            type_of[sink] = type_of[t.from]
        end
    end
    sort!(sinks; by = s -> tindex[type_of[s]])
    names = vcat(species, sinks)
    order = Dict(X => i for (i, X) in enumerate(names))
    key(a::Symbol, b::Symbol) = order[a] <= order[b] ? (a, b) : (b, a)
    Σ = susceptible_species(cmτ)

    # The reactions: contacts (R, J, P, τ) between joined types, and moves (W, Z, a).
    rate_names = unique!(reduce(vcat, (_rate_parameter_names(r.rate)
                                       for r in vcat(contacts(cmτ), node_transitions(cmτ)));
                                init = Symbol[]))
    params = Dict{Symbol,Any}(k => only(@parameters $(k)) for k in rate_names)
    rate(r) = _rate_symbolic(r, params, _TIME)
    cs = Tuple{Symbol,Symbol,Symbol,Any}[]
    for c in contacts(cmτ)
        R, J, P = c.recipient, c.infector, c.product
        R === J && throw(ArgumentError(
            "$where: the contact $(c.name) has its recipient $(R) as its infector"))
        type_of[P] === type_of[R] || throw(ArgumentError(
            "$where: the contact $(c.name) turns a node of type $(type_of[R]) ($(R)) into " *
            "$(P) of type $(type_of[P]); node types are fixed on a MultitypeNetwork"))
        joined(type_of[R], type_of[J]) || continue          # no R–J edges: no partners
        push!(cs, (R, J, P, rate(c.rate)))
    end
    ms = Tuple{Symbol,Symbol,Any}[]
    for t in node_transitions(cmτ)
        Z = something(t.to, Symbol(:removed_, type_of[t.from]))
        type_of[Z] === type_of[t.from] || throw(ArgumentError(
            "$where: the transition $(t.name) moves a node of type $(type_of[t.from]) " *
            "($(t.from)) into $(Z) of type $(type_of[Z]); node types are fixed on a " *
            "MultitypeNetwork (stratify has no transitions between strata)"))
        push!(ms, (t.from, Z, rate(t.rate)))
    end

    # The tracked pairs, the θ coordinates, and the generated names (distinct from each other and
    # from the rate parameters).
    tracked = if level === :s_anchored
        unique!([key(s, Y) for s in Σ for Y in names if joined(type_of[s], type_of[Y])])
    else
        [(names[i], names[j]) for i in eachindex(names) for j in i:length(names)
         if joined(type_of[names[i]], type_of[names[j]])]
    end
    sort!(tracked; by = k -> (order[k[1]], order[k[2]]))
    sus = Dict{Symbol,Symbol}()            # node type => its susceptible class (T_EB)
    if s_centred
        for s in Σ
            sus[type_of[s]] = s
        end
    end
    θkeys = Tuple{Symbol,Symbol}[]          # (b, a): θ_{b→a}
    if _closure_theta(closure)
        for a in types, b in types
            (haskey(sus, a) && joined(a, b)) && push!(θkeys, (b, a))
        end
    end
    θname(b, a) = Symbol(:θ_, b, :_, a)
    generated = vcat(names, Symbol[Symbol(a, b) for (a, b) in tracked],
                     Symbol[θname(k...) for k in θkeys], cumulative ? [:cumulative] : Symbol[])
    dup = unique!([x for x in generated if count(==(x), generated) > 1])
    isempty(dup) || throw(ArgumentError(
        "$where: the generated state names collide: $(join(dup, ", ")) (pairs are named by " *
        "joining the compartment names); rename the compartments"))
    bad = sort!([x for x in rate_names if x in generated])
    isempty(bad) || throw(ArgumentError(
        "$where: the parameter name(s) $(join(bad, ", ")) collide with state names of the " *
        "pairwise model; rename the parameter(s)"))

    singles = Dict{Symbol,Any}(X => _closed_state(X) for X in names)
    pairs = Dict{Tuple{Symbol,Symbol},Any}(k => _closed_state(Symbol(k...)) for k in tracked)
    θ = Dict{Tuple{Symbol,Symbol},Any}(k => _closed_state(θname(k...)) for k in θkeys)
    pair(X::Symbol, Y::Symbol) = joined(type_of[X], type_of[Y]) ? pairs[key(X, Y)] : 0

    # The closure factors K_a(c, b), cached by (a, c, b); θ_{·→a} is the argument x_b of ψ_a.
    Kcache = Dict{Tuple{Symbol,Symbol,Symbol},Any}()
    function Kfac(a::Symbol, c::Symbol, b::Symbol)
        return get!(Kcache, (a, c, b)) do
            x = b′ -> get(θ, (b′, a), 1.0)
            _mt_closure_factor(closure, law[a], c, b, x)
        end
    end
    # [Y R J]: T_EB closures divide by [s] > 0 (checked with the seeds); the constant closure of a
    # general model guards a recipient that may be empty, as population_pairwise.jl does.
    function triple(Y::Symbol, R::Symbol, J::Symbol)
        num = Kfac(type_of[R], type_of[Y], type_of[J]) * pair(Y, R) * pair(R, J)
        return s_centred || R in Σ ? num / singles[R] : _safe_div(num, singles[R])
    end

    # A(X, Y): the rate of change of [XY] through events at the first node.
    function first_node(X::Symbol, Y::Symbol)
        acc = Symbolics.Num(0)
        for (R, J, P, τ) in cs
            (X === P || X === R) || continue
            ev = τ * (Y === J ? pair(R, Y) + triple(Y, R, J) : triple(Y, R, J))
            acc += X === P ? ev : -ev
        end
        for (W, Z, a) in ms
            X === Z && (acc += a * pair(W, Y))
            X === W && (acc -= a * pair(W, Y))
        end
        return acc
    end

    dsingle = Dict{Symbol,Any}(X => Symbolics.Num(0) for X in names)
    for (R, J, P, τ) in cs
        flux = τ * pair(R, J)
        dsingle[R] -= flux
        dsingle[P] += flux
    end
    for (W, Z, a) in ms
        flux = a * singles[W]
        dsingle[W] -= flux
        dsingle[Z] += flux
    end
    states = Any[]
    rhs = Any[]
    for (b, a) in θkeys
        s = sus[a]
        hazard = sum((τ * pair(R, J) for (R, J, _, τ) in cs if R === s && type_of[J] === b);
                     init = Symbolics.Num(0))
        x = b′ -> get(θ, (b′, a), 1.0)
        push!(states, θ[(b, a)])
        push!(rhs, -(pgf(law[a], x) / (singles[s] * pgf_derivative(law[a], x, b))) * hazard)
    end
    for X in names
        push!(states, singles[X])
        push!(rhs, dsingle[X])
    end
    for k in tracked
        push!(states, pairs[k])
        push!(rhs, first_node(k[1], k[2]) + first_node(k[2], k[1]))
    end

    # The cumulative-incidence accumulator (DESIGN §J.8): entries from a non-infected into an
    # infected compartment, by contacts and by moves (the sinks are never infected).
    infected = infected_species(cmτ)
    eqs = Equation[_DT(x) ~ f for (x, f) in zip(states, rhs)]
    cum = nothing
    if cumulative
        cum = _closed_state(:cumulative)
        inc = Symbolics.Num(0)
        for (R, J, P, τ) in cs
            (!(R in infected) && P in infected) && (inc += τ * pair(R, J))
        end
        for (W, Z, a) in ms
            (!(W in infected) && Z in infected) && (inc += a * singles[W])
        end
        push!(eqs, _DT(cum) ~ inc)
    end
    sysname = something(name, Symbol(level === :s_anchored ? :s_anchored_multitype_ :
                                      :multitype_pairwise_, nameof(cm)))
    compiled = mtkcompile(System(eqs, _TIME; name = sysname))

    # Seeding (fractions of all nodes, per node type), then the π^PW image.
    backgrounds = _mt_backgrounds(cmτ, types, type_of, Σ)
    # the recipients that `triple` divides by without a guard must start non-empty
    centres = unique!(Symbol[R for (R, _, _, _) in cs if s_centred || R in Σ])
    ctx = (; where, cm = cmτ, types, n, type_of, species, names, backgrounds, infected, M, tindex)
    resolve = (init, ρ, state) -> _mt_resolve_seed(ctx, init, ρ, state, N)
    initial_state = (x, bg) -> _mt_u0(ctx, singles, pairs, θ, cum, Float64(N), x, centres)
    x, background = resolve(initial, seed_fraction, seed_state)
    u0 = initial_state(x, background)

    representation = level === :s_anchored ? :s_anchored :
                     _closure_theta(closure) ? :pgf_closure : :pairwise
    label = (level === :s_anchored ? "S-anchored multitype pairwise (" :
             "multitype pairwise (") * _mt_closure_label(closure) * ")"
    metadata = Dict{Symbol,Any}(
        :θ => nothing, :thetas => θ, :K => Kcache, :types => type_of, :sinks => sinks,
        :susceptible => Σ, :background => background, :contact_model => cmτ,
        :descriptor => net, :N => Float64(N), :seed => x, :resolve => resolve,
        :initial_state => initial_state, :default_fraction => Float64(seed_fraction),
        :seed_state => seed_state, :cumulative => cum, :infected => infected,
        :infectious => infectious_species(cmτ), :parameters => params, :states => states,
        :rhs => rhs,
        :domain => Pair{Any,Tuple{Float64,Float64}}[v => (0.05, 1.0) for v in values(θ)],
        :representation => representation, :label => label, :name => sysname,
        :level => level)
    vals = p === nothing ? Dict{Symbol,Float64}() :
           Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in _pairs_of(p))
    return MultitypePairwiseSystem(compiled, u0, (Float64(tspan[1]), Float64(tspan[2])), vals,
                                   source, net, closure, level, singles, pairs, metadata)
end

# ─── Seeding ─────────────────────────────────────────────────────────────────

# Node type => the class that holds its unseeded nodes: its susceptible class if it has one; of
# several, the unique one that no reaction produces (S₀ of a reinfection-counted model);
# otherwise `nothing` (the fractions of that type must then all be given).
function _mt_backgrounds(cm::ContactModel, types, type_of, Σ)
    produced = Set{Symbol}(c.product for c in contacts(cm))
    for t in node_transitions(cm)
        t.to === nothing || push!(produced, t.to)
    end
    out = Dict{Symbol,Union{Symbol,Nothing}}()
    for a in types
        own = [s for s in Σ if type_of[s] === a]
        free = [s for s in own if !(s in produced)]
        out[a] = length(own) == 1 ? only(own) : length(free) == 1 ? only(free) : nothing
    end
    return out
end

# The default seeded class of the node type a: the unique entry into infection from its
# background (`:entry`), or its first infectious species (`:first_infectious`).
function _mt_seed_class(ctx, a::Symbol, bg::Symbol, state::Symbol)
    if state === :entry
        entries = unique!(Symbol[c.product for c in contacts(ctx.cm)
                                 if c.recipient === bg && c.product in ctx.infected])
        length(entries) == 1 && return only(entries)
        throw(ArgumentError(
            "$(ctx.where): the node type $(a) has " *
            (isempty(entries) ? "no entry into infection from $(bg)" :
             "several entry states from $(bg) ($(join(entries, ", ")))") *
            ", so there is no default seed; pass an explicit `initial` with fractions of all " *
            "nodes, e.g. SeedFraction(:X_$(a) => ρ·n_$(a), …)"))
    end
    firsts = [X for X in infectious_species(ctx.cm) if ctx.type_of[X] === a]
    isempty(firsts) && throw(ArgumentError(
        "$(ctx.where): the node type $(a) has no infectious species to seed; pass an explicit " *
        "`initial`"))
    return first(firsts)
end

"""
    _mt_resolve_seed(ctx, initial, seed_fraction, seed_state, N) -> (x, backgrounds)

The initial fraction of **all** nodes in every compartment (DESIGN §J.6), `x`, and the background
class of every node type. Without `initial`, a fraction `seed_fraction` of the nodes of every type
a (ρ n_a of all nodes) starts in the type's default seeded class and the rest in its background.
With `initial` (a `SeedSpec` of fractions of all nodes; counts need an integer `N`), the unseeded
nodes of a type are its background; a background named explicitly must have the fraction the
other seeds of its type leave, and a type without a background must have all its fractions named.
"""
function _mt_resolve_seed(ctx, initial, ρ::Real, state::Symbol, N::Real)
    where = ctx.where
    x = Dict{Symbol,Float64}(X => 0.0 for X in ctx.names)
    if initial === nothing
        (isfinite(ρ) && 0 <= ρ <= 1) || throw(ArgumentError(
            "$where: seed_fraction must lie in [0, 1]; got $(ρ)"))
        for a in ctx.types
            bg = ctx.backgrounds[a]
            bg === nothing && throw(ArgumentError(
                "$where: the node type $(a) has no unique susceptible class to hold its unseeded " *
                "nodes; pass an explicit `initial` naming the fraction of all nodes in every " *
                "compartment of that type"))
            X = _mt_seed_class(ctx, a, bg, state)
            x[X] += ρ * ctx.n[a]
            x[bg] += (1 - ρ) * ctx.n[a]
        end
        return x, ctx.backgrounds
    end
    given = Dict{Symbol,Float64}()
    for (X, v) in seed_fractions(initial; N = _seed_population(initial, N))
        X in ctx.species || throw(ArgumentError(
            "$where: the initial condition $(initial) names $(X), which is not a species of the " *
            "model (species: $(join(ctx.species, ", ")))"))
        haskey(given, X) && throw(ArgumentError("$where: $(X) is seeded twice"))
        (isfinite(v) && v >= 0) || throw(ArgumentError(
            "$where: the seed fraction of $(X) must be finite and ≥ 0; got $(v)"))
        given[X] = v
    end
    for a in ctx.types
        bg = ctx.backgrounds[a]
        own = [X for X in ctx.species if ctx.type_of[X] === a]
        used = sum((get(given, X, 0.0) for X in own if X !== bg); init = 0.0)
        na = ctx.n[a]
        used <= na * (1 + 1e-12) || throw(ArgumentError(
            "$where: the seed fractions of the node type $(a) (size $(na)) sum to $(used), more " *
            "than $(na); seed fractions are fractions of all nodes (DESIGN §J.6)"))
        for X in own
            X === bg || (x[X] = get(given, X, 0.0))
        end
        if bg === nothing
            abs(used - na) <= 1e-8 || throw(ArgumentError(
                "$where: the node type $(a) has no unique susceptible class to hold its unseeded " *
                "nodes, so the fractions of its compartments must sum to its size $(na); they sum " *
                "to $(used)"))
        else
            rest = max(na - used, 0.0)
            (haskey(given, bg) && abs(given[bg] - rest) > 1e-8) && throw(ArgumentError(
                "$where: $(bg) is given the fraction $(given[bg]) of all nodes, but the other " *
                "seeds of the node type $(a) leave $(rest); seed fractions are fractions of all " *
                "nodes (DESIGN §J.6), so name only the seeded compartments"))
            x[bg] = rest
        end
    end
    return x, ctx.backgrounds
end

# The π^PW image of the fractions x of all nodes: [X] = N x_X, [XY] = N M_ab x_X x_Y / n_b
# (a = t(X), b = t(Y)), θ = 1, the accumulator N·Σ_{infected} x_X. `centres` are the susceptible
# classes that the closure divides by, which must start non-empty.
function _mt_u0(ctx, singles, pairs, θ, cum, N, x, centres)
    for s in centres
        get(x, s, 0.0) > 0 || throw(ArgumentError(
            "$(ctx.where): no node starts in the susceptible class $(s); the closure " *
            "[Y s J] = K[Ys][sJ]/[s] needs [s] > 0"))
    end
    u0 = Dict{Any,Float64}()
    for (X, v) in singles
        u0[v] = N * get(x, X, 0.0)
    end
    for ((X, Y), v) in pairs
        a, b = ctx.type_of[X], ctx.type_of[Y]
        u0[v] = N * ctx.M[ctx.tindex[a], ctx.tindex[b]] * get(x, X, 0.0) * get(x, Y, 0.0) / ctx.n[b]
    end
    for v in values(θ)
        u0[v] = 1.0
    end
    cum === nothing || (u0[cum] = N * sum((get(x, X, 0.0) for X in ctx.infected); init = 0.0))
    return u0
end

# ─── Observing ───────────────────────────────────────────────────────────────

# The compartments that refine `base` (their label's base), or the sinks for `:removed`.
function _mt_members(sys::MultitypePairwiseSystem, base::Symbol)
    base === REMOVED_COMPARTMENT && return copy(sys.metadata[:sinks])
    labels = species_labels(sys.metadata[:contact_model])
    return sort!([X for X in keys(sys.singles) if haskey(labels, X) && labels[X].base === base])
end

function _mt_theta(sys::MultitypePairwiseSystem, X::Symbol)
    for ((b, a), v) in sys.metadata[:thetas]
        X === Symbol(:θ_, b, :_, a) && return v
    end
    return nothing
end

"""
    compartment(sys::MultitypePairwiseSystem, sol, X::Symbol) -> Vector

The time series (on the population scale of the system) of the single [X] of the compartment
`X`, of the cumulative-incidence accumulator (`:cumulative`), of a θ state of `PGFClosure`
(`:θ_<b>_<a>`), or of the sum over the node types of a base compartment (`:S`, `:I`, … from the
species labels of a stratified model; `:removed` for the removal sinks).
"""
function compartment(sys::MultitypePairwiseSystem, sol, X::Symbol)
    haskey(sys.singles, X) && return sol[sys.singles[X]]
    md = sys.metadata
    (X === :cumulative && md[:cumulative] !== nothing) && return sol[md[:cumulative]]
    v = _mt_theta(sys, X)
    v === nothing || return sol[v]
    members = _mt_members(sys, X)
    isempty(members) || return sum(sol[sys.singles[Y]] for Y in members)
    θs = sort!([Symbol(:θ_, b, :_, a) for (b, a) in keys(md[:thetas])])
    throw(ArgumentError("unknown node-level compartment: $(X); available: " *
                        join(sort!(collect(keys(sys.singles))), ", ") *
                        (md[:cumulative] === nothing ? "" : ", cumulative") *
                        (isempty(θs) ? "" : ", " * join(θs, ", "))))
end

"""
    population_fraction(sys::MultitypePairwiseSystem, sol, X::Symbol; N = nothing)

The fraction of **all** nodes in the compartment `X`, in `:cumulative`, or in a base compartment
summed over the node types (`:I` = Σ_a I_a): the single divided by `N`, which defaults to the
population scale the system was built with.
"""
function population_fraction(sys::MultitypePairwiseSystem, sol, X::Symbol; N = nothing)
    _mt_theta(sys, X) === nothing || throw(ArgumentError(
        "$(X) is not a population fraction; use compartment(sys, sol, :$(X))"))
    return compartment(sys, sol, X) ./ something(N, sys.metadata[:N])
end

"""
    model_curves(sys::MultitypePairwiseSystem, sol; t = sol.t, label) -> ModelCurves

The observables of a solved multitype pairwise system on the grid `t` as fractions of all nodes:
every compartment, `:infectious` (the sum of the infectious compartments), `:cumulative` (the
fraction ever infected, seeds included, with infection status defined structurally as in
NetworkOutbreaks' `final_size`, DESIGN §J.8; present unless the system was built with
`cumulative = false`), and the totals over the node types of each base compartment of a
stratified model (`:S`, `:I`, `:R`, …, and `:removed` for the removal sinks) where that name is
not itself a compartment. The representation is `:pairwise`, `:pgf_closure` or `:s_anchored`.
"""
function model_curves(sys::MultitypePairwiseSystem, sol; t = sol.t,
                      label::AbstractString = sys.metadata[:label])
    tt = collect(Float64, t)
    md = sys.metadata
    scale = md[:N]
    at(v) = _values_at(sol, v, tt) ./ scale
    vals = Dict{Symbol,Vector{Float64}}(X => at(v) for (X, v) in sys.singles)
    vals[:infectious] = reduce(+, (vals[X] for X in md[:infectious]); init = zeros(length(tt)))
    md[:cumulative] === nothing || (vals[:cumulative] = at(md[:cumulative]))
    labels = species_labels(md[:contact_model])
    bases = unique!(Symbol[labels[X].base for X in keys(sys.singles) if haskey(labels, X)])
    isempty(md[:sinks]) || push!(bases, REMOVED_COMPARTMENT)
    for B in bases
        haskey(vals, B) && continue
        members = _mt_members(sys, B)
        isempty(members) || (vals[B] = reduce(+, (vals[Y] for Y in members)))
    end
    return ModelCurves(tt, vals; label, representation = md[:representation],
                       metadata = Dict{Symbol,Any}(:closure => string(sys.closure),
                                                   :network => "MultitypeNetwork(" *
                                                               join(sys.network.types, ", ") * ")"))
end
