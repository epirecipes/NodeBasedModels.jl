# s_anchored.jl — the S-anchored pairwise level, `node_based(…; level = :s_anchored)`, and the
# builder of the pairwise models of T_EB models closed at S-centred triples, which PGFClosure
# (closures_pgf.jl) uses at both levels. The solve/observe methods here are shared with
# `MeanFieldSystem` (wellmixed.jl).
#
# Owner: WP23 (DESIGN_NetworkEpiCore.md §A.4, §D.4, §D.5 M6/M7/M8, §E.2, §J.1, §J.2, §J.8).
#
# The mathematics
# ---------------
# In a T_EB model (one susceptible class s; every contact s + J → P + J converts an s node; no
# reaction produces s) a node changes state through a contact only while it is in s, so every
# triple of the pairwise equations is centred at an s node. [XY] is the ordered pair count, the
# number of ordered adjacent pairs (u, v) with u ∈ X, v ∈ Y (the NodeBasedModels convention of
# population_pairwise.jl: a cross pair counts each XY edge once, a self pair [XX] twice; DESIGN
# §E.2). Let A(X, Y) be the rate of change of [XY] through events at the first node u. Then
#
#     d[XY]/dt = A(X, Y) + A(Y, X),
#
#     contact  s + J → P + J (τ):  A(P, Y) += τ(δ_{YJ}[sY] + [Y s J]),   A(s, Y) −= the same
#     move     W → Z (a):          A(Z, Y) += a[WY],                     A(W, Y) −= a[WY]
#
# where a move is a progression X → Y, a removal X → ∅ (to the absorbing sink `:removed`, DESIGN
# §J.2) or an exit s → Y (vaccination), and the singles obey d[s] −= τ[sJ], d[P] += τ[sJ],
# d[W] −= a[W], d[Z] += a[W]. The S-anchored pairs {[sY]} ∪ {[ss]} form a closed subsystem PW^S
# (A(s, Y) and A(Y, s) involve only [sZ], [ss] and the S-centred triples), the S-anchored level
# (M7). The non-S pairs [XY] follow from the S-anchored ones, so the full set of pairs is closed as
# well. The S-centred triples are closed by
#
#     [Y s J] = K [Ys][sJ]/[s]
#
# - `PGFClosure` (closures_pgf.jl): K = K_ψ(θ) = ψ(θ)ψ''(θ)/ψ'(θ)², with the auxiliary state θ,
#   θ(0) = 1, θ̇ = −ψ(θ)/([s]ψ'(θ)) Σ_r τ_r [s J_r]. On the image of the edge-based model,
#   qξ = [s]/ψ(θ), so the field needs neither q nor the exit factor ξ. The map (M6)
#       π^PW(θ, ξ, φ, pop) = (θ, [s] = qξψ(θ), [sX] = qξψ'(θ)φ_X, [ss] = q²ξ²ψ'(θ)²/ψ'(1),
#                             [X] = pop_X)
#   carries the edge-based field (DESIGN §D.4) to this one for every C² ψ and every T_EB model
#   (Kiss, Kenah & Rempała 2023), so the node observables are those of the edge-based model. The
#   non-S pairs are exact too: the states of the partners of a susceptible node are independent,
#   so every S-centred triple is exact, and no other triple occurs.
# - `BernoulliClosure`: the constant K = K_ψ(1) = ψ''(1)/ψ'(1)² = `closure_constant(d)`, the
#   closure of the heterogeneous (and, (k − 1)/k, the homogeneous) population pairwise model. It
#   equals K_ψ(θ) for all θ iff ψ is Poisson type, ψ' = αψ^κ (M8), so it is exact on Poisson,
#   binomial, regular and negative binomial networks and biased otherwise (`:sir_bim`, `:sir_pl`).
#
# Initial condition (DESIGN §E.2): [X] = Nρ_X, [XY] = ⟨k⟩Nρ_Xρ_Y (ρ_s = q, the unseeded fraction;
# so [sX] = ⟨k⟩qρ_X and [ss] = ⟨k⟩q²), θ = 1: the image under π^PW of θ = ξ = 1, φ_X = pop_X = ρ_X.
# The cumulative-incidence accumulator counts the seeds in infected compartments and every entry
# into infection, with infection status decided from the typing (DESIGN §J.8).

export SAnchoredSystem

"""
    NodeBasedModels._NodeClosureSystem

Internal supertype of the ODE systems of this file and of wellmixed.jl ([`SAnchoredSystem`](@ref),
[`MeanFieldSystem`](@ref)), which share `solve_epidemic`, `default_initial_conditions`,
`model_curves`, `compartment`, `compartments`, `population_fraction` and `symbolic_ode`. Each has
the fields `system`, `u0`, `tspan`, `params`, `model`, `network`, `closure`, `singles` and
`metadata`.
"""
abstract type _NodeClosureSystem end

"""
    SAnchoredSystem

A pairwise model of a T_EB model on a `ConfigurationNetwork`, with every triple centred at the
susceptible class s and closed by [Y s J] = K [Ys][sJ]/[s] (see [`PGFClosure`](@ref)). It is what
[`node_based`](@ref) returns for

- `level = :s_anchored`: the S-anchored subsystem PW^S (DESIGN §D.4, §D.5 M7), with the singles
  [X] of every compartment, the pairs [sX] (every X ≠ s, including the removal sink) and [ss],
  and, for `PGFClosure`, the auxiliary θ. The closure is `PGFClosure()` (K = K_ψ(θ), exact for
  every degree distribution) or `BernoulliClosure()` (the constant K = `closure_constant(d)`,
  exact iff the degree distribution is Poisson type). Representation `:s_anchored`;
- `level = :population` with `closure = PGFClosure()`: the population pairwise model with every
  pair [XY] and θ (the non-S pairs follow from the S-centred triples, so the model is closed and,
  with K_ψ(θ), exact). Representation `:pgf_closure`.

The node observables of both are those of the edge-based model with `PGFClosure`. States are in
counts on the population scale `N` (fractions for `N = 1`), pairs in the ordered-pair convention
of [`PairwiseSystem`](@ref).

Fields:
- `system`: the compiled ModelingToolkit system; `u0`: the initial condition (`Dict`);
  `tspan`; `params`: parameter values given when the system was built;
- `model`: the model as passed; `network`: the `ConfigurationNetwork`; `closure`; `level`;
- `singles`: compartment => [X]; `pairs`: `(A, B)` => [AB], keys in compartment order;
- `metadata`: `:θ` (the state θ, or `nothing`), `:K` (the closure factor, symbolic in θ or a
  number), `:susceptible` (s), `:compartmental` (the lowered `CompartmentalModel`),
  `:contact_model` (the per-contact `ContactModel`), `:N`, `:background`, `:seed`,
  `:seed_state`, `:default_fraction`, `:cumulative` (the accumulator state, or `nothing`),
  `:infected` (DESIGN §J.8), `:infectious`, `:parameters` (MTK parameters by name), `:states`
  and `:rhs` (the uncompiled vector field without the accumulator), `:representation`,
  `:label`, `:name`.

Solve it with [`solve_epidemic`](@ref) and read it with [`model_curves`](@ref),
[`compartment`](@ref) and [`population_fraction`](@ref); `symbolic_ode(sys)` is its vector field.

```julia
sc  = scenario(:sir_bim)                                   # bimodal {2, 10}: not Poisson type
pgf = node_based(sc; closure = PGFClosure(), level = :s_anchored)
sol = solve_epidemic(pgf, sc)
model_curves(pgf, sol; t = sc.tgrid)                       # = the edge-based curves
```
"""
struct SAnchoredSystem <: _NodeClosureSystem
    system::Any
    u0::Dict{Any,Float64}
    tspan::Tuple{Float64,Float64}
    params::Dict{Symbol,Float64}
    model::Any
    network::ConfigurationNetwork
    closure::ClosureMethod
    level::Symbol
    singles::Dict{Symbol,Any}
    pairs::Dict{Tuple{Symbol,Symbol},Any}
    metadata::Dict{Symbol,Any}
end

function Base.show(io::IO, sys::SAnchoredSystem)
    θ = sys.metadata[:θ] === nothing ? "" : " and θ"
    print(io, "SAnchoredSystem(", nameof(sys.metadata[:contact_model]), "; level = :", sys.level,
          ", ", length(sys.singles), " singles, ", length(sys.pairs), " pairs", θ,
          ", closure = ", sys.closure, ")")
end

"""
    node_variables(sys::SAnchoredSystem) -> Dict{Symbol,Any}
    pair_variables(sys::SAnchoredSystem) -> Dict{Tuple{Symbol,Symbol},Any}

The single variables [X] (by compartment) and the pair variables [AB] (by `(A, B)`, in
compartment order) of an [`SAnchoredSystem`](@ref); θ is `sys.metadata[:θ]`.
"""
node_variables(sys::SAnchoredSystem) = sys.singles
pair_variables(sys::SAnchoredSystem) = sys.pairs

# ─── Closure hooks (PGFClosure adds its methods in closures_pgf.jl) ──────────

# Whether the closure has the auxiliary state θ.
_closure_theta(::ClosureMethod) = false

# The factor K of [Y s J] = K [Ys][sJ]/[s] for the degree distribution d (θ is the state θ, or
# `nothing` for a closure without it).
_closure_factor(::BernoulliClosure, d::DegreeDistribution, θ) = Float64(closure_constant(d))

# A short description of the closure for labels.
_closure_label(::BernoulliClosure, d::DegreeDistribution) =
    "constant K = $(round(Float64(closure_constant(d)); sigdigits = 4))"

# ─── node_based methods ──────────────────────────────────────────────────────

function _node_based(::Val{:s_anchored}, closure::BernoulliClosure, cm::ContactModel,
                     net::ConfigurationNetwork; kw...)
    return _closed_pairwise(cm, net, closure; level = :s_anchored, kw...)
end

# Every other closure or descriptor at the S-anchored level.
function _node_based(::Val{:s_anchored}, closure, cm::ContactModel, net; kw...)
    throw(ArgumentError("node_based(…; level = :s_anchored): " *
                        _s_anchored_unsupported(closure, net)))
end

_s_anchored_unsupported(closure, ::ConfigurationNetwork) =
    "the S-anchored subsystem closes the S-centred triples [Y s J] with PGFClosure() (K_ψ(θ), " *
    "exact for every degree distribution) or BernoulliClosure() (the constant K = " *
    "closure_constant(d), exact iff the degrees are Poisson type); got $(closure)"
_s_anchored_unsupported(closure, ::WellMixed) =
    "a well-mixed population has no persistent pairs; use node_based(model, WellMixed(κ); " *
    "closure = MeanFieldClosure()) (mass action)"
_s_anchored_unsupported(closure, ::ClusteredNetwork) =
    "on a clustered network the closure of an S-centred triple needs the non-S pair that closes " *
    "the triangle (Keeling), so there is no S-anchored subsystem; use level = :population with " *
    "KeelingClosure()"
_s_anchored_unsupported(closure, net) =
    "the S-anchored subsystem needs a ConfigurationNetwork (its closure is a function of the " *
    "degree distribution); got $(nameof(typeof(net)))"

# ─── The builder ─────────────────────────────────────────────────────────────

# The mean degree ψ'(1) of d as a positive Float64, or an ArgumentError.
function _numeric_mean_degree(d::DegreeDistribution, where)
    k̄ = mean_degree(d)
    (k̄ isa Real && !(k̄ isa Symbolics.Num)) || throw(ArgumentError(
        "$where: the degree distribution $(d) has a symbolic mean degree; the pairwise initial " *
        "condition [XY] = ⟨k⟩ρ_Xρ_Y needs numeric degree parameters"))
    (isfinite(k̄) && k̄ > 0) || throw(ArgumentError(
        "$where: the mean degree of $(d) is $(k̄), so the network has no pairs"))
    return Float64(k̄)
end

# The independent variable and the time derivative of the systems of this file and wellmixed.jl.
const _TIME = ModelingToolkit.t_nounits
const _DT = ModelingToolkit.D_nounits

_closed_state(name::Symbol) = only(@variables $(name)(_TIME))

"""
    _closed_pairwise(cm::ContactModel, net::ConfigurationNetwork, closure; level, source, p,
                     name, tspan, N, ε, seed_fraction, seed_state, initial, cumulative)
        -> SAnchoredSystem

Build the pairwise model of the T_EB model `cm` on `net` whose S-centred triples are closed by
`closure` (the header of src/s_anchored.jl gives the equations): the S-anchored pairs only for
`level = :s_anchored`, every pair for `level = :population`. Rates are converted to per-contact
rates with `per_contact_rates(cm, net)`, and `require_admissible(cm, :s_anchored; network)` is
checked first (SIS and SIRS raise an `AdmissibilityError`). The keywords are those of
[`node_based`](@ref) at the population level: `tspan = (0.0, 100.0)`, the population scale
`N = 1.0`, the seeding (`initial`, or a fraction `seed_fraction = ε = 1e-3` in `seed_state =
:entry`, the unique entry state) and `cumulative = true` (the cumulative-incidence accumulator).
"""
function _closed_pairwise(cm::ContactModel, net::ConfigurationNetwork, closure::ClosureMethod;
                          level::Symbol, source = cm, p = nothing,
                          name::Union{Nothing,Symbol} = nothing,
                          tspan::Tuple{Real,Real} = (0.0, 100.0), N::Real = 1.0,
                          ε::Real = 1e-3, seed_fraction::Real = ε, seed_state::Symbol = :entry,
                          initial::Union{Nothing,SeedSpec} = nothing, cumulative::Bool = true)
    level in (:s_anchored, :population) || throw(ArgumentError(
        "_closed_pairwise: level must be :s_anchored or :population; got :$(level)"))
    where = "node_based(:$(nameof(cm)), $(nameof(typeof(net))); level = :$(level), " *
            "closure = $(closure))"
    (isfinite(N) && N > 0) || throw(ArgumentError(
        "$where: the population scale N must be finite and > 0; got $(N)"))
    d = net.degrees
    k̄ = _numeric_mean_degree(d, where)
    cmτ = _per_contact_model(cm, net)
    require_admissible(cmτ, :s_anchored; network = net)
    s = only(susceptible_species(cmτ))
    cmodel = CompartmentalModel(cmτ)
    names = cmodel.compartment_names
    order = Dict(X => i for (i, X) in enumerate(names))
    key(a::Symbol, b::Symbol) = order[a] <= order[b] ? (a, b) : (b, a)

    # The reactions, per reaction as in the edge-based field: contacts (J, P, τ) with recipient s,
    # and moves (W, Z, a): progressions, removals to the sink, exits out of s.
    for c in contacts(cmτ)
        c.recipient === s || throw(ArgumentError(
            "$where: the contact $(c.name) has the recipient $(c.recipient), not $(s)"))
    end
    rate_names = unique!(reduce(vcat, (_rate_parameter_names(r.rate)
                                       for r in vcat(contacts(cmτ), node_transitions(cmτ)));
                                init = Symbol[]))
    params = Dict{Symbol,Any}(n => only(@parameters $(n)) for n in rate_names)
    rate(r) = _rate_symbolic(r, params, _TIME)
    cs = [(c.infector, c.product, rate(c.rate)) for c in contacts(cmτ)]
    ms = [(tr.from, something(tr.to, REMOVED_COMPARTMENT), rate(tr.rate))
          for tr in node_transitions(cmτ)]

    # The tracked pairs and the generated names (which must be distinct from each other and from
    # the rate parameters).
    tracked = level === :s_anchored ? unique!([key(s, X) for X in names]) :
              [(names[i], names[j]) for i in eachindex(names) for j in i:length(names)]
    uses_θ = _closure_theta(closure)
    generated = vcat(names, Symbol[Symbol(a, b) for (a, b) in tracked],
                     uses_θ ? [:θ] : Symbol[], cumulative ? [:cumulative] : Symbol[])
    dup = unique!([n for n in generated if count(==(n), generated) > 1])
    isempty(dup) || throw(ArgumentError(
        "$where: the generated state names collide: $(join(dup, ", ")) (pairs are named by " *
        "joining the compartment names); rename the compartments"))
    bad = sort!([n for n in rate_names if n in generated])
    isempty(bad) || throw(ArgumentError(
        "$where: the parameter name(s) $(join(bad, ", ")) collide with state names of the " *
        "pairwise model; rename the parameter(s)"))

    singles = Dict{Symbol,Any}(X => _closed_state(X) for X in names)
    pairs = Dict{Tuple{Symbol,Symbol},Any}(k => _closed_state(Symbol(k...)) for k in tracked)
    θ = uses_θ ? _closed_state(:θ) : nothing
    K = _closure_factor(closure, d, θ)
    S = singles[s]
    pair(a::Symbol, b::Symbol) = pairs[key(a, b)]
    triple(Y::Symbol, J::Symbol) = K * pair(Y, s) * pair(s, J) / S       # [Y s J]

    # A(X, Y): the rate of change of [XY] through events at the first node.
    function first_node(X::Symbol, Y::Symbol)
        acc = Symbolics.Num(0)
        for (J, P, τ) in cs
            (X === P || X === s) || continue
            ev = τ * (Y === J ? pair(s, Y) + triple(Y, J) : triple(Y, J))
            acc += X === P ? ev : -ev
        end
        for (W, Z, a) in ms
            X === Z && (acc += a * pair(W, Y))
            X === W && (acc -= a * pair(W, Y))
        end
        return acc
    end

    dsingle = Dict{Symbol,Any}(X => Symbolics.Num(0) for X in names)
    for (J, P, τ) in cs
        flux = τ * pair(s, J)
        dsingle[s] -= flux
        dsingle[P] += flux
    end
    for (W, Z, a) in ms
        flux = a * singles[W]
        dsingle[W] -= flux
        dsingle[Z] += flux
    end
    states = Any[]
    rhs = Any[]
    if θ !== nothing
        hazard = sum((τ * pair(s, J) for (J, _, τ) in cs); init = Symbolics.Num(0))
        push!(states, θ)
        push!(rhs, _theta_rhs(closure, d, θ, S, hazard))
    end
    for X in names
        push!(states, singles[X])
        push!(rhs, dsingle[X])
    end
    for k in tracked
        push!(states, pairs[k])
        push!(rhs, first_node(k[1], k[2]) + first_node(k[2], k[1]))
    end

    # The cumulative-incidence accumulator (DESIGN §J.8): contacts into an infected compartment
    # (s is never infected), and moves from a non-infected into an infected compartment.
    infected = _infected_compartments(cmτ)
    eqs = Equation[_DT(x) ~ f for (x, f) in zip(states, rhs)]
    cum = nothing
    if cumulative
        cum = _closed_state(:cumulative)
        inc = Symbolics.Num(0)
        for (J, P, τ) in cs
            P in infected && (inc += τ * pair(s, J))
        end
        for (W, Z, a) in ms
            (!(W in infected) && Z in infected) && (inc += a * singles[W])
        end
        push!(eqs, _DT(cum) ~ inc)
    end
    sysname = something(name, Symbol(level === :s_anchored ? :s_anchored_ : :pgf_pairwise_,
                                      nameof(cm)))
    compiled = mtkcompile(System(eqs, _TIME; name = sysname))

    # Seeding: the rules of the population pairwise model (lift.jl), then the π^PW image.
    bg = _seed_background(cmτ)
    resolve = (init, ρ, state) -> _cm_resolve_seed(cmτ, cmodel, init, ρ, state, N, bg)
    initial_state = (x, background) -> _closed_pairwise_u0(names, s, singles, pairs, θ, cum,
                                                           Float64(N), k̄, x, background,
                                                           infected, where)
    x, background = resolve(initial, seed_fraction, seed_state)
    u0 = initial_state(x, background)

    representation = level === :s_anchored ? :s_anchored : :pgf_closure
    label = (level === :s_anchored ? "S-anchored pairwise (" : "pairwise (") *
            _closure_label(closure, d) * ")"
    metadata = Dict{Symbol,Any}(
        :θ => θ, :K => K, :susceptible => s, :compartmental => cmodel, :contact_model => cmτ,
        :descriptor => net, :N => Float64(N), :background => background, :seed => x,
        :resolve => resolve, :initial_state => initial_state,
        :default_fraction => Float64(seed_fraction), :seed_state => seed_state,
        :cumulative => cum, :infected => infected, :infectious => infectious_species(cmτ),
        :parameters => params, :states => states, :rhs => rhs,
        :domain => θ === nothing ? Pair{Any,Tuple{Float64,Float64}}[] :
                   Pair{Any,Tuple{Float64,Float64}}[θ => (0.05, 1.0)],
        :representation => representation, :label => label, :name => sysname,
        :level => level)
    vals = p === nothing ? Dict{Symbol,Float64}() :
           Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in _pairs_of(p))
    return SAnchoredSystem(compiled, u0, (Float64(tspan[1]), Float64(tspan[2])), vals, source,
                           net, closure, level, singles, pairs, metadata)
end

# Iterate `name => value` over a Dict or a NamedTuple.
_pairs_of(p::AbstractDict) = pairs(p)
_pairs_of(p::NamedTuple) = pairs(p)
_pairs_of(p) = throw(ArgumentError(
    "parameter values must be a Dict or a NamedTuple; got $(typeof(p))"))

# The π^PW image of the seeding x (fractions by compartment; DESIGN §E.2).
function _closed_pairwise_u0(names, s, singles, pairs, θ, cum, N, k̄, x, background, infected,
                             where)
    get(x, s, 0.0) > 0 || throw(ArgumentError(
        "$where: no node starts in the susceptible class $(s); the closure [Y s J] = " *
        "K[Ys][sJ]/[s] needs [s] > 0"))
    u0 = Dict{Any,Float64}()
    for X in names
        u0[singles[X]] = N * get(x, X, 0.0)
    end
    for ((a, b), v) in pairs
        u0[v] = k̄ * N * get(x, a, 0.0) * get(x, b, 0.0)
    end
    θ === nothing || (u0[θ] = 1.0)
    cum === nothing || (u0[cum] = N * _seeded_infections(x, background, infected))
    return u0
end

# ─── Solving and observing (shared with MeanFieldSystem) ─────────────────────

"""
    default_initial_conditions(sys::SAnchoredSystem; initial = nothing, seed_fraction = nothing,
                               seed_state = nothing)
    default_initial_conditions(sys::MeanFieldSystem; …)

The initial condition of `sys`: the stored `sys.u0` when no keyword is given, otherwise that of
the seeding `initial` (a `SeedSpec`) or of a fraction `seed_fraction` in `seed_state` (the
defaults are those the system was built with), by the rules of [`node_based`](@ref). For an
`SAnchoredSystem` it is the π^PW image of the edge-based initial condition: [X] = Nρ_X,
[XY] = ⟨k⟩Nρ_Xρ_Y and θ = 1 (DESIGN §E.2). The `cumulative` accumulator starts at the seeds in
infected compartments.
"""
function default_initial_conditions(sys::_NodeClosureSystem; initial = nothing,
                                    seed_fraction = nothing,
                                    seed_state::Union{Nothing,Symbol} = nothing)
    initial === nothing && seed_fraction === nothing && seed_state === nothing && return sys.u0
    md = sys.metadata
    x, bg = md[:resolve](initial, something(seed_fraction, md[:default_fraction]),
                         something(seed_state, md[:seed_state]))
    return md[:initial_state](x, bg)
end

"""
    solve_epidemic(sys::SAnchoredSystem; p = nothing, initial = nothing, tspan = sys.tspan,
                   saveat = nothing, solver = nothing, reltol = 1e-8, abstol = 1e-10, kwargs...)
    solve_epidemic(sys::SAnchoredSystem, p::AbstractDict; kwargs...)
    solve_epidemic(sys::SAnchoredSystem, sc::Scenario; kwargs...)

(and the same for [`MeanFieldSystem`](@ref)). Solve the system with the parameter values `p`
(by name; the model's defaults and the values given to `node_based` fill in the rest, and an
unknown name is an error), an optional seeding `initial` (see
[`default_initial_conditions`](@ref)), the time span and the save grid. The tolerances default
to those of [`solve_pairwise`](@ref); `solver` selects the algorithm. The scenario form takes
`p`, `initial`, `tspan` and `saveat` from `sc`.
"""
function solve_epidemic(sys::_NodeClosureSystem; p = nothing, initial = nothing,
                        tspan::Tuple{Real,Real} = sys.tspan, saveat = nothing, solver = nothing,
                        reltol = 1e-8, abstol = 1e-10, kwargs...)
    u0 = initial === nothing ? sys.u0 : default_initial_conditions(sys; initial)
    vals = _closed_parameter_values(sys, p)
    byname = Dict(Symbol(q) => q for q in ModelingToolkit.parameters(sys.system))
    op = Dict{Any,Float64}(u0)
    for (k, v) in vals
        op[byname[k]] = v
    end
    prob = ModelingToolkit.ODEProblem(sys.system, op, (Float64(tspan[1]), Float64(tspan[2])))
    opts = saveat === nothing ? (; reltol, abstol, kwargs...) :
                                (; reltol, abstol, saveat, kwargs...)
    solver === nothing && return OrdinaryDiffEqDefault.solve(prob; opts...)
    return OrdinaryDiffEqDefault.solve(prob, solver; opts...)
end

solve_epidemic(sys::_NodeClosureSystem, p::AbstractDict; kwargs...) =
    solve_epidemic(sys; p, kwargs...)

solve_epidemic(sys::_NodeClosureSystem, sc::Scenario; kwargs...) =
    solve_epidemic(sys; p = sc.params, initial = sc.initial, tspan = sc.tspan,
                   saveat = sc.tgrid, kwargs...)

# Parameter values by name: the model's defaults, then `sys.params`, then `p`, restricted to the
# parameters of the system; unknown names and missing values are errors.
function _closed_parameter_values(sys::_NodeClosureSystem, p)
    cm = sys.metadata[:contact_model]
    vals = Dict{Symbol,Float64}(parameter_defaults(cm))
    needed = Set(Symbol(q) for q in ModelingToolkit.parameters(sys.system))
    known = union(needed, Set{Symbol}(keys(sys.metadata[:parameters])),
                  Set{Symbol}(_parameter_symbol(q) for q in rate_parameters(cm)))
    unknown = Symbol[]
    for given in (sys.params, something(p, Dict{Symbol,Float64}())), (k, v) in _pairs_of(given)
        n = Symbol(k)
        n in known || push!(unknown, n)
        vals[n] = Float64(v)
    end
    isempty(unknown) || throw(ArgumentError(
        "Unknown parameter names: $(join(string.(sort!(unique!(unknown))), ", ")); the " *
        "parameters of this system are $(join(string.(sort!(collect(needed))), ", "))"))
    missing_ = sort!([n for n in needed if !haskey(vals, n)])
    isempty(missing_) || throw(ArgumentError(
        "Missing parameter values for: $(join(string.(missing_), ", "))"))
    return Dict{Symbol,Float64}(k => v for (k, v) in vals if k in needed)
end

"""
    compartment(sys::SAnchoredSystem, sol, X::Symbol) -> Vector
    compartment(sys::MeanFieldSystem, sol, X::Symbol) -> Vector

The time series of the single [X] of compartment `X` in the solution `sol` (on the population
scale of the system), of the accumulator (`:cumulative`) or, for `PGFClosure`, of `:θ`. Pairs are
read with `sol[pair_variables(sys)[(A, B)]]`.
"""
function compartment(sys::_NodeClosureSystem, sol, X::Symbol)
    haskey(sys.singles, X) && return sol[sys.singles[X]]
    md = sys.metadata
    (X === :cumulative && md[:cumulative] !== nothing) && return sol[md[:cumulative]]
    (X === :θ && get(md, :θ, nothing) !== nothing) && return sol[md[:θ]]
    throw(ArgumentError("unknown node-level compartment: $(X); available: " *
                        "$(join(sort!(collect(keys(sys.singles))), ", "))" *
                        (md[:cumulative] === nothing ? "" : ", cumulative") *
                        (get(md, :θ, nothing) === nothing ? "" : ", θ")))
end

"""
    compartments(sys::SAnchoredSystem, sol, Xs::AbstractVector{Symbol}) -> Dict
    compartments(sys::MeanFieldSystem, sol, Xs::AbstractVector{Symbol}) -> Dict

Several compartments at once (see [`compartment`](@ref)).
"""
compartments(sys::_NodeClosureSystem, sol, Xs::AbstractVector{Symbol}) =
    Dict(X => compartment(sys, sol, X) for X in Xs)

"""
    population_fraction(sys::SAnchoredSystem, sol, X::Symbol; N = nothing)
    population_fraction(sys::MeanFieldSystem, sol, X::Symbol; N = nothing)

The fraction of the population in compartment `X` (or `:cumulative`): the single [X] divided by
`N`, which defaults to the population scale the system was built with.
"""
function population_fraction(sys::_NodeClosureSystem, sol, X::Symbol; N = nothing)
    X === :θ && throw(ArgumentError(
        "θ is not a population fraction; use compartment(sys, sol, :θ)"))
    return compartment(sys, sol, X) ./ something(N, sys.metadata[:N])
end

"""
    model_curves(sys::SAnchoredSystem, sol; t = sol.t, label) -> ModelCurves
    model_curves(sys::MeanFieldSystem, sol; t = sol.t, label) -> ModelCurves

The observables of a solved system on the grid `t` as population fractions: every compartment,
`:infectious` (the sum of the infectious compartments) and `:cumulative` (the fraction ever
infected, seeds included, with infection status defined structurally as in NetworkOutbreaks'
`final_size`, DESIGN §J.8; present unless the system was built with `cumulative = false`). The
result is a `NetworkEpiCore.ModelCurves` with representation `:s_anchored` (level
`:s_anchored`), `:pgf_closure` (`PGFClosure` at the population level) or `:mean_field`, which
`compare` and the plot recipes accept. The default label names the level and the closure.
"""
function model_curves(sys::_NodeClosureSystem, sol; t = sol.t,
                      label::AbstractString = sys.metadata[:label])
    tt = collect(Float64, t)
    md = sys.metadata
    scale = md[:N]
    at(v) = _values_at(sol, v, tt) ./ scale
    vals = Dict{Symbol,Vector{Float64}}(X => at(v) for (X, v) in sys.singles)
    vals[:infectious] = reduce(+, (vals[X] for X in md[:infectious]); init = zeros(length(tt)))
    md[:cumulative] === nothing || (vals[:cumulative] = at(md[:cumulative]))
    return ModelCurves(tt, vals; label, representation = md[:representation],
                       metadata = Dict{Symbol,Any}(:closure => string(sys.closure),
                                                   :network => string(sys.network)))
end

"""
    symbolic_ode(sys::SAnchoredSystem) -> SymbolicODE
    symbolic_ode(sys::MeanFieldSystem) -> SymbolicODE

The uncompiled vector field of `sys` as a `NetworkEpiCore.SymbolicODE`, for
`vector_fields_equal` and `verify`: for an `SAnchoredSystem`, θ (with `PGFClosure`), the singles
and the tracked pairs; for a `MeanFieldSystem`, the singles. The cumulative-incidence
accumulator, an observer of the trajectory, is not part of the field (as for edge-based
systems). With `PGFClosure` the probe box of θ is (0.05, 1], where ψ and ψ' do not vanish.
"""
function symbolic_ode(sys::_NodeClosureSystem)
    md = sys.metadata
    params = Any[md[:parameters][n] for n in sort!(collect(keys(md[:parameters])))]
    return SymbolicODE(md[:name]; states = md[:states], rhs = md[:rhs], parameters = params,
                       domain = md[:domain])
end
