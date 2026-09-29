# wellmixed.jl — `MeanFieldClosure`, [XY] = κ[X][Y] on `WellMixed(κ)` (mass action), and
# `default_closure(::WellMixed)`.
#
# Owner: WP23 (DESIGN_NetworkEpiCore.md §A.4, §C.2, §C.3, §D.5 M1/M14, Λ5; work package in §G.2).
#
# On WellMixed(κ) every node has κ contacts at a time with partners drawn afresh, uniformly at
# random (fleeting contacts): the per-capita hazard of a contact r + J → P + J at per-contact rate
# τ is κτ[J]/N (DESIGN §C.2), i.e. the pair count [rJ] is κ[r][J]/N. Closing the node equations of
# the pairwise model with [XY] = κ[X][Y]/N gives
#
#     d[r]/dt −= κτ[r][J]/N,   d[P]/dt += κτ[r][J]/N,   d[W]/dt −= a[W],   d[Z]/dt += a[W]
#
# for the contacts and the moves W → Z (removals to the sink `:removed`), which is the mass-action
# model MA(c_κ P), β = κτ (NetworkEpiCore's `mass_action(cm; κ)`), for every IR model: the
# well-mixed unit of the edge-based lift (M1) and the symmetric subspace of the individual-based
# model on the complete graph (M14). On a configuration network the same closure with κ = ⟨k⟩ is
# only the dense-limit approximation Λ5; it is available as node_based(model, WellMixed(⟨k⟩)).

export MeanFieldClosure, MeanFieldSystem

"""
    MeanFieldClosure()

The mean-field pair closure [XY] = κ[X][Y]/N of a well-mixed population `WellMixed(κ)` (DESIGN
§A.4, §C.3): each node has κ fleeting contacts with uniformly random partners, so a contact
`r + J → P + J` at per-contact rate τ has the per-capita hazard κτ[J]/N, and
`node_based(model, WellMixed(κ))` (this closure is `default_closure(WellMixed(κ))`) is the
mass-action model with β = κτ, `NetworkEpiCore.mass_action(model; κ)`, for every model (SIS and
SIRS included). Rates given with the frequency-dependent convention (β) are converted with
`per_contact_rates(model, WellMixed(κ))` (τ = β/κ), so they are mass action with that β for every
κ. It returns a [`MeanFieldSystem`](@ref).

On a configuration network the closure is only the dense limit Λ5 (an approximation), so it is
an `ArgumentError` there; `node_based(model, WellMixed(mean_degree(net)))` gives that
approximation explicitly.

```julia
sc  = scenario(:sir_wm5)                        # WellMixed(5), τ = 1/10: MA(β = 1/2, γ = 1/4)
sys = node_based(sc)                            # MeanFieldClosure by default
sol = solve_epidemic(sys, sc)
vector_fields_equal(symbolic_ode(sys), mass_action(sc.model; κ = 5))   # true
```
"""
struct MeanFieldClosure <: ClosureMethod end

"""
    default_closure(::WellMixed) -> MeanFieldClosure

A well-mixed population is closed by the mean-field pair closure [XY] = κ[X][Y]/N
([`MeanFieldClosure`](@ref)), which gives mass action.
"""
default_closure(::WellMixed) = MeanFieldClosure()

"""
    MeanFieldSystem

The mean-field (mass-action) system of a model on `WellMixed(κ)`, returned by
`node_based(model, WellMixed(κ); closure = MeanFieldClosure())` (the default closure there): the
singles [X] of every compartment (removals go to the absorbing compartment `:removed`, DESIGN
§J.2) and the cumulative-incidence accumulator, closed with [XY] = κ[X][Y]/N (see
[`MeanFieldClosure`](@ref)). States are counts on the population scale `N` (fractions for
`N = 1`).

Fields: `system` (the compiled ModelingToolkit system), `u0`, `tspan`, `params`, `model` (as
passed), `network` (the `WellMixed` descriptor), `closure`, `singles` (compartment => [X]) and
`metadata` (`:κ`, `:compartmental`, `:contact_model` (the per-contact `ContactModel`), `:N`,
`:background`, `:seed`, `:seed_state`, `:default_fraction`, `:cumulative`, `:infected`,
`:infectious`, `:parameters`, `:states` and `:rhs` (the uncompiled field without the
accumulator), `:representation` (`:mean_field`), `:label`, `:name`).

It is solved and read like an [`SAnchoredSystem`](@ref): [`solve_epidemic`](@ref),
[`model_curves`](@ref), [`compartment`](@ref), [`population_fraction`](@ref) and
`symbolic_ode(sys)`, which equals `mass_action(cm; κ)` for a model without removals.
"""
struct MeanFieldSystem <: _NodeClosureSystem
    system::Any
    u0::Dict{Any,Float64}
    tspan::Tuple{Float64,Float64}
    params::Dict{Symbol,Float64}
    model::Any
    network::WellMixed
    closure::MeanFieldClosure
    singles::Dict{Symbol,Any}
    metadata::Dict{Symbol,Any}
end

Base.show(io::IO, sys::MeanFieldSystem) =
    print(io, "MeanFieldSystem(", nameof(sys.metadata[:contact_model]), "; κ = ",
          sys.metadata[:κ], ", ", length(sys.singles), " singles)")

"""
    node_variables(sys::MeanFieldSystem) -> Dict{Symbol,Any}

The single variables [X] of a [`MeanFieldSystem`](@ref), by compartment.
"""
node_variables(sys::MeanFieldSystem) = sys.singles

# ─── node_based methods ──────────────────────────────────────────────────────

function _node_based(::Val{:population}, closure::MeanFieldClosure, cm::ContactModel,
                     net::WellMixed; kw...)
    return _mean_field(cm, net, closure; kw...)
end

const _MEAN_FIELD_ONLY_WELL_MIXED =
    "MeanFieldClosure ([XY] = κ[X][Y]) is the closure of a well-mixed population, WellMixed(κ)"

function _node_based(::Val{:population}, ::MeanFieldClosure, cm::ContactModel,
                     net::Union{ConfigurationNetwork,ClusteredNetwork}; kw...)
    throw(ArgumentError("node_based: $(_MEAN_FIELD_ONLY_WELL_MIXED). On a " *
                        "$(nameof(typeof(net))) partnerships persist, and the mean-field closure " *
                        "is only the dense-limit approximation (Λ5); for that approximation use " *
                        "node_based(model, WellMixed(mean_degree(net))), for the network the " *
                        "default closure or PGFClosure()"))
end

function _node_based(::Val{:population}, ::MeanFieldClosure, cm::ContactModel,
                     net::NetworkStructure; kw...)
    throw(ArgumentError("node_based: $(_MEAN_FIELD_ONLY_WELL_MIXED); got a " *
                        "$(nameof(typeof(net))). Use node_based(model, WellMixed(κ))"))
end

# The 0.1 builder (`generate_pairwise` on a NetworkStructure) cannot use MeanFieldClosure.
function triple_closure(::Symbol, ::Symbol, ::Symbol, ::Dict, ::Dict, net::NetworkStructure,
                        ::MeanFieldClosure)
    throw(ArgumentError("$(_MEAN_FIELD_ONLY_WELL_MIXED), not a triple closure of the " *
                        "pairwise model on a $(nameof(typeof(net))). Use " *
                        "node_based(model, WellMixed(κ); closure = MeanFieldClosure())"))
end

# ─── The builder ─────────────────────────────────────────────────────────────

"""
    _mean_field(cm::ContactModel, net::WellMixed, closure::MeanFieldClosure; source, p, name,
                tspan, N, ε, seed_fraction, seed_state, initial, cumulative) -> MeanFieldSystem

Build the mean-field system of `cm` on `net` (the equations are in the header of
src/wellmixed.jl). Rates are converted to per-contact rates with `per_contact_rates(cm, net)`;
every IR model is admissible (`require_admissible(cm, :pairwise)`, total on T_net). The keywords
are those of [`node_based`](@ref) at the population level.
"""
function _mean_field(cm::ContactModel, net::WellMixed, closure::MeanFieldClosure;
                     source = cm, p = nothing, name::Union{Nothing,Symbol} = nothing,
                     tspan::Tuple{Real,Real} = (0.0, 100.0), N::Real = 1.0, ε::Real = 1e-3,
                     seed_fraction::Real = ε, seed_state::Symbol = :entry,
                     initial::Union{Nothing,SeedSpec} = nothing, cumulative::Bool = true)
    where = "node_based(:$(nameof(cm)), WellMixed($(net.κ)); closure = MeanFieldClosure())"
    (isfinite(N) && N > 0) || throw(ArgumentError(
        "$where: the population scale N must be finite and > 0; got $(N)"))
    κ = Float64(net.κ)
    cmτ = _per_contact_model(cm, net)
    require_admissible(cmτ, :pairwise; network = net)
    cmodel = CompartmentalModel(cmτ)
    names = cmodel.compartment_names

    rate_names = unique!(reduce(vcat, (_rate_parameter_names(r.rate)
                                       for r in vcat(contacts(cmτ), node_transitions(cmτ)));
                                init = Symbol[]))
    generated = vcat(names, cumulative ? [:cumulative] : Symbol[])
    cumulative && :cumulative in names && throw(ArgumentError(
        "$where: a compartment is named `cumulative`, which clashes with the accumulator " *
        "state; rename it or pass cumulative = false"))
    bad = sort!([n for n in rate_names if n in generated])
    isempty(bad) || throw(ArgumentError(
        "$where: the parameter name(s) $(join(bad, ", ")) collide with state names; rename " *
        "the parameter(s)"))
    params = Dict{Symbol,Any}(n => only(@parameters $(n)) for n in rate_names)
    rate(r) = _rate_symbolic(r, params, _TIME)
    singles = Dict{Symbol,Any}(X => _closed_state(X) for X in names)

    # The node equations with [rJ] = κ[r][J]/N (κ/N is folded into one number).
    c = κ / N
    dX = Dict{Symbol,Any}(X => Symbolics.Num(0) for X in names)
    infected = _infected_compartments(cmτ)
    inc = Symbolics.Num(0)
    for ct in contacts(cmτ)
        flux = rate(ct.rate) * c * singles[ct.recipient] * singles[ct.infector]
        dX[ct.recipient] -= flux
        dX[ct.product] += flux
        (!(ct.recipient in infected) && ct.product in infected) && (inc += flux)
    end
    for tr in node_transitions(cmτ)
        W, Z = tr.from, something(tr.to, REMOVED_COMPARTMENT)
        flux = rate(tr.rate) * singles[W]
        dX[W] -= flux
        dX[Z] += flux
        (!(W in infected) && Z in infected) && (inc += flux)
    end
    states = Any[singles[X] for X in names]
    rhs = Any[dX[X] for X in names]
    eqs = Equation[_DT(x) ~ f for (x, f) in zip(states, rhs)]
    cum = nothing
    if cumulative
        cum = _closed_state(:cumulative)
        push!(eqs, _DT(cum) ~ inc)
    end
    sysname = something(name, Symbol(:mean_field_, nameof(cm)))
    compiled = mtkcompile(System(eqs, _TIME; name = sysname))

    bg = _seed_background(cmτ)
    resolve = (init, ρ, state) -> _cm_resolve_seed(cmτ, cmodel, init, ρ, state, N, bg)
    scale = Float64(N)
    initial_state = function (x, background)
        u0 = Dict{Any,Float64}(singles[X] => scale * get(x, X, 0.0) for X in names)
        cum === nothing || (u0[cum] = scale * _seeded_infections(x, background, infected))
        return u0
    end
    x, background = resolve(initial, seed_fraction, seed_state)
    u0 = initial_state(x, background)

    metadata = Dict{Symbol,Any}(
        :κ => κ, :compartmental => cmodel, :contact_model => cmτ, :descriptor => net,
        :N => scale, :background => background, :seed => x, :resolve => resolve,
        :initial_state => initial_state, :default_fraction => Float64(seed_fraction),
        :seed_state => seed_state, :cumulative => cum, :infected => infected,
        :infectious => infectious_species(cmτ), :parameters => params, :states => states,
        :rhs => rhs, :domain => Pair{Any,Tuple{Float64,Float64}}[],
        :representation => :mean_field, :label => "mean field (κ = $(net.κ))",
        :name => sysname)
    vals = p === nothing ? Dict{Symbol,Float64}() :
           Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in _pairs_of(p))
    return MeanFieldSystem(compiled, u0, (Float64(tspan[1]), Float64(tspan[2])), vals, source,
                           net, closure, singles, metadata)
end
