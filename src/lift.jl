# lift.jl — `node_based`, the NodeBasedModels verb (owner: WP15; DESIGN §A.4, §A.6)
#
# node_based(model, net; closure, level, …) turns any model that `contact_model` accepts
# (a ContactModel, a Catalyst ReactionSystem, an MTK System, a legacy CompartmentalModel) on a
# NetworkEpiCore network descriptor (or a legacy NetworkStructure) into a node-based system.
#
# It dispatches through the internal hook
#
#     _node_based(::Val{level}, closure, cm::ContactModel, net; kw...)
#
# so that later work packages add levels, closures and descriptors with methods in their own
# files (WP23: `PGFClosure`, `level = :s_anchored`, `MeanFieldClosure` on `WellMixed`;
# WP36e: population multitype pairwise). Conventions for those methods: `cm` is always a
# `ContactModel` (annotate it `cm::ContactModel`, to avoid method ambiguities); dispatch on the
# closure and the descriptor types; accept the keywords `source` (the model as passed), `p`
# (parameter values or `nothing`) and `name` that `node_based` always passes, and reject
# unknown keywords (no catch-all `kw...`, so that a misspelt keyword is an error); call
# `require_admissible(cm, backend; network)` before lowering. `default_closure(net)` has methods
# per descriptor type, and the fallback below throws.

"""
    NODE_BASED_LEVELS

The levels of [`node_based`](@ref): `:population` (population pairwise, the default),
`:s_anchored` (the S-anchored pairwise subsystem), `:individual` and `:pair` (individual- and
pair-based models on an `ExplicitGraph`), `:motif` and `:neighbourhood` (SIS on a regular
network).
"""
const NODE_BASED_LEVELS = (:population, :s_anchored, :individual, :pair, :motif, :neighbourhood)

"""
    default_closure(net) -> ClosureMethod

The closure that [`node_based`](@ref) uses when none is given:

- `ConfigurationNetwork`: `BernoulliClosure()`, the constant-K closure
  K = ⟨k(k − 1)⟩/⟨k⟩² = `closure_constant(net.degrees)` ((k − 1)/k on a k-regular network). It
  is exact against the edge-based model iff the degree distribution is Poisson type
  (`is_poisson_type`; Kiss, Kenah & Rempała 2023);
- `ClusteredNetwork`: `KeelingClosure()` with ϕ = `clustering_coefficient(net)`;
- `ExplicitGraph`: `KirkwoodClosure()` (used by `level = :pair`);
- a legacy `HomogeneousNetwork`/`HeterogeneousNetwork`: `KeelingClosure()` if its ϕ > 0,
  otherwise `BernoulliClosure()`;
- `WellMixed`: `MeanFieldClosure()` (mass action; wellmixed.jl);
- `MultitypeNetwork`: `BernoulliClosure()` (multitype_pairwise.jl).

Other descriptors have no population-level closure in NodeBasedModels (an `ArgumentError`).
"""
default_closure(::ConfigurationNetwork) = BernoulliClosure()
default_closure(::ClusteredNetwork) = KeelingClosure()
default_closure(::ExplicitGraph) = KirkwoodClosure()
default_closure(net::Union{HomogeneousNetwork,HeterogeneousNetwork}) =
    net.ϕ > 0 ? KeelingClosure() : BernoulliClosure()
default_closure(net::GraphNetwork) = KirkwoodClosure()
default_closure(net) = throw(ArgumentError(
    "node_based: no default closure for $(nameof(typeof(net))) — " * _unsupported_hint(net)))

_unsupported_hint(::MultitypeNetwork) =
    "population-level multitype pairwise models are not implemented yet; use " *
    "level = :individual on a sampled typed graph"
_unsupported_hint(::DynamicNetwork) = "NodeBasedModels has no dynamic-network models; use " *
    "EdgeBasedModels.edge_based or NetworkOutbreaks.simulate"
_unsupported_hint(::MultiplexNetwork) = "population-level multiplex pairwise models are not " *
    "implemented; use level = :individual on the layered graph"
_unsupported_hint(::WellMixed) = "a well-mixed population needs MeanFieldClosure (mass action)"
_unsupported_hint(net) = "pass `closure = …` explicitly"

"""
    node_based(model, net; closure = default_closure(net), level = :population, p = nothing,
               name = nothing, rates = nothing, population = nothing, kwargs...)
    node_based(sc::Scenario; closure = default_closure(sc.network), level = :population, kwargs...)

The node-based representation of `model` on the network `net`. `model` is anything that
`contact_model` accepts (a `ContactModel` such as `sir_model()`, a Catalyst `ReactionSystem`, a
ModelingToolkit `System`, a legacy `CompartmentalModel`); `rates` (`:per_contact`,
`:frequency`, `:density`) and `population` (e.g. `:N`) are passed to `contact_model` for
Catalyst and MTK inputs. `net` is a NetworkEpiCore
descriptor (`ConfigurationNetwork`, `ClusteredNetwork`, `ExplicitGraph`, …) or a legacy
`NetworkStructure`. `p` holds parameter values (by name) used as defaults when solving, and
needed by the graph-level levels. Contact rates are converted to per-contact rates τ with
`per_contact_rates(cm, net)`, like every other back end.

`level` is one of [`NODE_BASED_LEVELS`](@ref):

- `:population` (default): the population pairwise model, a [`PairwiseSystem`](@ref). It is
  total on T_net models (SIS and SIRS included). Keywords: `tspan = (0.0, 100.0)`, the
  population scale `N = 1.0`, the seeding `initial` (a `SeedSpec`; by default a fraction
  `seed_fraction = ε = 1e-3` in the entry state), `seed_state` (see [`generate_pairwise`](@ref)),
  `cumulative = true` (adds the cumulative-incidence state used by [`model_curves`](@ref): the
  seeds in infected compartments plus every entry into infection, with infection status decided
  from the typing as in NetworkOutbreaks' `final_size`, DESIGN §J.8) and
  `tol` (truncation of infinite degree distributions). The initial condition is the π^PW image
  of the edge-based one (DESIGN §E.2);
- `:individual` and `:pair`: the individual-based (NIMFA) and pair-based (Kirkwood, SIR) models
  on an `ExplicitGraph`, solved at once; they need numeric rates `p` and return an
  `IndividualBasedResult` / `PairBasedResult`;
- `:motif` (with `closure = MotifClosure(k, m)`) and `:neighbourhood` (keyword `n = 2`): the SIS
  approximations of Keeling, House, Cooper & Pellis (2016) on `ConfigurationNetwork(RegularDegree(k))`;
- `:s_anchored`: the S-anchored pairwise subsystem of a T_EB model (one susceptible class),
  an `SAnchoredSystem`, with `BernoulliClosure` or `PGFClosure` on a `ConfigurationNetwork`
  (see `s_anchored.jl`).

The default seeded compartment is the unique entry state of the contacts (I for SIR, E for
SEIR); for a model with several susceptible classes the unseeded nodes start in the one that
no reaction produces (S₀ of a reinfection-counted model), and the entry state is the one
reached from it. A model with several entry states (two strains) needs an explicit `initial`.

The scenario form takes the model, network, parameters, seeding and time span from `sc`.

```julia
sys = node_based(seir_model(), ConfigurationNetwork(PoissonDegree(5)))   # constant K = 1: exact
sol = solve_epidemic(sys; p = Dict(:τ => 1/6, :σ => 1/5, :γ => 1/4), tspan = (0.0, 150.0))
model_curves(sys, sol; t = 0:1:150)
```
"""
function node_based(model, net; closure = default_closure(net), level::Symbol = :population,
                    p = nothing, name::Union{Nothing,Symbol} = nothing, rates = nothing,
                    population = nothing, kwargs...)
    level in NODE_BASED_LEVELS || throw(ArgumentError(
        "node_based: level must be one of $(NODE_BASED_LEVELS); got $(repr(level))"))
    front = (; (k => v for (k, v) in pairs((; rates, population)) if v !== nothing)...)
    cm = isempty(front) ? contact_model(model) : contact_model(model; front...)
    return _node_based(Val(level), closure, cm, net; source = model, p, name, kwargs...)
end

function node_based(sc::Scenario; closure = default_closure(sc.network),
                    level::Symbol = :population, kwargs...)
    return node_based(sc.model, sc.network; closure, level, p = sc.params,
                      initial = sc.initial, tspan = sc.tspan, kwargs...)
end

"""
    generate_pairwise(model::ContactModel, net, closure::ClosureMethod; kwargs...)
    generate_pairwise(model::CompartmentalModel, net::NetworkDescriptor, closure::ClosureMethod; kwargs...)

The factory form of `node_based(model, net; closure, level = :population, cumulative = false,
kwargs...)` (DESIGN §A.6): the same system as the 0.1 `generate_pairwise` on the equivalent
`CompartmentalModel`, without the cumulative accumulator unless `cumulative = true`.
"""
generate_pairwise(model::ContactModel, net, closure::ClosureMethod; cumulative::Bool = false,
                  kwargs...) =
    node_based(model, net; closure, level = :population, cumulative, kwargs...)
generate_pairwise(model::CompartmentalModel, net::NetworkDescriptor, closure::ClosureMethod;
                  cumulative::Bool = false, kwargs...) =
    node_based(model, net; closure, level = :population, cumulative, kwargs...)

"""
    solve_epidemic(psys::PairwiseSystem, sc::Scenario; kwargs...)

Solve with the parameters, seeding, time span and save grid of the scenario `sc`
(`p = sc.params`, `initial = sc.initial`, `tspan = sc.tspan`, `saveat = sc.tgrid`).
"""
solve_epidemic(psys::PairwiseSystem, sc::Scenario; kwargs...) =
    solve_epidemic(psys; p = sc.params, initial = sc.initial, tspan = sc.tspan,
                   saveat = sc.tgrid, kwargs...)

# ─── The dispatch hook ────────────────────────────────────────────────────────

function _node_based(::Val{L}, closure, cm::ContactModel, net; kw...) where {L}
    what = L === :population ? "population pairwise models with $(nameof(typeof(closure)))" :
           "level = :$(L) with $(nameof(typeof(closure)))"
    throw(ArgumentError("node_based: $(what) on $(nameof(typeof(net))) is not supported " *
                        "(level :$(L); " * _level_hint(Val(L), net) * ")"))
end

_level_hint(::Val{:population}, ::ExplicitGraph) =
    "a fixed graph needs a graph-level model: level = :individual or :pair"
_level_hint(::Val{:population}, net::NetworkDescriptor) = _unsupported_hint(net)
_level_hint(::Val{:population}, net) =
    "population closures are Bernoulli, Keeling, Barnard and Power on NetworkStructures, " *
    "Bernoulli and Keeling on heterogeneous networks"
_level_hint(::Val{:s_anchored}, net) =
    "the S-anchored level takes BernoulliClosure or PGFClosure on a ConfigurationNetwork"
_level_hint(::Val{L}, net) where {L} =
    L in (:individual, :pair) ? "graph-level models need an ExplicitGraph" :
    "motif and neighbourhood models need SIS on ConfigurationNetwork(RegularDegree(k))"

# Population pairwise on a legacy network structure or a descriptor that converts to one.
function _node_based(::Val{:population}, closure::ClosureMethod, cm::ContactModel,
                     net::NetworkStructure; kw...)
    return _population_pairwise(cm, net, closure; kw...)
end

function _node_based(::Val{:population}, closure::ClosureMethod, cm::ContactModel,
                     net::Union{ConfigurationNetwork,ClusteredNetwork}; N::Real = 1.0,
                     tol::Real = 1e-12, kw...)
    return _population_pairwise(cm, network_structure(net; N, tol), closure; N,
                                descriptor = net, kw...)
end

"""
    _population_pairwise(cm::ContactModel, net::NetworkStructure, closure; source, p, name,
                         tspan, N, ε, seed_fraction, seed_state, initial, cumulative,
                         descriptor) -> PairwiseSystem

Lower a `ContactModel` to the population pairwise model: rates to per-contact rates (with the
mean degree of the descriptor, or of `net`), admissibility, `CompartmentalModel(cm)`, the
seeding (entry state and background of the contact model) and the infected compartments of the
contact model's typing (DESIGN §J.8, [`_infected_compartments`](@ref)), which fix the seeds and
the transitions that the `cumulative` accumulator counts.
"""
function _population_pairwise(cm::ContactModel, net::NetworkStructure, closure::ClosureMethod;
                              source = cm, p = nothing, name = nothing,
                              tspan::Tuple{Real,Real} = (0.0, 100.0), N::Real = 1.0,
                              ε::Real = 1e-3, seed_fraction::Real = ε,
                              seed_state::Symbol = :entry,
                              initial::Union{Nothing,SeedSpec} = nothing,
                              cumulative::Bool = true, descriptor = nothing)
    network = descriptor === nothing ? net : descriptor
    cmτ = _per_contact_model(cm, network)
    require_admissible(cmτ, :pairwise; network)
    cmodel = CompartmentalModel(cmτ)
    bg = _seed_background(cmτ)
    resolve = (init, ρ, state) -> _cm_resolve_seed(cmτ, cmodel, init, ρ, state, N, bg)
    params = p === nothing ? Dict{Symbol,Float64}() : p
    return _build_pairwise(cmodel, net, closure; tspan, N,
                           seed = resolve(initial, seed_fraction, seed_state), resolve,
                           default_fraction = seed_fraction, seed_state, cumulative,
                           infected = _infected_compartments(cmτ), source, contact = cmτ,
                           descriptor, params, name)
end

# The model with every contact rate converted to its per-contact rate τ on `net`.
function _per_contact_model(cm::ContactModel, net)
    rate_convention(cm) isa PerContact && return cm
    τs = per_contact_rates(cm, net)
    cs = [Contact(c.recipient, c.infector, c.product, τ; layer = c.layer, name = c.name)
          for (c, τ) in zip(contacts(cm), τs)]
    note = "per-contact rates for $(nameof(typeof(net))) (⟨k⟩ = $(mean_degree(net))) from the " *
           "$(rate_convention(cm)) convention"
    pv = provenance(cm)
    return ContactModel(nameof(cm); contacts = cs, transitions = node_transitions(cm),
                        species = species_names(cm), susceptible = susceptible_species(cm),
                        defaults = parameter_defaults(cm), labels = species_labels(cm),
                        provenance = Provenance(pv.source; method = pv.method,
                                                assumptions = vcat(pv.assumptions, note)))
end

"""
    _seed_background(cm::ContactModel) -> Union{Symbol,Nothing}

The compartment that holds the unseeded nodes: the susceptible class if there is one, else the
unique susceptible class that no reaction produces (S₀ of a reinfection-counted model, S when
R can be reinfected), else `nothing` (an explicit `initial` naming every compartment is then
required).
"""
function _seed_background(cm::ContactModel)
    sus = susceptible_species(cm)
    length(sus) == 1 && return only(sus)
    produced = Set{Symbol}(c.product for c in contacts(cm))
    for t in node_transitions(cm)
        t.to === nothing || push!(produced, t.to)
    end
    free = [s for s in sus if !(s in produced)]
    return length(free) == 1 ? only(free) : nothing
end

"""
    _cm_entry_state(cm::ContactModel, background) -> Symbol

The default seeded compartment (DESIGN §E.2): the unique entry state of the contacts
(`NetworkEpiCore.default_seed_state`), or, when there are several, the unique entry state
reached from the background compartment (I₁ from S₀ after reinfection counting). Otherwise the
`default_seed_state` error asks for an explicit `initial`.
"""
function _cm_entry_state(cm::ContactModel, background)
    if length(entry_species(cm)) != 1 && background !== nothing
        from_bg = unique!([c.product for c in contacts(cm) if c.recipient === background])
        length(from_bg) == 1 && return only(from_bg)
    end
    return default_seed_state(cm)
end

function _cm_resolve_seed(cm::ContactModel, cmodel::CompartmentalModel, initial, ρ::Real,
                          state::Symbol, N::Real, background)
    if initial === nothing
        background === nothing && throw(ArgumentError(
            "model :$(nameof(cm)) has several susceptible classes " *
            "($(join(susceptible_species(cm), ", "))) and none of them is the natural " *
            "background; pass an explicit `initial` naming the fraction of every compartment " *
            "(they must sum to 1), or a SeedFraction with `default`"))
        if state === :entry
            (isfinite(ρ) && 0 <= ρ <= 1) || throw(ArgumentError(
                "seed_fraction must lie in [0, 1]; got $(ρ)"))
            X = _cm_entry_state(cm, background)
            return Dict{Symbol,Float64}(background => 1 - ρ, X => ρ), background
        end
    end
    return _resolve_seed(cmodel, initial, ρ, state, N; background)
end

# The infected species of the contact model's typing (DESIGN §J.8): NetworkEpiCore's
# `infected_species`, the one rule behind the final size and the `:cumulative` observable of
# every back end (the edge-based accumulator, NetworkOutbreaks' `final_size`), so the pairwise
# accumulators cannot drift from it. A removal X → ∅ goes to the absorbing `:removed`
# compartment of the lowering, which is never infected (nor listed by `infected_species`).
_infected_compartments(cm::ContactModel) = NetworkEpiCore.infected_species(cm)

# ─── Graph-level models on an explicit graph ─────────────────────────────────

# The one numeric value of the rates `rs` under `p` (the 0.1 graph-level builders take a
# single infection rate and a single spontaneous rate), or an error naming the problem.
function _single_rate(rs, p, what, level)
    vals = unique([Float64(rate_value(r, p)) for r in rs])
    length(vals) == 1 || throw(ArgumentError(
        "node_based(…; level = :$(level)): the graph-level builder uses one $(what) for every " *
        "reaction of that kind, but the model has $(length(vals)) different values " *
        "($(join(vals, ", "))); verified issue B02"))
    return only(vals)
end

# The per-contact rate τ and spontaneous rate γ of the 0.1 graph-level builders, from the
# per-contact model `cm` (rates converted) with parameter values `p` (defaults of `cm` fill in).
function _graph_level_rates(cm::ContactModel, cmodel::CompartmentalModel, p, level)
    p === nothing && throw(ArgumentError(
        "node_based(…; level = :$(level)) solves at once and needs numeric parameter values `p`"))
    p = merge(parameter_defaults(cm), Dict{Symbol,Float64}(Symbol(k) => v for (k, v) in p))
    all(tr -> tr.type === :spontaneous || isempty(tr.via), cmodel.transitions) ||
        throw(ArgumentError("node_based(…; level = :$(level)): the graph-level builder lets " *
                            "every infectious compartment infect through every infection " *
                            "transition, so per-infector contacts (via) are not supported"))
    froms = [tr.from for tr in cmodel.transitions if tr.type === :infection]
    allunique(froms) || throw(ArgumentError(
        "node_based(…; level = :$(level)): a compartment has several infection transitions, " *
        "which the graph-level builder would count twice"))
    τ = _single_rate([c.rate for c in contacts(cm)], p, "infection rate", level)
    ts = node_transitions(cm)
    γ = isempty(ts) ? 0.0 : _single_rate([t.rate for t in ts], p, "spontaneous rate", level)
    return τ, γ
end

# The seed of the graph-level builders: a fraction in the first infectious compartment, with
# the unseeded nodes in the first susceptible compartment.
function _graph_level_seed(cm::ContactModel, cmodel::CompartmentalModel, initial, ρ, level)
    bg = _seed_background(cm)
    X = if initial === nothing
        _cm_entry_state(cm, bg)
    else
        (initial isa SeedFraction && length(seed_fractions(initial)) == 1) ||
            throw(ArgumentError("node_based(…; level = :$(level)) takes a single-compartment " *
                                "SeedFraction as `initial`"))
        ρ = last(only(seed_fractions(initial)))
        first(only(seed_fractions(initial)))
    end
    (X === first(cmodel.infectious_compartments) && bg === _background(cmodel)) ||
        throw(ArgumentError("node_based(…; level = :$(level)): the graph-level builder seeds " *
                            "the first infectious compartment ($(first(cmodel.infectious_compartments))), " *
                            "not $(X)"))
    return Float64(ρ)
end

function _node_based(::Val{:individual}, closure, cm::ContactModel, net::ExplicitGraph;
                     p = nothing, initial::Union{Nothing,SeedSpec} = nothing, ε::Real = 1e-3,
                     seed_fraction::Real = ε, tspan::Tuple{Real,Real} = (0.0, 100.0),
                     saveat::Real = 1.0, transmission_matrix = nothing, source = nothing,
                     name = nothing)
    require_admissible(cm, :individual; network = net)
    cmτ = _per_contact_model(cm, net)
    cmodel = CompartmentalModel(cmτ)
    τ, γ = _graph_level_rates(cmτ, cmodel, p, :individual)
    ρ = _graph_level_seed(cm, cmodel, initial, seed_fraction, :individual)
    g = GraphNetwork(net; transmission_matrix)
    return generate_individual_based(cmodel, g; tspan, infection_rate = τ, recovery_rate = γ,
                                     seed_fraction = ρ, saveat = Float64(saveat))
end

function _node_based(::Val{:pair}, closure, cm::ContactModel, net::ExplicitGraph;
                     p = nothing, initial::Union{Nothing,SeedSpec} = nothing, ε::Real = 1e-3,
                     seed_fraction::Real = ε, tspan::Tuple{Real,Real} = (0.0, 100.0),
                     saveat::Real = 1.0, transmission_matrix = nothing, source = nothing,
                     name = nothing)
    closure isa KirkwoodClosure || throw(ArgumentError(
        "node_based(…; level = :pair) uses the Kirkwood closure; got $(closure)"))
    require_admissible(cm, :pair; network = net)
    cmτ = _per_contact_model(cm, net)
    cmodel = CompartmentalModel(cmτ)
    τ, γ = _graph_level_rates(cmτ, cmodel, p, :pair)
    ρ = _graph_level_seed(cm, cmodel, initial, seed_fraction, :pair)
    g = GraphNetwork(net; transmission_matrix)
    return generate_pair_based(cmodel, g; tspan, infection_rate = τ, recovery_rate = γ,
                               seed_fraction = ρ, saveat = Float64(saveat))
end

# ─── SIS motif and neighbourhood models on a regular network ─────────────────

# (k, τ, γ, ε) of an SIS model S + I → 2I, I → S on a k-regular configuration network.
function _regular_sis(cm::ContactModel, net::ConfigurationNetwork, p, initial, ρ, level)
    k = _regular_degree(net.degrees)
    k === nothing && throw(ArgumentError(
        "node_based(…; level = :$(level)) needs a regular network, " *
        "ConfigurationNetwork(RegularDegree(k)); got $(net.degrees)"))
    cs, ts = contacts(cm), node_transitions(cm)
    (sort(species_names(cm)) == [:I, :S] && length(cs) == 1 && length(ts) == 1 &&
     (only(cs).recipient, only(cs).infector, only(cs).product) == (:S, :I, :I) &&
     (only(ts).from, only(ts).to) == (:I, :S)) || throw(ArgumentError(
        "node_based(…; level = :$(level)) implements SIS only (S + I → 2I, I → S); got " *
        "model :$(nameof(cm))"))
    p === nothing && throw(ArgumentError(
        "node_based(…; level = :$(level)) needs numeric parameter values `p`"))
    p = merge(parameter_defaults(cm), Dict{Symbol,Float64}(Symbol(k) => v for (k, v) in p))
    cmτ = _per_contact_model(cm, net)
    τ = Float64(rate_value(only(contacts(cmτ)).rate, p))
    γ = Float64(rate_value(only(ts).rate, p))
    if initial !== nothing
        fr = seed_fractions(initial)
        (length(fr) == 1 && first(only(fr)) === :I) || throw(ArgumentError(
            "node_based(…; level = :$(level)) seeds I only; pass SeedFraction(:I => ρ)"))
        ρ = last(only(fr))
    end
    return k, τ, γ, Float64(ρ)
end

function _node_based(::Val{:motif}, closure::MotifClosure, cm::ContactModel,
                     net::ConfigurationNetwork; p = nothing, initial = nothing,
                     ε::Real = 1e-3, seed_fraction::Real = ε,
                     tspan::Tuple{Real,Real} = (0.0, 100.0), N::Real = 1.0,
                     source = nothing, name = nothing)
    k, τ, γ, ρ = _regular_sis(cm, net, p, initial, seed_fraction, :motif)
    closure.k == k || throw(ArgumentError(
        "node_based(…; level = :motif): MotifClosure($(closure.k), $(closure.m)) is for " *
        "$(closure.k)-regular networks; the network is $(k)-regular"))
    return motif_based_sis(; β = τ, γ, k, m = closure.m, tspan, N, ε = ρ)
end

function _node_based(::Val{:neighbourhood}, closure, cm::ContactModel,
                     net::ConfigurationNetwork; p = nothing, initial = nothing,
                     ε::Real = 1e-3, seed_fraction::Real = ε, n::Integer = 2,
                     tspan::Tuple{Real,Real} = (0.0, 100.0), N::Real = 1.0,
                     source = nothing, name = nothing)
    k, τ, γ, ρ = _regular_sis(cm, net, p, initial, seed_fraction, :neighbourhood)
    return generate_neighbourhood(CompartmentalModel(cm), k, n; β = τ, γ, tspan, N, ε = ρ)
end
