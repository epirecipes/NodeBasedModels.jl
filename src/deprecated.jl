# deprecated.jl — NodeBasedModels 0.1 names kept for one release (owner: WP15; DESIGN §A.7)
#
# Renamed but correct functionality forwards with a `Base.depwarn`; functionality that gave
# wrong numbers throws with a migration message. MIGRATION.md lists every entry.

# ─── 0.1 canned models (returned CompartmentalModel) ─────────────────────────

function _depwarn_canned(old::Symbol, new::Symbol)
    Base.depwarn("`$(old)()` is deprecated: `$(new)()` (NetworkEpiCore, re-exported by " *
                 "NodeBasedModels) returns a ContactModel that node_based, generate_pairwise " *
                 "and the other builders accept. `CompartmentalModel($(new)())` is the " *
                 "CompartmentalModel.", old)
end

"""
    node_sir_model(; τ = :τ, γ = :γ) -> CompartmentalModel

Deprecated: the NodeBasedModels 0.1 `sir_model()`, S →[τ] I →[γ] R, as a `CompartmentalModel`
named `:SIR`. Use `sir_model()` (a `ContactModel`), or `CompartmentalModel(sir_model())`.
"""
function node_sir_model(; τ = :τ, γ = :γ)
    _depwarn_canned(:node_sir_model, :sir_model)
    CompartmentalModel(
        [Compartment(:S), Compartment(:I; infectious=true), Compartment(:R)],
        [Transition(:S, :I, τ, :infection),
         Transition(:I, :R, γ, :spontaneous)];
        name = :SIR
    )
end

"""
    node_sis_model(; τ = :τ, γ = :γ) -> CompartmentalModel

Deprecated: the NodeBasedModels 0.1 `sis_model()`, S →[τ] I →[γ] S, as a `CompartmentalModel`
named `:SIS`. Use `sis_model()`, or `CompartmentalModel(sis_model())`.
"""
function node_sis_model(; τ = :τ, γ = :γ)
    _depwarn_canned(:node_sis_model, :sis_model)
    CompartmentalModel(
        [Compartment(:S), Compartment(:I; infectious=true)],
        [Transition(:S, :I, τ, :infection),
         Transition(:I, :S, γ, :spontaneous)];
        name = :SIS
    )
end

"""
    node_seir_model(; τ = :τ, σ = :σ, γ = :γ) -> CompartmentalModel

Deprecated: the NodeBasedModels 0.1 `seir_model()`, S →[τ] E →[σ] I →[γ] R, as a
`CompartmentalModel` named `:SEIR`. Use `seir_model()`, or `CompartmentalModel(seir_model())`.
"""
function node_seir_model(; τ = :τ, σ = :σ, γ = :γ)
    _depwarn_canned(:node_seir_model, :seir_model)
    CompartmentalModel(
        [Compartment(:S), Compartment(:E), Compartment(:I; infectious=true),
         Compartment(:R)],
        [Transition(:S, :E, τ, :infection),
         Transition(:E, :I, σ, :spontaneous),
         Transition(:I, :R, γ, :spontaneous)];
        name = :SEIR
    )
end

"""
    node_sirs_model(; τ = :τ, γ = :γ, ε = :ε) -> CompartmentalModel

Deprecated: the NodeBasedModels 0.1 `sirs_model()`, S →[τ] I →[γ] R →[ε] S, as a
`CompartmentalModel` named `:SIRS`. Use `sirs_model()`, or `CompartmentalModel(sirs_model())`.
"""
function node_sirs_model(; τ = :τ, γ = :γ, ε = :ε)
    _depwarn_canned(:node_sirs_model, :sirs_model)
    CompartmentalModel(
        [Compartment(:S), Compartment(:I; infectious=true), Compartment(:R)],
        [Transition(:S, :I, τ, :infection),
         Transition(:I, :R, γ, :spontaneous),
         Transition(:R, :S, ε, :spontaneous)];
        name = :SIRS
    )
end

# ─── Catalyst conversion ──────────────────────────────────────────────────────

"""
    model_from_catalyst(rn; infectious = nothing) -> CompartmentalModel

Deprecated: use `contact_model(rn)` (NetworkEpiCore's Catalyst front end, loaded with Catalyst),
which classifies contacts by their catalyst, and pass it to `node_based`. This wrapper returns
`CompartmentalModel(contact_model(rn))`: infection transitions keep their infectors (`via`),
expression rates stay expressions, and the infectious compartments are the catalysts of the
contacts (`infectious`, if given, must name the same set).

It warns when the 0.1 conversion gave a different model: 0.1 marked the compartments in
`infectious` (default `[:I]`) as infectious, applied every infection rate to every infectious
compartment (verified issue B04), read a contact whose catalyst was not in `infectious` as a
spontaneous transition, and turned expression rates into bogus parameter names.
"""
function model_from_catalyst(rn; infectious::Union{Nothing,AbstractVector{Symbol}} = nothing)
    Base.depwarn("`model_from_catalyst(rn)` is deprecated: use `contact_model(rn)` " *
                 "(NetworkEpiCore, with Catalyst loaded) and pass it to node_based or " *
                 "generate_pairwise; `CompartmentalModel(contact_model(rn))` is the " *
                 "CompartmentalModel.", :model_from_catalyst)
    cm = contact_model(rn)
    m = CompartmentalModel(cm; name = nameof(cm))
    legacy_inf = something(infectious, [:I])
    notes = String[]
    if infectious !== nothing && Set(infectious) != Set(m.infectious_compartments)
        throw(ArgumentError(
            "model_from_catalyst: `infectious = $(infectious)` does not match the catalysts of " *
            "the contacts ($(join(m.infectious_compartments, ", "))); the infectious " *
            "compartments are now read from the reaction network. Drop the keyword."))
    end
    Set(legacy_inf) == Set(m.infectious_compartments) ||
        push!(notes, "0.1 marked $(join(legacy_inf, ", ")) as infectious; the catalysts are " *
                     "$(join(m.infectious_compartments, ", "))")
    any(tr -> !isempty(tr.via), m.transitions) &&
        push!(notes, "0.1 applied every infection rate to every infectious compartment; " *
                     "the infections now act through their own catalysts (via, B04)")
    any(tr -> tr.type === :infection && !(any(in(legacy_inf), infectors(m, tr))),
        m.transitions) &&
        push!(notes, "0.1 read contacts whose catalyst is not in $(legacy_inf) as spontaneous " *
                     "transitions")
    any(tr -> !(tr.rate isa Symbol), m.transitions) &&
        push!(notes, "0.1 turned non-parameter rates into parameter names")
    isempty(notes) || @warn "model_from_catalyst: the 0.1 conversion of this reaction network " *
                            "gave a different model: " * join(notes, "; ") * "."
    return m
end

# ─── GraphNetwork (use ExplicitGraph) ─────────────────────────────────────────

function _depwarn_graph_network()
    Base.depwarn("`GraphNetwork(g, …)` is deprecated: wrap the graph as `ExplicitGraph(g)` " *
                 "(NetworkEpiCore) and use `node_based(model, ExplicitGraph(g); level = " *
                 ":individual | :pair, p)` or `generate_individual_based(model, ExplicitGraph(g); " *
                 "…)`; the per-contact rate comes from the model. A per-edge rate matrix is " *
                 "`GraphNetwork(ExplicitGraph(g); transmission_matrix = T)`.", :GraphNetwork)
end

"""
    GraphNetwork(g; transmission_rate = nothing, transmission_matrix = nothing)
    GraphNetwork(g, transmission_matrix)

Deprecated constructors of the graph-instance network: use `ExplicitGraph(g)` (see
[`GraphNetwork`](@ref)). `transmission_rate = nothing` means the solver's `infection_rate`; an
explicit rate, including 1.0, is kept (verified issue B04).
"""
function GraphNetwork(g; transmission_rate::Union{Nothing,Real} = nothing,
                      transmission_matrix::Union{Nothing,AbstractMatrix{<:Real}} = nothing)
    _depwarn_graph_network()
    return _graph_network(g; transmission_rate, transmission_matrix)
end

function GraphNetwork(g, transmission_matrix::Union{Nothing,AbstractMatrix{<:Real}})
    _depwarn_graph_network()
    return _graph_network(g; transmission_matrix)
end

# ─── Reinfection counting on a CompartmentalModel ────────────────────────────

"""
    with_reinfection_counting(model::CompartmentalModel, L::Integer) -> CompartmentalModel

Deprecated: use `with_reinfection_counting(cm::ContactModel, L)` (NetworkEpiCore, e.g.
`with_reinfection_counting(sis_model(), L)`) and `node_based`. This method returns the
NodeBasedModels 0.1 lift (Keeling, House, Cooper & Pellis 2016, approximation 1; compartments
`X_p` by infection count p ≤ L). A model with per-infector transitions (`via`) is lifted
through `CompartmentalModel(with_reinfection_counting(contact_model(model), L))`, which keeps
the infectors; the 0.1 lift dropped them.
"""
function with_reinfection_counting(model::CompartmentalModel, L::Integer)
    Base.depwarn("`with_reinfection_counting(::CompartmentalModel, L)` is deprecated: use " *
                 "`with_reinfection_counting(cm::ContactModel, L)` (e.g. with sis_model()) and " *
                 "node_based.", :with_reinfection_counting)
    any(tr -> !isempty(tr.via), model.transitions) &&
        return CompartmentalModel(with_reinfection_counting(contact_model(model), L))
    return _LegacyReinfection.with_reinfection_counting(model, L)
end

# ─── Removed ─────────────────────────────────────────────────────────────────

"""
    mass_action(psys::PairwiseSystem)

Not defined for a pairwise system: a pairwise model reduces to mass action only in the
mean-field limit of a well-mixed population. Build that model directly with
`node_based(cm, WellMixed(κ); closure = MeanFieldClosure())`, or use
`NetworkEpiCore.mass_action(cm; κ)` for the mass-action ODE of a model.
"""
function mass_action(psys::PairwiseSystem; kwargs...)
    throw(ArgumentError(
        "mass_action(::PairwiseSystem) is not defined: a pairwise model is mass action only " *
        "in the mean-field closure [XY] = κ[X][Y] on a well-mixed population. Use " *
        "node_based(cm, WellMixed(κ); closure = MeanFieldClosure()), or " *
        "NetworkEpiCore.mass_action(cm; κ) for the mass-action ODE of the model."))
end
