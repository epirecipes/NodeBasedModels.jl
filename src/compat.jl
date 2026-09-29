# compat.jl — converters between NodeBasedModels 0.1 types and NetworkEpiCore (owner: WP15)
#
# - models: `contact_model(::CompartmentalModel)` and `CompartmentalModel(::ContactModel)`;
# - networks: `HomogeneousNetwork` / `HeterogeneousNetwork` from configuration and clustered
#   descriptors, `GraphNetwork(::ExplicitGraph)`;
# - the 0.1 entry points that take a model or a graph also accept a `ContactModel` or an
#   `ExplicitGraph`, so that `generate_individual_based(sir_model(), ExplicitGraph(g); …)`
#   reproduces the 0.1 numbers (DESIGN §A.4, §A.7).

# ─── Models ───────────────────────────────────────────────────────────────────

"""
    contact_model(m::CompartmentalModel) -> ContactModel

The intermediate representation of a NodeBasedModels 0.1 model: one
`Contact(from, Z, to, rate)` per infection transition and infector `Z` (its `via`, or every
infectious compartment when `via` is empty), and one `NodeTransition(from, to, rate)` per
spontaneous transition. This reproduces the 0.1 semantics exactly: two infection transitions
with the same source, target and infectors add their rates (`merge_duplicates`), which is what
the 0.1 pairwise equations did. The susceptible classes are inferred (recipients that are never
a contact product).

An infection transition whose source is one of its own infectors (X + X → Y + X, possible in
0.1 with `via = []` when `from` is infectious) is not a network contact; restrict its `via`.
"""
function contact_model(m::CompartmentalModel)
    cs = Contact[]
    ts = NodeTransition[]
    cmap = Int[]
    tmap = Int[]
    for (k, tr) in enumerate(m.transitions)
        if tr.type === :infection
            for Z in infectors(m, tr)
                Z === tr.from && throw(ArgumentError(
                    "contact_model(:$(m.name)): the infection transition $(tr.from) → $(tr.to) " *
                    "has its own source $(Z) among its infectors" *
                    (isempty(tr.via) ? " (via = [] means every infectious compartment)" : "") *
                    "; a network contact s + J → X + J needs s ≠ J. List the other infectors in " *
                    "`via`."))
                push!(cs, Contact(tr.from, Z, tr.to, tr.rate))
                push!(cmap, k)
            end
        else
            push!(ts, NodeTransition(tr.from, tr.to, tr.rate))
            push!(tmap, k)
        end
    end
    notes = ["converted from NodeBasedModels.CompartmentalModel :$(m.name): one contact per " *
             "infection transition and infector (empty via = every infectious compartment)"]
    return ContactModel(m.name; contacts = cs, transitions = ts, species = m.compartment_names,
                        merge_duplicates = true,
                        provenance = Provenance(:legacy_nbm; method = :explicit,
                                                assumptions = notes,
                                                reaction_map = vcat(cmap, tmap)))
end

"""
    REMOVED_COMPARTMENT

`:removed`, the absorbing compartment that `CompartmentalModel(::ContactModel)` adds as the
target of removals `X → ∅` (DESIGN §J.2; NetworkOutbreaks does the same).
"""
const REMOVED_COMPARTMENT = :removed

"""
    CompartmentalModel(cm::ContactModel; name = nameof(cm)) -> CompartmentalModel

The NodeBasedModels compartmental form of a `ContactModel` (the lowering target of the
population pairwise builder):

- compartments in the species order of `cm`; the infectors of the contacts are infectious;
- the contacts with the same recipient, product and rate become one infection transition whose
  `via` lists their infectors, or is empty when they are every infector of the model (so
  `CompartmentalModel(sir_model())` is the 0.1 `sir_model()`, named `:sir`); contacts with
  different rates (SEAIR's τI via I and τA via A) stay separate transitions;
- each node transition becomes a spontaneous transition, and a removal `X → ∅` goes to the
  absorbing compartment [`REMOVED_COMPARTMENT`](@ref) (`:removed`), which is added.

The rates are the model's per-contact rates, so the model must use the `PerContact`
convention; `node_based` first converts other conventions with
`per_contact_rates(cm, net)`. Contacts on a multiplex layer cannot be represented.
"""
function CompartmentalModel(cm::ContactModel; name::Symbol = nameof(cm))
    where = "CompartmentalModel(:$(nameof(cm)))"
    rate_convention(cm) isa PerContact || throw(ArgumentError(
        "$where: the contact rates use the $(rate_convention(cm)) convention; convert them to " *
        "per-contact rates first (node_based does this with the mean degree of the network, " *
        "through per_contact_rates(cm, net))"))
    for c in contacts(cm)
        c.layer === :all || throw(ArgumentError(
            "$where: the contact `$(c.name)` acts on the layer :$(c.layer); a " *
            "CompartmentalModel has no layers"))
    end
    sp = species_names(cm)
    infectious = infectious_species(cm)
    removal = any(t -> t.to === nothing, node_transitions(cm))
    removal && REMOVED_COMPARTMENT in sp && throw(ArgumentError(
        "$where: the model has a removal X → ∅ and a species named :$(REMOVED_COMPARTMENT), " *
        "the name of the absorbing compartment that removals go to; rename the species"))
    names = removal ? vcat(sp, REMOVED_COMPARTMENT) : sp
    comps = [Compartment(x; infectious = x in infectious) for x in names]

    order = Dict(x => i for (i, x) in enumerate(sp))
    groups = Tuple{Symbol,Symbol,Any,Vector{Symbol}}[]
    for c in contacts(cm)
        k = findfirst(g -> g[1] === c.recipient && g[2] === c.product && isequal(g[3], c.rate),
                      groups)
        k === nothing ? push!(groups, (c.recipient, c.product, c.rate, [c.infector])) :
                        push!(groups[k][4], c.infector)
    end
    trs = Transition[]
    for (s, X, rate, js) in groups
        via = Set(js) == Set(infectious) ? Symbol[] : sort!(unique(js); by = j -> order[j])
        push!(trs, Transition(s, X, rate, :infection; via))
    end
    for t in node_transitions(cm)
        push!(trs, Transition(t.from, something(t.to, REMOVED_COMPARTMENT), t.rate, :spontaneous))
    end
    return CompartmentalModel(comps, trs; name)
end

# ─── Networks ─────────────────────────────────────────────────────────────────

# The common degree of a distribution that is a point mass (nothing otherwise).
_regular_degree(d::RegularDegree) = d.k
function _regular_degree(d::DegreeDistribution)
    p = try
        degree_probabilities(d)
    catch err
        err isa ArgumentError || rethrow()
        return nothing
    end
    ks = findall(>(1 - 1e-12), p)
    return length(ks) == 1 ? only(ks) - 1 : nothing
end

"""
    HomogeneousNetwork(net::ConfigurationNetwork; N = 1.0)
    HomogeneousNetwork(net::ClusteredNetwork; N = 1.0)

The homogeneous (k-regular) network structure of a descriptor whose nodes all have degree k:
`ConfigurationNetwork(RegularDegree(k))` gives `HomogeneousNetwork(k)` (clustering ϕ = 0), and a
clustered network whose degree k = s + 2t is constant (e.g.
`ClusteredNetwork(RegularDegree(2), RegularDegree(2))`, k = 6) gives `HomogeneousNetwork(k)` with
Keeling's ϕ = `clustering_coefficient(net)` (2/15 in that example). Other degree distributions
are an `ArgumentError` (use [`HeterogeneousNetwork`](@ref)).
"""
function HomogeneousNetwork(net::ConfigurationNetwork; N::Real = 1.0)
    k = _regular_degree(net.degrees)
    k === nothing && throw(ArgumentError(
        "HomogeneousNetwork needs a regular degree distribution; got $(net.degrees). Use " *
        "HeterogeneousNetwork(net)"))
    return HomogeneousNetwork(k; ϕ = 0.0, N)
end

function HomogeneousNetwork(net::ClusteredNetwork; N::Real = 1.0)
    p = _clustered_degree_probabilities(net, 1e-12)
    ks = findall(>(1 - 1e-12), p)
    length(ks) == 1 || throw(ArgumentError(
        "HomogeneousNetwork needs a constant degree k = s + 2t; the clustered network " *
        "$(net) has several degrees. Use HeterogeneousNetwork(net)"))
    return HomogeneousNetwork(only(ks) - 1; ϕ = clustering_coefficient(net), N)
end

"""
    HeterogeneousNetwork(net::ConfigurationNetwork; tol = 1e-12, ϕ = 0.0, N = 1.0)
    HeterogeneousNetwork(net::ClusteredNetwork; tol = 1e-12, N = 1.0)

The heterogeneous network structure of a configuration or clustered descriptor. The moments
⟨k⟩, ⟨k²⟩ and the excess degree, which are all the heterogeneous closures use, are the exact
values of the degree distribution (for a clustered network, of k = s + 2t), so the constant
closure K = ⟨k(k − 1)⟩/⟨k⟩² is `closure_constant(net.degrees)` exactly (1 for Poisson degrees).
`degree_probs` is the probability vector, truncated where the tail mass is below `tol` for an
infinite support, and ⟨k³⟩ is computed from it. A clustered network has Keeling's
ϕ = `clustering_coefficient(net)`.
"""
function HeterogeneousNetwork(net::ConfigurationNetwork; tol::Real = 1e-12, ϕ::Real = 0.0,
                              N::Real = 1.0)
    d = net.degrees
    p = degree_probabilities(d; tol)
    mk = Float64(mean_degree(d))
    mk2 = Float64(degree_moment(d, 2))
    mk3 = Float64(degree_moment(d, 3))
    excess = mk > 0 ? Float64(excess_degree(d)) : 0.0
    return HeterogeneousNetwork(p, length(p) - 1, Float64(ϕ), Float64(N), mk, mk2, mk3, excess)
end

function HeterogeneousNetwork(net::ClusteredNetwork; tol::Real = 1e-12, N::Real = 1.0)
    p = _clustered_degree_probabilities(net, tol)
    mk = Float64(mean_degree(net))
    excess = mk > 0 ? Float64(excess_degree(net)) : 0.0
    mk2 = excess * mk + mk
    mk3 = sum(k^3 * p[k + 1] for k in 0:(length(p) - 1))
    return HeterogeneousNetwork(p, length(p) - 1, Float64(clustering_coefficient(net)),
                                Float64(N), mk, mk2, mk3, excess)
end

# P(k = s + 2t) of a clustered network, truncated where each marginal's tail is below tol.
function _clustered_degree_probabilities(net::ClusteredNetwork, tol)
    joint = net.joint.joint
    P = joint isa Tuple ?
        degree_probabilities(joint[1]; tol) * transpose(degree_probabilities(joint[2]; tol)) :
        joint
    p = zeros(Float64, (size(P, 1) - 1) + 2 * (size(P, 2) - 1) + 1)
    for a in axes(P, 1), b in axes(P, 2)
        p[(a - 1) + 2 * (b - 1) + 1] += P[a, b]
    end
    return p
end

"""
    network_structure(net::NetworkDescriptor; N = 1.0, tol = 1e-12) -> NetworkStructure

The NodeBasedModels network structure that the population pairwise model uses for a
descriptor: `HomogeneousNetwork` when every node has the same degree (a regular configuration
network, or a clustered network with constant s + 2t), otherwise `HeterogeneousNetwork`.
"""
network_structure(net::NetworkStructure; kw...) = net
function network_structure(net::ConfigurationNetwork; N::Real = 1.0, tol::Real = 1e-12)
    return _regular_degree(net.degrees) === nothing ? HeterogeneousNetwork(net; tol, N) :
           HomogeneousNetwork(net; N)
end
function network_structure(net::ClusteredNetwork; N::Real = 1.0, tol::Real = 1e-12)
    p = _clustered_degree_probabilities(net, tol)
    return count(>(1 - 1e-12), p) == 1 ? HomogeneousNetwork(net; N) :
           HeterogeneousNetwork(net; tol, N)
end
network_structure(net::NetworkDescriptor; kw...) = throw(ArgumentError(
    "NodeBasedModels has no population-level network structure for $(nameof(typeof(net))); " *
    "population pairwise models take ConfigurationNetwork or ClusteredNetwork descriptors " *
    "(graph-level models take an ExplicitGraph)"))

"""
    GraphNetwork(net::ExplicitGraph; transmission_rate = nothing, transmission_matrix = nothing)

The graph-instance network of the graph-level builders (`generate_individual_based`,
`generate_pair_based`) for an `ExplicitGraph`: a uniform per-edge rate supplied by the solver
keywords (the default), a uniform `transmission_rate`, or an explicit
`transmission_matrix` (`T[i, j]` = rate from node j to node i). An explicit rate 1.0 is kept
(verified issue B04).
"""
GraphNetwork(net::ExplicitGraph; transmission_rate::Union{Nothing,Real} = nothing,
             transmission_matrix::Union{Nothing,AbstractMatrix{<:Real}} = nothing) =
    _graph_network(net.graph; transmission_rate, transmission_matrix)

# ─── 0.1 entry points that also take a ContactModel or an ExplicitGraph ───────

"""
    generate_individual_based(model::ContactModel, net; kwargs...)
    generate_individual_based(model, net::ExplicitGraph; transmission_rate, transmission_matrix, kwargs...)

The 0.1 individual-based builder on `CompartmentalModel(model)` and
`GraphNetwork(net; transmission_rate, transmission_matrix)`; see the `CompartmentalModel`
method for the keywords. Every spontaneous transition still runs at `recovery_rate`
(verified issue B02); `node_based(model, net; level = :individual, p)` refuses the models
where that matters.
"""
generate_individual_based(model::ContactModel, net::GraphNetwork; kwargs...) =
    generate_individual_based(CompartmentalModel(model), net; kwargs...)
function generate_individual_based(model::Union{CompartmentalModel,ContactModel},
                                   net::ExplicitGraph;
                                   transmission_rate::Union{Nothing,Real} = nothing,
                                   transmission_matrix = nothing, kwargs...)
    return generate_individual_based(model, GraphNetwork(net; transmission_rate,
                                                         transmission_matrix); kwargs...)
end

"""
    generate_pair_based(model::ContactModel, net; kwargs...)
    generate_pair_based(model, net::ExplicitGraph; transmission_rate, transmission_matrix, kwargs...)

The 0.1 pair-based (SIR only) builder on `CompartmentalModel(model)` and
`GraphNetwork(net; transmission_rate, transmission_matrix)`.
"""
generate_pair_based(model::ContactModel, net::GraphNetwork; kwargs...) =
    generate_pair_based(CompartmentalModel(model), net; kwargs...)
function generate_pair_based(model::Union{CompartmentalModel,ContactModel}, net::ExplicitGraph;
                             transmission_rate::Union{Nothing,Real} = nothing,
                             transmission_matrix = nothing, kwargs...)
    return generate_pair_based(model, GraphNetwork(net; transmission_rate, transmission_matrix);
                               kwargs...)
end

"""
    generate_neighbourhood(model::ContactModel, k, n; kwargs...)

The neighbourhood model of `CompartmentalModel(model)` (SIS only).
"""
generate_neighbourhood(model::ContactModel, k::Integer, n::Integer; kwargs...) =
    generate_neighbourhood(CompartmentalModel(model), k, n; kwargs...)

"""
    basic_reproduction_number(model::ContactModel, network::NetworkStructure,
                              closure = BernoulliClosure())

The 0.1 pairwise R₀ formula on `CompartmentalModel(model)`. For the next-generation R₀ of a
model on a network descriptor use `basic_reproduction_number(cm, net, p)` (NetworkEpiCore).
"""
basic_reproduction_number(model::ContactModel, network::NetworkStructure,
                          closure::ClosureMethod = BernoulliClosure()) =
    basic_reproduction_number(CompartmentalModel(model), network, closure)

"""
    disease_free_equilibrium(model::ContactModel, network::NetworkStructure; N = 1.0)

The pairwise disease-free equilibrium of `CompartmentalModel(model)`.
"""
disease_free_equilibrium(model::ContactModel, network::NetworkStructure; N::Real = 1.0) =
    disease_free_equilibrium(CompartmentalModel(model), network; N)

"""
    reinfection_totals(psys::PairwiseSystem, sol) -> Dict{Symbol,Vector{Float64}}

Sum the reinfection-counted compartments `X_p` of a pairwise solution back to the compartments
before counting, along the saved trajectory (`S = Σ_p S_p`, …). For a system built from a
`ContactModel` (`node_based(with_reinfection_counting(sis_model(), L), net)`) the lumping reads
the model's labels (NetworkEpiCore `reinfection_totals`); for a legacy `CompartmentalModel`
lift it parses the names (`:S_3` ↦ `:S`) as in 0.1.
"""
function reinfection_totals(psys::PairwiseSystem, sol)
    cm = get(psys.metadata, :contact_model, nothing)
    cm === nothing && return _LegacyReinfection.reinfection_totals(psys, sol)
    series = Dict{Symbol,Vector{Float64}}(X => collect(Float64, sol[psys.singles[X]])
                                          for X in species_names(cm))
    return Dict{Symbol,Vector{Float64}}(reinfection_totals(cm, series))
end
