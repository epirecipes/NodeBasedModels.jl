# individual_based.jl — Order-1 moment closure: individual-based model (owner: WP24)
#
# Per-node ODEs on a specific graph (Sharkey 2008, 2011).
# Each node i has K-1 state variables (one per compartment minus conservation).
# Pairwise independence: ⟨A_i B_j⟩ ≈ ⟨A_i⟩⟨B_j⟩
#
# For SIR:
#   d⟨S_i⟩/dt = -Σ_j T_ij ⟨S_i⟩⟨I_j⟩
#   d⟨I_i⟩/dt = Σ_j T_ij ⟨S_i⟩⟨I_j⟩ - g_i⟨I_i⟩
#
# This is also known as NIMFA (N-Intertwined Mean Field Approximation)
# or the quenched mean field (QMF) model.
#
# In general (Sharkey 2008, Eq. 3/7, one term per transition of the model):
#   an infection X → Y at per-contact rate τ_r through the infectors Z ∈ via_r contributes
#       ∓ τ_r ⟨X_i⟩ Σ_j A_ij Σ_{Z ∈ via_r} ⟨Z_j⟩
#   to d⟨X_i⟩/dt and d⟨Y_i⟩/dt (A_ij = 1 for an edge j → i), and a spontaneous transition
#   X → Y at rate a contributes ∓ a⟨X_i⟩, which is exact (no closure).
# Changes in 0.2: every transition runs at the value of its own rate parameter (verified issue
# B02: 0.1 ran every spontaneous transition at `recovery_rate`), each infection transition has
# its own force through its own infectors (`Transition.via`, verified issue B04: 0.1 applied the
# summed force of every infectious compartment once per infection transition), and seeded nodes
# start in the entry state of the infections (E for SEIR; DESIGN §E.2).

"""
    IndividualBasedResult

Result container for individual-based (order-1) model on a graph.

# Fields
- `sol` — ODE solution
- `graph` — the graph used
- `N` — number of nodes
- `K` — number of tracked states per node
- `state_names` — names of tracked states (e.g., [:S, :I])
- `model` — the `CompartmentalModel` that was solved (a `ContactModel` is lowered with
  `CompartmentalModel(cm)`)

Node `i` in tracked state `k` is `sol[K*(i-1) + k, :]`; the last compartment of the model is
derived by conservation, ⟨X_i⟩ = 1 − Σ_k (tracked states of node i).
"""
struct IndividualBasedResult
    sol::Any
    graph::Any
    N::Int
    K::Int
    state_names::Vector{Symbol}
    model::CompartmentalModel
end

function Base.show(io::IO, r::IndividualBasedResult)
    tspan = (first(r.sol.t), last(r.sol.t))
    print(io, "IndividualBasedResult(N=$(r.N), states=$(r.state_names), tspan=$tspan)")
end

"""
    node_state(result::IndividualBasedResult, i, state, t_idx)

Get the probability that node `i` is in compartment `state` at time index `t_idx`.
"""
function node_state(r::IndividualBasedResult, i::Int, state::Symbol, t_idx::Int)
    k = findfirst(==(state), r.state_names)
    if isnothing(k)
        _check_ib_state(r, state)
        # Derived state (e.g., R = 1 - S - I)
        return 1.0 - sum(r.sol[r.K*(i-1) + j, t_idx] for j in 1:r.K)
    end
    return r.sol[r.K*(i-1) + k, t_idx]
end

function _check_ib_state(r::IndividualBasedResult, state::Symbol)
    state === last(r.model.compartment_names) || throw(ArgumentError(
        "unknown compartment $(state); the compartments are " *
        "$(join(r.model.compartment_names, ", "))"))
    return nothing
end

"""
    aggregate(result::IndividualBasedResult, state)

Aggregate a state across all nodes: [X](t) = Σ_i ⟨X_i⟩(t).
Returns a vector over time points.
"""
function aggregate(r::IndividualBasedResult, state::Symbol)
    nt = length(r.sol.t)
    k = findfirst(==(state), r.state_names)
    if isnothing(k)
        _check_ib_state(r, state)
        return [sum(1.0 - sum(r.sol[r.K*(i-1) + j, t_idx] for j in 1:r.K)
                     for i in 1:r.N) for t_idx in 1:nt]
    end
    return [sum(r.sol[r.K*(i-1) + k, t_idx] for i in 1:r.N) for t_idx in 1:nt]
end

"""
    compartment(r::IndividualBasedResult, state) -> Vector{Float64}

The expected number of nodes in `state` over the saved times (the same as
[`aggregate`](@ref)).
"""
compartment(r::IndividualBasedResult, state::Symbol) = aggregate(r, state)

"""
    compartments(r::IndividualBasedResult, states) -> Dict{Symbol,Vector{Float64}}

[`compartment`](@ref) for several states, keyed by name.
"""
function compartments(r::IndividualBasedResult, states::AbstractVector{Symbol})
    return Dict(state => compartment(r, state) for state in states)
end

"""
    population_fraction(r::IndividualBasedResult, state) -> Vector{Float64}

The expected fraction of the nodes in `state` over the saved times.
"""
population_fraction(r::IndividualBasedResult, state::Symbol) = aggregate(r, state) ./ r.N

"""
    model_curves(r::IndividualBasedResult, sol = r.sol; t = sol.t, label = "individual") -> ModelCurves

The expected population fractions of every compartment and `:infectious` (the sum over the
infectious compartments) on the time grid `t`, as a `NetworkEpiCore.ModelCurves` with
representation `:individual`.
"""
function model_curves(r::IndividualBasedResult, sol = r.sol; t = sol.t,
                      label::AbstractString = "individual")
    tt = collect(Float64, t)
    names = r.model.compartment_names
    # Σ_i ⟨X_i⟩ for every tracked state, evaluating the solution once per time point
    tot = zeros(r.K, length(tt))
    for (m, ti) in enumerate(tt)
        u = sol(ti)
        for i in 1:r.N, k in 1:r.K
            tot[k, m] += u[r.K*(i-1) + k]
        end
    end
    vals = Dict{Symbol,Vector{Float64}}()
    for (k, s) in enumerate(r.state_names)
        vals[s] = tot[k, :] ./ r.N
    end
    vals[last(names)] = 1 .- sum(vals[s] for s in r.state_names; init = zeros(length(tt)))
    vals[:infectious] = sum(vals[s] for s in r.model.infectious_compartments)
    return ModelCurves(tt, vals; label, representation = :individual,
                       metadata = Dict{Symbol,Any}(:N => r.N))
end

# ─── Rates ────────────────────────────────────────────────────────────────────

# The numeric value (or, for a time-dependent rate, the function of t) of every transition's
# rate, in the order of `model.transitions`, from the parameter values `p` and the legacy
# keywords `infection_rate` / `recovery_rate`.
#
# - `p` maps rate parameter names to values; a key that is no rate parameter of the model is an
#   error.
# - `infection_rate` / `recovery_rate` are the 0.1 keywords: the value of the rate of every
#   infection / spontaneous transition. They need a model whose transitions of that kind share
#   one rate (τ and γ of SIR and SIS): a model with several spontaneous rates (SEIR, SIRS) needs
#   `p` (verified issue B02: 0.1 ran them all at recovery_rate). A keyword that contradicts `p`
#   is an error.
# - When `p === nothing` (the 0.1 call), a kind of transition that shares one rate parameter falls
#   back to the 0.1 defaults τ = 0.5, γ = 0.1.
# - `infections_needed = false` (an explicit transmission matrix, which replaces the per-contact
#   rate): the infection rates are not evaluated and are set to 1.
function _graph_rate_values(model::CompartmentalModel, p, infection_rate, recovery_rate,
                            builder::AbstractString; infections_needed::Bool = true)
    trs = model.transitions
    names_of(ks) = unique!(reduce(vcat, (_rate_parameter_names(trs[k].rate) for k in ks);
                                  init = Symbol[]))
    known = names_of(eachindex(trs))
    vals = Dict{Symbol,Float64}()
    if p !== nothing
        unknown = Symbol[]
        for (k, v) in p
            name = Symbol(k)
            name in known || push!(unknown, name)
            vals[name] = Float64(v)
        end
        isempty(unknown) || throw(ArgumentError(
            "$builder: $(join(sort!(unknown), ", ")) " *
            (length(unknown) == 1 ? "is not a rate parameter" : "are not rate parameters") *
            " of model :$(model.name) (its rate parameters are $(join(known, ", ")))"))
    end
    rates = Vector{Any}(undef, length(trs))
    from_p = Int[]
    for (kind, kw, value, default) in ((:infection, :infection_rate, infection_rate, 0.5),
                                       (:spontaneous, :recovery_rate, recovery_rate, 0.1))
        idx = [k for (k, tr) in enumerate(trs) if tr.type === kind]
        isempty(idx) && continue
        if kind === :infection && !infections_needed
            rates[idx] .= 1.0
            continue
        end
        r1 = trs[first(idx)].rate
        shared = all(k -> isequal(trs[k].rate, r1), idx)
        if value === nothing && p === nothing && shared && r1 isa Symbol
            value = default
        end
        if value === nothing
            append!(from_p, idx)
            continue
        end
        shared || throw(ArgumentError(
            "$builder: `$kw` is the rate of every $kind transition, but the $kind transitions " *
            "of model :$(model.name) have the different rates " *
            "$(join(unique(string(trs[k].rate) for k in idx), ", ")); pass the parameter " *
            "values in `p = Dict(" * join(("$(repr(n)) => …" for n in names_of(idx)), ", ") * ")`" *
            (kind === :spontaneous ? " (0.1 ran every spontaneous transition at recovery_rate, " *
                                     "verified issue B02)" : "")))
        (r1 isa Symbol && haskey(vals, r1) && vals[r1] != value) && throw(ArgumentError(
            "$builder: `$kw = $value` contradicts `p[$(repr(r1))] = $(vals[r1])`; give the rate " *
            "once"))
        rates[idx] .= Float64(value)
    end
    missing = [n for n in names_of(from_p) if !haskey(vals, n)]
    isempty(missing) || throw(ArgumentError(
        "$builder: no value for the rate parameter(s) $(join(missing, ", ")) of model " *
        ":$(model.name); pass `p = Dict(" * join(("$(repr(n)) => …" for n in known), ", ") * ")`" *
        (length(names_of([k for (k, tr) in enumerate(trs) if tr.type === :spontaneous])) > 1 ?
         " (each spontaneous transition runs at its own rate; 0.1 ran them all at " *
         "recovery_rate, verified issue B02)" : "")))
    for k in from_p
        rates[k] = _graph_rate(trs[k].rate, vals)
    end
    return rates
end

# A constant rate as a Float64, a time-dependent one as a function of t.
function _graph_rate(rate, vals::Dict{Symbol,Float64})
    rate isa Real && !(rate isa Symbolics.Num) && return Float64(rate)
    rate isa Symbol && rate !== :t && return vals[rate]
    v = try
        rate_value(rate, vals)
    catch err
        err isa ArgumentError || rethrow()
        nothing   # depends on t
    end
    v === nothing || return Float64(Symbolics.value(v))
    return t -> Float64(Symbolics.value(rate_value(rate, vals; t)))
end

_rate_at(r::Float64, t) = r
_rate_at(r, t) = r(t)

# ─── Seeding ──────────────────────────────────────────────────────────────────

# The initial probability of every compartment for every node: `x[i]` maps compartment names
# to probabilities for node i.
function _graph_initial_state(model::CompartmentalModel, N::Int, initial_infected, initial,
                              seed_fraction::Real, seed_state::Symbol, builder::AbstractString)
    background = _background(model)
    seedX() = seed_state === :entry ? _entry_state(model, background) :
              seed_state === :first_infectious ? first(model.infectious_compartments) :
              throw(ArgumentError("$builder: seed_state must be :entry or :first_infectious; " *
                                  "got $(repr(seed_state))"))
    point(X) = Dict{Symbol,Float64}(X => 1.0)
    if initial_infected !== nothing
        initial === nothing || throw(ArgumentError(
            "$builder: pass either `initial_infected` or `initial`, not both"))
        X = seedX()
        seeded = Set{Int}(initial_infected)
        all(i -> 1 <= i <= N, seeded) || throw(ArgumentError(
            "$builder: initial_infected lists nodes outside 1:$N"))
        return [i in seeded ? point(X) : point(background) for i in 1:N]
    elseif initial isa SeedNodes
        bg = something(initial.default, background)
        x = [point(bg) for _ in 1:N]
        for (X, nodes) in initial.assignments
            X in model.compartment_names || throw(ArgumentError(
                "$builder: the initial condition names $(X), which is not a compartment of " *
                "model :$(model.name)"))
            for i in nodes
                1 <= i <= N || throw(ArgumentError("$builder: SeedNodes lists node $i outside 1:$N"))
                x[i] = point(X)
            end
        end
        return x
    elseif initial === nothing || initial isa SeedFraction
        frac, _ = _resolve_seed(model, initial, seed_fraction, seed_state, N)
        return [frac for _ in 1:N]
    end
    throw(ArgumentError("$builder: `initial` must be a SeedFraction (the same probabilities for " *
                        "every node) or SeedNodes (named nodes); got $(nameof(typeof(initial)))"))
end

# ─── Builder ──────────────────────────────────────────────────────────────────

# The work buffers of the individual-based right-hand side (the parameter object of its ODE
# problem): the current value of every transition rate and the neighbour sums of one node.
struct _IBWork
    rate_t::Vector{Float64}
    pressure::Vector{Float64}
end

"""
    generate_individual_based(model, net; p = nothing, tspan = (0.0, 100.0),
                              initial_infected = nothing, initial = nothing,
                              seed_fraction = 1e-3, seed_state = :entry, saveat = 1.0,
                              reltol = nothing, abstol = nothing,
                              infection_rate = nothing, recovery_rate = nothing)

Generate and solve the order-1 (individual-based / NIMFA) ODE system on a
specific graph. Uses pairwise independence: ⟨S_i I_j⟩ = ⟨S_i⟩⟨I_j⟩.

This is exact in the limit of infinite graph connectivity and provides an
upper bound on the true epidemic. Errors arise from "anomalous terms"
identified by Sharkey (2011) corresponding to 2-cycles.

Every transition contributes its own term (Sharkey 2008, Eq. 3/7): an infection X → Y with
per-contact rate τ through the infectors `via` (every infectious compartment when empty) moves
node i at rate τ Σ_j A_ij Σ_{Z ∈ via} ⟨Z_j⟩ (A_ij = 1 for an edge from j to i), and a
spontaneous transition X → Y at rate a moves it at rate a. Each rate takes the value of its own
parameter.

# Arguments
- `model` — a `CompartmentalModel` or a NetworkEpiCore `ContactModel` (e.g. `seir_model()`)
- `net` — a NetworkEpiCore `ExplicitGraph` or a `GraphNetwork` (an optional per-edge
  transmission matrix `T[i, j]`, the rate from node j to node i, **replaces** the per-contact
  rate of the infection transitions; it needs a model whose infections share one rate value)

# Keyword Arguments
- `p` — rate parameter values by name, e.g. `Dict(:τ => 0.3, :σ => 1.0, :γ => 0.25)`. A key
  that is not a rate parameter of the model is an error.
- `infection_rate`, `recovery_rate` — the 0.1 keywords: the rate of every infection /
  spontaneous transition, for a model whose transitions of that kind share one rate (τ and γ of
  SIR and SIS). A model with several spontaneous rates (SEIR, SIRS) needs `p`: 0.1 ran every
  spontaneous transition at `recovery_rate` (verified issue B02). A keyword that contradicts
  `p` is an error. Without `p` and keywords, SIR/SIS use the 0.1 defaults τ = 0.5, γ = 0.1.
- `tspan` — time span (default: (0.0, 100.0))
- `initial_infected` — nodes that start in the seed state (the others start in the background
  compartment, the first susceptible compartment)
- `initial` — alternatively `NetworkEpiCore.SeedNodes(:E => [1, 2])` (named nodes) or
  `SeedFraction(:I => 0.01, …)` (the same probabilities for every node)
- `seed_fraction` (alias `ε`) — probability of the seed state for every node when neither
  `initial_infected` nor `initial` is given
- `seed_state` — `:entry` (default): the entry state of the infections, E for SEIR (DESIGN
  §E.2); `:first_infectious`: the first infectious compartment, the 0.1 rule
- `saveat` — time points to save solution (default: 1.0)
- `reltol`, `abstol` — solver tolerances (the solver defaults if `nothing`)

# Returns
`IndividualBasedResult` containing the ODE solution and metadata.

# Thread safety
The right-hand side of the ODE problem `prob = result.sol.prob` reuses two work buffers (the
rate values and the neighbour sums), which are the problem's parameter object `prob.p`. A
problem must therefore not be solved in several tasks at once with the same `p` (`remake(prob)`
shares it). To solve it concurrently, e.g. in an `EnsembleThreads` ensemble, give each task its
own buffers: `remake(prob; p = deepcopy(prob.p))`, for instance as the ensemble's
`prob_func = (prob, i, repeat) -> remake(prob; p = deepcopy(prob.p))`.

# Example
```julia
using Graphs
g = random_regular_graph(100, 6)
result = generate_individual_based(seir_model(), ExplicitGraph(g);
    p = Dict(:τ => 0.15, :σ => 0.5, :γ => 0.1), initial_infected = [1, 2, 3])
S_total = aggregate(result, :S)
```

# References
- Sharkey (2008) "Deterministic epidemiological models at the individual level", Eq. (3), (7)
- Sharkey (2011) "Deterministic epidemic models on contact networks" Eq. (30)
- Van Mieghem et al. (2009) "Virus spread in networks" (N-intertwined model)
"""
function generate_individual_based(model::CompartmentalModel,
                                    net::GraphNetwork;
                                    tspan::Tuple{Real,Real} = (0.0, 100.0),
                                    p::Union{Nothing,AbstractDict} = nothing,
                                    infection_rate::Union{Nothing,Real} = nothing,
                                    recovery_rate::Union{Nothing,Real} = nothing,
                                    initial_infected::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                    initial::Union{Nothing,SeedSpec} = nothing,
                                    ε::Real = 1e-3,
                                    seed_fraction::Real = ε,
                                    seed_state::Symbol = :entry,
                                    saveat::Real = 1.0,
                                    reltol::Union{Nothing,Real} = nothing,
                                    abstol::Union{Nothing,Real} = nothing)
    builder = "generate_individual_based(:$(model.name))"
    g = net.graph
    N = nv(g)
    # An explicit transmission matrix T[i, j] of the network is the per-contact rate from j to
    # i, so it replaces the rates of the infection transitions (their values are not needed and
    # not used); the infections must then share one rate.
    explicit_T = !isnothing(net.transmission_matrix)
    inf_rates = unique([tr.rate for tr in model.transitions if tr.type === :infection])
    (explicit_T && length(inf_rates) > 1) && throw(ArgumentError(
        "$builder: the transmission matrix of the network replaces the per-contact rate, so " *
        "every infection transition must have the same rate; this model has " *
        "$(join(string.(inf_rates), ", ")). Use an unweighted graph and the rates in `p`"))
    rates = _graph_rate_values(model, p, infection_rate, recovery_rate, builder;
                               infections_needed = !explicit_T)

    # Identify tracked states (all except the last, which is derived by conservation)
    all_names = model.compartment_names
    tracked = all_names[1:end-1]
    derived = all_names[end]
    K = length(tracked)
    index(X) = something(findfirst(==(X), tracked), 0)   # 0: the derived compartment

    # Edge weights: W[i, j] = 1 for an edge j → i (the per-contact rate is the transition's),
    # or the explicit transmission matrix T[i, j], which is the rate itself (the infection
    # rates are then 1).
    adj = [_infection_sources(g, i) for i in 1:N]
    weights = explicit_T ? [Float64[net.transmission_matrix[i, j] for j in adj[i]] for i in 1:N] :
                           [ones(length(adj[i])) for i in 1:N]

    # The infectious compartments whose neighbour sums the infections read, and for each
    # infection transition the positions of its infectors among them.
    infectious = model.infectious_compartments
    inf_pos = Dict(Z => m for (m, Z) in enumerate(infectious))
    inf_state = [index(Z) for Z in infectious]
    tr_from = [index(tr.from) for tr in model.transitions]
    tr_to = [index(tr.to) for tr in model.transitions]
    tr_infection = [tr.type === :infection for tr in model.transitions]
    tr_infectors = [tr.type === :infection ? [inf_pos[Z] for Z in infectors(model, tr)] : Int[]
                    for tr in model.transitions]
    nZ = length(infectious)

    # The work buffers of the right-hand side, reused by every call: the rate values (the
    # time-dependent ones are refreshed at each t) and the neighbour sums (a fresh vector only for
    # a non-Float64 state, e.g. the dual numbers of an implicit solver's Jacobian). They are the
    # problem's parameter object, so a copy of the problem with `p = deepcopy(prob.p)` has its own
    # (the function itself must not run in two tasks at once on the same buffers).
    work = _IBWork(Float64[r isa Float64 ? r : 0.0 for r in rates], zeros(nZ))
    time_dependent = [k for (k, r) in enumerate(rates) if !(r isa Float64)]

    function rhs!(du, u, work::_IBWork, t)
        rate_t, pressure_buffer = work.rate_t, work.pressure
        for k in time_dependent
            rate_t[k] = _rate_at(rates[k], t)
        end
        # prob(base, s): probability that the node at offset `base` is in tracked state s
        # (s = 0: the derived compartment)
        prob(base, s) = s == 0 ? 1 - sum(@view u[base+1:base+K]) : u[base + s]
        pressure = eltype(u) === Float64 ? pressure_buffer : zeros(eltype(u), nZ)
        @inbounds for i in 1:N
            base = K * (i - 1)
            # Σ_j W_ij ⟨Z_j⟩ for every infectious compartment Z
            fill!(pressure, 0.0)
            for (jj, j) in enumerate(adj[i])
                w = weights[i][jj]
                w == 0 && continue
                base_j = K * (j - 1)
                for m in 1:nZ
                    pressure[m] += w * prob(base_j, inf_state[m])
                end
            end
            for k in 1:K
                du[base + k] = 0.0
            end
            for k in eachindex(rates)
                s_from, s_to = tr_from[k], tr_to[k]
                rate = rate_t[k]
                if tr_infection[k]
                    force = zero(eltype(u))
                    for m in tr_infectors[k]
                        force += pressure[m]
                    end
                    rate *= force
                end
                flux = rate * prob(base, s_from)
                s_from == 0 || (du[base + s_from] -= flux)
                s_to == 0 || (du[base + s_to] += flux)
            end
        end
        nothing
    end

    # Initial conditions
    x = _graph_initial_state(model, N, initial_infected, initial, seed_fraction, seed_state,
                             builder)
    u0 = zeros(K * N)
    for i in 1:N, (k, X) in enumerate(tracked)
        u0[K * (i - 1) + k] = get(x[i], X, 0.0)
    end

    # (not named `prob`: rhs! defines a local function of that name, and an outer local of the
    # same name would be captured, boxed and reassigned on every call)
    odeprob = ODEProblem(rhs!, u0, (Float64(tspan[1]), Float64(tspan[2])), work)
    tol = (; (k => Float64(v) for (k, v) in pairs((; reltol, abstol)) if v !== nothing)...)
    sol = solve(odeprob; saveat = Float64(saveat), tol...)

    return IndividualBasedResult(sol, g, N, K, tracked, model)
end
