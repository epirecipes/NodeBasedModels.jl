# population_pairwise.jl — population-level pairwise ODE generation (owner: WP15)
#
# Given a CompartmentalModel, a NetworkStructure, and a ClosureMethod,
# symbolically generates the full pairwise ODE system as a compiled
# ModelingToolkit system.
#
# The key insight: for K compartments, we track:
#   - K single (node) variables: [S], [I], [R], ...
#   - K(K+1)/2 pair variables: [SS], [SI], [SR], [II], [IR], [RR], ...
#     (symmetric: [AB] = [BA])
# Equations for pairs depend on triples [ABC], which are closed using
# the chosen ClosureMethod.
#
# `node_based(model, net)` (lift.jl) lowers any model to a CompartmentalModel and calls the
# builder below; `generate_pairwise(::CompartmentalModel, ::NetworkStructure, closure)` is the
# 0.1 entry point. Changes in 0.2: infection transitions act through their own infectors
# (`Transition.via`, verified issue B04), rates may be numbers, `Expr`s or symbolic expressions,
# the default initial condition seeds the entry state of the infections (E for SEIR; 0.1 seeded
# the first infectious compartment), and the initial condition is the image of the edge-based
# initial condition under π^PW (DESIGN §E.2).

"""
    PairwiseSystem

Container for the generated pairwise ODE system.

Fields:
- `system` — compiled ModelingToolkit system
- `u0` — initial condition Dict
- `tspan` — time span
- `params` — parameter Dict (values given when the system was built; empty by default)
- `model` — the model the system was built from, as passed (a `CompartmentalModel`, or any
  model that `node_based` accepts, e.g. a `NetworkEpiCore.ContactModel`)
- `network` — the `NetworkStructure` used by the closure (a network descriptor given to
  `node_based` is converted; the descriptor itself is `metadata[:descriptor]`)
- `closure` — ClosureMethod used
- `singles` — Dict of single variable symbols
- `pairs` — Dict of pair variable symbols
- `metadata` — `Dict{Symbol,Any}`: `:compartmental` (the lowered `CompartmentalModel`),
  `:contact_model` (the per-contact `ContactModel`, or `nothing` for a legacy build),
  `:descriptor`, `:N` (population scale of the state), `:background` (the compartment holding
  the unseeded nodes), `:seed` (initial fractions by compartment), `:seed_state` and
  `:default_fraction` (the default seeding), `:cumulative` (the accumulator variable, or
  `nothing`), `:infected` (the infected compartments, DESIGN §J.8) and `:infections` (the
  transitions the accumulator counts), `:parameters` (MTK parameters by name), `:states` and
  `:equations` (the uncompiled vector field), `:representation` (`:pairwise`)
"""
struct PairwiseSystem
    system::Any     # ODESystem
    u0::Dict
    tspan::Tuple{Float64,Float64}
    params::Dict
    model::Any
    network::NetworkStructure
    closure::ClosureMethod
    singles::Dict{Symbol,Any}
    pairs::Dict{Tuple{Symbol,Symbol},Any}
    metadata::Dict{Symbol,Any}
end

PairwiseSystem(system, u0, tspan, params, model, network, closure, singles, pairs) =
    PairwiseSystem(system, u0, tspan, params, model, network, closure, singles, pairs,
                   Dict{Symbol,Any}())

function Base.show(io::IO, psys::PairwiseSystem)
    model = _compartmental(psys)
    print(io, "PairwiseSystem(", model === nothing ? "?" : model.name, "; ",
          length(psys.singles), " singles, ", length(psys.pairs), " pairs, closure = ",
          psys.closure, ")")
end

# The CompartmentalModel that the equations were generated from.
_compartmental(psys::PairwiseSystem) =
    get(psys.metadata, :compartmental, psys.model isa CompartmentalModel ? psys.model : nothing)

function _validate_pairwise_closure_support(network::NetworkStructure,
                                            closure::ClosureMethod)
    closure isa EamesClosure && throw(ArgumentError(
        "EamesClosure was removed in NodeBasedModels 0.2: the hybrid closure of Eames (2008) " *
        "needs separate regular and random contact rates that NodeBasedModels never encoded, " *
        "and it was never validated. Use KeelingClosure (clustering ϕ) or BarnardClosure."))
    closure isa KirkwoodClosure && throw(ArgumentError(
        "KirkwoodClosure is only used by generate_pair_based on explicit graphs, not by population-level generate_pairwise."))
    if network isa HeterogeneousNetwork &&
       !(closure isa BernoulliClosure || closure isa KeelingClosure)
        throw(ArgumentError(
            "Only BernoulliClosure and KeelingClosure are implemented for HeterogeneousNetwork."))
    end
end

"""
    generate_pairwise(model, network, closure; tspan=(0.0,100.0), N=1.0, ε=1e-3,
                      seed_fraction=ε, seed_state=:entry, initial=nothing, cumulative=false)

Generate a pairwise ODE system from a compartmental model, network structure,
and triple closure approximation.

Returns a `PairwiseSystem` containing the MTK ODESystem and initial conditions.

`model` may also be a `NetworkEpiCore.ContactModel` (e.g. `sir_model()`) and `network` a
network descriptor; that method forwards to [`node_based`](@ref).

Closure support matrix:

| Closure | HomogeneousNetwork | HeterogeneousNetwork |
|---|:---:|:---:|
| `BernoulliClosure` | ✓ | ✓ |
| `KeelingClosure` | ✓ | ✓ |
| `BarnardClosure` | ✓ | ✗ |
| `PowerClosure` | ✓ | ✗ |
| `EamesClosure` | ✗ (removed in 0.2) | ✗ |

`KirkwoodClosure` is reserved for `generate_pair_based` on explicit graphs.

# Initial condition

Singles and pairs are in counts on the population scale `N` (`N = 1` gives fractions). A
fraction ε = `seed_fraction` of the nodes is seeded and every other node starts in the
**background** compartment, the first susceptible compartment (the first source of an infection
transition):

- `seed_state = :entry` (the default) seeds the **entry state**, the unique target of the
  infection transitions from the background compartment: I for SIR, E for SEIR. A model whose
  background has several entry states (two strains) needs an explicit `initial`.
- `seed_state = :first_infectious` seeds the first infectious compartment, the
  NodeBasedModels 0.1 default (I for SEIR).
- `initial = SeedFraction(:E => 0.01, …)` (or `SeedCount`/`SeedNodes` with an integer `N`)
  seeds the named compartments; unnamed nodes are background.

Pairs are then those of a random arrangement, [XY] = ⟨k⟩ N x_X x_Y with x_X = [X]/N in the
ordered-pair convention (a cross pair [XY], X ≠ Y, counts each undirected XY edge once; a self
pair [XX] counts it twice), so Σ over ordered pairs is ⟨k⟩N. This is the image under π^PW of
the edge-based initial condition θ = 1, φ_X = pop_X = ρ_X: [s] = q, [sX] = ⟨k⟩qρ_X,
[ss] = ⟨k⟩q² (DESIGN §E.2).

`cumulative = true` adds the state `cumulative`, the cumulative number of infections: the seeds
in *infected* compartments plus the flux of every transition from a non-infected compartment
into an infected one. Infection status is structural (DESIGN §J.8, the rule of NetworkOutbreaks'
`final_size`; see [`_infected_compartments`](@ref)): E and I of SEIR are infected, a recovered R,
a vaccinated V or a traced Q are not, so seeding R or V does not count as infection. For models
without an arrow back into a susceptible class, `cumulative/N` is the fraction ever infected.
`node_based` adds the accumulator by default.

# Example
```julia
sys = generate_pairwise(sir_model(), regular_network(6), KeelingClosure())
sol = solve_pairwise(sys, Dict(:τ => 0.2, :γ => 0.1))
```
"""
function generate_pairwise(model::CompartmentalModel,
                            network::NetworkStructure,
                            closure::ClosureMethod;
                            tspan::Tuple{Real,Real} = (0.0, 100.0),
                            N::Real = 1.0,
                            ε::Real = 1e-3,
                            seed_fraction::Real = ε,
                            seed_state::Symbol = :entry,
                            initial::Union{Nothing,SeedSpec} = nothing,
                            cumulative::Bool = false)
    resolve = (init, ρ, state) -> _resolve_seed(model, init, ρ, state, N)
    return _build_pairwise(model, network, closure; tspan, N, seed = resolve(initial,
                           seed_fraction, seed_state), resolve, default_fraction = seed_fraction,
                           seed_state, cumulative, infected = _infected_compartments(model),
                           source = model)
end

"""
    _build_pairwise(model::CompartmentalModel, network, closure; tspan, N, seed, resolve,
                    default_fraction, seed_state, cumulative, infected, source,
                    contact = nothing, descriptor = nothing, params = Dict(), name = nothing)
        -> PairwiseSystem

The population pairwise builder shared by `generate_pairwise` and `node_based`. `seed =
(x, background)` gives the initial fraction `x[X]` of every compartment (see
[`_resolve_seed`](@ref)); `resolve(initial, seed_fraction, seed_state)` recomputes it for
[`default_initial_conditions`](@ref), with `default_fraction` and `seed_state` as defaults.
`infected` are the infected compartments (DESIGN §J.8: [`_infected_compartments`](@ref) of the
contact model on the `node_based` path, of `model` on the legacy path). The `cumulative`
accumulator starts at the seeds in `infected` and counts the transitions from outside
`infected` into it; both are stored in the metadata (`:infected`, `:infections`) for
`default_initial_conditions`.
"""
function _build_pairwise(model::CompartmentalModel, network::NetworkStructure,
                         closure::ClosureMethod; tspan, N::Real, seed, resolve,
                         default_fraction::Real, seed_state::Symbol, cumulative::Bool,
                         infected::Vector{Symbol}, source, contact = nothing,
                         descriptor = nothing, params::AbstractDict = Dict{Any,Float64}(),
                         name::Union{Nothing,Symbol} = nothing)
    _validate_pairwise_closure_support(network, closure)
    names = model.compartment_names
    K = length(names)
    unknown = setdiff(infected, names)
    isempty(unknown) || throw(ArgumentError(
        "the infected compartments $(join(unknown, ", ")) are not compartments of model " *
        ":$(model.name)"))
    infections = _incidence_transitions(model, infected)
    cumulative && :cumulative in names && throw(ArgumentError(
        "a compartment is named `cumulative`, which clashes with the accumulator state; " *
        "rename it or pass cumulative = false"))

    # ─── Create symbolic variables ────────────────────────────────────────
    @independent_variables t
    D = Differential(t)

    # Node-level variables [A] for each compartment A
    singles = Dict{Symbol,Any}()
    single_vars = []
    for name in names
        var = only(@variables $(name)(t))
        singles[name] = var
        push!(single_vars, var)
    end

    # Pair-level variables [AB] for each unordered pair (A,B)
    pairs = Dict{Tuple{Symbol,Symbol},Any}()
    pair_vars = []
    for i in 1:K, j in i:K
        a, b = names[i], names[j]
        sym = Symbol(a, b)
        var = only(@variables $(sym)(t))
        pairs[(a, b)] = var
        push!(pair_vars, var)
    end

    # ─── Create parameters ────────────────────────────────────────────────
    # One MTK parameter per name appearing in a rate (for Symbol rates, one per rate symbol in
    # order of first appearance, as in 0.1).
    rate_names = unique!(reduce(vcat, (_rate_parameter_names(tr.rate) for tr in model.transitions);
                                init = Symbol[]))
    param_syms = Dict{Symbol,Any}()
    for rname in rate_names
        p = only(@parameters $(rname))
        param_syms[rname] = p
    end
    rates = Any[_rate_symbolic(tr.rate, param_syms, t) for tr in model.transitions]

    # ─── Build node equations ─────────────────────────────────────────────
    node_eqs = _build_node_equations(model, singles, pairs, rates, D)

    # ─── Build pair equations ─────────────────────────────────────────────
    pair_eqs = _build_pair_equations(model, network, closure,
                                      singles, pairs, rates, D, names)

    all_eqs = vcat(node_eqs, pair_eqs)
    states = Any[single_vars..., pair_vars...]
    cum = nothing
    if cumulative
        cum = only(@variables cumulative(t))
        push!(all_eqs, D(cum) ~ _incidence(model, singles, pairs, rates, infections))
        push!(states, cum)
    end

    # ─── Create ODESystem ─────────────────────────────────────────────────
    sysname = something(name, Symbol(:pairwise_, model.name))
    sys = System(all_eqs, t; name = sysname)
    sys = mtkcompile(sys)

    # ─── Initial conditions ───────────────────────────────────────────────
    x, background = seed
    cum0 = _seeded_infections(x, background, infected)
    u0 = _pairwise_u0(model, network, singles, pairs, N, x; cumulative = cum, cum0)

    metadata = Dict{Symbol,Any}(
        :compartmental => model, :contact_model => contact, :descriptor => descriptor,
        :N => Float64(N), :background => background, :seed => x, :resolve => resolve,
        :default_fraction => Float64(default_fraction), :seed_state => seed_state,
        :cumulative => cum, :infected => infected, :infections => infections,
        :parameters => param_syms, :states => states,
        :equations => all_eqs, :representation => :pairwise, :name => sysname)
    return PairwiseSystem(sys, u0, (Float64(tspan[1]), Float64(tspan[2])),
                          Dict{Any,Float64}(k => Float64(v) for (k, v) in params), source,
                          network, closure, singles, pairs, metadata)
end

# ─── Node equations ───────────────────────────────────────────────────────────
# d[A]/dt = Σ (inflows to A) - Σ (outflows from A)
#   infection: S → I via Z contributes -τ[SZ] to d[S]/dt, +τ[SZ] to d[I]/dt, for each
#              infector Z of the transition (Transition.via; all infectious if empty)
#   spontaneous: I → R contributes -γ[I] to d[I]/dt, +γ[I] to d[R]/dt

function _build_node_equations(model, singles, pairs, rates, D)
    eqs = Equation[]
    names = model.compartment_names

    for name in names
        rhs = Num(0)

        for (k, tr) in enumerate(model.transitions)
            rate = rates[k]

            if tr.type == :infection
                # Infection: from → to at rate τ per [from, infector] edge
                # This creates -τ·Σ_Z [from·Z] for each infector Z of the transition
                for inf_comp in infectors(model, tr)
                    pair_var = _get_pair(pairs, tr.from, inf_comp)
                    if name == tr.from
                        rhs -= rate * pair_var
                    elseif name == tr.to
                        rhs += rate * pair_var
                    end
                end
            else  # :spontaneous
                if name == tr.from
                    rhs -= rate * singles[tr.from]
                elseif name == tr.to
                    rhs += rate * singles[tr.from]
                end
            end
        end

        push!(eqs, D(singles[name]) ~ rhs)
    end

    return eqs
end

# The flux counted by the `cumulative` accumulator: the transitions `infections` (from a
# non-infected compartment into an infected one, `_incidence_transitions`), τ[XZ] per infector Z
# of an infection, a[X] for a spontaneous transition (importation S → E).
function _incidence(model, singles, pairs, rates, infections)
    flux = Num(0)
    for k in infections
        tr = model.transitions[k]
        if tr.type === :infection
            for Z in infectors(model, tr)
                flux += rates[k] * _get_pair(pairs, tr.from, Z)
            end
        else
            flux += rates[k] * singles[tr.from]
        end
    end
    return flux
end

# ─── Pair equations ───────────────────────────────────────────────────────────
# Canonical Keeling-style pair approximation under the "mixed" convention.
#
# Convention (Keeling/Eames "mixed"):
#   - Cross pair [XY] for X ≠ Y: counts each undirected XY edge ONCE.
#   - Self  pair [XX]:           counts each undirected XX edge TWICE
#                                (i.e., directed-pair count).
#   This convention makes the moment closure   [XYZ] ≈ κ·[XY][YZ]/[Y]
#   exact in the absence of clustering and yields the standard
#       d[SS] = -2τ[SSI],  d[II] = 2τ([ISI] + [SI]) - 2γ[II], …
#   It is the ordered-pair convention of DESIGN §E.2: [XY] = number of ordered adjacent pairs
#   (x, y) with x ∈ X, y ∈ Y, so no rescaling is needed against the edge-based model.
#
# Per-event accounting rule:
#   For every event with rate R that destroys one undirected source edge and
#   creates one undirected target edge:
#       d[source] += -factor_src · R
#       d[target] += +factor_tgt · R
#   where factor = 2 if the pair is a self-pair (XX), else 1.
#
# Event-rate formulas (in the same MIXED convention):
#   External infection X→Y via infectious Z, source pair (X, other):
#       R = τ · κ · [XZ] · [X, other] / [X]   ( = τ · [other X Z], `_infection_triple`, infector last )
#   Direct infection X→Y along pair (X, Z), X ≠ Z:
#       R = τ · [X, Z]
#   Spontaneous X→Y on source pair (X, other):
#       R = γ · [X, other]
#   Z ranges over the infectors of the transition (Transition.via; all infectious if empty).
#
# Verified for SIR against both PairwiseInvariantRegion.lean (Results 130–141)
# and node-level conservation Σ_Y d[XY]/dt = k · d[X]/dt for k-regular networks.

# Self-pair convention factor (2 for [XX], 1 for [XY] X≠Y).
_pair_factor(A::Symbol, B::Symbol) = (A == B) ? 2 : 1

function _build_pair_equations(model, network, closure,
                                singles, pairs, rates, D, names)
    K = length(names)
    # Public pair-state ordering (preserved): (names[i], names[j]) for i ≤ j.
    pair_states = [(names[i], names[j]) for i in 1:K for j in i:K]
    order = Dict(name => idx for (idx, name) in enumerate(names))

    # `normalize_pair` returns the canonical (i ≤ j) public key for two
    # compartment names — preserving the public pair-key ordering used
    # throughout the package.
    normalize_pair(a::Symbol, b::Symbol) =
        order[a] <= order[b] ? (a, b) : (b, a)

    rhs = Dict(state => Num(0) for state in pair_states)

    function add_ext_event!(X::Symbol, Y::Symbol, Z::Symbol, other::Symbol,
                             rate_param)
        # Event: endpoint X of pair (X, other) gets infected externally
        # via an infectious Z-neighbor. Source pair: (X, other);
        # target pair: (Y, other).
        triple = _infection_triple(Z, X, other, pairs, singles, network, closure)
        R = rate_param * triple
        f_src = _pair_factor(X, other)
        f_tgt = _pair_factor(Y, other)
        rhs[normalize_pair(X, other)] -= f_src * R
        rhs[normalize_pair(Y, other)] += f_tgt * R
    end

    function add_direct_event!(X::Symbol, Y::Symbol, Z::Symbol, rate_param)
        # Direct event: pair (X, Z) transmits along itself. Caller ensures
        # X != Z, so source pair is always a cross-pair (f_src = 1).
        pair_var = _get_pair(pairs, X, Z)
        R = rate_param * pair_var
        f_tgt = _pair_factor(Y, Z)
        rhs[normalize_pair(X, Z)] -= R
        rhs[normalize_pair(Y, Z)] += f_tgt * R
    end

    function add_spontaneous_event!(X::Symbol, Y::Symbol, other::Symbol,
                                     rate_param)
        # Event: endpoint X of pair (X, other) transitions spontaneously.
        pair_var = _get_pair(pairs, X, other)
        R = rate_param * pair_var
        f_src = _pair_factor(X, other)
        f_tgt = _pair_factor(Y, other)
        rhs[normalize_pair(X, other)] -= f_src * R
        rhs[normalize_pair(Y, other)] += f_tgt * R
    end

    for (k, tr) in enumerate(model.transitions)
        rate = rates[k]

        if tr.type == :infection
            X = tr.from
            Y = tr.to
            for Z in infectors(model, tr)
                # External (triple) events: for each possible "other" compartment.
                for other in names
                    add_ext_event!(X, Y, Z, other, rate)
                end
                # Direct event along the (X, Z) pair itself, when X != Z.
                if X != Z
                    add_direct_event!(X, Y, Z, rate)
                end
            end
        else  # :spontaneous
            X = tr.from
            Y = tr.to
            for other in names
                add_spontaneous_event!(X, Y, other, rate)
            end
        end
    end

    eqs = Equation[]
    for state in pair_states
        push!(eqs, D(_get_pair(pairs, state...)) ~ Symbolics.simplify(rhs[state]))
    end

    return eqs
end

# ─── Initial conditions ─────────────────────────────────────────────────────

# The compartment that holds every unseeded node: the first susceptible compartment.
function _background(model::CompartmentalModel)
    isempty(model.susceptible_compartments) && return nothing
    return first(model.susceptible_compartments)
end

"""
    _entry_state(model::CompartmentalModel, background::Symbol) -> Symbol

The compartment a seeded node starts in by default: the unique target of the infection
transitions out of the background compartment (I for SIR, E for SEIR, I₁ for staged or
reinfection-counted models). Throws an `ArgumentError` asking for an explicit `initial` if
there are several (two strains) or none.
"""
function _entry_state(model::CompartmentalModel, background::Symbol)
    targets = unique([tr.to for tr in model.transitions
                      if tr.type === :infection && tr.from === background])
    isempty(targets) && throw(ArgumentError(
        "model :$(model.name) has no infection transition out of the background compartment " *
        "$(background), so there is no entry state to seed; pass an explicit `initial`, " *
        "e.g. SeedFraction(:X => ρ)"))
    length(targets) == 1 || throw(ArgumentError(
        "model :$(model.name) has several entry states from $(background) " *
        "($(join(targets, ", "))), so there is no default seed; pass an explicit `initial`, " *
        "e.g. SeedFraction(" * join(("$(repr(X)) => ρ_$X" for X in targets), ", ") * ")"))
    return only(targets)
end

"""
    _resolve_seed(model::CompartmentalModel, initial, seed_fraction, seed_state, N;
                  background = _background(model)) -> (x::Dict{Symbol,Float64}, background)

The initial fraction x_X of every compartment of `model`: from `initial` (a `SeedSpec`; the
background compartment receives the unseeded nodes), or else a fraction `seed_fraction` in
the entry state (`seed_state = :entry`) or the first infectious compartment
(`seed_state = :first_infectious`, the 0.1 default) and the rest in `background`.
"""
function _resolve_seed(model::CompartmentalModel, initial, seed_fraction::Real,
                       seed_state::Symbol, N::Real; background = _background(model))
    names = model.compartment_names
    if initial === nothing
        (isfinite(seed_fraction) && 0 <= seed_fraction <= 1) || throw(ArgumentError(
            "seed_fraction must lie in [0, 1]; got $(seed_fraction)"))
        background === nothing && throw(ArgumentError(
            "model :$(model.name) has no infection transition, so no background " *
            "compartment; pass an explicit `initial` naming the fraction of every compartment"))
        X = if seed_state === :entry
            _entry_state(model, background)
        elseif seed_state === :first_infectious
            first(model.infectious_compartments)
        else
            throw(ArgumentError("seed_state must be :entry or :first_infectious; got " *
                                repr(seed_state)))
        end
        X === background && throw(ArgumentError(
            "the seeded compartment $(X) is the background compartment; pass an explicit " *
            "`initial`"))
        return Dict{Symbol,Float64}(background => 1 - seed_fraction, X => seed_fraction),
               background
    end
    bg = initial.default === nothing ? background : initial.default
    fr = seed_fractions(initial; N = _seed_population(initial, N), background = bg)
    x = Dict{Symbol,Float64}()
    for (X, ρ) in fr
        X in names || throw(ArgumentError(
            "the initial condition names $(X), which is not a compartment of model " *
            ":$(model.name) (compartments: $(join(names, ", ")))"))
        x[X] = get(x, X, 0.0) + ρ
    end
    total = sum(values(x); init = 0.0)
    abs(total - 1) <= 1e-9 || throw(ArgumentError(
        "the initial fractions sum to $(total), not 1" *
        (bg === nothing ? "; the model has no unique background compartment, so name the " *
                          "fraction of every compartment or set `default` in the SeedSpec" : "")))
    return x, bg
end

_seed_population(::SeedFraction, N::Real) = nothing
function _seed_population(seed::SeedSpec, N::Real)
    (isinteger(N) && N >= 1) || throw(ArgumentError(
        "$(nameof(typeof(seed))) counts nodes, so the population scale N must be a whole " *
        "number of nodes (got N = $(N)); use SeedFraction for fractions"))
    return Int(N)
end

# ─── Infection status (DESIGN §J.8) ──────────────────────────────────────────

"""
    _infected_compartments(model::CompartmentalModel) -> Vector{Symbol}
    _infected_compartments(cm::ContactModel) -> Vector{Symbol}

The compartments counted as *infected*, in model order (DESIGN §J.8): infection status is
structural and decided from the typing, as for NetworkOutbreaks' `final_size` and the
edge-based `:cumulative` accumulator.

1. The susceptible classes Σ are the Sus-typed species of the contact model
   (`susceptible_species(cm)`; for a `CompartmentalModel`, the sources of infection transitions
   that are never the target of one, which is how `ContactModel` infers Σ).
2. The *infection chain* is the set of species from which an infector can be reached through
   node transitions that do not start in Σ (NetworkEpiCore's rule for reinfection counting).
   An *infection* is a contact from a class in Σ into the chain. A contact whose product is not
   in the chain (tracing `S + D → Q + D`) is not an infection, and neither is a contact whose
   recipient is not in Σ (superinfection `I₁ + I₂ → I₁₂ + I₂`, quarantine `E + I → E_q + I`).
3. Every reaction other than an infection preserves infection status. A compartment is
   infected if it is infectious, or if it lies on a status-preserving path from the product of
   an infection to an infectious compartment (the forward and backward closures of
   NetworkOutbreaks' `_infected_mask`). No class in Σ is infected.

So E and I of SEIR are infected, E and E_q of the quarantine model are too, and S, R, a
vaccinated V (`S → V`), a traced Q and the absorbing `:removed` compartment are not. In the
superinfection model the recipient I₁ is infected (not susceptible), and I₁₂, which never
infects, is not. With the inferred Σ the result is the same on a `ContactModel` and on its
lowering `CompartmentalModel(cm)`. On a `ContactModel` this is `NetworkEpiCore.infected_species(cm)`
(the method delegates, so every back end shares one rule); the `CompartmentalModel` method applies
the same rule to the lowered transitions.
"""
function _infected_compartments(model::CompartmentalModel)
    targets = Set{Symbol}(tr.to for tr in model.transitions if tr.type === :infection)
    Σ = [X for X in model.susceptible_compartments if !(X in targets)]
    cs = [(tr.from, tr.to) for tr in model.transitions
          if tr.type === :infection && !isempty(infectors(model, tr))]
    ts = [(tr.from, tr.to) for tr in model.transitions if tr.type === :spontaneous]
    return _structural_infected(model.compartment_names, model.infectious_compartments, Σ, cs, ts)
end

# The §J.8 rule on the contacts `cs` and node transitions `ts`, as (from, to) pairs.
function _structural_infected(names, infectious, susceptible, cs, ts)
    Σ = Set{Symbol}(susceptible)
    inf = Set{Symbol}(infectious)
    chain = _backward_closure!(copy(inf), ts, Σ)
    infection = [s in Σ && !(x in Σ) && x in chain for (s, x) in cs]
    preserving = vcat([c for (c, f) in zip(cs, infection) if !f], ts)
    fwd = _forward_closure!(Set{Symbol}(x for ((_, x), f) in zip(cs, infection) if f),
                            preserving, Σ)
    bwd = _backward_closure!(copy(inf), preserving, Σ)
    return [X for X in names if !(X in Σ) && (X in inf || (X in fwd && X in bwd))]
end

# Grow `set` backwards (from `to` to `from`) along the reactions `rs`, never adding a
# compartment in `barrier`.
function _backward_closure!(set::Set{Symbol}, rs, barrier::Set{Symbol})
    changed = true
    while changed
        changed = false
        for (from, to) in rs
            (to in set && !(from in set) && !(from in barrier)) || continue
            push!(set, from)
            changed = true
        end
    end
    return set
end

# Grow `set` forwards along the reactions `rs`, never entering a compartment in `barrier`.
function _forward_closure!(set::Set{Symbol}, rs, barrier::Set{Symbol})
    changed = true
    while changed
        changed = false
        for (from, to) in rs
            (from in set && !(to in set) && !(to in barrier)) || continue
            push!(set, to)
            changed = true
        end
    end
    return set
end

# Indices of the transitions that the `cumulative` accumulator counts: from a compartment that
# is not infected into an infected one (contacts and node transitions alike, as NetworkOutbreaks
# counts entries into infection).
_incidence_transitions(model::CompartmentalModel, infected) =
    [k for (k, tr) in enumerate(model.transitions) if !(tr.from in infected) && tr.to in infected]

# The seeds that the cumulative accumulator counts at t = 0: the seeded infected compartments.
_seeded_infections(x, background, infected) =
    sum((ρ for (X, ρ) in x if X !== background && X in infected); init = 0.0)

# Singles N·x_X; pairs [AB] = ⟨k⟩·[A][B]/N (the 0.1 random-mixing formula, kept verbatim);
# the accumulator N·cum0.
function _pairwise_u0(model, network, singles, pairs, N, x; cumulative = nothing, cum0 = 0.0)
    names = model.compartment_names
    n = mean_degree(network)
    u0 = Dict{Any,Float64}()
    for name in names
        u0[singles[name]] = N * get(x, name, 0.0)
    end

    # Pairs: [AB] ≈ n · [A] · [B] / N (random mixing at t=0)
    K = length(names)
    for i in 1:K, j in i:K
        a, b = names[i], names[j]
        na = u0[singles[a]]
        nb = u0[singles[b]]
        pair_val = n * na * nb / N
        if a == b
            pair_val = n * na * (na / N)  # [AA] = n·[A]²/N
        end
        u0[_get_pair(pairs, a, b)] = pair_val
    end
    cumulative === nothing || (u0[cumulative] = N * cum0)

    return u0
end

"""
    default_initial_conditions(psys::PairwiseSystem; initial = nothing, seed_fraction = nothing,
                               seed_state = nothing)

The initial condition of `psys`: the stored `psys.u0` (the same object) when no keyword is
given, otherwise the π^PW image of the seeding `initial` (a `SeedSpec`, e.g.
`SeedFraction(:E => 0.01)`) or of a fraction `seed_fraction` in `seed_state` (see
[`generate_pairwise`](@ref)), on the population scale of the system. `seed_fraction` and
`seed_state` default to those the system was built with (so a system built with
`seed_state = :first_infectious` reseeds I, not E). The `cumulative` accumulator, if present,
starts at the seeds in infected compartments (DESIGN §J.8).
"""
function default_initial_conditions(psys::PairwiseSystem; initial = nothing,
                                    seed_fraction = nothing,
                                    seed_state::Union{Nothing,Symbol} = nothing)
    initial === nothing && seed_fraction === nothing && seed_state === nothing && return psys.u0
    md = psys.metadata
    haskey(md, :resolve) || throw(ArgumentError(
        "this PairwiseSystem was built without a seeding rule; set its initial condition " *
        "directly"))
    state = something(seed_state, get(md, :seed_state, :entry))
    x, bg = md[:resolve](initial, something(seed_fraction, md[:default_fraction]), state)
    model = _compartmental(psys)
    infected = get(() -> _infected_compartments(model), md, :infected)
    return _pairwise_u0(model, psys.network, psys.singles, psys.pairs, md[:N], x;
                        cumulative = md[:cumulative],
                        cum0 = _seeded_infections(x, bg, infected))
end

# ─── Convenience solver ──────────────────────────────────────────────────────

"""
    solve_pairwise(psys::PairwiseSystem, params::Dict;
                   solver=nothing, reltol=1e-8, abstol=1e-10, u0=psys.u0, tspan=psys.tspan,
                   kwargs...)

Solve a PairwiseSystem with given parameter values.

The default tolerances (`reltol=1e-8`, `abstol=1e-10`) are tighter than
`OrdinaryDiffEq`'s defaults (`1e-3`, `1e-6`). Looser tolerances are
unsuitable for moment-closure systems with many tightly-coupled small
components (e.g. reinfection-counting lifts at L≥2), where they can let
components drift negative and the integrator report `Unstable`. Pass
`reltol`/`abstol` explicitly to override. `u0` and `tspan` override the stored initial
condition and time span.
"""
function solve_pairwise(psys::PairwiseSystem, param_values::AbstractDict;
                        solver = nothing,
                        reltol = 1e-8, abstol = 1e-10,
                        u0::AbstractDict = psys.u0,
                        tspan::Tuple{Real,Real} = psys.tspan,
                        kwargs...)
    p = Dict{Any,Float64}()
    required_params = collect(ModelingToolkit.parameters(psys.system))
    required_by_symbol = Dict(Symbol(sp) => sp for sp in required_params)
    unknown = Symbol[]

    # Map Symbol keys to symbolic parameter variables
    for (k, v) in param_values
        if k isa Symbol
            if haskey(required_by_symbol, k)
                p[required_by_symbol[k]] = Float64(v)
            else
                push!(unknown, k)
            end
        else
            k in required_params || throw(ArgumentError("Unknown parameter key: $(k)"))
            p[k] = Float64(v)
        end
    end

    isempty(unknown) || throw(ArgumentError(
        "Unknown parameter names: $(join(string.(sort(unique(unknown))), ", "))"))
    missing = [sp for sp in required_params if !haskey(p, sp)]
    isempty(missing) || throw(ArgumentError(
        "Missing parameter values for: $(join(string.(sort(Symbol.(missing))), ", "))"))

    tsp = (Float64(tspan[1]), Float64(tspan[2]))
    prob = ModelingToolkit.ODEProblem(psys.system, merge(u0, p), tsp)
    if isnothing(solver)
        return OrdinaryDiffEqDefault.solve(prob; reltol = reltol, abstol = abstol, kwargs...)
    else
        return OrdinaryDiffEqDefault.solve(prob, solver; reltol = reltol, abstol = abstol, kwargs...)
    end
end

"""
    solve_epidemic(psys::PairwiseSystem; p = nothing, initial = nothing, tspan = psys.tspan,
                   saveat = nothing, kwargs...)
    solve_epidemic(psys::PairwiseSystem, params::AbstractDict; kwargs...)

Solve a pairwise system. The keyword form takes the parameter values `p` (a `Dict` keyed by
name; the values stored in `psys.params` and the model's parameter defaults fill the rest), an
optional seeding `initial` (see [`default_initial_conditions`](@ref)), the time span and the
save grid; the positional form is [`solve_pairwise`](@ref). For a scenario,
`solve_epidemic(psys, sc::Scenario)` takes all of these from `sc`.
"""
solve_epidemic(psys::PairwiseSystem, param_values::AbstractDict; kwargs...) =
    solve_pairwise(psys, param_values; kwargs...)

function solve_epidemic(psys::PairwiseSystem; p = nothing, initial = nothing,
                        tspan::Tuple{Real,Real} = psys.tspan, saveat = nothing, kwargs...)
    u0 = initial === nothing ? psys.u0 : default_initial_conditions(psys; initial)
    vals = _parameter_values(psys, p)
    saveat === nothing && return solve_pairwise(psys, vals; u0, tspan, kwargs...)
    return solve_pairwise(psys, vals; u0, tspan, saveat, kwargs...)
end

# Parameter values by name: the model's defaults, then `psys.params`, then `p`, restricted to
# the parameters of the system. A name in `psys.params` or `p` that is neither a parameter of
# the system nor a rate parameter of the model is an error (a misspelt key would otherwise be
# ignored, and a default used silently); defaults of the model that the system does not use are
# dropped.
function _parameter_values(psys::PairwiseSystem, p)
    vals = Dict{Symbol,Float64}()
    cm = get(psys.metadata, :contact_model, nothing)
    cm === nothing || merge!(vals, parameter_defaults(cm))
    needed = Set(Symbol(sp) for sp in ModelingToolkit.parameters(psys.system))
    known = union(needed, Set{Symbol}(keys(get(psys.metadata, :parameters, Dict{Symbol,Any}()))))
    cm === nothing || union!(known, (_parameter_symbol(q) for q in rate_parameters(cm)))
    unknown = Symbol[]
    for given in (psys.params, something(p, Dict{Symbol,Float64}())), (k, v) in given
        name = Symbol(k)
        name in known || push!(unknown, name)
        vals[name] = Float64(v)
    end
    isempty(unknown) || throw(ArgumentError(
        "Unknown parameter names: $(join(string.(sort!(unique!(unknown))), ", ")); the " *
        "parameters of this system are $(join(string.(sort!(collect(needed))), ", "))"))
    return Dict{Symbol,Float64}(k => v for (k, v) in vals if k in needed)
end

"""
    node_variables(psys::PairwiseSystem)

Return the Dict of node-level symbolic variables.
"""
node_variables(psys::PairwiseSystem) = psys.singles

"""
    pair_variables(psys::PairwiseSystem)

Return the Dict of pair-level symbolic variables.
"""
pair_variables(psys::PairwiseSystem) = psys.pairs

"""
    compartment(psys::PairwiseSystem, sol, state::Symbol)

Look up the time series for the node-level compartment `state` (e.g. `:S`, `:I`)
in the ModelingToolkit solution `sol`. Throws `ArgumentError` if `state` is not
a known node-level compartment.
"""
function compartment(psys::PairwiseSystem, sol, state::Symbol)
    haskey(psys.singles, state) ||
        throw(ArgumentError("unknown node-level compartment: $state; available: $(collect(keys(psys.singles)))"))
    return sol[psys.singles[state]]
end

"""
    compartments(psys::PairwiseSystem, sol, states::AbstractVector{Symbol}) -> Dict

The time series of several node-level compartments, keyed by name (see
[`compartment`](@ref)).
"""
compartments(psys::PairwiseSystem, sol, states::AbstractVector{Symbol}) =
    Dict(state => compartment(psys, sol, state) for state in states)

"""
    population_fraction(psys::PairwiseSystem, sol, state::Symbol; N = nothing)

Return the time series of the fraction of the population in compartment `state`: the single
[state] divided by `N`, which defaults to the population scale the system was built with (1 by
default, so the result then equals [`compartment`](@ref)). If `state` is not a compartment but
the base of refined compartments of the model (reinfection counts `I_1, I_2, …`, Erlang stages,
strata, read from the `ContactModel` labels), their sum is returned.
"""
function population_fraction(psys::PairwiseSystem, sol, state::Symbol; N = nothing)
    scale = N === nothing ? get(psys.metadata, :N, 1.0) : N
    haskey(psys.singles, state) && return compartment(psys, sol, state) ./ scale
    members = _refinements(psys, state)
    isempty(members) && throw(ArgumentError(
        "unknown node-level compartment: $state; available: $(collect(keys(psys.singles)))"))
    return sum(compartment(psys, sol, X) for X in members) ./ scale
end

# The compartments of the model that refine `base` (by their NetworkEpiCore labels).
function _refinements(psys::PairwiseSystem, base::Symbol)
    cm = get(psys.metadata, :contact_model, nothing)
    cm === nothing && return Symbol[]
    labels = species_labels(cm)
    return [X for X in keys(psys.singles)
            if haskey(labels, X) && (labels[X].base === base || _uncounted(cm, X) === base)]
end

# The species of a reinfection-counted model before counting (`:I_2` ↦ `:I`), else X.
function _uncounted(cm::ContactModel, X::Symbol)
    X in species_names(cm) || return X
    return only(keys(reinfection_totals(cm, Dict(X => 0.0))))
end

"""
    model_curves(psys::PairwiseSystem, sol; t = sol.t, label = "pairwise") -> ModelCurves

The observables of a solved pairwise system on the time grid `t` as population fractions: every
compartment, `:infectious` (the sum of the infectious compartments), `:cumulative` (the
cumulative-incidence accumulator, when the system has one: `node_based` adds it. It counts the
seeds in infected compartments and every entry into an infected compartment, with infection
status defined structurally as in NetworkOutbreaks' `final_size` (DESIGN §J.8), so seeded R or
V nodes are not counted; for models without an arrow into a susceptible class it is the fraction
ever infected), and for a reinfection-counted model the totals over infection counts (`:S`, `:I`, …).
The result is a `NetworkEpiCore.ModelCurves` with representation `:pairwise` that `compare` and
the plot recipes accept.
"""
function model_curves(psys::PairwiseSystem, sol; t = sol.t, label::AbstractString = "pairwise")
    tt = collect(Float64, t)
    scale = get(psys.metadata, :N, 1.0)
    at(v) = _values_at(sol, v, tt) ./ scale
    vals = Dict{Symbol,Vector{Float64}}(X => at(v) for (X, v) in psys.singles)
    model = _compartmental(psys)
    vals[:infectious] = sum(vals[X] for X in model.infectious_compartments)
    cum = get(psys.metadata, :cumulative, nothing)
    cum === nothing || (vals[:cumulative] = at(cum))
    cm = get(psys.metadata, :contact_model, nothing)
    if cm !== nothing && any(l -> l.count > 0, values(species_labels(cm)))
        for (X, v) in reinfection_totals(cm, Dict(X => vals[X] for X in species_names(cm)))
            haskey(vals, X) || (vals[X] = v)
        end
    end
    return ModelCurves(tt, vals; label, representation = get(psys.metadata, :representation,
                                                               :pairwise),
                       metadata = Dict{Symbol,Any}(:closure => string(psys.closure),
                                                   :network => string(psys.network)))
end

function _values_at(sol, v, tt::Vector{Float64})
    length(sol.t) == length(tt) && sol.t == tt && return collect(Float64, sol[v])
    return Float64[sol(ti; idxs = v) for ti in tt]
end

"""
    symbolic_ode(psys::PairwiseSystem) -> SymbolicODE

The uncompiled pairwise vector field of `psys` (singles, pairs and, if present, the cumulative
accumulator) as a `NetworkEpiCore.SymbolicODE`, for `vector_fields_equal` and `verify`.
"""
function symbolic_ode(psys::PairwiseSystem)
    md = psys.metadata
    haskey(md, :equations) || throw(ArgumentError(
        "this PairwiseSystem does not store its uncompiled equations"))
    eqs = md[:equations]
    return SymbolicODE(md[:name]; states = md[:states], rhs = Any[eq.rhs for eq in eqs],
                       parameters = Any[md[:parameters][n] for n in sort!(collect(keys(md[:parameters])))])
end
