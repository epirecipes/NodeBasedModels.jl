# pairwise.jl — Automatic pairwise ODE system generation
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

"""
    PairwiseSystem

Container for the generated pairwise ODE system.

Fields:
- `system` — compiled ModelingToolkit system
- `u0` — initial condition Dict
- `tspan` — time span
- `params` — parameter Dict
- `model` — original CompartmentalModel
- `network` — original NetworkStructure
- `closure` — ClosureMethod used
- `singles` — Dict of single variable symbols
- `pairs` — Dict of pair variable symbols
"""
struct PairwiseSystem
    system::Any     # ODESystem
    u0::Dict
    tspan::Tuple{Float64,Float64}
    params::Dict
    model::CompartmentalModel
    network::NetworkStructure
    closure::ClosureMethod
    singles::Dict{Symbol,Any}
    pairs::Dict{Tuple{Symbol,Symbol},Any}
end

function _validate_pairwise_closure_support(network::NetworkStructure,
                                            closure::ClosureMethod)
    closure isa EamesClosure && throw(ArgumentError(
        "EamesClosure is not supported by generate_pairwise because NodeBasedModels.jl does not encode the separate regular and random contact parameters required by the Eames hybrid closure."))
    closure isa KirkwoodClosure && throw(ArgumentError(
        "KirkwoodClosure is only used by generate_pair_based on explicit graphs, not by population-level generate_pairwise."))
    if network isa HeterogeneousNetwork &&
       !(closure isa BernoulliClosure || closure isa KeelingClosure)
        throw(ArgumentError(
            "Only BernoulliClosure and KeelingClosure are implemented for HeterogeneousNetwork."))
    end
end

"""
    generate_pairwise(model, network, closure; tspan=(0.0,100.0), N=1.0, ε=1e-3, seed_fraction=ε)

Generate a pairwise ODE system from a compartmental model, network structure,
and triple closure approximation.

Returns a `PairwiseSystem` containing the MTK ODESystem and initial conditions.

Closure support matrix:

| Closure | HomogeneousNetwork | HeterogeneousNetwork |
|---|:---:|:---:|
| `BernoulliClosure` | ✓ | ✓ |
| `KeelingClosure` | ✓ | ✓ |
| `BarnardClosure` | ✓ | ✗ |
| `PowerClosure` | ✓ | ✗ |
| `EamesClosure` | ✗ | ✗ |

`KirkwoodClosure` is reserved for `generate_pair_based` on explicit graphs.

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
                            ε::Float64 = 1e-3,
                            seed_fraction::Float64 = ε)
    _validate_pairwise_closure_support(network, closure)
    names = model.compartment_names
    K = length(names)

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
    rate_names = unique([t.rate for t in model.transitions])
    param_syms = Dict{Symbol,Any}()
    for rname in rate_names
        p = only(@parameters $(rname))
        param_syms[rname] = p
    end

    # ─── Build node equations ─────────────────────────────────────────────
    node_eqs = _build_node_equations(model, singles, pairs, param_syms, D)

    # ─── Build pair equations ─────────────────────────────────────────────
    pair_eqs = _build_pair_equations(model, network, closure,
                                      singles, pairs, param_syms, D, names)

    all_eqs = vcat(node_eqs, pair_eqs)

    # ─── Create ODESystem ─────────────────────────────────────────────────
    sys = System(all_eqs, t; name = Symbol(:pairwise_, model.name))
    sys = mtkcompile(sys)

    # ─── Initial conditions ───────────────────────────────────────────────
    u0, p = _default_ic(model, network, singles, pairs, param_syms, N; seed_fraction = seed_fraction)

    return PairwiseSystem(sys, u0, (Float64(tspan[1]), Float64(tspan[2])),
                          p, model, network, closure, singles, pairs)
end

# ─── Node equations ───────────────────────────────────────────────────────────
# d[A]/dt = Σ (inflows to A) - Σ (outflows from A)
#   infection: S → I contributes -τ[SI] to d[S]/dt, +τ[SI] to d[I]/dt
#   spontaneous: I → R contributes -γ[I] to d[I]/dt, +γ[I] to d[R]/dt

function _build_node_equations(model, singles, pairs, params, D)
    eqs = Equation[]
    names = model.compartment_names

    for name in names
        rhs = Num(0)

        for tr in model.transitions
            rate = params[tr.rate]

            if tr.type == :infection
                # Infection: from → to at rate τ per [from, infectious] edge
                # This creates -τ·Σ_j [from·j] for each infectious j
                for inf_comp in model.infectious_compartments
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
#       R = τ · κ · [XZ] · [X, other] / [X]   ( = τ · triple_closure(Z, X, other) )
#   Direct infection X→Y along pair (X, Z), X ≠ Z:
#       R = τ · [X, Z]
#   Spontaneous X→Y on source pair (X, other):
#       R = γ · [X, other]
#
# Verified for SIR against both PairwiseInvariantRegion.lean (Results 130–141)
# and node-level conservation Σ_Y d[XY]/dt = k · d[X]/dt for k-regular networks.

# Self-pair convention factor (2 for [XX], 1 for [XY] X≠Y).
_pair_factor(A::Symbol, B::Symbol) = (A == B) ? 2 : 1

function _build_pair_equations(model, network, closure,
                                singles, pairs, params, D, names)
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
        triple = triple_closure(Z, X, other, pairs, singles, network, closure)
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

    for tr in model.transitions
        rate = params[tr.rate]

        if tr.type == :infection
            X = tr.from
            Y = tr.to
            for Z in model.infectious_compartments
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

# ─── Default initial conditions ──────────────────────────────────────────────

function _default_ic(model, network, singles, pairs, params, N; seed_fraction::Float64 = 1e-3)
    names = model.compartment_names
    n = mean_degree(network)
    ε = seed_fraction

    u0 = Dict{Any,Float64}()
    p = Dict{Any,Float64}()

    # Singles: almost all susceptible, tiny fraction infected
    first_susc = model.susceptible_compartments[1]
    first_inf = model.infectious_compartments[1]

    for name in names
        if name == first_susc
            u0[singles[name]] = N * (1 - ε)
        elseif name == first_inf
            u0[singles[name]] = N * ε
        else
            u0[singles[name]] = 0.0
        end
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

    return u0, p
end

# ─── Convenience solver ──────────────────────────────────────────────────────

"""
    solve_pairwise(psys::PairwiseSystem, params::Dict;
                   solver=nothing, reltol=1e-8, abstol=1e-10, kwargs...)

Solve a PairwiseSystem with given parameter values.

The default tolerances (`reltol=1e-8`, `abstol=1e-10`) are tighter than
`OrdinaryDiffEq`'s defaults (`1e-3`, `1e-6`). Looser tolerances are
unsuitable for moment-closure systems with many tightly-coupled small
components (e.g. reinfection-counting lifts at L≥2), where they can let
components drift negative and the integrator report `Unstable`. Pass
`reltol`/`abstol` explicitly to override.
"""
function solve_pairwise(psys::PairwiseSystem, param_values::Dict;
                        solver = nothing,
                        reltol = 1e-8, abstol = 1e-10,
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

    prob = ModelingToolkit.ODEProblem(psys.system, merge(psys.u0, p), psys.tspan)
    if isnothing(solver)
        return OrdinaryDiffEqDefault.solve(prob; reltol = reltol, abstol = abstol, kwargs...)
    else
        return OrdinaryDiffEqDefault.solve(prob, solver; reltol = reltol, abstol = abstol, kwargs...)
    end
end

solve_epidemic(psys::PairwiseSystem, param_values::Dict; kwargs...) =
    solve_pairwise(psys, param_values; kwargs...)

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
    default_initial_conditions(psys::PairwiseSystem)

Return the default initial conditions.
"""
default_initial_conditions(psys::PairwiseSystem) = psys.u0

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
    population_fraction(psys::PairwiseSystem, sol, state::Symbol; N=nothing)

Return the time series of the fractional population in compartment `state`.
If the system was set up with single-totals summing to `N` (the default), pass
`N = sum_of_initial_singles` to normalise; otherwise this returns the same value
as [`compartment`](@ref).
"""
function population_fraction(psys::PairwiseSystem, sol, state::Symbol; N = nothing)
    counts = compartment(psys, sol, state)
    isnothing(N) && return counts
    return counts ./ N
end
