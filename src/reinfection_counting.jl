#=
reinfection_counting.jl

Lifts a `CompartmentalModel` into an expanded model whose nodes carry an
extra integer "infection count" `p ∈ {0, …, L}`. Each infection event
increments p (capped at L); spontaneous events preserve p.

This implements **Approximation 1** from
Keeling, House, Cooper & Pellis (2016)
*Systematic Approximations to SIS Dynamics on Networks*,
PLoS Comp Biol 12(12): e1005296, doi:10.1371/journal.pcbi.1005296

The lifted model can be fed to `generate_pairwise` unchanged: the existing
generator already sums force-of-infection contributions over all
infectious compartments (every `I_p` is marked infectious in the lifted
model), so no closure changes are needed.

Reachability pruning: the unphysical `I_0` (an infected state with zero
infections so far) is excluded whenever `L ≥ 1`. At `L = 0` the cap
collapses everything to `p = 0` and the lifted model is isomorphic to the
base model.
=#

"""
    with_reinfection_counting(model::CompartmentalModel, L::Integer)

Return a lifted `CompartmentalModel` in which every base compartment `X`
becomes a family of compartments `X_p` for `p ∈ {p_min(X), …, L}`, where
`p` is the number of infections the node has experienced so far (capped
at `L`).

- `L::Integer ≥ 0` is the maximum tracked infection count. `L = 0` is a
  no-op (returns a model isomorphic to `model` with names suffixed by
  `_0`). Choose `L = 0` to recover the base pairwise dynamics.
- An infection transition `from → to` lifts to
  `from_p → to_{min(p+1, L)}` for each reachable `p`.
- A spontaneous transition `from → to` lifts to `from_p → to_p` for each
  reachable `p` (when `to`'s reachable set permits it).
- All `I_p` (for any base infectious `I`) are marked infectious in the
  lifted model so the existing pair-equation generator picks them up as
  catalysts.
- Parameter symbols are preserved, so a parameter dict that worked for the
  base model still works for the lifted model.

# Reachability rule
For each base compartment `X`, the minimum reachable infection-count
`p_min(X)` is `0` if `X` is reachable from a susceptible via only
spontaneous transitions (susceptibles themselves, plus anything
spontaneously reachable from a susceptible without ever being infected),
otherwise `min(1, L)`. This prunes the unphysical `I_0` whenever
`L ≥ 1`.

# Example (SIS with up to 4 infections tracked)
```julia
base   = sis_model()
lifted = with_reinfection_counting(base, 4)
psys   = generate_pairwise(lifted, regular_network(3; n_nodes = 1_000),
                           BernoulliClosure(); tspan = (0.0, 80.0))
sol    = solve_pairwise(psys, Dict(:τ => 0.5, :γ => 1.0))
totals = reinfection_totals(psys, sol)         # Dict(:S => …, :I => …)
```
"""
function with_reinfection_counting(model::CompartmentalModel, L::Integer)
    L >= 0 || throw(ArgumentError("L must be non-negative"))

    p_min_map = _compute_p_min(model, L)

    new_compartments = Compartment[]
    name_to_infectious = Dict{Symbol, Bool}(c.name => c.infectious for c in model.compartments)
    new_names = Symbol[]
    for c in model.compartments
        for p in p_min_map[c.name]:L
            sym = _lifted_name(c.name, p)
            push!(new_compartments, Compartment(sym; infectious = name_to_infectious[c.name]))
            push!(new_names, sym)
        end
    end

    new_transitions = Transition[]
    for t in model.transitions
        p_from_lo = p_min_map[t.from]
        for p in p_from_lo:L
            if t.type == :infection
                p_target = min(p + 1, L)
            elseif t.type == :spontaneous
                p_target = p
            else
                throw(ArgumentError("Unsupported transition type for reinfection counting: $(t.type)"))
            end

            p_target >= p_min_map[t.to] || continue

            from_sym = _lifted_name(t.from, p)
            to_sym   = _lifted_name(t.to,   p_target)

            push!(new_transitions, Transition(from_sym, to_sym, t.rate, t.type))
        end
    end

    new_name = Symbol(string(model.name) * "_reinf_L$(L)")

    CompartmentalModel(new_compartments, new_transitions; name = new_name)
end

# ─── Internal helpers ─────────────────────────────────────────────────────────

_lifted_name(base::Symbol, p::Integer) = Symbol(string(base) * "_" * string(p))

"""
    _compute_p_min(model, L) :: Dict{Symbol, Int}

Compute the minimum reachable infection count for each base compartment.
A compartment is `p_min = 0` iff it is reachable from a susceptible
without traversing any infection transition; otherwise `p_min = min(1, L)`.
"""
function _compute_p_min(model::CompartmentalModel, L::Integer)
    # BFS over spontaneous transitions starting from susceptible_compartments
    reachable_via_spontaneous = Set{Symbol}(model.susceptible_compartments)
    spontaneous_edges = [(t.from, t.to) for t in model.transitions if t.type == :spontaneous]
    changed = true
    while changed
        changed = false
        for (from, to) in spontaneous_edges
            if from in reachable_via_spontaneous && !(to in reachable_via_spontaneous)
                push!(reachable_via_spontaneous, to)
                changed = true
            end
        end
    end

    p_min_susceptible = 0
    p_min_infected    = min(1, L)

    Dict(c.name => (c.name in reachable_via_spontaneous ? p_min_susceptible : p_min_infected)
         for c in model.compartments)
end

"""
    base_compartment_of(name::Symbol) :: Symbol

Recover the base compartment from a lifted name like `:S_3 -> :S`.
Returns `name` itself when no `_<digits>` suffix is present.
"""
function base_compartment_of(name::Symbol)
    s = string(name)
    m = match(r"^(.+)_(\d+)$", s)
    m === nothing ? name : Symbol(m.captures[1])
end

"""
    infection_count_of(name::Symbol) :: Union{Int, Nothing}

Recover the infection count `p` from a lifted name. Returns `nothing` when
the name is not in lifted form.
"""
function infection_count_of(name::Symbol)
    m = match(r"^(.+)_(\d+)$", string(name))
    m === nothing ? nothing : parse(Int, m.captures[2])
end

"""
    reinfection_totals(psys::PairwiseSystem, sol) :: Dict{Symbol, Vector{Float64}}

Aggregate the lifted node compartments back into base-compartment totals
across the saved trajectory. Each value vector has length
`length(sol.t)`; the keys are the base compartment symbols.

Useful for plotting: instead of plotting `S_0, S_1, S_2, …` separately,
plot one curve for total `S = Σ_p S_p`.
"""
function reinfection_totals(psys::PairwiseSystem, sol)
    nodes = node_variables(psys)
    base_groups = Dict{Symbol, Vector{Symbol}}()
    for nv in keys(nodes)
        b = base_compartment_of(nv)
        push!(get!(base_groups, b, Symbol[]), nv)
    end

    totals = Dict{Symbol, Vector{Float64}}()
    for (base, members) in base_groups
        acc = zeros(length(sol.t))
        for m in members
            v = nodes[m]
            traj = sol[v]
            acc .+= traj
        end
        totals[base] = acc
    end
    totals
end
