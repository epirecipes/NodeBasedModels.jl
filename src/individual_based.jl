# individual_based.jl — Order-1 moment closure: individual-based model
#
# Per-node ODEs on a specific graph (Sharkey 2008, 2011).
# Each node i has K-1 state variables (one per compartment minus conservation).
# Pairwise independence: ⟨A_i B_j⟩ ≈ ⟨A_i⟩⟨B_j⟩
#
# For SIR:
#   d⟨S_i⟩/dt = -Σ_j T_ji ⟨S_i⟩⟨I_j⟩
#   d⟨I_i⟩/dt = Σ_j T_ji ⟨S_i⟩⟨I_j⟩ - g_i⟨I_i⟩
#
# This is also known as NIMFA (N-Intertwined Mean Field Approximation)
# or the quenched mean field (QMF) model.

"""
    IndividualBasedResult

Result container for individual-based (order-1) model on a graph.

# Fields
- `sol` — ODE solution
- `graph` — the graph used
- `N` — number of nodes
- `K` — number of tracked states per node
- `state_names` — names of tracked states (e.g., [:S, :I])
- `model` — original CompartmentalModel
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
        # Derived state (e.g., R = 1 - S - I)
        return 1.0 - sum(r.sol[r.K*(i-1) + j, t_idx] for j in 1:r.K)
    end
    return r.sol[r.K*(i-1) + k, t_idx]
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
        return [sum(1.0 - sum(r.sol[r.K*(i-1) + j, t_idx] for j in 1:r.K)
                     for i in 1:r.N) for t_idx in 1:nt]
    end
    return [sum(r.sol[r.K*(i-1) + k, t_idx] for i in 1:r.N) for t_idx in 1:nt]
end

compartment(r::IndividualBasedResult, state::Symbol) = aggregate(r, state)

function compartments(r::IndividualBasedResult, states::AbstractVector{Symbol})
    return Dict(state => compartment(r, state) for state in states)
end

population_fraction(r::IndividualBasedResult, state::Symbol) = aggregate(r, state) ./ r.N

"""
    generate_individual_based(model, graph_net; kwargs...)

Generate and solve the order-1 (individual-based / NIMFA) ODE system on a
specific graph. Uses pairwise independence: ⟨S_i I_j⟩ = ⟨S_i⟩⟨I_j⟩.

This is exact in the limit of infinite graph connectivity and provides an
upper bound on the true epidemic. Errors arise from "anomalous terms"
identified by Sharkey (2011) corresponding to 2-cycles.

# Arguments
- `model::CompartmentalModel` — disease model (e.g., `sir_model()`)
- `net::GraphNetwork` — graph with transmission matrix

# Keyword Arguments
- `tspan` — time span (default: (0.0, 100.0))
- `infection_rate` — per-edge transmission rate τ (default: 0.5)
- `recovery_rate` — recovery rate γ (default: 0.1)
- `initial_infected` — vector of initially infected node indices (or nothing for random)
- `ε` — backward-compatible alias for `seed_fraction`
- `seed_fraction` — fraction of random initial infected if `initial_infected` is nothing
- `saveat` — time points to save solution (default: 1.0)

# Returns
`IndividualBasedResult` containing the ODE solution and metadata.

# Example
```julia
using Graphs
g = random_regular_graph(100, 6)
net = GraphNetwork(g)
result = generate_individual_based(sir_model(), net;
    infection_rate=0.15, recovery_rate=0.1, initial_infected=[1,2,3])
S_total = aggregate(result, :S)
```

# References
- Sharkey (2011) "Deterministic epidemic models on contact networks" Eq. (30)
- Van Mieghem et al. (2009) "Virus spread in networks" (N-intertwined model)
"""
function generate_individual_based(model::CompartmentalModel,
                                    net::GraphNetwork;
                                    tspan::Tuple{Real,Real} = (0.0, 100.0),
                                    infection_rate::Float64 = 0.5,
                                    recovery_rate::Float64 = 0.1,
                                    initial_infected::Union{Vector{Int}, Nothing} = nothing,
                                    ε::Float64 = 1e-3,
                                    seed_fraction::Float64 = ε,
                                    saveat::Float64 = 1.0)
    g = net.graph
    N = nv(g)

    # Build rate lookup: parameter name → value
    rate_values = Dict{Symbol, Float64}()
    for tr in model.transitions
        if tr.type == :infection
            rate_values[tr.rate] = infection_rate
        else
            rate_values[tr.rate] = recovery_rate
        end
    end

    # Identify tracked states (all except the last, which is derived by conservation)
    all_names = model.compartment_names
    K_total = length(all_names)
    tracked = all_names[1:end-1]
    derived = all_names[end]
    K = length(tracked)

    # T[i,j] is the per-edge infection rate from node j to node i.
    T = _effective_transmission_matrix(net, infection_rate)

    # Precompute infection-source lists.
    adj = [_infection_sources(g, i) for i in 1:N]

    # Identify infectious compartment indices in tracked states
    infectious_in_tracked = Int[]
    for c in model.infectious_compartments
        k = findfirst(==(c), tracked)
        if !isnothing(k)
            push!(infectious_in_tracked, k)
        end
    end
    derived_is_infectious = derived in model.infectious_compartments

    function rhs!(du, u, p, t)
        @inbounds for i in 1:N
            base = K * (i - 1)

            # Probability of derived state for node i
            p_derived_i = 1.0
            for k in 1:K
                p_derived_i -= u[base + k]
            end

            # Total infection pressure on node i from neighbors
            force_i = 0.0
            for j in adj[i]
                base_j = K * (j - 1)
                for k in infectious_in_tracked
                    force_i += T[i, j] * u[base_j + k]
                end
                if derived_is_infectious
                    p_derived_j = 1.0
                    for kk in 1:K
                        p_derived_j -= u[base_j + kk]
                    end
                    force_i += T[i, j] * p_derived_j
                end
            end

            # Zero derivatives
            for k in 1:K
                du[base + k] = 0.0
            end

            # Apply transitions
            for tr in model.transitions
                rate_val = rate_values[tr.rate]
                from_k = findfirst(==(tr.from), tracked)
                to_k = findfirst(==(tr.to), tracked)

                if tr.type == :infection
                    p_from = if !isnothing(from_k)
                        u[base + from_k]
                    elseif tr.from == derived
                        p_derived_i
                    else
                        0.0
                    end

                    flux = force_i * p_from
                    if !isnothing(from_k)
                        du[base + from_k] -= flux
                    end
                    if !isnothing(to_k)
                        du[base + to_k] += flux
                    end
                else  # :spontaneous
                    p_from = if !isnothing(from_k)
                        u[base + from_k]
                    elseif tr.from == derived
                        p_derived_i
                    else
                        0.0
                    end

                    flux = rate_val * p_from
                    if !isnothing(from_k)
                        du[base + from_k] -= flux
                    end
                    if !isnothing(to_k)
                        du[base + to_k] += flux
                    end
                end
            end
        end
        nothing
    end

    # Initial conditions
    u0 = zeros(K * N)
    first_susc = model.susceptible_compartments[1]
    first_inf = model.infectious_compartments[1]
    susc_k = findfirst(==(first_susc), tracked)
    inf_k = findfirst(==(first_inf), tracked)

    if !isnothing(initial_infected)
        for i in 1:N
            base = K * (i - 1)
            if i in initial_infected
                if !isnothing(inf_k)
                    u0[base + inf_k] = 1.0
                end
            else
                if !isnothing(susc_k)
                    u0[base + susc_k] = 1.0
                end
            end
        end
    else
        for i in 1:N
            base = K * (i - 1)
            if !isnothing(susc_k)
                u0[base + susc_k] = 1.0 - seed_fraction
            end
            if !isnothing(inf_k)
                u0[base + inf_k] = seed_fraction
            end
        end
    end

    prob = ODEProblem(rhs!, u0, (Float64(tspan[1]), Float64(tspan[2])))
    sol = solve(prob; saveat=saveat)

    return IndividualBasedResult(sol, g, N, K, tracked, model)
end
