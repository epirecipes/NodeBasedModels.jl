# pair_based.jl — Order-2 moment closure: pair-based model
#
# Per-node AND per-edge ODEs on a specific graph (Sharkey 2008, 2011).
# Node variables: ⟨S_i⟩, ⟨I_i⟩ for each node
# Edge variables: [SS]_ij, [SI]_ij for each directed edge (i,j)
#
# Triple closure (conditional independence through middle node):
#   ⟨A_k B_i C_j⟩ ≈ [A_kB_i] × [B_iC_j] / ⟨B_i⟩
#
# For SIR:
#   d⟨S_i⟩/dt = -τ Σ_{j∈N(i)} [SI]_ij
#   d⟨I_i⟩/dt = τ Σ_{j∈N(i)} [SI]_ij - γ⟨I_i⟩
#   d[SS]_ij/dt = -τ [SS]_ij × (Q_i^{-j} + Q_j^{-i})
#   d[SI]_ij/dt = -(τ+γ)[SI]_ij - τ[SI]_ij × Q_i^{-j} + τ[SS]_ij × Q_j^{-i}
#
# where Q_i^{-j} = Σ_{k∈N(i)\{j}} [SI]_ik / ⟨S_i⟩
#
# Exact on tree graphs. Errors proportional to 3-cycle density.

"""
    PairBasedResult

Result container for pair-based (order-2) model on a graph.

# Fields
- `sol` — ODE solution
- `graph` — the graph used
- `N` — number of nodes
- `n_directed_edges` — number of directed edges (2 × undirected edges)
- `directed_edges` — list of (i,j) pairs
- `edge_index` — Dict mapping (i,j) → edge index
- `model` — original CompartmentalModel
"""
struct PairBasedResult
    sol::Any
    graph::Any
    N::Int
    n_directed_edges::Int
    directed_edges::Vector{Tuple{Int,Int}}
    edge_index::Dict{Tuple{Int,Int}, Int}
    model::CompartmentalModel
end

function Base.show(io::IO, r::PairBasedResult)
    tspan = (first(r.sol.t), last(r.sol.t))
    print(io, "PairBasedResult(N=$(r.N), edges=$(r.n_directed_edges), tspan=$tspan)")
end

"""
    node_state(result::PairBasedResult, i, state, t_idx)

Get the probability that node `i` is in compartment `state` at time index `t_idx`.
"""
function node_state(r::PairBasedResult, i::Int, state::Symbol, t_idx::Int)
    if state == :S
        return r.sol[2*(i-1)+1, t_idx]
    elseif state == :I
        return r.sol[2*(i-1)+2, t_idx]
    else  # :R (derived)
        return 1.0 - r.sol[2*(i-1)+1, t_idx] - r.sol[2*(i-1)+2, t_idx]
    end
end

"""
    aggregate(result::PairBasedResult, state)

Aggregate a state across all nodes. Returns a vector over time points.
"""
function aggregate(r::PairBasedResult, state::Symbol)
    nt = length(r.sol.t)
    [sum(node_state(r, i, state, ti) for i in 1:r.N) for ti in 1:nt]
end

compartment(r::PairBasedResult, state::Symbol) = aggregate(r, state)

function compartments(r::PairBasedResult, states::AbstractVector{Symbol})
    return Dict(state => compartment(r, state) for state in states)
end

population_fraction(r::PairBasedResult, state::Symbol) = aggregate(r, state) ./ r.N

"""
    pair_prob(result, i, j, a, b, t_idx)

Get the probability P(node i in state a, node j in state b) at time index `t_idx`.
Requires (i,j) to be a directed edge in the graph.
"""
function pair_prob(r::PairBasedResult, i::Int, j::Int, a::Symbol, b::Symbol, t_idx::Int)
    e = r.edge_index[(i,j)]
    base = 2*r.N + 2*(e-1)
    if a == :S && b == :S
        return r.sol[base + 1, t_idx]
    elseif a == :S && b == :I
        return r.sol[base + 2, t_idx]
    elseif a == :I && b == :S
        # [IS]_ij = [SI]_ji (swap direction)
        re = r.edge_index[(j,i)]
        rbase = 2*r.N + 2*(re-1)
        return r.sol[rbase + 2, t_idx]
    elseif a == :S && b == :R
        return node_state(r, i, :S, t_idx) - r.sol[base+1, t_idx] - r.sol[base+2, t_idx]
    elseif a == :R && b == :S
        re = r.edge_index[(j,i)]
        rbase = 2*r.N + 2*(re-1)
        return node_state(r, j, :S, t_idx) - r.sol[base+1, t_idx] - r.sol[rbase+2, t_idx]
    else
        error("Pair ($a,$b) not directly tracked; available: SS, SI, IS, SR, RS")
    end
end

function _supports_pair_based_sir(model::CompartmentalModel)
    model.compartment_names == [:S, :I, :R] || return false
    model.infectious_compartments == [:I] || return false
    model.susceptible_compartments == [:S] || return false

    inf_trans = infection_transitions(model)
    spon_trans = spontaneous_transitions(model)
    return length(inf_trans) == 1 &&
        length(spon_trans) == 1 &&
        inf_trans[1].from == :S &&
        inf_trans[1].to == :I &&
        spon_trans[1].from == :I &&
        spon_trans[1].to == :R
end

function _validate_pair_based_setup(model::CompartmentalModel,
                                    net::GraphNetwork,
                                    closure)
    closure isa KirkwoodClosure || throw(ArgumentError(
        "generate_pair_based only supports KirkwoodClosure()."))
    Graphs.is_directed(net.graph) && throw(ArgumentError(
        "generate_pair_based currently supports undirected graphs only."))
    _supports_pair_based_sir(model) || throw(ArgumentError(
        "generate_pair_based currently supports only the canonical SIR model with transitions S -> I -> R."))
end

"""
    generate_pair_based(model, net; kwargs...)

Generate and solve the order-2 (pair-based) ODE system on a specific graph.
Uses Kirkwood triple closure: ⟨A_k B_i C_j⟩ ≈ [A_kB_i][B_iC_j]/⟨B_i⟩.

Currently supports the canonical SIR model on undirected graphs only.
Exact on tree graphs; errors proportional to the density of 3-cycles
(triangles).

# Arguments
- `model::CompartmentalModel` — must be SIR
- `net::GraphNetwork` — graph with transmission rates

# Keyword Arguments
- `closure` — triple closure method (must be `KirkwoodClosure()`)
- `tspan` — time span (default: (0.0, 100.0))
- `infection_rate` — per-edge transmission rate τ (default: 0.5)
- `recovery_rate` — recovery rate γ (default: 0.1)
- `initial_infected` — vector of initially infected node indices
- `ε` — backward-compatible alias for `seed_fraction`
- `seed_fraction` — fraction of random initial infected if `initial_infected` is nothing
- `saveat` — time points to save solution (default: 1.0)

# Returns
`PairBasedResult` containing the ODE solution and metadata.

# Variable layout
- `u[1:2N]` — node variables: S_1, I_1, S_2, I_2, ..., S_N, I_N
- `u[2N+1:2N+2M]` — edge variables: SS_e1, SI_e1, SS_e2, SI_e2, ...
  where M is the number of directed edges.

# References
- Sharkey (2011) Eq. (41): pair-based SIR on a graph
- Kiss, Miller & Simon (2017) Ch. 5: pair-based models on networks
"""
function generate_pair_based(model::CompartmentalModel,
                              net::GraphNetwork;
                              closure = KirkwoodClosure(),
                              tspan::Tuple{Real,Real} = (0.0, 100.0),
                              infection_rate::Float64 = 0.5,
                              recovery_rate::Float64 = 0.1,
                              initial_infected::Union{Vector{Int}, Nothing} = nothing,
                              ε::Float64 = 1e-3,
                              seed_fraction::Float64 = ε,
                              saveat::Float64 = 1.0)
    _validate_pair_based_setup(model, net, closure)

    g = net.graph
    N = nv(g)
    γ = recovery_rate
    T = _effective_transmission_matrix(net, infection_rate)

    # Enumerate directed edges: for each undirected edge {i,j}, create (i,j) and (j,i)
    directed_edges = Tuple{Int,Int}[]
    for e in edges(g)
        push!(directed_edges, (src(e), dst(e)))
        push!(directed_edges, (dst(e), src(e)))
    end
    M = length(directed_edges)

    edge_idx = Dict{Tuple{Int,Int}, Int}()
    for (e, (i,j)) in enumerate(directed_edges)
        edge_idx[(i,j)] = e
    end
    edge_rates = [T[i, j] for (i, j) in directed_edges]

    adj = [neighbors(g, i) for i in 1:N]

    # Precompute neighbor lists excluding specific nodes for each directed edge
    ni_exc_j = Vector{Vector{Int}}(undef, M)
    nj_exc_i = Vector{Vector{Int}}(undef, M)
    for (e, (i,j)) in enumerate(directed_edges)
        ni_exc_j[e] = [k for k in adj[i] if k != j]
        nj_exc_i[e] = [k for k in adj[j] if k != i]
    end

    # Precompute edge index lookups for the closure terms
    # For edge e=(i,j): edge ids of SI_ik for k ∈ N(i)\{j}
    si_ik_edges = Vector{Vector{Int}}(undef, M)
    si_jk_edges = Vector{Vector{Int}}(undef, M)
    for (e, (i,j)) in enumerate(directed_edges)
        si_ik_edges[e] = [edge_idx[(i,k)] for k in ni_exc_j[e]]
        si_jk_edges[e] = [edge_idx[(j,k)] for k in nj_exc_i[e]]
    end

    n_vars = 2*N + 2*M
    eps_safe = 1e-12

    function rhs!(du, u, p, t)
        @inbounds begin
            # Zero all derivatives
            for idx in 1:n_vars
                du[idx] = 0.0
            end

            # Node equations
            for i in 1:N
                s_idx = 2*(i-1) + 1
                i_idx = 2*(i-1) + 2

                # Sum of [SI_ij] over j ∈ N(i)
                sum_SI = 0.0
                for j in adj[i]
                    e_ij = edge_idx[(i,j)]
                    sum_SI += edge_rates[e_ij] * u[2*N + 2*(e_ij-1) + 2]
                end

                du[s_idx] = -sum_SI
                du[i_idx] = sum_SI - γ * u[i_idx]
            end

            # Pair equations
            for e in 1:M
                i, j = directed_edges[e]
                τ_ij = edge_rates[e]
                ss_idx = 2*N + 2*(e-1) + 1
                si_idx = 2*N + 2*(e-1) + 2

                S_i = max(u[2*(i-1)+1], eps_safe)
                S_j = max(u[2*(j-1)+1], eps_safe)
                SS_ij = u[ss_idx]
                SI_ij = u[si_idx]

                # Infection pressure on i from neighbors k ≠ j (Kirkwood closure):
                #   Q_i^{-j} = Σ_{k∈N(i)\{j}} [SI]_ik / S_i
                pressure_i = 0.0
                for edge in si_ik_edges[e]
                    pressure_i += edge_rates[edge] * u[2*N + 2*(edge-1) + 2]
                end
                pressure_i /= S_i

                # Infection pressure on j from neighbors k ≠ i:
                #   Q_j^{-i} = Σ_{k∈N(j)\{i}} [SI]_jk / S_j
                pressure_j = 0.0
                for edge in si_jk_edges[e]
                    pressure_j += edge_rates[edge] * u[2*N + 2*(edge-1) + 2]
                end
                pressure_j /= S_j

                # d[SS_ij]/dt = -SS_ij (Q_i^{-j} + Q_j^{-i})
                du[ss_idx] = -SS_ij * (pressure_i + pressure_j)

                # d[SI_ij]/dt = -(τ_ij+γ)SI_ij - SI_ij Q_i^{-j} + SS_ij Q_j^{-i}
                du[si_idx] = -(τ_ij + γ) * SI_ij - SI_ij * pressure_i + SS_ij * pressure_j
            end
        end
        nothing
    end

    # Initial conditions
    u0 = zeros(n_vars)
    if !isnothing(initial_infected)
        inf_set = Set(initial_infected)
        for i in 1:N
            if i in inf_set
                u0[2*(i-1)+2] = 1.0  # I_i = 1
            else
                u0[2*(i-1)+1] = 1.0  # S_i = 1
            end
        end
        for (e, (i,j)) in enumerate(directed_edges)
            s_i = i ∉ inf_set ? 1.0 : 0.0
            s_j = j ∉ inf_set ? 1.0 : 0.0
            i_j = j ∈ inf_set ? 1.0 : 0.0
            u0[2*N + 2*(e-1) + 1] = s_i * s_j       # SS_ij
            u0[2*N + 2*(e-1) + 2] = s_i * i_j        # SI_ij
        end
    else
        for i in 1:N
            u0[2*(i-1)+1] = 1.0 - seed_fraction
            u0[2*(i-1)+2] = seed_fraction
        end
        for e in 1:M
            u0[2*N + 2*(e-1) + 1] = (1-seed_fraction)^2   # SS
            u0[2*N + 2*(e-1) + 2] = (1-seed_fraction)*seed_fraction    # SI
        end
    end

    prob = ODEProblem(rhs!, u0, (Float64(tspan[1]), Float64(tspan[2])))
    sol = solve(prob; saveat=saveat)

    return PairBasedResult(sol, g, N, M, directed_edges, edge_idx, model)
end
