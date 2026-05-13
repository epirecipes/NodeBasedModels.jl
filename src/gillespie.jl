# gillespie.jl — Exact stochastic SIR simulation on graphs via JumpProcesses.jl
#
# Uses ConstantRateJump with the Direct (Gillespie) aggregator for exact
# stochastic simulation of SIR dynamics on a specific graph.
#
# State vector u[1:2N]:
#   u[i] = 1 if node i is susceptible, 0 otherwise
#   u[N+i] = 1 if node i is infected, 0 otherwise
#   Recovered is derived: R_i = 1 - u[i] - u[N+i]
#
# Jump processes:
#   For each edge {i,j}: two infection jumps (i→j and j→i)
#     rate = τ × u[susceptible] × u[N+infected]
#   For each node i: one recovery jump
#     rate = γ × u[N+i]

"""
    GillespieResult

Result of an exact Gillespie (SSA) simulation of SIR on a graph.

# Fields
- `sol` — JumpProcesses solution (piecewise-constant interpolation)
- `graph` — the graph used
- `N` — number of nodes
"""
struct GillespieResult
    sol::Any
    graph::Any
    N::Int
end

function Base.show(io::IO, r::GillespieResult)
    tspan = (first(r.sol.t), last(r.sol.t))
    print(io, "GillespieResult(N=$(r.N), tspan=$tspan)")
end

"""
    node_state(r::GillespieResult, i, t)

Get the state of node `i` at time `t`. Returns :S, :I, or :R.
"""
function node_state(r::GillespieResult, i::Int, t::Float64)
    u = r.sol(t)
    if u[i] == 1
        return :S
    elseif u[r.N + i] == 1
        return :I
    else
        return :R
    end
end

"""
    aggregate(r::GillespieResult, state; saveat=1.0)

Count nodes in `state` over time. Returns `(times, counts)`.
"""
function aggregate(r::GillespieResult, state::Symbol; saveat::Float64 = 1.0)
    tmax = r.sol.t[end]
    ts = collect(0.0:saveat:tmax)
    counts = zeros(Int, length(ts))
    N = r.N
    for (ti, t) in enumerate(ts)
        u = r.sol(t)
        if state == :S
            counts[ti] = sum(u[1:N])
        elseif state == :I
            counts[ti] = sum(u[N+1:2N])
        elseif state == :R
            counts[ti] = N - sum(u[1:N]) - sum(u[N+1:2N])
        end
    end
    return (times=ts, counts=counts)
end

compartment(r::GillespieResult, state::Symbol; saveat::Float64 = 1.0) =
    aggregate(r, state; saveat = saveat)

function compartments(r::GillespieResult, states::AbstractVector{Symbol}; saveat::Float64 = 1.0)
    return Dict(state => compartment(r, state; saveat = saveat) for state in states)
end

function population_fraction(r::GillespieResult, state::Symbol; saveat::Float64 = 1.0)
    values = aggregate(r, state; saveat = saveat)
    return (times = values.times, counts = values.counts ./ r.N)
end

"""
    gillespie_sir(net; kwargs...)

Run an exact Gillespie (SSA) SIR simulation on a specific graph using
JumpProcesses.jl with the Direct aggregator.

Each infected node can transmit to each susceptible neighbor at rate `infection_rate`.
Each infected node recovers at rate `recovery_rate`.

# Arguments
- `net::GraphNetwork` — graph network

# Keyword Arguments
- `infection_rate` — per-edge transmission rate τ (default: 0.5)
- `recovery_rate` — per-capita recovery rate γ (default: 0.1)
- `initial_infected` — vector of initially infected node indices (default: [1])
- `tmax` — maximum simulation time (default: 100.0)
- `seed` — random seed (default: nothing)

# Returns
`GillespieResult` with the full stochastic trajectory.

# Example
```julia
using Graphs
g = random_regular_graph(100, 6)
net = GraphNetwork(g)
result = gillespie_sir(net; infection_rate=0.15, recovery_rate=0.1,
                       initial_infected=[1,2,3])
ts, S_counts = aggregate(result, :S)
```

# References
- Gillespie (1977) "Exact stochastic simulation of coupled chemical reactions"
"""
function gillespie_sir(net::GraphNetwork;
                        infection_rate::Float64 = 0.5,
                        recovery_rate::Float64 = 0.1,
                        initial_infected::Vector{Int} = [1],
                        tmax::Float64 = 100.0,
                        seed::Union{Int, Nothing} = nothing)
    g = net.graph
    N = nv(g)
    T = _effective_transmission_matrix(net, infection_rate)

    # State: u[1:N] = S indicators, u[N+1:2N] = I indicators
    u0 = zeros(Int, 2N)
    for i in 1:N
        if i in initial_infected
            u0[N+i] = 1
        else
            u0[i] = 1
        end
    end

    # Build jump processes
    jumps = ConstantRateJump[]

    # Infection jumps: one jump per transmission direction.
    for e in edges(g)
        let s = src(e), d = dst(e), n = N
            rate_sd = T[d, s]
            if rate_sd > 0
                push!(jumps, ConstantRateJump(
                    (u, p, t) -> rate_sd * u[d] * u[n+s],
                    integrator -> begin
                        integrator.u[d] -= 1
                        integrator.u[n+d] += 1
                    end
                ))
            end
            if !Graphs.is_directed(g)
                rate_ds = T[s, d]
                if rate_ds > 0
                    push!(jumps, ConstantRateJump(
                        (u, p, t) -> rate_ds * u[s] * u[n+d],
                        integrator -> begin
                            integrator.u[s] -= 1
                            integrator.u[n+s] += 1
                        end
                    ))
                end
            end
        end
    end

    # Recovery jumps: for each node
    for i in 1:N
        let node = i, n = N
            push!(jumps, ConstantRateJump(
                (u, p, t) -> p[2] * u[n+node],
                integrator -> begin
                    integrator.u[n+node] -= 1
                end
            ))
        end
    end

    p = [0.0, recovery_rate]
    dprob = DiscreteProblem(u0, (0.0, tmax), p)
    jprob = JumpProblem(dprob, Direct(), jumps...)

    if !isnothing(seed)
        sol = solve(jprob, SSAStepper(); seed=seed)
    else
        sol = solve(jprob, SSAStepper())
    end

    return GillespieResult(sol, g, N)
end

"""
    gillespie_sir_average(net; nruns, kwargs...)

Run multiple Gillespie simulations and compute time-averaged trajectories.

Returns a NamedTuple with fields:
- `t_grid` — regular time grid
- `S_mean`, `I_mean`, `R_mean` — mean counts (Float64)
- `S_q05`, `S_q95`, `I_q05`, `I_q95` — 5th/95th percentile bounds
- `final_sizes` — vector of final epidemic sizes from each run

# Example
```julia
avg = gillespie_sir_average(net; nruns=200, infection_rate=0.15,
                            recovery_rate=0.1, initial_infected=[1])
# avg.t_grid, avg.I_mean, avg.I_q05, avg.I_q95
```
"""
function gillespie_sir_average(net::GraphNetwork;
                                nruns::Int = 100,
                                dt::Float64 = 1.0,
                                tmax_grid::Float64 = 100.0,
                                kwargs...)
    N = nv(net.graph)
    t_grid = collect(0.0:dt:tmax_grid)
    nt = length(t_grid)

    S_all = zeros(nruns, nt)
    I_all = zeros(nruns, nt)
    R_all = zeros(nruns, nt)
    final_sizes = zeros(Int, nruns)

    for run in 1:nruns
        res = gillespie_sir(net; kwargs...)

        # Interpolate onto grid
        for (ti, tval) in enumerate(t_grid)
            u = res.sol(min(tval, res.sol.t[end]))
            S_all[run, ti] = sum(u[1:N])
            I_all[run, ti] = sum(u[N+1:2N])
            R_all[run, ti] = N - S_all[run, ti] - I_all[run, ti]
        end

        # Final size
        u_final = res.sol(res.sol.t[end])
        final_sizes[run] = round(Int, N - sum(u_final[1:N]))
    end

    S_mean = vec(mean(S_all; dims=1))
    I_mean = vec(mean(I_all; dims=1))
    R_mean = vec(mean(R_all; dims=1))

    q05_idx = max(1, round(Int, 0.05 * nruns))
    q95_idx = min(nruns, round(Int, 0.95 * nruns))

    S_q05 = vec(mapslices(x -> sort(x)[q05_idx], S_all; dims=1))
    S_q95 = vec(mapslices(x -> sort(x)[q95_idx], S_all; dims=1))
    I_q05 = vec(mapslices(x -> sort(x)[q05_idx], I_all; dims=1))
    I_q95 = vec(mapslices(x -> sort(x)[q95_idx], I_all; dims=1))

    return (t_grid=t_grid, S_mean=S_mean, I_mean=I_mean, R_mean=R_mean,
            S_q05=S_q05, S_q95=S_q95, I_q05=I_q05, I_q95=I_q95,
            final_sizes=final_sizes)
end

# ─── SIS Gillespie ────────────────────────────────────────────────────────────
#
# Hand-rolled exact SSA (Direct method). We avoid JumpProcesses for SIS
# because constructing thousands of per-edge `ConstantRateJump`s overflows
# the compiler's type-level recursion. The native loop is also faster for
# this style of model.
#
# The trajectory is exposed via a `GillespieSISResult` whose `sol(t)` method
# performs piecewise-constant lookup over the recorded jump times.

"""
    GillespieSISResult

Result of an exact Gillespie SIS simulation on a graph. Stores per-jump
state snapshots so `sol(t)` can return the full N-vector at any time
within `[0, tmax]`. Also stores `infection_times[i]` — the sorted list of
times node `i` underwent an S→I event — so quantities like `[S_p](t)` and
`[I_p](t)` (number of nodes with infection count `p` at time `t`) can be
reconstructed for direct comparison with the reinfection-counting closure.

# Fields
- `times::Vector{Float64}` — recorded event times (with `times[1] = 0`).
- `states::Vector{BitVector}` — `states[k][i] = true` iff node `i` is
   infectious in the interval `[times[k], times[k+1])`.
- `graph` — the graph used.
- `N::Int` — number of nodes.
- `infection_times::Vector{Vector{Float64}}` — per-node S→I event times.
"""
struct GillespieSISResult
    times::Vector{Float64}
    states::Vector{BitVector}
    graph::Any
    N::Int
    infection_times::Vector{Vector{Float64}}
end

Base.show(io::IO, r::GillespieSISResult) = print(io,
    "GillespieSISResult(N=$(r.N), tspan=(0.0, $(last(r.times))))")

"""
    (r::GillespieSISResult)(t::Real) -> BitVector

Piecewise-constant interpolation: returns the state vector at time `t`.
"""
function (r::GillespieSISResult)(t::Real)
    if t <= r.times[1]
        return copy(r.states[1])
    elseif t >= last(r.times)
        return copy(last(r.states))
    end
    # binary search for largest index k with times[k] <= t
    lo, hi = 1, length(r.times)
    while lo < hi
        mid = (lo + hi + 1) >>> 1
        r.times[mid] <= t ? (lo = mid) : (hi = mid - 1)
    end
    copy(r.states[lo])
end

"""
    gillespie_sis(net; kwargs...) -> GillespieSISResult

Exact Gillespie SSA for SIS on a graph. Each S–I edge fires infection at
rate `infection_rate`; each I node recovers (back to S) at rate
`recovery_rate`. Per-node S→I event times are recorded so reinfection
counts can be reconstructed.

!!! note "Migration path"
    This is the package's first-generation SSA, optimised for SIS only.
    For new code, prefer the companion package
    [`NetworkOutbreaks.jl`](https://github.com/) which provides a
    general-purpose Direct-method SSA that works for any compartmental
    model (SIS, SIR, SEIR, …) and integrates with both
    `NodeBasedModels.CompartmentalModel` and
    `EdgeBasedModels.DiseaseProgression`. The integration testset
    `"NetworkOutbreaks integration"` in this package exercises that
    adapter; see also `vignettes/15_reinfection_counting/index.qmd` for
    a SIS reinfection-counting workflow that uses NetworkOutbreaks for
    ground-truth simulation. `gillespie_sis` will continue to be
    supported for back-compat, but no new features will be added.

# Keyword Arguments
- `infection_rate::Float64` — per-edge τ (default `0.5`)
- `recovery_rate::Float64`  — per-node γ (default `1.0`)
- `initial_infected::Vector{Int}` — initial I nodes (default `[1]`)
- `tmax::Float64` — simulation horizon (default `100.0`)
- `seed::Union{Int, Nothing}` — RNG seed (default `nothing`)
"""
function gillespie_sis(net::GraphNetwork;
                        infection_rate::Float64 = 0.5,
                        recovery_rate::Float64 = 1.0,
                        initial_infected::Vector{Int} = [1],
                        tmax::Float64 = 100.0,
                        seed::Union{Int, Nothing} = nothing)
    g = net.graph
    N = nv(g)
    Tmat = _effective_transmission_matrix(net, infection_rate)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    # Build per-node neighbour list once
    neigh = [collect(neighbors(g, i)) for i in 1:N]

    # Current state
    is_inf = falses(N)
    for i in initial_infected
        is_inf[i] = true
    end

    # Edge list as ordered pairs (s, d) with rate τ_{d,s}.
    # For undirected graphs, both directions are stored.
    edge_src = Int[]; edge_dst = Int[]; edge_rate = Float64[]
    for e in edges(g)
        s, d = src(e), dst(e)
        r_sd = Tmat[d, s]
        if r_sd > 0
            push!(edge_src, s); push!(edge_dst, d); push!(edge_rate, r_sd)
        end
        if !Graphs.is_directed(g)
            r_ds = Tmat[s, d]
            if r_ds > 0
                push!(edge_src, d); push!(edge_dst, s); push!(edge_rate, r_ds)
            end
        end
    end
    nedges = length(edge_src)

    infection_times = [Float64[] for _ in 1:N]
    for i in initial_infected
        push!(infection_times[i], 0.0)
    end

    times  = Float64[0.0]
    states = BitVector[copy(is_inf)]

    # SSA loop
    t = 0.0
    while t < tmax
        # Infection rates: rate[k] = edge_rate[k] if src is I and dst is S, else 0
        # Recovery rates: γ for each I node
        inf_rate_total = 0.0
        @inbounds for k in 1:nedges
            if is_inf[edge_src[k]] && !is_inf[edge_dst[k]]
                inf_rate_total += edge_rate[k]
            end
        end
        nI = count(is_inf)
        rec_rate_total = recovery_rate * nI
        total_rate = inf_rate_total + rec_rate_total

        if total_rate <= 0
            # Absorbing (no I left)
            push!(times, tmax); push!(states, copy(is_inf))
            break
        end

        dt = -log(rand(rng)) / total_rate
        t_next = t + dt
        if t_next > tmax
            push!(times, tmax); push!(states, copy(is_inf))
            break
        end
        t = t_next

        # Sample event
        u = rand(rng) * total_rate
        if u < inf_rate_total
            # Choose infection edge
            cum = 0.0
            chosen = -1
            @inbounds for k in 1:nedges
                if is_inf[edge_src[k]] && !is_inf[edge_dst[k]]
                    cum += edge_rate[k]
                    if u < cum
                        chosen = k; break
                    end
                end
            end
            d = edge_dst[chosen]
            is_inf[d] = true
            push!(infection_times[d], t)
        else
            # Recovery: choose an I node uniformly weighted by γ
            u2 = u - inf_rate_total
            idx = floor(Int, u2 / recovery_rate) + 1
            # Find idx-th infectious node
            seen = 0; chosen = -1
            @inbounds for i in 1:N
                if is_inf[i]
                    seen += 1
                    if seen == idx
                        chosen = i; break
                    end
                end
            end
            chosen == -1 && (chosen = findlast(is_inf))
            is_inf[chosen] = false
        end

        push!(times, t); push!(states, copy(is_inf))
    end

    GillespieSISResult(times, states, g, N, infection_times)
end

"""
    sis_state(r::GillespieSISResult, i::Int, t::Real) -> Symbol

Return the state of node `i` at time `t` (`:S` or `:I`).
"""
function sis_state(r::GillespieSISResult, i::Int, t::Real)
    u = r(t)
    u[i] ? :I : :S
end

"""
    infection_count(r::GillespieSISResult, i::Int, t::Real) -> Int

Number of S→I events node `i` has undergone by time `t`.
"""
function infection_count(r::GillespieSISResult, i::Int, t::Real)
    times = r.infection_times[i]
    n = 0
    @inbounds for τ in times
        τ <= t && (n += 1)
    end
    n
end

"""
    reinfection_histogram(r::GillespieSISResult, t::Real, L::Integer)

Compute `(S_counts, I_counts)` where `S_counts[p+1]` is the number of
nodes currently susceptible with infection-count `min(p, L)`, and
similarly for `I_counts`. `p` ranges from `0` to `L`.
"""
function reinfection_histogram(r::GillespieSISResult, t::Real, L::Integer)
    S = zeros(Int, L + 1)
    I = zeros(Int, L + 1)
    u = r(t)
    for i in 1:r.N
        p = min(infection_count(r, i, t), L)
        if u[i]
            I[p + 1] += 1
        else
            S[p + 1] += 1
        end
    end
    (S = S, I = I)
end

"""
    reinfection_histogram_series(r::GillespieSISResult, ts::AbstractVector,
                                  L::Integer) -> (S, I)

Compute reinfection histograms over a time grid. Returns
`S::Matrix{Int}` and `I::Matrix{Int}` of shape `(L+1, length(ts))`.
"""
function reinfection_histogram_series(r::GillespieSISResult, ts::AbstractVector,
                                       L::Integer)
    S = zeros(Int, L + 1, length(ts))
    I = zeros(Int, L + 1, length(ts))
    for (k, t) in enumerate(ts)
        h = reinfection_histogram(r, t, L)
        S[:, k] = h.S
        I[:, k] = h.I
    end
    (S = S, I = I)
end

"""
    gillespie_sis_average(net; nruns, kwargs...)

Run `nruns` independent Gillespie SIS simulations and return averaged
prevalence trajectories. Returns a NamedTuple
`(t_grid, S_mean, I_mean, S_q05, S_q95, I_q05, I_q95)`.
"""
function gillespie_sis_average(net::GraphNetwork;
                                nruns::Int = 100,
                                dt::Float64 = 1.0,
                                tmax_grid::Float64 = 100.0,
                                seed::Union{Int, Nothing} = nothing,
                                kwargs...)
    N = nv(net.graph)
    t_grid = collect(0.0:dt:tmax_grid)
    nt = length(t_grid)
    S_all = zeros(nruns, nt)
    I_all = zeros(nruns, nt)

    for run in 1:nruns
        # Derive a distinct per-run seed so the ensemble is genuinely an
        # ensemble. If the caller passes `seed`, use it as a base; otherwise
        # leave each run's RNG to its own default.
        run_seed = isnothing(seed) ? nothing : seed + run - 1
        res = gillespie_sis(net; tmax = tmax_grid, seed = run_seed, kwargs...)
        for (ti, tval) in enumerate(t_grid)
            u = res(min(tval, last(res.times)))
            I_all[run, ti] = count(u)
            S_all[run, ti] = N - I_all[run, ti]
        end
    end

    S_mean = vec(mean(S_all; dims = 1))
    I_mean = vec(mean(I_all; dims = 1))
    q05_idx = max(1, round(Int, 0.05 * nruns))
    q95_idx = min(nruns, round(Int, 0.95 * nruns))
    S_q05 = vec(mapslices(x -> sort(x)[q05_idx], S_all; dims = 1))
    S_q95 = vec(mapslices(x -> sort(x)[q95_idx], S_all; dims = 1))
    I_q05 = vec(mapslices(x -> sort(x)[q05_idx], I_all; dims = 1))
    I_q95 = vec(mapslices(x -> sort(x)[q95_idx], I_all; dims = 1))

    return (t_grid = t_grid, S_mean = S_mean, I_mean = I_mean,
            S_q05 = S_q05, S_q95 = S_q95, I_q05 = I_q05, I_q95 = I_q95)
end
