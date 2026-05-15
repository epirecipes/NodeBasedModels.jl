"""
Shared validation helpers for NodeBasedModels.jl vignettes — uses NetworkOutbreaks
SSA as ground truth for node/pairwise/closure ODE predictions.
"""

using NetworkOutbreaks
using Graphs
using StableRNGs
using Statistics
using Random

"""
    gillespie_ribbon(prog_or_model, params, graph_builder; ...)

Run an SSA ensemble across `n_graphs` host graphs × `nsims_per_graph`
epidemic replicates. `prog_or_model` can be an NBM `CompartmentalModel`
(handled via the `OutbreakModel` adapter) or a pre-built `OutbreakModel`.

Returns `(tgrid, mean_dict, std_dict)` keyed by compartment symbol with
values in fractions of N.
"""
function gillespie_ribbon(prog_or_model, params, graph_builder;
                          N::Int = 1000,
                          n_graphs::Int = 5,
                          nsims_per_graph::Int = 20,
                          tspan = (0.0, 40.0),
                          seed_fraction::Real = 0.01,
                          tgrid = collect(tspan[1]:0.5:tspan[2]),
                          base_seed::Int = 20240501,
                          infected::Symbol = :I,
                          major_outbreak_thresh::Real = 0.0,
                          susceptible::Symbol = :S)
    no_model = prog_or_model isa OutbreakModel ? prog_or_model :
               OutbreakModel(prog_or_model, params)
    rng = StableRNG(base_seed)
    nsamples = n_graphs * nsims_per_graph
    comps = collect(keys(no_model.index_of))
    series = Dict(c => Matrix{Float64}(undef, nsamples, length(tgrid)) for c in comps)
    row = 1
    for gi in 1:n_graphs
        g = graph_builder(rng)
        spec = OutbreakSpec(model = no_model, network = g,
                            initial = SeedFraction(infected => seed_fraction),
                            tspan = tspan)
        ens = simulate_ensemble(spec; nsims = nsims_per_graph,
                                seed = base_seed + 1000 * gi,
                                parallel = true)
        for traj in ens.trajectories
            for (j, t) in enumerate(tgrid)
                st = state_at(traj, t)
                for c in comps
                    series[c][row, j] = Float64(st[no_model.index_of[c]])
                end
            end
            row += 1
        end
    end
    # Optional major-outbreak filter: require attack-rate at final tgrid point
    # to exceed `major_outbreak_thresh`. Drops stochastic die-outs that would
    # otherwise drag the SSA mean below the deterministic prediction.
    keep = trues(nsamples)
    if major_outbreak_thresh > 0 && haskey(series, susceptible)
        S_end = series[susceptible][:, end] ./ N
        keep = (1.0 .- S_end) .>= major_outbreak_thresh
    end
    means = Dict(c => vec(mean(M[keep, :]; dims = 1)) ./ N for (c, M) in series)
    stds  = Dict(c => vec(std(M[keep, :];  dims = 1)) ./ N for (c, M) in series)
    return tgrid, means, stds
end

poisson_graph_builder(N::Int, κ::Real) =
    rng -> erdos_renyi(N, κ / (N - 1); rng = rng)

regular_graph_builder(N::Int, k::Int) =
    rng -> random_regular_graph(N, k; rng = rng)

# Barabási–Albert preferential attachment: each new node adds `k_add` edges,
# producing mean degree ≈ 2*k_add and a power-law degree tail.
barabasi_albert_graph_builder(N::Int, k_add::Int) =
    rng -> barabasi_albert(N, k_add; rng = rng)

# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy (MMD) — kernel two-sample test for trajectories
# Mirrors the EdgeBasedModels.jl/vignettes/_validation.jl machinery so that
# NBM vignettes can quantify ODE-vs-SSA discrepancy on the same scale.
# ---------------------------------------------------------------------------

function pairwise_sqdist(X::AbstractMatrix, Y::AbstractMatrix)
    n, m = size(X, 1), size(Y, 1)
    D = Matrix{Float64}(undef, n, m)
    sx = sum(X.^2; dims = 2)[:]
    sy = sum(Y.^2; dims = 2)[:]
    XY = X * Y'
    @inbounds for i in 1:n, j in 1:m
        D[i, j] = max(0.0, sx[i] + sy[j] - 2 * XY[i, j])
    end
    return D
end

function median_heuristic_sigma(X::AbstractMatrix)
    D = pairwise_sqdist(X, X)
    n = size(D, 1)
    vals = Float64[]
    @inbounds for i in 1:n, j in (i+1):n
        push!(vals, sqrt(D[i, j]))
    end
    isempty(vals) && return 1.0
    σ = median(vals)
    return σ > 0 ? σ : 1.0
end

"""
    mmd2_gaussian(X, Y; σ = nothing) -> Float64

Biased two-sample MMD² with a Gaussian RBF kernel
`k(x,y) = exp(-|x-y|² / (2σ²))`. Each row of `X`/`Y` is one trajectory
flattened over the time grid.
"""
function mmd2_gaussian(X::AbstractMatrix, Y::AbstractMatrix; σ = nothing)
    σ = σ === nothing ? median_heuristic_sigma(vcat(X, Y)) : σ
    s2 = 2 * σ^2
    Kxx = exp.(-pairwise_sqdist(X, X) ./ s2)
    Kyy = exp.(-pairwise_sqdist(Y, Y) ./ s2)
    Kxy = exp.(-pairwise_sqdist(X, Y) ./ s2)
    return mean(Kxx) - 2 * mean(Kxy) + mean(Kyy)
end

"""
    mmd2_permutation_pvalue(X, Y; n_perm = 200) -> (mmd2, p_value, σ)

Permutation-test p-value for the two-sample MMD² statistic. Small p-value
indicates the two ensembles are statistically distinguishable.
"""
function mmd2_permutation_pvalue(X::AbstractMatrix, Y::AbstractMatrix;
                                 σ = nothing, n_perm::Int = 200,
                                 rng = StableRNG(20240701))
    Z = vcat(X, Y)
    σ = σ === nothing ? median_heuristic_sigma(Z) : σ
    obs = mmd2_gaussian(X, Y; σ = σ)
    n = size(X, 1); m = size(Y, 1)
    nz = n + m
    count_ge = 0
    for _ in 1:n_perm
        perm = randperm(rng, nz)
        Xp = Z[perm[1:n], :]
        Yp = Z[perm[(n+1):end], :]
        mp = mmd2_gaussian(Xp, Yp; σ = σ)
        mp >= obs && (count_ge += 1)
    end
    return obs, (count_ge + 1) / (n_perm + 1), σ
end

"""
    ssa_feature_matrix(prog_or_model, params, graph_builder, comps, tgrid; ...)

Run an SSA ensemble and return a feature matrix where each row is one
trajectory's concatenation `[c1(tgrid)... ; c2(tgrid)... ; ...]` for the
list of compartments `comps`, normalised by `N`. Optionally filter to
major outbreaks (final attack rate ≥ `major_outbreak_thresh`).
"""
function ssa_feature_matrix(prog_or_model, params, graph_builder,
                            comps::AbstractVector{Symbol}, tgrid;
                            N::Int, n_graphs::Int, nsims_per_graph::Int,
                            tspan, seed_fraction::Real,
                            base_seed::Int = 20240701,
                            infected::Symbol = :I,
                            susceptible::Symbol = :S,
                            major_outbreak_thresh::Real = 0.0)
    no_model = prog_or_model isa OutbreakModel ? prog_or_model :
               OutbreakModel(prog_or_model, params)
    rng = StableRNG(base_seed)
    nsamp = n_graphs * nsims_per_graph
    nT = length(tgrid)
    nfeat = length(comps) * nT
    F = Matrix{Float64}(undef, nsamp, nfeat)
    s_idx = haskey(no_model.index_of, susceptible) ?
            no_model.index_of[susceptible] : 0
    S_end = Vector{Float64}(undef, nsamp)
    row = 1
    for gi in 1:n_graphs
        g = graph_builder(rng)
        spec = OutbreakSpec(model = no_model, network = g,
                            initial = SeedFraction(infected => seed_fraction),
                            tspan = tspan)
        ens = simulate_ensemble(spec; nsims = nsims_per_graph,
                                seed = base_seed + 1000 * gi,
                                parallel = true)
        for traj in ens.trajectories
            col = 1
            for c in comps
                cidx = no_model.index_of[c]
                for tn in tgrid
                    st = state_at(traj, tn)
                    F[row, col] = Float64(st[cidx]) / N
                    col += 1
                end
            end
            if s_idx > 0
                st_end = state_at(traj, tgrid[end])
                S_end[row] = Float64(st_end[s_idx]) / N
            else
                S_end[row] = 0.0
            end
            row += 1
        end
    end
    if major_outbreak_thresh > 0 && s_idx > 0
        keep = (1.0 .- S_end) .>= major_outbreak_thresh
        F = F[keep, :]
    end
    return F
end

"""
    ode_feature_row(t_curve, dict_of_curves, comps, tgrid)

Pack one ODE/IB trajectory (given as time vector + dict of compartment
trajectories already on `t_curve`) into a 1×F feature row, matching the
layout of `ssa_feature_matrix`. Linear interpolation onto `tgrid`.
"""
function ode_feature_row(t_curve::AbstractVector,
                         curves::AbstractDict,
                         comps::AbstractVector{Symbol}, tgrid)
    nT = length(tgrid)
    nfeat = length(comps) * nT
    row = Vector{Float64}(undef, nfeat)
    col = 1
    for c in comps
        v = curves[c]
        for tn in tgrid
            row[col] = _interp_at(t_curve, v, tn)
            col += 1
        end
    end
    return reshape(row, 1, nfeat)
end

function _interp_at(ts::AbstractVector, vs::AbstractVector, t)
    t <= ts[1]   && return vs[1]
    t >= ts[end] && return vs[end]
    j = searchsortedfirst(ts, t)
    t0, t1 = ts[j-1], ts[j]
    v0, v1 = vs[j-1], vs[j]
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
end
