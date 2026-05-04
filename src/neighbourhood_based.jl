# neighbourhood_based.jl — Phase C: neighbourhood model (Approximation 3,
# n = 2) of Keeling, House, Cooper & Pellis (2016, PLoS Comp Biol 12(12):
# e1005296).
#
# State variables for SIS on a k-regular network:
#
#   [S_y] = expected number of S nodes whose y of k neighbours are infectious
#   [I_y] = same for I nodes,                          y ∈ {0, 1, …, k}
#
# Closure: the per-S-neighbour infection rates ω_S, ω_I are obtained from
# the consistent-overlap closure (Eq 10 of the paper) — see
# `NEIGHBOURHOOD_SPEC.md` for the derivation.
#
# Layout / API closely mirrors `motif_based.jl` (Phase B) for consistency.

"""
    NeighbourhoodSystem

Container for an n=2 neighbourhood-closure ODE system on a k-regular
host network.

Fields:
- `k::Int` — host degree.
- `n::Int` — neighbourhood order (currently always 2).
- `var_names::Vector{Symbol}` — `[:S_0, :S_1, …, :S_k, :I_0, :I_1, …, :I_k]`.
- `index::Dict{Tuple{Symbol,Int},Int}` — `(:S, y) → row index in u`.
- `rhs!::Function` — `(du, u, p, t) → nothing` consuming `p = (β, γ, k)`.
- `u0::Vector{Float64}` — initial condition.
- `tspan::Tuple{Float64,Float64}`
- `params::NamedTuple` — `(β, γ, k, N, ε)`.
- `model`, `network` — references to the underlying `CompartmentalModel`
  and `NetworkStructure`.
"""
struct NeighbourhoodSystem
    k::Int
    n::Int
    var_names::Vector{Symbol}
    index::Dict{Tuple{Symbol,Int},Int}
    rhs!::Function
    u0::Vector{Float64}
    tspan::Tuple{Float64,Float64}
    params::NamedTuple
    model::Any
    network::Any
end

# ─── Indexing ──────────────────────────────────────────────────────────────

function _build_neighbourhood_index(k::Int)
    var_names = Symbol[]
    index = Dict{Tuple{Symbol,Int},Int}()
    for y in 0:k
        push!(var_names, Symbol("S_", y))
        index[(:S, y)] = length(var_names)
    end
    for y in 0:k
        push!(var_names, Symbol("I_", y))
        index[(:I, y)] = length(var_names)
    end
    return var_names, index
end

# ─── Numeric RHS ───────────────────────────────────────────────────────────
#
# The RHS is a closure over the precomputed singleton index map. Inner loops
# walk y = 0..k, accumulating two ω-closure ratios and applying the four
# transition contributions to dS_y/dt and dI_y/dt.

function _build_neighbourhood_rhs(k::Int, index::Dict{Tuple{Symbol,Int},Int})
    iS = [index[(:S, y)] for y in 0:k]
    iI = [index[(:I, y)] for y in 0:k]

    function rhs!(du, u, p, t)
        β = p.β; γ = p.γ
        # Closure numerators / denominators
        num_SS = 0.0; den_SS = 0.0
        num_IS = 0.0; den_IS = 0.0
        @inbounds for y in 0:k
            Sy = u[iS[y+1]]
            num_SS += y * (k - y) * Sy
            den_SS += (k - y) * Sy
            num_IS += y * y * Sy
            den_IS += y * Sy
        end
        ω_S = safe_ratio(num_SS, den_SS) * β
        ω_I = safe_ratio(num_IS, den_IS) * β

        @inbounds for y in 0:k
            Sy   = u[iS[y+1]]
            Iy   = u[iI[y+1]]
            Sym1 = y > 0       ? u[iS[y]]   : 0.0   # S_{y-1}
            Syp1 = y < k       ? u[iS[y+2]] : 0.0   # S_{y+1}
            Iym1 = y > 0       ? u[iI[y]]   : 0.0   # I_{y-1}
            Iyp1 = y < k       ? u[iI[y+2]] : 0.0   # I_{y+1}

            # dS_y
            dS  = γ * Iy - β * y * Sy
            dS += γ * ((y + 1) * Syp1 - y * Sy)
            dS += ω_S * ((k - y + 1) * Sym1 - (k - y) * Sy)

            # dI_y
            dI  = β * y * Sy - γ * Iy
            dI += γ * ((y + 1) * Iyp1 - y * Iy)
            dI += ω_I * ((k - y + 1) * Iym1 - (k - y) * Iy)

            du[iS[y+1]] = dS
            du[iI[y+1]] = dI
        end
        return nothing
    end
    return rhs!
end

# ─── Initial conditions ────────────────────────────────────────────────────
#
# Random-mixing IC with infected fraction ε:
#
#   S_y = N (1 − ε) · C(k, y) · ε^y · (1−ε)^(k−y)
#   I_y = N  ε      · C(k, y) · ε^y · (1−ε)^(k−y)

function _binomial_pmf(k::Int, y::Int, p::Float64)
    return binomial(k, y) * p^y * (1 - p)^(k - y)
end

function _build_neighbourhood_ic(k::Int, index::Dict{Tuple{Symbol,Int},Int},
                                  N::Float64, ε::Float64)
    nvars = 2 * (k + 1)
    u0 = zeros(Float64, nvars)
    for y in 0:k
        b = _binomial_pmf(k, y, ε)
        u0[index[(:S, y)]] = N * (1 - ε) * b
        u0[index[(:I, y)]] = N *      ε  * b
    end
    return u0
end

# ─── Public API ────────────────────────────────────────────────────────────

"""
    generate_neighbourhood(model, k::Integer, n::Integer; β, γ,
                           tspan=(0.0,100.0), N=1.0, ε=1e-3) -> NeighbourhoodSystem

Build a Keeling/House/Cooper/Pellis 2016 neighbourhood-closure SIS
system on a `k`-regular host at neighbourhood order `n`.

Currently only **`n = 2`** and SIS (`sis_model()`) are supported;
other values throw `ArgumentError`.

`β` is the per-edge transmission rate (the paper's `τ`) and `γ` is the
per-node recovery rate.  The initial condition is random mixing with
infected fraction `ε`.
"""
function generate_neighbourhood(model, k::Integer, n::Integer;
                                 β::Real, γ::Real,
                                 tspan = (0.0, 100.0),
                                 N::Real = 1.0,
                                 ε::Real = 1e-3)
    n == 2 || throw(ArgumentError(
        "generate_neighbourhood: only n = 2 is implemented (got n = $n)."))
    k = Int(k)
    k ≥ 1 || throw(ArgumentError("generate_neighbourhood: k must be ≥ 1, got $k."))

    # Validate that the model is SIS-shaped: two compartments {S, I},
    # one infection S→I, one recovery I→S.
    cnames = sort(model.compartment_names)
    if cnames != [:I, :S]
        throw(ArgumentError(
            "generate_neighbourhood: only the canonical SIS model is "
            * "supported (compartments must be [:S, :I]); got $cnames."))
    end

    var_names, index = _build_neighbourhood_index(k)
    rhs!  = _build_neighbourhood_rhs(k, index)
    u0    = _build_neighbourhood_ic(k, index, Float64(N), Float64(ε))

    network = regular_network(k)
    params  = (β = Float64(β), γ = Float64(γ), k = k,
               N = Float64(N), ε = Float64(ε))
    return NeighbourhoodSystem(k, 2, var_names, index, rhs!, u0,
                               (Float64(tspan[1]), Float64(tspan[2])),
                               params, model, network)
end

"""
    solve_neighbourhood(sys::NeighbourhoodSystem; saveat=nothing,
                        alg=Tsit5(), reltol=1e-8, abstol=1e-10, kwargs...)

Solve the neighbourhood ODE.  Returns the `OrdinaryDiffEq` solution.

Default tolerances are tightened from `OrdinaryDiffEq`'s defaults
(`reltol=1e-3, abstol=1e-6`) to `1e-8`/`1e-10` for consistency with the
other moment-closure solvers in this package; pass `reltol`/`abstol`
explicitly to override.
"""
function solve_neighbourhood(sys::NeighbourhoodSystem; saveat = nothing,
                              alg = nothing,
                              reltol = 1e-8, abstol = 1e-10,
                              kwargs...)
    prob = OrdinaryDiffEqDefault.ODEProblem(sys.rhs!, sys.u0, sys.tspan, sys.params)
    if isnothing(alg)
        if saveat === nothing
            return OrdinaryDiffEqDefault.solve(prob; reltol = reltol, abstol = abstol, kwargs...)
        else
            return OrdinaryDiffEqDefault.solve(prob; saveat = saveat,
                                               reltol = reltol, abstol = abstol, kwargs...)
        end
    elseif saveat === nothing
        return OrdinaryDiffEqDefault.solve(prob, alg; reltol = reltol, abstol = abstol, kwargs...)
    else
        return OrdinaryDiffEqDefault.solve(prob, alg; saveat = saveat,
                                           reltol = reltol, abstol = abstol, kwargs...)
    end
end

"""
    neighbourhood_compartment(sys::NeighbourhoodSystem, sol, base::Symbol)
        -> Vector{Float64}

Return the population-level trajectory of base state `base` (`:S` or
`:I`) by summing over the neighbour-count axis.
"""
function neighbourhood_compartment(sys::NeighbourhoodSystem, sol, base::Symbol)
    base in (:S, :I) || throw(ArgumentError(
        "neighbourhood_compartment: base must be :S or :I (got :$base)."))
    rows = [sys.index[(base, y)] for y in 0:sys.k]
    return [sum(u[i] for i in rows) for u in sol.u]
end
