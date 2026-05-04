# neighbourhood_symbolic.jl — Symbolics.jl-based independent validator
# for the n=2 neighbourhood-closure RHS implemented in
# `neighbourhood_based.jl`.
#
# Purpose: emit a symbolic ODE RHS that, by independent first-principles
# derivation from Eqs 9–10 of Keeling, House, Cooper & Pellis (2016),
# matches the numeric RHS produced by `generate_neighbourhood(model, k, 2)`
# element-wise at every state vector — including near the disease-free
# equilibrium where the `safe_ratio` semantics matter.
#
# Independence: this file does NOT call any of `_build_neighbourhood_rhs`
# nor the numeric IC builder.  Layout is re-derived from `k` only; closure
# expressions are independently typed in below.
#
# Closure semantics: `safe_ratio_sym(num, den)` from `motif_symbolic.jl`
# is reused — it compiles to `ifelse(den < 1e-12, 0, num/den)` so the
# symbolic RHS yields exactly zero (not NaN) when `den` underflows.

using Symbolics: @variables, build_function, Num

"""
    build_neighbourhood_symbolic_rhs(k::Integer)
        -> (rhs_sym!, var_keys::Vector{Tuple{Symbol,Int}}, raw_exprs::Vector)

Independently build a symbolic RHS for the n = 2 neighbourhood-closure
SIS dynamics on a `k`-regular network.

Returns a tuple `(rhs_sym!, var_keys, raw_exprs)`:

* `rhs_sym!(du, u, p, t)` — compiled function consuming a 2(k+1)-vector
  `u` in the canonical `[S_0, …, S_k, I_0, …, I_k]` order and a tuple
  `p = (β, γ)`.
* `var_keys::Vector{Tuple{Symbol,Int}}` — the same canonical ordering as
  `(:S, 0), …, (:S, k), (:I, 0), …, (:I, k)`.
* `raw_exprs::Vector{Num}` — the symbolic RHS expressions in the same
  order (useful for inspection / debugging).
"""
function build_neighbourhood_symbolic_rhs(k::Integer)
    k = Int(k)
    k ≥ 1 || throw(ArgumentError(
        "build_neighbourhood_symbolic_rhs: k must be ≥ 1, got $k."))

    n_var = 2 * (k + 1)
    Symbolics.@variables β_sym γ_sym
    u_sym = [first(@eval Symbolics.@variables $(Symbol("u_", i)))
             for i in 1:n_var]

    var_keys = Tuple{Symbol,Int}[]
    for y in 0:k; push!(var_keys, (:S, y)); end
    for y in 0:k; push!(var_keys, (:I, y)); end
    idx = Dict(k_ => i for (i, k_) in enumerate(var_keys))

    Sy(y) = u_sym[idx[(:S, y)]]
    Iy(y) = u_sym[idx[(:I, y)]]
    bnd(y, base) = (0 ≤ y ≤ k) ?
        (base === :S ? Sy(y) : Iy(y)) :
        zero(β_sym)

    # Closure ratios — independently re-derived from Eq 10 of the paper.
    # Numerator / denominator construction:
    #   ω_S = β · Σ y(k−y) S_y / Σ (k−y) S_y
    #   ω_I = β · Σ y²     S_y / Σ y       S_y
    num_SS = sum(y * (k - y) * Sy(y) for y in 0:k)
    den_SS = sum((k - y) * Sy(y) for y in 0:k)
    num_IS = sum(y * y * Sy(y) for y in 0:k)
    den_IS = sum(y * Sy(y) for y in 0:k)
    ω_S = β_sym * safe_ratio_sym(num_SS, den_SS)
    ω_I = β_sym * safe_ratio_sym(num_IS, den_IS)

    raw_exprs = Vector{Num}(undef, n_var)

    for y in 0:k
        # dS_y from Eq 9
        dS  = γ_sym * Iy(y) - β_sym * y * Sy(y)
        dS += γ_sym * ((y + 1) * bnd(y + 1, :S) - y * Sy(y))
        dS += ω_S   * ((k - y + 1) * bnd(y - 1, :S) - (k - y) * Sy(y))
        raw_exprs[idx[(:S, y)]] = dS

        # dI_y from Eq 9
        dI  = β_sym * y * Sy(y) - γ_sym * Iy(y)
        dI += γ_sym * ((y + 1) * bnd(y + 1, :I) - y * Iy(y))
        dI += ω_I   * ((k - y + 1) * bnd(y - 1, :I) - (k - y) * Iy(y))
        raw_exprs[idx[(:I, y)]] = dI
    end

    # Compile to in-place RHS over (u, p) where p = (β, γ).
    # build_function with target Vector returns (oop, ip).
    p_sym = [β_sym, γ_sym]
    _, rhs_ip = build_function(raw_exprs, u_sym, p_sym; expression = Val{false})

    function rhs_sym!(du, u, p, t)
        # p is allowed to be either a NamedTuple (β, γ, …) or a Tuple (β, γ).
        β = p isa NamedTuple ? p.β : p[1]
        γ = p isa NamedTuple ? p.γ : p[2]
        rhs_ip(du, u, (β, γ))
        return nothing
    end

    return rhs_sym!, var_keys, raw_exprs
end
