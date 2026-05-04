# closures.jl — Triple closure approximations
#
# In pairwise models, the equation for d[AB]/dt depends on triples [ABC].
# Closure approximations express [ABC] in terms of pairs [AB], [BC], [AC]
# and singles [A], [B], [C]. Different closures make different assumptions
# about local network structure.

"""
    ClosureMethod

Abstract type for triple closure approximations.
"""
abstract type ClosureMethod end

"""
    BernoulliClosure()

Ordinary pair approximation (OPA). Assumes conditional independence of
neighbor states. No clustering correction.

    [ABC] ≈ (n-1)/n · [AB][BC] / [B]

Reference: Rand (1999), Matsuda et al (1992)
"""
struct BernoulliClosure <: ClosureMethod end

"""
    KeelingClosure()

Keeling's closure incorporating clustering coefficient ϕ.
Extends the Bernoulli closure with a triangular correction term.

    [ABC] ≈ (n-1)/n · [AB][BC]/[B] · ( (1-ϕ) + ϕ · N·[AC]/(n·[A][C]) )

Reference: Keeling (1999), Morris (1997)
"""
struct KeelingClosure <: ClosureMethod end

"""
    BarnardClosure()

Improved closure from Barnard et al. (2019) that conserves pairs.
Uses a normalized correction ensuring Σ_A [ABC] = (n-1)[BC].

    [ABC] = (n-1) · ( (1-ϕ)·[AB][BC]/(n[B]) + ϕ·[AB][BC][CA] / ([A]·Σ_a [aB][aC]/[a]) )

Reference: Barnard et al. (2019), J. Math. Biol. 79:823–860
"""
struct BarnardClosure <: ClosureMethod end

"""
    EamesClosure()

Placeholder for Eames' hybrid closure for populations with regular
(network) and random (mass-action) contacts.

    [ABC] ≈ (k_N-1)/k_N · [AB][BC]/[B] · ( (1-ϕ) + ϕ·[AC]·P/(k_N·[A][C]) )

The current package does not encode the additional regular/random-contact
parameters required by the Eames model, so `generate_pairwise` rejects this
closure instead of silently using a misleading surrogate.

Reference: Eames (2008), Theor. Popul. Biol. 73:104–111
"""
struct EamesClosure <: ClosureMethod end

"""
    PowerClosure(p)

Power closure with exponent p. Interpolates between mean-field (p=1)
and Bernoulli (p→∞).

    [ABC] ≈ (n-1)/n · [AB]^p · [BC]^p / [B]^(2p-1)

Reference: Rogers (2011)
"""
struct PowerClosure <: ClosureMethod
    p::Float64
end

"""
    KirkwoodClosure()

Kirkwood superposition closure for triples on specific graphs.
For a path k-i-j, uses conditional independence through the middle node i:

    ⟨A_k B_i C_j⟩ ≈ [A_kB_i] × [B_iC_j] / ⟨B_i⟩

This is exact when there is no edge (k,j), i.e., on tree graphs.
`generate_pair_based` uses this closure for graph-instance SIR models.
"""
struct KirkwoodClosure <: ClosureMethod end

# ─── Symbolic triple closure computation ──────────────────────────────────────

"""
    triple_closure(A, B, C, pairs, singles, net, closure)

Compute the symbolic expression for [ABC] given:
- `A`, `B`, `C` — compartment name symbols
- `pairs` — Dict mapping (X,Y) => symbolic pair variable [XY]
- `singles` — Dict mapping X => symbolic single variable [X]
- `net` — NetworkStructure
- `closure` — ClosureMethod

Returns a Symbolics expression.
"""
function triple_closure end

_safe_div(num, denom) = ifelse(denom == 0, 0, num / denom)

function triple_closure(A::Symbol, B::Symbol, C::Symbol,
                         pairs::Dict, singles::Dict,
                         net::HomogeneousNetwork, ::BernoulliClosure)
    n = net.n
    AB = _get_pair(pairs, A, B)
    BC = _get_pair(pairs, B, C)
    B_s = singles[B]
    return ((n - 1) / n) * _safe_div(AB * BC, B_s)
end

function triple_closure(A::Symbol, B::Symbol, C::Symbol,
                         pairs::Dict, singles::Dict,
                         net::HomogeneousNetwork, ::KeelingClosure)
    n = net.n
    ϕ = net.ϕ
    N = net.N
    AB = _get_pair(pairs, A, B)
    BC = _get_pair(pairs, B, C)
    AC = _get_pair(pairs, A, C)
    A_s = singles[A]
    B_s = singles[B]
    C_s = singles[C]

    pair_term = _safe_div(AB * BC, B_s)
    clustering_correction = (1 - ϕ) + ϕ * _safe_div(N * AC, n * A_s * C_s)
    return ((n - 1) / n) * pair_term * clustering_correction
end

function triple_closure(A::Symbol, B::Symbol, C::Symbol,
                         pairs::Dict, singles::Dict,
                         net::HomogeneousNetwork, ::BarnardClosure)
    n = net.n
    ϕ = net.ϕ
    AB = _get_pair(pairs, A, B)
    BC = _get_pair(pairs, B, C)
    CA = _get_pair(pairs, C, A)
    A_s = singles[A]
    B_s = singles[B]
    C_s = singles[C]

    # Unclustered part
    open_term = (1 - ϕ) * _safe_div(AB * BC, n * B_s)

    # Clustered part: [AB][BC][CA] / ([A] · Σ_a [aB][aC]/[a])
    # The denominator sums over all compartment states 'a'
    # For simplicity, we compute using all singles
    denom_sum = sum(
        _safe_div(_get_pair(pairs, a, B) * _get_pair(pairs, a, C), singles[a])
        for a in keys(singles)
    )
    closed_term = ϕ * _safe_div(AB * BC * CA, A_s * denom_sum)

    return (n - 1) * (open_term + closed_term)
end

function triple_closure(::Symbol, ::Symbol, ::Symbol,
                         ::Dict, ::Dict,
                         ::HomogeneousNetwork, ::EamesClosure)
    throw(ArgumentError(
        "EamesClosure is not supported by NodeBasedModels.jl because the package does not encode the separate regular and random contact rates required by the Eames hybrid closure."))
end

function triple_closure(A::Symbol, B::Symbol, C::Symbol,
                         pairs::Dict, singles::Dict,
                         net::HomogeneousNetwork, cl::PowerClosure)
    n = net.n
    p = cl.p
    AB = _get_pair(pairs, A, B)
    BC = _get_pair(pairs, B, C)
    B_s = singles[B]
    return ((n - 1) / n) * _safe_div(AB^p * BC^p, B_s^(2p - 1))
end

# Heterogeneous network closure (degree-stratified)
function triple_closure(A::Symbol, B::Symbol, C::Symbol,
                         pairs::Dict, singles::Dict,
                         net::HeterogeneousNetwork, ::BernoulliClosure)
    # For compact heterogeneous: use excess degree ratio
    q = net.excess_degree
    mk = net.mean_degree
    AB = _get_pair(pairs, A, B)
    BC = _get_pair(pairs, B, C)
    B_s = singles[B]
    return q * _safe_div(AB * BC, mk * B_s)
end

function triple_closure(A::Symbol, B::Symbol, C::Symbol,
                         pairs::Dict, singles::Dict,
                         net::HeterogeneousNetwork, ::KeelingClosure)
    q = net.excess_degree
    mk = net.mean_degree
    ϕ = net.ϕ
    N = net.N
    AB = _get_pair(pairs, A, B)
    BC = _get_pair(pairs, B, C)
    AC = _get_pair(pairs, A, C)
    A_s = singles[A]
    B_s = singles[B]
    C_s = singles[C]

    pair_term = _safe_div(AB * BC, B_s)
    clustering_correction = (1 - ϕ) + ϕ * _safe_div(N * AC, mk * A_s * C_s)
    return (q / mk) * pair_term * clustering_correction
end

function triple_closure(::Symbol, ::Symbol, ::Symbol,
                         ::Dict, ::Dict,
                         ::HeterogeneousNetwork, ::BarnardClosure)
    throw(ArgumentError(
        "BarnardClosure is currently implemented only for HomogeneousNetwork."))
end

function triple_closure(::Symbol, ::Symbol, ::Symbol,
                         ::Dict, ::Dict,
                         ::HeterogeneousNetwork, ::EamesClosure)
    throw(ArgumentError(
        "EamesClosure is not supported by NodeBasedModels.jl because the package does not encode the separate regular and random contact rates required by the Eames hybrid closure."))
end

function triple_closure(::Symbol, ::Symbol, ::Symbol,
                         ::Dict, ::Dict,
                         ::HeterogeneousNetwork, ::PowerClosure)
    throw(ArgumentError(
        "PowerClosure is currently implemented only for HomogeneousNetwork."))
end

# ─── Helper ────────────────────────────────────────────────────────────────────

"""Get pair variable, treating [AB] and [BA] as the same (undirected)."""
function _get_pair(pairs::Dict, A::Symbol, B::Symbol)
    if haskey(pairs, (A, B))
        return pairs[(A, B)]
    elseif haskey(pairs, (B, A))
        return pairs[(B, A)]
    else
        error("No pair variable for ($A, $B)")
    end
end
