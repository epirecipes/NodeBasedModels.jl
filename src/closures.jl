# closures.jl — Triple closure approximations (owner: WP24)
#
# In pairwise models, the equation for d[AB]/dt depends on triples [ABC].
# Closure approximations express [ABC] in terms of pairs [AB], [BC], [AC]
# and singles [A], [B], [C]. Different closures make different assumptions
# about local network structure.
#
# Every closure here is homogeneous of degree 1 in (singles, pairs): scaling the state from
# fractions to counts scales [ABC] by the same factor. KeelingClosure's population N is
# therefore Σ_X [X], read from the state, not the `N` field of the network structure (verified
# issue B03: 0.1 used `network.N`, which need not match the scale of the state).

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

N is the number of nodes in the units of the state, N = Σ_X [X] (the correlation factor
C_AC = N[AC]/(n[A][C]) of Keeling 1999 and Barnard's thesis §4.2). It is read from the singles,
so the closure gives the same per-capita dynamics for fractions and for counts, whatever the
`N` field of the network structure says (verified issue B03). On a heterogeneous network n is
the mean degree and (n − 1)/n becomes q/⟨k⟩ with q the mean excess degree.

Reference: Keeling (1999), Morris (1997)
"""
struct KeelingClosure <: ClosureMethod end

"""
    BarnardClosure()

Improved closure from Barnard et al. (2019) that conserves pairs. Written for the triple [ASI]
of Barnard's thesis (eq. 4.23): the B–C link is the one the infection crosses (C = I is the
infector of the middle node B = S) and A is the state of another neighbour of B,

    [ABC] = (n-1) · ( (1-ϕ)·[AB][BC]/(n[B]) + ϕ·[AB][BC][CA] / ([A]·Σ_a [aB][aC]/[a]) )

The clustered term is the probability [AB]C_AC/(n[B]) of the neighbour's state A, normalised over
A, so that Σ_A [ABC] = (n-1)[BC] for every B–C link. It is not symmetric in A and C: in the
pairwise equations the infector is the **last** argument of [`triple_closure`](@ref) (the event
"the X end of an (X, Y) pair is infected by a Z neighbour" has the triple [Y X Z]).

Reference: Barnard et al. (2019), J. Math. Biol. 79:823–860; Barnard (2019), PhD thesis, §4.3.2
"""
struct BarnardClosure <: ClosureMethod end

"""
    EamesClosure()

Removed in NodeBasedModels 0.2: every use throws an `ArgumentError`. The type is kept only so
that 0.1 code fails with a migration message instead of an `UndefVarError`.

Eames' hybrid closure for populations with regular (network) and random (mass-action) contacts,

    [ABC] ≈ (k_N-1)/k_N · [AB][BC]/[B] · ( (1-ϕ) + ϕ·[AC]·P/(k_N·[A][C]) ),

needs the separate regular and random contact rates, which NodeBasedModels never encoded, and it
was never validated. Use [`KeelingClosure`](@ref) or [`BarnardClosure`](@ref) for clustered
networks.

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

The closed triple [ABC] (a path A–B–C with B in the middle). In the pairwise equations C is the
infector: the X end of an (X, Y) pair is infected by a Z neighbour at rate τ[Y X Z]. The
Bernoulli, Keeling and power closures are symmetric in A and C; `BarnardClosure` is not (it
normalises over the state A, Σ_A [ABC] = (n − 1)[BC]). The arguments are:
- `A`, `B`, `C` — compartment name symbols
- `pairs` — Dict mapping (X,Y) => pair variable [XY] (either order may be stored)
- `singles` — Dict mapping X => single variable [X], for **every** compartment of the model:
  `KeelingClosure` takes the population as N = Σ_X [X] and `BarnardClosure` sums over all
  compartments, so a partial `singles` gives a wrong closure
- `net` — NetworkStructure
- `closure` — ClosureMethod

Returns a Symbolics expression for symbolic variables, or a number for numeric ones (the
analysis functions evaluate closures numerically). Every closure except `PowerClosure(p ≠ 1)`
is homogeneous of degree 1 in (`singles`, `pairs`).
"""
function triple_closure end

_safe_div(num, denom) = ifelse(denom == 0, 0, num / denom)

# The population N = Σ_X [X] in the units of the state, summed in a fixed (sorted) order.
_population(singles::Dict) = sum(singles[X] for X in sort!(collect(keys(singles))))

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
    N = _population(singles)   # not net.N (B03)
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

    # Clustered part: [AB][BC][CA] / ([A] · Σ_a [aB][aC]/[a]), the probability of state A for the
    # other neighbour of B normalised over every compartment a (C is the infector, so this is
    # not symmetric in A and C; see the BarnardClosure docstring)
    denom_sum = sum(
        _safe_div(_get_pair(pairs, a, B) * _get_pair(pairs, a, C), singles[a])
        for a in keys(singles)
    )
    closed_term = ϕ * _safe_div(AB * BC * CA, A_s * denom_sum)

    return (n - 1) * (open_term + closed_term)
end

function triple_closure(::Symbol, ::Symbol, ::Symbol,
                         ::Dict, ::Dict,
                         ::NetworkStructure, ::EamesClosure)
    throw(ArgumentError(_EAMES_REMOVED))
end

const _EAMES_REMOVED =
    "EamesClosure was removed in NodeBasedModels 0.2: the hybrid closure of Eames (2008) needs " *
    "separate regular and random contact rates that NodeBasedModels never encoded, and it was " *
    "never validated. Use KeelingClosure (clustering ϕ) or BarnardClosure."

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
    N = _population(singles)   # not net.N (B03)
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
                         ::HeterogeneousNetwork, ::PowerClosure)
    throw(ArgumentError(
        "PowerClosure is currently implemented only for HomogeneousNetwork."))
end

# ─── Helper ────────────────────────────────────────────────────────────────────

# The closed triple of the event "the X end of an (X, other) pair is infected by a Z neighbour":
# [other X Z], with the infector Z last (Barnard's [ASI] with A = other, S = X, I = Z). For
# BarnardClosure the order matters; the pairwise builders should use this helper rather than
# ordering the arguments of `triple_closure` themselves.
_infection_triple(Z::Symbol, X::Symbol, other::Symbol, pairs::Dict, singles::Dict, net, closure) =
    triple_closure(other, X, Z, pairs, singles, net, closure)

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
