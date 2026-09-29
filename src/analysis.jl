# analysis.jl — threshold quantities of the population pairwise models (owner: WP24)
#
# The epidemic threshold τ_c, the early growth rate r and the threshold ratio R = 1 + r/γ of
# the pairwise SIR and SIS models that `generate_pairwise` / `node_based` build, for the
# Bernoulli, Keeling and Barnard closures on homogeneous (k-regular) and heterogeneous networks.
#
# SIR and SIS are separate (verified issue B01: 0.1 used SIR formulas on homogeneous networks and
# SIS formulas on heterogeneous ones, whatever the model). With q = ⟨k(k−1)⟩/⟨k⟩ the mean excess
# degree (n − 1 on an n-regular network) the Bernoulli closure gives
#
#   SIR: r = τ(q − 1) − γ,                              τ_c = γ/(q − 1)
#   SIS: r = (a − 2γ + √((a − 2γ)² + 8γ(a + τ)))/2,     τ_c = γ/q,      a = τ(q − 1) − γ
#
# (the leading eigenvalue of the [SI] block, and of the [SI], [II] block; Keeling 1999, Kiss,
# Miller & Simon 2017 §4). The clustered closures are singular at the disease-free state
# ([II]/[I]² is 0/0), so r comes from the quasi-equilibrium of the fast variables α = [SI]/[I],
# δ = [II]/[I] (Barnard, thesis ch. 4; Barnard et al. 2019), with, for Barnard's SIR closure, the
# recovered-block ratios η = [IR]/[I] and ρ = [SR]/[R] that its denominator Σ_a [aS][aI]/[a]
# reads. For Keeling's closure the fast variables are a planar system whose attractors are known
# in closed form (a cubic's roots, the origin, and for SIS δ = ∞, where [II] outgrows [I]); r is
# that of the attractor reached from a random seed. The threshold is where r changes sign; the
# thresholds of Keeling's closure and of Barnard's SIS closure are closed forms. Both clustered
# closures are the Bernoulli closure when ϕ = 0.

"""
    PAIRWISE_DYNAMICS

The dynamics of the pairwise threshold analysis: `:SIR` (S → I → R) and `:SIS` (S → I → S).
`basic_reproduction_number`, `epidemic_threshold` and `early_growth_rate` on a network structure
take one of them as the keyword `dynamics`.
"""
const PAIRWISE_DYNAMICS = (:SIR, :SIS)

# ─── Dynamics ────────────────────────────────────────────────────────────────

function _check_dynamics(d::Symbol)
    d in PAIRWISE_DYNAMICS || throw(ArgumentError(
        "dynamics must be one of $(PAIRWISE_DYNAMICS); got $(repr(d))"))
    return d
end

# The dynamics of the network-level methods: SIR by default on a homogeneous network (the 0.1
# meaning); explicit on a heterogeneous network, where 0.1 returned the SIS R₀ and threshold but
# the SIR growth rate (verified issue B01).
_resolve_dynamics(::NetworkStructure, d::Symbol, fname) = _check_dynamics(d)
_resolve_dynamics(::HomogeneousNetwork, ::Nothing, fname) = :SIR
function _resolve_dynamics(net::NetworkStructure, ::Nothing, fname)
    q = _excess(net)
    throw(ArgumentError(
        "$(fname) on a $(nameof(typeof(net))) needs `dynamics = :SIR` or `dynamics = :SIS`. " *
        "NodeBasedModels 0.1 returned the SIS R₀ and threshold on heterogeneous networks but the " *
        "SIR growth rate (verified issue B01); the two thresholds differ, τ_c = γ/(q − 1) for SIR " *
        "and γ/q for SIS, with q = ⟨k(k−1)⟩/⟨k⟩ = $(q) here."))
end

"""
    _pairwise_dynamics(model::CompartmentalModel) -> (dynamics, infection, recovery)

Classify a model for the closed-form pairwise analysis, strictly: exactly one infection
transition X → Y whose only infector is Y (X not infectious), exactly one spontaneous transition,
out of Y, and no other compartment. It is `:SIS` if the spontaneous transition goes back to X,
and `:SIR` if it goes to a third compartment. Anything else (SEIR, SIRS, two strains, a
vaccination exit, …) is an `ArgumentError`: those thresholds are not the SIR or SIS formulas
(for SIRS, τ_c = γ/(q − 1 + ε/(γ + ε)); verified issue B01).
"""
function _pairwise_dynamics(model::CompartmentalModel)
    inf = infection_transitions(model)
    spon = spontaneous_transitions(model)
    fail(why) = throw(ArgumentError(
        "the pairwise threshold analysis covers the SIR model S → I → R and the SIS model " *
        "S → I → S (one infection transition and one spontaneous transition, out of the " *
        "infected compartment); model :$(model.name) $(why)"))
    length(inf) == 1 || fail("has $(length(inf)) infection transitions")
    length(spon) == 1 || fail("has $(length(spon)) spontaneous transitions")
    tr, rec = only(inf), only(spon)
    X, Y = tr.from, tr.to
    infectors(model, tr) == [Y] && model.infectious_compartments == [Y] ||
        fail("does not infect through its infected compartment $(Y) alone")
    rec.from === Y || fail("recovers from $(rec.from), not from the infected compartment $(Y)")
    if rec.to === X
        length(model.compartment_names) == 2 || fail("has compartments besides $(X) and $(Y)")
        return (:SIS, tr, rec)
    end
    length(model.compartment_names) == 3 ||
        fail("has compartments besides $(X), $(Y) and $(rec.to)")
    return (:SIR, tr, rec)
end

# ─── Network constants ───────────────────────────────────────────────────────

# q = ⟨k(k−1)⟩/⟨k⟩ (the mean excess degree) and ⟨k⟩: n − 1 and n on an n-regular network.
_excess(net::HomogeneousNetwork) = Float64(net.n - 1)
_excess(net::HeterogeneousNetwork) = net.excess_degree
_excess(net::NetworkStructure) = throw(ArgumentError(
    "the pairwise threshold analysis needs a HomogeneousNetwork or a HeterogeneousNetwork; got " *
    "$(nameof(typeof(net)))"))
_meandeg(net::HomogeneousNetwork) = Float64(net.n)
_meandeg(net::HeterogeneousNetwork) = net.mean_degree
_meandeg(net::NetworkStructure) = _excess(net)   # throws

# ─── Small nonlinear solver (fast-variable quasi-equilibria) ─────────────────

# Newton's method with a central-difference Jacobian and step halving on ‖F‖.
function _newton(F, x0::Vector{Float64}; tol = 1e-13, maxiter = 100)
    x = copy(x0)
    fx = F(x)
    m = length(x)
    for _ in 1:maxiter
        norm(fx) <= tol * (1 + norm(x)) && return x
        J = zeros(m, m)
        for j in 1:m
            h = 1e-7 * max(1.0, abs(x[j]))
            e = zeros(m)
            e[j] = h
            J[:, j] = (F(x .+ e) .- F(x .- e)) ./ (2h)
        end
        dx = J \ fx
        λ = 1.0
        while true
            xn = x .- λ .* dx
            fn = F(xn)
            if all(isfinite, fn) && norm(fn) < norm(fx)
                x, fx = xn, fn
                break
            end
            λ /= 2
            λ < 1e-12 && (norm(fx) <= 1e-9 * (1 + norm(x)) ? (return x) :
                          throw(ArgumentError("fast-variable quasi-equilibrium: Newton's method " *
                                              "stalled at $(x) (residual $(norm(fx)))")))
        end
    end
    norm(fx) <= 1e-9 * (1 + norm(x)) && return x
    throw(ArgumentError("fast-variable quasi-equilibrium: Newton's method did not converge " *
                        "(residual $(norm(fx)))"))
end

# Follow the root of F(x, s) from the known root x0 at s = 0 to s = 1 (natural continuation), so
# the clustered quasi-equilibrium is the branch that continues the unclustered one. Returns
# `nothing` if the branch leaves the region where `admissible(x)` holds.
function _continuation(F, x0::Vector{Float64}; steps::Int = 32, admissible = x -> true)
    x = x0
    for k in 1:steps
        s = k / steps
        x = _newton(z -> F(z, s), x)
        admissible(x) || return nothing
    end
    return x
end

# ─── Early growth rate r(τ) ──────────────────────────────────────────────────

# The growth rate of [I] at the disease-free state: the dominant rate of the infected block, which
# is never below −γ (the rate of [I] itself when the pairs decay faster).

function _growth(::Val{:SIR}, net::NetworkStructure, ::BernoulliClosure, τ, γ)
    _meandeg(net) > 0 || return -γ
    return max(τ * (_excess(net) - 1) - γ, -γ)
end

function _growth(::Val{:SIS}, net::NetworkStructure, ::BernoulliClosure, τ, γ)
    _meandeg(net) > 0 || return -γ
    a = τ * (_excess(net) - 1) - γ
    return (a - 2γ + sqrt((a - 2γ)^2 + 8γ * (a + τ))) / 2
end

# Keeling's closure. At the disease-free state ([S] = N, [SS] = ⟨k⟩N at leading order) its
# triples are [SSI] ≈ qα(1 − ϕ + ϕα/⟨k⟩)[I] and [ISI] ≈ cα²δ[I], c = qϕ/⟨k⟩², in the fast
# variables α = [SI]/[I] and δ = [II]/[I]. The pairwise equations divided by [I] are then the
# autonomous system (σ = 1 for SIS, 0 for SIR)
#   dα/dt = τα(A + Bα) − τcα²δ + σγδ,   dδ/dt = δ(2τcα² − γ − τα) + 2τα,
# A = q(1 − ϕ) − 1, B = qϕ/⟨k⟩ − 1, and [I] grows at the rate r = τα − γ. The early growth rate is
# r at the attractor that (α, δ) reaches. The attractors are
# - the stable fixed points with α > 0 and δ > 0: δ = 2τα/Δ(α) with Δ(α) = γ + τα − 2τcα² > 0,
#   and α a root of the cubic (A + Bα)Δ(α) + 2σγ − 2τcα² = 0;
# - for SIR with A < 0 (or A = 0 and B ≤ 0), the origin: [SI]/[I] → 0 and [I] decays at the
#   rate γ;
# - for SIS with τ ≤ γc, δ → ∞: [II] outgrows [I], the terms in δ of dα/dt balance at
#   α∞ = √(γ/(τc)), and δ keeps growing (at the rate γ − τα∞ ≥ 0), so r = √(τγ/c) − γ ≤ 0. The
#   fixed-point branch reaches δ = ∞ only at τ = γc, α = 1/c, where Δ = 0 and r = 0.
# When there are several attractors, r is that of the one the fast variables reach from a random
# seed, (α, δ) = (⟨k⟩, 0) (the product-state seeding of generate_pairwise).

_keeling_constants(net::NetworkStructure) =
    (A = _excess(net) * (1 - net.ϕ) - 1, B = _excess(net) * net.ϕ / _meandeg(net) - 1,
     c = _excess(net) * net.ϕ / _meandeg(net)^2, mk = _meandeg(net))

function _keeling_field!(du, u, K, τ, γ, σ)
    α, δ = u[1], u[2]
    du[1] = τ * α * (K.A + K.B * α) - τ * K.c * α^2 * δ + σ * γ * δ
    du[2] = δ * (2τ * K.c * α^2 - γ - τ * α) + 2τ * α
    return du
end

# The real roots of the polynomial with ascending coefficients `p` (the eigenvalues of its
# companion matrix with a negligible imaginary part).
function _real_roots(p::AbstractVector{Float64})
    m = findlast(!iszero, p)
    (m === nothing || m == 1) && return Float64[]
    n = m - 1
    C = zeros(n, n)
    for i in 2:n
        C[i, i-1] = 1.0
    end
    C[:, n] .= -p[1:n] ./ p[m]
    return [real(z) for z in eigvals(C) if abs(imag(z)) <= 1e-8 * (1 + abs(z))]
end

# The fixed points (α, δ) of the fast variables with α > 0 and δ > 0, and their stability.
function _keeling_fixed_points(K, τ, γ, σ)
    A, B, c = K.A, K.B, K.c
    p = [A * γ + 2σ * γ, A * τ + B * γ, B * τ - 2τ * c * (A + 1), -2τ * c * B]
    P(α) = evalpoly(α, p)
    dP(α) = evalpoly(α, (p[2], 2p[3], 3p[4]))
    pts = NamedTuple{(:α, :δ, :stable),Tuple{Float64,Float64,Bool}}[]
    for α0 in _real_roots(p)
        α = α0
        for _ in 1:4                       # polish the root (Newton on the cubic)
            d = dP(α)
            d == 0 && break
            αn = α - P(α) / d
            abs(P(αn)) < abs(P(α)) || break
            α = αn
        end
        Δ = γ + τ * α - 2τ * c * α^2
        # (a root at α = 0, for SIR with A = 0, is the origin, which is not one of these)
        (α > 1e-9 * (1 + K.mk) && Δ > 0) || continue
        δ = 2τ * α / Δ
        # the Jacobian of the field at (α, δ)
        fα, fδ = τ * A + 2τ * B * α - 2τ * c * α * δ, σ * γ - τ * c * α^2
        gα, gδ = δ * (4τ * c * α - τ) + 2τ, 2τ * c * α^2 - γ - τ * α
        push!(pts, (α = α, δ = δ, stable = fα + gδ < 0 && fα * gδ - fδ * gα > 0))
    end
    return pts
end

# The attractors of the fast variables: (kind, point, r) with kind :fixed, :origin or :infinite.
function _keeling_attractors(K, τ, γ, σ)
    pts = _keeling_fixed_points(K, τ, γ, σ)
    atts = [(kind = :fixed, x = [p.α, p.δ], r = τ * p.α - γ) for p in pts if p.stable]
    # the origin attracts for SIR when dα/dt < 0 near it: A < 0, or A = 0 and B ≤ 0
    σ == 0 && (K.A < 0 || (K.A == 0 && K.B <= 0)) &&
        push!(atts, (kind = :origin, x = [0.0, 0.0], r = -γ))
    σ == 1 && τ <= γ * K.c &&
        push!(atts, (kind = :infinite, x = [sqrt(γ / (τ * K.c)), Inf], r = sqrt(τ * γ / K.c) - γ))
    return atts, pts
end

# The attractor that the fast variables reach from the seed (⟨k⟩, 0), by integrating their field
# until the state is at a stable fixed point, at the origin, or beyond every fixed point in δ.
function _keeling_seeded_attractor(K, τ, γ, σ, atts, pts)
    f!(du, u, _, t) = (_keeling_field!(du, u, K, τ, γ, σ); nothing)
    δbig = 1e6 * (1 + maximum((p.δ for p in pts); init = K.mk))
    αsmall = 1e-6 * minimum((p.α for p in pts); init = K.mk)
    u = [K.mk, 0.0]
    Δt = 10 / γ
    for _ in 1:2000
        sol = solve(ODEProblem(f!, u, (0.0, Δt)); reltol = 1e-10, abstol = 1e-12,
                    save_everystep = false)
        sol.t[end] == Δt || break
        u = sol.u[end]
        for a in atts
            reached = a.kind === :fixed ? norm(u .- a.x, Inf) <= 1e-7 * (1 + norm(a.x, Inf)) :
                      a.kind === :infinite ? u[2] >= δbig : u[1] <= αsmall
            reached && return a
        end
    end
    throw(ArgumentError(
        "early_growth_rate: the fast variables [SI]/[I], [II]/[I] of Keeling's closure did not " *
        "settle at one of their attractors from a random seed (τ = $(τ), γ = $(γ))"))
end

function _growth(::Val{D}, net::NetworkStructure, ::KeelingClosure, τ, γ) where {D}
    # ϕ = 0, or q = 0 (every triple is 0): the Bernoulli closure
    (net.ϕ == 0 || _excess(net) == 0) && return _growth(Val(D), net, BernoulliClosure(), τ, γ)
    (τ > 0 && _meandeg(net) > 0) || return -γ
    K = _keeling_constants(net)
    σ = D === :SIS ? 1 : 0
    atts, pts = _keeling_attractors(K, τ, γ, σ)
    isempty(atts) && throw(ArgumentError(
        "early_growth_rate: the fast variables [SI]/[I], [II]/[I] of Keeling's closure have no " *
        "attracting state here (τ = $(τ), γ = $(γ), $(nameof(typeof(net))) with q = " *
        "$(_excess(net)), ⟨k⟩ = $(K.mk), ϕ = $(net.ϕ)): τ is at a bifurcation of the " *
        "quasi-equilibrium"))
    rs = [a.r for a in atts]
    r = (length(atts) == 1 || maximum(rs) - minimum(rs) <= 1e-12 * (1 + γ)) ? first(rs) :
        _keeling_seeded_attractor(K, τ, γ, σ, atts, pts).r
    return max(r, -γ)
end

# The unclustered quasi-equilibrium (α, δ): α = (r + γ)/τ with the Bernoulli r, and
# δ = 2τα/(γ + τα) from E2.
function _unclustered_fast(::Val{D}, net, τ, γ) where {D}
    α = (_growth(Val(D), net, BernoulliClosure(), τ, γ) + γ) / τ
    return [α, 2τ * α / (γ + τ * α)]
end

# Barnard's closure on an n-regular network (thesis eq. 4.23): [ASI] = (n − 1)((1 − ϕ)[AS][SI]/(n[S])
# + ϕ[AS][SI][IA]/([A] Σ_a [aS][aI]/[a])), where the S–I link carries the infection and A is the
# state of the other neighbour of S. At leading order ([S] = N, [SS] = nN, [SI] = α[I],
# [II] = δ[I], [IR] = η[I], [SR] = ρ[R]) the normalisation is Σ_a [aS][aI]/[a] = D[I] with
# D = nα + αδ + ρη (the recovered term [RS][RI]/[R] = ρη[I]; SIS has no R, D = nα + αδ), and
#   [SSI] ≈ (n − 1)α(1 − ϕ + ϕnα/D)[I],  [ISI] ≈ (n − 1)ϕα²δ[I]/D,  [RSI] ≈ (n − 1)ϕαρη[I]/D.
# The equations are E1, E2 as for Keeling with these triples, and for SIR
#   E3: γδ − γη + τ[RSI]/[I] − rη = 0          (d[IR]/dt ÷ [I])
#   E4: γα − τ[RSI]/[I] − γρ = 0               (d[SR]/dt = r[SR] with r[R] = γ[I], ÷ [I])
function _growth(::Val{D}, net::HomogeneousNetwork, ::BarnardClosure, τ, γ) where {D}
    n, ϕ = net.n, net.ϕ
    ϕ == 0 && return _growth(Val(D), net, BernoulliClosure(), τ, γ)
    τ > 0 || return -γ
    D === :SIR && n <= 2 && return -γ
    x0 = _unclustered_fast(Val(D), net, τ, γ)
    if D === :SIS
        function Fsis(x, s)
            α, δ = x
            φ = s * ϕ
            r = τ * α - γ
            ssi = (n - 1) * α * (1 - φ + φ * n / (n + δ))         # [SSI]/[I]
            isi = (n - 1) * φ * α * δ / (n + δ)                   # [ISI]/[I]
            e1 = τ * ssi - τ * isi - τ * α - γ * α + γ * δ - r * α
            e2 = 2τ * isi + 2τ * α - 2γ * δ - r * δ
            return [e1, e2]
        end
        x = _continuation(Fsis, x0; admissible = x -> all(>(0), x))
        x === nothing && throw(ArgumentError(_barnard_branch_lost(net, τ, γ, D)))
        return max(τ * first(x) - γ, -γ)
    end
    α0, δ0 = x0
    r0 = τ * α0 - γ
    function Fsir(x, s)
        α, δ, η, ρ = x
        φ = s * ϕ
        r = τ * α - γ
        Dn = n * α + α * δ + ρ * η
        ssi = (n - 1) * α * (1 - φ + φ * n * α / Dn)          # [SSI]/[I]
        isi = (n - 1) * φ * α^2 * δ / Dn                       # [ISI]/[I]
        rsi = (n - 1) * φ * α * ρ * η / Dn                     # [RSI]/[I]
        e1 = τ * ssi - τ * isi - τ * α - γ * α - r * α
        e2 = 2τ * isi + 2τ * α - 2γ * δ - r * δ
        e3 = γ * δ - γ * η + τ * rsi - r * η
        e4 = γ * α - τ * rsi - γ * ρ
        return [e1, e2, e3, e4]
    end
    x = _continuation(Fsir, [α0, δ0, γ * δ0 / (γ + r0), α0]; admissible = x -> all(>(0), x))
    x === nothing && throw(ArgumentError(_barnard_branch_lost(net, τ, γ, D)))
    r = τ * first(x) - γ
    # Below the threshold the recovered block keeps the history of the early epidemic, so the
    # decay rate is that of the state reached from the seed, not the continued branch.
    r < 0 && (r = _barnard_seeded_sir(n, ϕ, τ, γ))
    return max(r, -γ)
end

# Barnard's SIR closure below the threshold. [R] and [SR] stop changing once [I] has decayed,
# so ρ = [SR]/[R] freezes at a value set by the whole early epidemic, and [I] then decays at the
# rate of the fast-variable quasi-equilibrium (α, δ, η) at that ρ. When (n − 1)(1 − ϕ) < 1 that
# attractor can be the origin α → 0, where [I] decays at the rate γ; the ϕ-continued branch misses
# both effects (it is off by O(γ) there, and by ~1e-2 deep below the threshold elsewhere). So r is
# the asymptotic log-slope of [I] from the product-state seed of generate_pairwise ([S] = N,
# [SS] = nN, [SI] = n[I], all other infected-block pairs 0). The closure is homogeneous of
# degree 1, so at leading order the infected block is autonomous and its seed size is immaterial.
# It is integrated briefly as it is (so [R] > 0), then in the scale-free variables
# (α, δ, η, ρ, ψ) = ([SI], [II], [IR])/[I], [SR]/[R], [I]/[R]:
#   α' = τ(ssi − isi − α) − γα − rα,   δ' = 2τ(isi + α) − 2γδ − rδ,
#   η' = γδ − γη + τ rsi − rη,         ρ' = ψ(γα − τ rsi − γρ),      ψ' = ψ(r − γψ),
# with r = τα − γ and the triples ÷ [I] of the comment above, until r has converged and ψ ≈ 0.
function _barnard_seeded_sir(n, ϕ, τ, γ)
    sdiv(a, b) = b == 0 ? zero(a) : a / b
    function block!(du, u, _, t)                       # [I], [SI], [II], [R], [SR], [IR]
        I, SI, II, R, SR, IR = u
        Dn = n * SI + sdiv(SI * II, I) + sdiv(SR * IR, R)
        ssi = (n - 1) * SI * (1 - ϕ + ϕ * sdiv(n * SI, Dn))
        isi = (n - 1) * ϕ * sdiv(SI^2 * II, I * Dn)
        rsi = (n - 1) * ϕ * sdiv(SR * SI * IR, R * Dn)
        du[1] = τ * SI - γ * I
        du[2] = τ * (ssi - isi - SI) - γ * SI
        du[3] = 2τ * (isi + SI) - 2γ * II
        du[4] = γ * I
        du[5] = γ * SI - τ * rsi
        du[6] = τ * rsi + γ * II - γ * IR
        return nothing
    end
    function fast!(dv, v, _, t)
        α, δ, η, ρ, ψ = v
        Dn = n * α + α * δ + ρ * η
        ssi = (n - 1) * α * (1 - ϕ + ϕ * n * α / Dn)
        isi = (n - 1) * ϕ * α^2 * δ / Dn
        rsi = (n - 1) * ϕ * α * ρ * η / Dn
        r = τ * α - γ
        dv[1] = τ * (ssi - isi - α) - γ * α - r * α
        dv[2] = 2τ * (isi + α) - 2γ * δ - r * δ
        dv[3] = γ * δ - γ * η + τ * rsi - r * η
        dv[4] = ψ * (γ * α - τ * rsi - γ * ρ)
        dv[5] = ψ * (r - γ * ψ)
        return nothing
    end
    t0 = 1 / (τ + γ)
    sol = solve(ODEProblem(block!, [1.0, n, 0.0, 0.0, 0.0, 0.0], (0.0, t0));
                reltol = 1e-12, abstol = 1e-14, save_everystep = false)
    I, SI, II, R, SR, IR = sol.u[end]
    v = [SI / I, II / I, IR / I, SR / R, I / R]
    t, Δt, rprev = t0, 5 / γ, NaN
    # the approach is at the rate |r|, so just below the threshold it takes ~ 20/|r|
    while t < 1e8 / γ
        sol = solve(ODEProblem(fast!, v, (0.0, Δt)); reltol = 1e-11, abstol = 1e-13,
                    save_everystep = false)
        sol.t[end] == Δt || break
        v = sol.u[end]
        t += Δt
        r = τ * v[1] - γ
        abs(r - rprev) <= 1e-10 * (1 + γ) && v[5] <= 1e-9 && return r
        rprev = r
        Δt = max(5 / γ, t / 4)
    end
    isnan(rprev) && throw(ArgumentError(
        "early_growth_rate: the infected block of Barnard's SIR closure could not be integrated " *
        "from the seed (n = $(n), ϕ = $(ϕ), τ = $(τ), γ = $(γ))"))
    return rprev                                       # τ within ~1e-8 of τ_c: r ≈ 0⁻
end

# The continued quasi-equilibrium left the region where the fast variables are positive: the
# unclustered branch does not continue to a physical state (never a wrong-branch number).
_barnard_branch_lost(net, τ, γ, D) =
    "early_growth_rate: the quasi-equilibrium of the fast variables of Barnard's $(D) closure, " *
    "continued in ϕ from the unclustered one, leaves the region where they are positive " *
    "(n = $(net.n), ϕ = $(net.ϕ), τ = $(τ), γ = $(γ)); the growth rate is not available here"

function _growth(::Val{D}, net::NetworkStructure, closure::ClosureMethod, τ, γ) where {D}
    throw(ArgumentError(_unsupported_analysis(net, closure)))
end

_unsupported_analysis(net, closure) =
    "the pairwise threshold analysis is implemented for BernoulliClosure and KeelingClosure on " *
    "HomogeneousNetwork and HeterogeneousNetwork, and for BarnardClosure on HomogeneousNetwork; " *
    "got $(nameof(typeof(closure))) on $(nameof(typeof(net)))" *
    (closure isa PowerClosure ? " (PowerClosure is not homogeneous of degree 1, so it has no " *
                                "linearisation at the disease-free state)" : "")

# ─── Threshold τ_c ───────────────────────────────────────────────────────────

function _threshold(::Val{:SIR}, net::NetworkStructure, ::BernoulliClosure, γ)
    q = _meandeg(net) > 0 ? _excess(net) : 0.0
    return q > 1 ? γ / (q - 1) : Inf
end

function _threshold(::Val{:SIS}, net::NetworkStructure, ::BernoulliClosure, γ)
    q = _meandeg(net) > 0 ? _excess(net) : 0.0
    return q > 0 ? γ / q : Inf
end

# Keeling's closure at r = 0 (τα = γ, x = α = γ/τ; the fast-variable field above). A fixed point
# has δ = 1/(1 − cx), and its α-equation becomes
#   SIR: (1 − cx)(A + Bx) − cx = 0,   SIS: (1 − cx)(A + 1 + Bx) = 0.
# SIR: for A ≥ 0 exactly one root has 0 < x < 1/c (δ > 0), and τ_c = γ/x; for A < 0 and B ≤ 0
# the field has dα/dt < 0, so [SI]/[I] → 0 and there is no epidemic; for A < 0 < B the origin
# and a fixed point attract together (below).
# SIS: the roots are the linear x = (A + 1)/(−B) and the δ-singular x = 1/c (τ = γc, where the
# fixed-point branch meets δ = ∞). To first order in τ − γc the branch through it has
# Δ = 2(τ − γc)/(c(3s + 1)) and r = 2s(τ − γc)/(c(3s + 1)), s = A + 1 + B/c. For s > 0, i.e.
# qϕ(q(1 − ϕ) + ⟨k⟩) > ⟨k⟩² (the linear root has x > 1/c, or B ≥ 0; never on a homogeneous
# network, where the left side is at most n(n − 1)), the branch is admissible above γc with r > 0,
# and below γc the attractor is δ = ∞ with r = √(τγ/c) − γ < 0: τ_c = γc.
# For s ≤ 0, r < 0 on both sides of γc, the linear root has 0 < x ≤ 1/c and τ_c = γ(−B)/(A + 1),
# none when A + 1 = q(1 − ϕ) = 0.
function _threshold(::Val{D}, net::NetworkStructure, ::KeelingClosure, γ) where {D}
    (net.ϕ == 0 || _excess(net) == 0) && return _threshold(Val(D), net, BernoulliClosure(), γ)
    _meandeg(net) > 0 || return Inf
    K = _keeling_constants(net)
    A, B, c = K.A, K.B, K.c
    if D === :SIS
        A + 1 + B / c > 0 && return γ * c
        return A + 1 > 0 ? γ * (-B) / (A + 1) : Inf
    end
    A < 0 < B && throw(ArgumentError(
        "epidemic_threshold: for SIR with Keeling's closure and q(1 − ϕ) < 1 < qϕ/⟨k⟩ (q = " *
        "$(_excess(net)), ⟨k⟩ = $(K.mk), ϕ = $(net.ϕ)) the fast variables [SI]/[I], [II]/[I] " *
        "have two attractors at the disease-free state (they die out, or settle where the " *
        "closed-triangle term of the closure sustains transmission), so whether [I] grows " *
        "depends on the initial condition and is not monotone in τ: there is no threshold. " *
        "early_growth_rate(network, KeelingClosure(), τ, γ; dynamics = :SIR) gives the growth " *
        "rate from a random seed at a given τ"))
    # c·B x² − (B − cA − c) x − A = 0
    a2, a1, a0 = c * B, -(B - c * A - c), -A
    roots = if a2 == 0
        a1 == 0 ? Float64[] : [-a0 / a1]
    else
        disc = a1^2 - 4a2 * a0
        disc < 0 ? Float64[] : [(-a1 - sqrt(disc)) / (2a2), (-a1 + sqrt(disc)) / (2a2)]
    end
    xs = [x for x in roots if x > 0 && 1 - c * x > 0]
    return isempty(xs) ? Inf : γ / minimum(xs)
end

# Barnard's closure at r = 0 (τα = γ, x = α = γ/τ). SIS: E2 gives (n − 1)ϕδ/(n + δ) = δ − 1, i.e.
# δ² + (n − 1)(1 − ϕ)δ − n = 0, and E1 then reduces to x = (n − 1)(1 − ϕ + ϕn/(n + δ)). SIR: the
# equations (E1 ÷ τx, E2 ÷ 2τx, E3 ÷ γ, E4 ÷ γ)
#   (n − 1)(1 − ϕ + ϕnx/D) − (n − 1)ϕxδ/D − 1 − x = 0,  1 − δ + (n − 1)ϕxδ/D = 0,
#   δ − η + (n − 1)ϕρη/D = 0,  x − ρ − (n − 1)ϕρη/D = 0,   D = nx + xδ + ρη,
# continued in ϕ from the unclustered root x = n − 2, δ = η = 1, ρ = n − 2.
function _threshold(::Val{D}, net::HomogeneousNetwork, ::BarnardClosure, γ) where {D}
    n, ϕ = net.n, net.ϕ
    ϕ == 0 && return _threshold(Val(D), net, BernoulliClosure(), γ)
    if D === :SIS
        n > 1 || return Inf
        b = (n - 1) * (1 - ϕ)
        δ = 2n / (b + sqrt(b^2 + 4n))              # the positive root, without cancellation
        return γ / ((n - 1) * (1 - ϕ + ϕ * n / (n + δ)))
    end
    n > 2 || return Inf
    function F(z, s)
        x, δ, η, ρ = z
        φ = s * ϕ
        Dn = n * x + x * δ + ρ * η
        g = (n - 1) * φ * ρ * η / Dn
        return [(n - 1) * (1 - φ + φ * n * x / Dn) - (n - 1) * φ * x * δ / Dn - 1 - x,
                1 - δ + (n - 1) * φ * x * δ / Dn,
                δ - η + g,
                x - ρ - g]
    end
    z = _continuation(F, [n - 2.0, 1.0, 1.0, n - 2.0]; admissible = z -> all(>(0), z))
    # no threshold when the branch leaves x > 0, or ends at x ≈ 0 (a numerically zero root; a
    # guard only: the marginal case (n − 1)(1 − ϕ) = 1, e.g. n = 5, ϕ = 0.75, has τ_c = 0.7779γ)
    return (z === nothing || first(z) <= 1e-7 * n) ? Inf : γ / first(z)
end

function _threshold(::Val{D}, net::NetworkStructure, closure::ClosureMethod, γ) where {D}
    throw(ArgumentError(_unsupported_analysis(net, closure)))
end

# ─── Network-level API ───────────────────────────────────────────────────────

"""
    basic_reproduction_number(network, closure, τ, γ; dynamics = nothing) -> Float64

The threshold ratio R = 1 + r/γ of the pairwise model with per-contact rate `τ` and
recovery rate `γ`, where r is the early growth rate ([`early_growth_rate`](@ref)). R > 1 exactly
when τ > τ_c ([`epidemic_threshold`](@ref)), and R = 1 at τ_c. It is a growth-scaled threshold
ratio, not the generation-based R₀ of a branching process (for SIR on an n-regular network
R = τ(n − 2)/γ, while the per-generation R₀ is (n − 1)τ/(τ + γ); both equal 1 at τ_c).

`dynamics` is `:SIR` or `:SIS` (`NodeBasedModels.PAIRWISE_DYNAMICS`). On a `HomogeneousNetwork` it
defaults to `:SIR`; on a `HeterogeneousNetwork` it must be given (NodeBasedModels 0.1 returned
the SIS value there, verified issue B01). With q = ⟨k(k−1)⟩/⟨k⟩ and a = τ(q − 1) − γ:

- Bernoulli, SIR: R = τ(q − 1)/γ (0 when q ≤ 1);
- Bernoulli, SIS: R = (a + √((2γ − a)² + 8γ(a + τ)))/(2γ);
- `KeelingClosure` (both network types) and `BarnardClosure` (homogeneous): R = τα*/γ with α*
  the quasi-equilibrium of the fast variable [SI]/[I] (Barnard et al. 2019), computed numerically
  (the Bernoulli value when ϕ = 0; see [`early_growth_rate`](@ref) for the attractor it is).
"""
function basic_reproduction_number(network::NetworkStructure, closure::ClosureMethod,
                                    τ::Real, γ::Real; dynamics::Union{Nothing,Symbol} = nothing)
    d = _resolve_dynamics(network, dynamics, "basic_reproduction_number")
    return 1 + _growth(Val(d), network, closure, Float64(τ), Float64(γ)) / γ
end

"""
    epidemic_threshold(network, closure, γ; dynamics = nothing) -> Float64

The critical per-contact rate τ_c at which the disease-free state of the pairwise model loses
stability (the early growth rate changes sign), or `Inf` if the network cannot sustain an
epidemic. `dynamics` is `:SIR` (the default on a `HomogeneousNetwork`) or `:SIS` (required on a
`HeterogeneousNetwork`; verified issue B01). With q = ⟨k(k−1)⟩/⟨k⟩ (n − 1 on an n-regular
network):

- `BernoulliClosure`: τ_c = γ/(q − 1) for SIR and γ/q for SIS, so τ_c = γ/(k − 2) and γ/(k − 1)
  on a k-regular network (0.125 for SIS with k = 3, γ = 1/4);
- `KeelingClosure`, with A = q(1 − ϕ) − 1, B = qϕ/⟨k⟩ − 1 and c = qϕ/⟨k⟩²: for SIR, γ/x with x
  the root in (0, 1/c) of (1 − cx)(A + Bx) = cx (Barnard et al. 2019), e.g. τ_c = 0.31917 for
  n = 6, ϕ = 0.3, γ = 1, and `Inf` when A < 0 and B ≤ 0; for SIS, τ_c = γ(−B)/(A + 1), or τ_c = γc
  when A + 1 + B/c > 0, i.e. qϕ(q(1 − ϕ) + ⟨k⟩) > ⟨k⟩² (strong clustering on a heterogeneous
  network; below γc, [II]/[I] grows without bound and r = √(τγ/c) − γ). For SIR with
  A < 0 < B, i.e. q(1 − ϕ) < 1 < qϕ/⟨k⟩, the early dynamics are bistable and there is no
  threshold: an `ArgumentError`;
- `BarnardClosure` (homogeneous, the improved closure of Barnard's thesis eq. 4.23): for SIS
  τ_c = γ/((n − 1)(1 − ϕ + ϕn/(n + δ))) with δ the positive root of δ² + (n − 1)(1 − ϕ)δ − n = 0;
  for SIR the four fast-variable conditions at r = 0 are solved numerically, e.g. τ_c = 0.29948
  for n = 6, ϕ = 0.3, γ = 1 (below Keeling's: the improved closure spreads more readily).

Both clustered closures are the Bernoulli closure when ϕ = 0.
"""
function epidemic_threshold(network::NetworkStructure, closure::ClosureMethod, γ::Real;
                            dynamics::Union{Nothing,Symbol} = nothing)
    d = _resolve_dynamics(network, dynamics, "epidemic_threshold")
    return _threshold(Val(d), network, closure, Float64(γ))
end

"""
    early_growth_rate(network, closure, τ, γ; dynamics = nothing) -> Float64

The early exponential growth rate r of [I] in the pairwise model at the disease-free state: the
leading eigenvalue of the linearised infected block for the Bernoulli closure, and the growth
rate τα* − γ at the fast-variable quasi-equilibrium for Keeling's and Barnard's closures (whose
closure terms are 0/0 at the disease-free state). It is never below −γ, the decay rate of [I]
itself. `dynamics` as in [`epidemic_threshold`](@ref).

For Keeling's closure, r is that of the attractor that the fast variables α = [SI]/[I],
δ = [II]/[I] reach: a stable fixed point (from the roots of a cubic), for SIR the origin
(r = −γ), or for SIS below τ = γc (c = qϕ/⟨k⟩²) the state δ = ∞, where [II] outgrows [I],
α → √(γ/(τc)) and r = √(τγ/c) − γ. When several of them attract (below the threshold, and for
SIR with q(1 − ϕ) < 1 < qϕ/⟨k⟩), r is that of the one reached from a random seed,
[SI] = ⟨k⟩[I] and [II] = 0 (the seeding of `generate_pairwise`), found by integrating the fast
variables. The approach to δ = ∞ is slow (at the rate −r), so the ODE's log-slope approaches r
slowly there.

Below the threshold, Barnard's SIR closure reads the recovered-block ratio [SR]/[R] that the
early epidemic leaves behind, so the decay depends on the history: r is the asymptotic log-slope
of [I] from the product-state seed of `generate_pairwise` ([SI] = n[I], the other infected-block
pairs 0), found by integrating the leading-order infected block. When (n − 1)(1 − ϕ) < 1 this can
be r = −γ ([SI]/[I] → 0), where the continuation of the supercritical branch is O(γ) off. Just
below the threshold [R] settles at the rate |r|, so the ODE's log-slope approaches r slowly.

- Bernoulli, SIR: r = τ(q − 1) − γ;
- Bernoulli, SIS: r = (a − 2γ + √((a − 2γ)² + 8γ(a + τ)))/2 with a = τ(q − 1) − γ.
"""
function early_growth_rate(network::NetworkStructure, closure::ClosureMethod, τ::Real, γ::Real;
                           dynamics::Union{Nothing,Symbol} = nothing)
    d = _resolve_dynamics(network, dynamics, "early_growth_rate")
    return _growth(Val(d), network, closure, Float64(τ), Float64(γ))
end

# ─── Model-based API ─────────────────────────────────────────────────────────

"""
    basic_reproduction_number(model, network::NetworkStructure, closure = BernoulliClosure())

The threshold ratio R of an SIR or SIS model (a `CompartmentalModel` or a `ContactModel`) as a
symbolic expression in its rate parameters, for the Bernoulli closure. The dynamics are read
from the model, strictly: one infection transition X → Y through Y alone and one spontaneous
transition out of Y, to X (SIS) or to a third compartment (SIR); other models (SEIR, SIRS, …)
are an `ArgumentError`. So `sis_model()` gets the SIS formula (0.1 used the network type
instead, verified issue B01):

- SIR: R = τ(q − 1)/γ; SIS: R = (a + √((2γ − a)² + 8γ(a + τ)))/(2γ), a = τ(q − 1) − γ.

`KeelingClosure` and `BarnardClosure` give the same formula when the network has ϕ = 0 (they
are then the Bernoulli closure). With clustering they have no closed form (the fast-variable
quasi-equilibrium is the root of a cubic), so this method throws; evaluate them with
`basic_reproduction_number(network, closure, τ, γ; dynamics)`, or
`basic_reproduction_number(psys)` on a built system.
"""
function basic_reproduction_number(model::CompartmentalModel, network::NetworkStructure,
                                    closure::ClosureMethod = BernoulliClosure())
    d, tr, rec = _pairwise_dynamics(model)
    _symbolic_closure(network, closure, d)
    (tr.rate isa Symbol && rec.rate isa Symbol) || throw(ArgumentError(
        "basic_reproduction_number(model, …) builds a formula in the rate parameters, so the " *
        "rates must be parameter names; model :$(model.name) has $(tr.rate) and $(rec.rate)"))
    τ, γ = Symbolics.variable(tr.rate), Symbolics.variable(rec.rate)
    return _symbolic_R(Val(d), network, τ, γ)
end

# The closure of the symbolic formula: Bernoulli, or a clustered closure that reduces to it
# (ϕ = 0); anything else throws.
function _symbolic_closure(net::NetworkStructure, closure::ClosureMethod, d::Symbol)
    closure isa BernoulliClosure && return closure
    reduces = (closure isa KeelingClosure && net isa Union{HomogeneousNetwork,HeterogeneousNetwork}) ||
              (closure isa BarnardClosure && net isa HomogeneousNetwork)
    reduces || throw(ArgumentError(_unsupported_analysis(net, closure)))
    net.ϕ == 0 && return BernoulliClosure()
    throw(ArgumentError(
        "basic_reproduction_number(model, network, $(nameof(typeof(closure)))): the threshold " *
        "ratio of a clustered closure (ϕ = $(net.ϕ)) has no closed form (the quasi-equilibrium " *
        "of the fast variables is the root of a cubic, Barnard et al. 2019); evaluate it " *
        "numerically with basic_reproduction_number(network, closure, τ, γ; dynamics = :$(d)), " *
        "or on a built system with basic_reproduction_number(psys)"))
end

function _symbolic_R(::Val{:SIR}, net::NetworkStructure, τ, γ)
    q = _meandeg(net) > 0 ? _excess(net) : 0.0
    return q > 1 ? τ * (q - 1) / γ : 0 * τ
end

function _symbolic_R(::Val{:SIS}, net::NetworkStructure, τ, γ)
    _meandeg(net) > 0 || return 0 * τ
    a = τ * (_excess(net) - 1) - γ
    return (a + sqrt((2γ - a)^2 + 8γ * (a + τ))) / (2γ)
end

# The SIR/SIS dynamics and the numeric (τ, γ) of a model with parameter values `p` (τ is `NaN`
# when `infection = false`: a threshold needs only γ).
function _dynamics_and_rates(model::CompartmentalModel, p, defaults = Dict{Symbol,Float64}();
                             infection::Bool = true)
    d, tr, rec = _pairwise_dynamics(model)
    vals = merge(Dict{Symbol,Float64}(defaults),
                 Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in something(p, Dict())))
    rate(r) = try
        Float64(Symbolics.value(rate_value(r, vals)))
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("model :$(model.name): cannot evaluate the rate $(r) with the " *
                            "parameter values $(vals) (" * sprint(showerror, err) * ")"))
    end
    return d, infection ? rate(tr.rate) : NaN, rate(rec.rate)
end

# A model and network for the model-based numeric methods: the per-contact CompartmentalModel
# (the rates of a frequency- or density-dependent ContactModel converted to τ with the mean
# degree of the network structure or descriptor) and the NetworkStructure.
_analysis_model(model::CompartmentalModel, net) = (model, Dict{Symbol,Float64}())
function _analysis_model(cm::ContactModel, net)
    return CompartmentalModel(_per_contact_model(cm, net)), parameter_defaults(cm)
end
_analysis_structure(net::NetworkStructure) = net
_analysis_structure(net::Union{ConfigurationNetwork,ClusteredNetwork}) = network_structure(net)

const _AnalysisModel = Union{CompartmentalModel,ContactModel}
const _AnalysisNetwork = Union{ConfigurationNetwork,ClusteredNetwork}

"""
    epidemic_threshold(model, network::NetworkStructure, closure = default_closure(network); p = nothing)
    epidemic_threshold(model, net::Union{ConfigurationNetwork,ClusteredNetwork}, closure; p = nothing)

The critical per-contact rate τ_c of an SIR or SIS model (a `CompartmentalModel` or a
`ContactModel`, e.g. `scenario(:sis_reg3).model`) on a network structure or a configuration /
clustered descriptor, with the dynamics read from the model and the recovery rate γ from the
parameter values `p` (a `ContactModel`'s defaults fill in). The threshold is a per-contact rate
whatever the model's rate convention: the rates of a frequency- or density-dependent
`ContactModel` are converted to per-contact rates with the mean degree of the network. See
`epidemic_threshold(network, closure, γ; dynamics)` for the formulas.

```julia
sc = scenario(:sis_reg3)                       # SIS, 3-regular, γ = 1/4
epidemic_threshold(sc.model, sc.network, BernoulliClosure(); p = sc.params)   # 0.125
```
"""
function epidemic_threshold(model::_AnalysisModel, network::NetworkStructure,
                            closure::ClosureMethod = default_closure(network); p = nothing)
    m, defaults = _analysis_model(model, network)
    d, _, γ = _dynamics_and_rates(m, p, defaults; infection = false)
    return _threshold(Val(d), network, closure, γ)
end
epidemic_threshold(model::_AnalysisModel, net::_AnalysisNetwork, closure::ClosureMethod;
                   p = nothing) = _model_threshold(model, net, closure, p)
function _model_threshold(model, net, closure, p)
    m, defaults = _analysis_model(model, net)
    d, _, γ = _dynamics_and_rates(m, p, defaults; infection = false)
    return _threshold(Val(d), _analysis_structure(net), closure, γ)
end

"""
    early_growth_rate(model, network::NetworkStructure, closure = default_closure(network); p)
    early_growth_rate(model, net::Union{ConfigurationNetwork,ClusteredNetwork}, closure; p)

The early growth rate r of an SIR or SIS model with the parameter values `p` (see
`early_growth_rate(network, closure, τ, γ; dynamics)`). A frequency- or density-dependent
`ContactModel` has its contact rates converted to per-contact rates with the mean degree of the
network (τ = β/⟨k⟩ for frequency dependence).
"""
function early_growth_rate(model::_AnalysisModel, network::NetworkStructure,
                           closure::ClosureMethod = default_closure(network); p = nothing)
    m, defaults = _analysis_model(model, network)
    d, τ, γ = _dynamics_and_rates(m, p, defaults)
    return _growth(Val(d), network, closure, τ, γ)
end
function early_growth_rate(model::_AnalysisModel, net::_AnalysisNetwork, closure::ClosureMethod;
                           p = nothing)
    m, defaults = _analysis_model(model, net)
    d, τ, γ = _dynamics_and_rates(m, p, defaults)
    return _growth(Val(d), _analysis_structure(net), closure, τ, γ)
end

# ─── System-level API ────────────────────────────────────────────────────────

# The SIR/SIS dynamics and numeric (τ, γ) of a built pairwise system (parameter values: the
# model's defaults, then psys.params, then p).
function _system_rates(psys::PairwiseSystem, p; infection::Bool = true)
    model = _compartmental(psys)
    model === nothing && throw(ArgumentError(
        "this PairwiseSystem does not record the model it was built from"))
    return _dynamics_and_rates(model, _parameter_values(psys, p); infection)
end

"""
    epidemic_threshold(psys::PairwiseSystem; p = nothing) -> Float64

The critical per-contact rate τ_c of a built SIR or SIS pairwise system (its network structure,
closure and model; γ from the parameter values: the model's defaults, `psys.params`, then `p`).

```julia
sys = node_based(scenario(:sis_reg3))   # SIS on a 3-regular network, γ = 1/4
epidemic_threshold(sys)                 # γ/(k − 1) = 0.125
```
"""
function epidemic_threshold(psys::PairwiseSystem; p = nothing)
    d, _, γ = _system_rates(psys, p; infection = false)
    return _threshold(Val(d), psys.network, psys.closure, γ)
end

"""
    early_growth_rate(psys::PairwiseSystem; p = nothing) -> Float64

The early growth rate r of a built SIR or SIS pairwise system at the parameter values (the
model's defaults, `psys.params`, then `p`); r > 0 exactly when τ > [`epidemic_threshold`](@ref).
"""
function early_growth_rate(psys::PairwiseSystem; p = nothing)
    d, τ, γ = _system_rates(psys, p)
    return _growth(Val(d), psys.network, psys.closure, τ, γ)
end

"""
    basic_reproduction_number(psys::PairwiseSystem; p = nothing) -> Float64

The threshold ratio R = 1 + r/γ of a built SIR or SIS pairwise system (see
`basic_reproduction_number(network, closure, τ, γ)`).
"""
function basic_reproduction_number(psys::PairwiseSystem; p = nothing)
    d, τ, γ = _system_rates(psys, p)
    return 1 + _growth(Val(d), psys.network, psys.closure, τ, γ) / γ
end

# ─── Disease-free equilibrium ────────────────────────────────────────────────

"""
    disease_free_equilibrium(model, network; N=1.0)

Compute the disease-free equilibrium (DFE) for a pairwise system.
At DFE: all nodes are susceptible, [S]=N, [SS]=nN, all other
variables are zero.
"""
function disease_free_equilibrium(model::CompartmentalModel,
                                   network::NetworkStructure;
                                   N::Real = 1.0)
    names = model.compartment_names
    n = mean_degree(network)

    dfe = Dict{String,Float64}()

    # Singles
    first_susc = model.susceptible_compartments[1]
    for name in names
        dfe["[$name]"] = name == first_susc ? Float64(N) : 0.0
    end

    # Pairs
    for i in eachindex(names), j in i:length(names)
        a, b = names[i], names[j]
        if a == first_susc && b == first_susc
            dfe["[$a$b]"] = n * Float64(N)
        else
            dfe["[$a$b]"] = 0.0
        end
    end

    return dfe
end
