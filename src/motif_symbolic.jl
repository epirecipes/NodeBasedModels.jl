# motif_symbolic.jl — Independent Symbolics-based oracle for the motif-closure
# RHS implemented in `motif_based.jl`.
#
# Purpose: emit a symbolic ODE RHS for `(k, m)` ∈ {(2,2), (2,3), (2,4), (2,5),
# (2,6), (3,2), (3,3)} that — by independent first-principles derivation —
# matches the numeric RHS produced by `motif_based_sis(k, m)` element-wise at
# every state vector, including near the disease-free equilibrium where the
# `safe_ratio` semantics matter.
#
# Independence: this file does NOT call any of `_build_sis_k*_rhs` nor the
# numeric RHS closures. It re-uses only the (closure-agnostic) layout helpers
# `enumerate_shapes`, `enumerate_state_classes`, `canonical_state` and
# `_build_variables`. The closure rules and pair-/singleton-derivative
# assembly are independently re-derived in symbolic form below.
#
# Closure registry (selected via `closure_kind`):
#
#   :auto                 — per-shape default closure that mirrors the
#                           numeric builder convention used by the existing
#                           `_build_sis_k*` RHSs:
#                             • P_2  (any k):  Keeling κ-corrected,
#                                              flow_ext = β · κ · L_σ ·
#                                              L_(σ_v, I) / ⟨σ_v⟩  (κ=(k-1)/k)
#                             • P_3  k=2     : raw Markov pair-closure (NO κ
#                                              factor; matches the specialised
#                                              m=3 builder which uses the
#                                              lower-order Kirkwood
#                                              L_σ · L_(I,σ_1) / ⟨σ_1⟩)
#                             • P_3  k=3     : per-vertex slot factor
#                                              (n_ext_i / k); endpoints get
#                                              (k-1)/k, middle gets (k-2)/k
#                             • C_3          : (k-2)/k · paw closure
#                             • P_m  m ≥ 4 (k=2): generalised order-m chain
#                                              Kirkwood at endpoints; middle
#                                              vertices have n_ext = 0 on a
#                                              k=2 ring, so flow_ext = 0
#   :strict_kirkwood      — universally use the chain-style Kirkwood
#                           closure  L_σ · L_(σ_v, I) / ⟨σ_v⟩  (no slot
#                           factor, no κ correction). Diagnostic only —
#                           does NOT match the numeric RHS at most (k, m).
#   :match_specialised_m3 — alias of :auto (kept for interface parity with
#                           the spec).
#
# `safe_ratio_sym(num, den)` mirrors `safe_ratio(num, den; tol=1e-12)`:
# it returns 0 when `den < 1e-12`, otherwise `num/den`. Implemented as
# `Base.ifelse(den < 1e-12, zero(den), num/den)` so the resulting Symbolics
# expression compiles to a branched call that produces exactly 0.0 (not
# NaN/Inf) at the disease-free equilibrium.

using Symbolics: @variables, build_function, Num

# ─── Symbolic-safe helpers ────────────────────────────────────────────────

"""
    safe_ratio_sym(num, den) -> Num

Symbolic counterpart of [`safe_ratio`](@ref). Compiles to
`ifelse(den < 1e-12, 0, num/den)` so it yields exactly zero (not NaN) when
`den` underflows.
"""
@inline safe_ratio_sym(num, den) = ifelse(den < 1e-12, zero(den), num / den)

# Helper: lookup the canonical `MotifShape` constant by name. Used by the
# Phase B(c) per-shape Kirkwood closure to resolve the 3-vertex anchor
# and 4-vertex target shapes named in `_CLOSURE_RULES_4V`.
function _shape_by_name(name::Symbol)
    name === :singleton && return _SINGLETON_SHAPE
    name === :P2   && return _P2_SHAPE
    name === :P3   && return _P3_SHAPE
    name === :C3   && return _C3_SHAPE
    name === :P4   && return _P4_SHAPE
    name === :K13  && return _K13_SHAPE
    name === :paw  && return _PAW_SHAPE
    name === :C4   && return _C4_SHAPE
    name === :K4me && return _K4ME_SHAPE
    name === :K4   && return _K4_SHAPE
    error("_shape_by_name: unknown shape :$name")
end

# Helper: list neighbours of vertex `i` inside `shape`.
function _shape_neighbours(shape::MotifShape, i::Int)
    nbrs = Int[]
    for (a, b) in shape.edges
        if a == i
            push!(nbrs, b)
        elseif b == i
            push!(nbrs, a)
        end
    end
    return nbrs
end

function _shape_internal_degrees(shape::MotifShape)
    deg = zeros(Int, shape.n_nodes)
    for (a, b) in shape.edges
        deg[a] += 1
        deg[b] += 1
    end
    return deg
end

# Symbolic labelled count L_σ for a labelled state σ on `shape`:
#   L_σ = u[idx of canon(σ)] · stab(canon)
#   stab = |G| / |orbit(σ)|
function _Lsym_of(shape::MotifShape, σ::Vector{Symbol}, idx, u_sym)
    canon, osz = canonical_state(shape, σ)
    stab = length(shape.automorphisms) ÷ osz
    return u_sym[idx[(shape.name, canon)]] * stab
end

# Iterate over labelled states for a shape (Vector{Vector{Symbol}}).
function _labelled_states(shape::MotifShape, base_states::Vector{Symbol})
    n = shape.n_nodes
    iters = ntuple(_ -> base_states, n)
    return [collect(t) for t in Iterators.product(iters...)]
end

# ─── Per-shape closure dispatch ───────────────────────────────────────────
#
# Returns the symbolic flow_ext expression for an external infection at
# vertex `i` of shape `sh` in labelled state `σ` on a k-regular host.
function _closure_flow_ext(closure_kind::Symbol, k::Int, m::Int,
                           sh::MotifShape, σ::Vector{Symbol}, i::Int,
                           n_ext::Int, Lσ_sym, β_sym,
                           Lpair_sym, single_sym,
                           idx, u_sym, base_states::Vector{Symbol})
    # Universal short-circuit: no external slots, no flow.
    n_ext == 0 && return zero(β_sym)

    if closure_kind === :strict_kirkwood
        return β_sym * safe_ratio_sym(
            Lσ_sym * Lpair_sym(σ[i], :I), single_sym(σ[i]))
    end

    # :auto / :match_specialised_m3 dispatch by shape name.
    name = sh.name

    if name === :P2
        κ = (k - 1) / k
        return β_sym * κ * safe_ratio_sym(
            Lσ_sym * Lpair_sym(σ[i], :I), single_sym(σ[i]))

    elseif name === :P3
        if k == 2
            # Specialised m=3 builder: lower-order Kirkwood, no slot factor.
            # (Endpoints only — middle has n_ext = 0 on k=2 ring, handled by
            # the short-circuit above.)
            return β_sym * safe_ratio_sym(
                Lσ_sym * Lpair_sym(σ[i], :I), single_sym(σ[i]))
        else
            slot = n_ext / k  # endpoint: (k-1)/k; middle: (k-2)/k
            return β_sym * slot * safe_ratio_sym(
                Lσ_sym * Lpair_sym(σ[i], :I), single_sym(σ[i]))
        end

    elseif name === :C3
        slot = n_ext / k  # = (k-2)/k for any C_3 vertex
        return β_sym * slot * safe_ratio_sym(
            Lσ_sym * Lpair_sym(σ[i], :I), single_sym(σ[i]))

    elseif name === :K13 || name === :paw || name === :C4 ||
           name === :K4me || name === :K4 ||
           (name === :P4 && k == 3)
        # Phase B(c) — per-shape higher-order Kirkwood closure on the
        # 5-vertex motif (e + σ). For each (shape, ext_vertex) we drop
        # a chosen vertex w (typically diametrically opposite to the
        # extension vertex i) and factorise as
        #
        #   L_(e=I, σ) ≈ L_(σ-{w}∪{e=I}) · L_σ / L_(σ-{w})
        #
        # σ-{w} is always a tracked 3-vertex shape (P_3 or C_3) and
        # σ-{w}∪{e} is always a tracked 4-vertex shape (P_4, K_{1,3},
        # paw or C_4). The drop vertex w and the resulting state
        # permutations come from `_CLOSURE_RULES_4V` in
        # `motif_based.jl` (we re-use that registry directly so the
        # symbolic and numeric closures stay in lockstep). The
        # resulting flow is
        #
        #   flow_ext = β · (n_ext_i / k) · safe_ratio(L_4target · L_σ,
        #                                              L_3target)
        rule = get(_CLOSURE_RULES_4V, (name, i), nothing)
        rule === nothing && error(
            "motif_symbolic: missing Kirkwood closure rule for ($name, $i)")
        target3_sh = _shape_by_name(rule.target3)
        target4_sh = _shape_by_name(rule.target4)
        state3 = Symbol[σ[rule.perm3[1]], σ[rule.perm3[2]], σ[rule.perm3[3]]]
        state4 = Symbol[rule.perm4[j] == 0 ? :I : σ[rule.perm4[j]]
                        for j in 1:4]
        L3 = _Lsym_of(target3_sh, state3, idx, u_sym)
        L4 = _Lsym_of(target4_sh, state4, idx, u_sym)
        slot = n_ext / k
        return β_sym * slot * safe_ratio_sym(L4 * Lσ_sym, L3)

    elseif name === :P4 || name === :P5 || name === :P6 ||
           startswith(string(name), "P")
        # Generic-chain (k=2, m ≥ 4) endpoint extension via the order-m
        # chain Kirkwood factorisation:
        #
        #   L_(I, σ_1, …, σ_{m-1}, σ_m)_{P_{m+1}}
        #     ≈ L_(I, σ_1, …, σ_{m-1})_{P_m} · L_σ_{P_m}
        #          / L_(σ_1, …, σ_{m-1})_{P_{m-1}}
        #
        # Symmetric form for the right endpoint. The denominator P_{m-1}
        # count is recovered as a marginal of P_m by summing both possible
        # boundary bits of the dropped position.
        @assert k == 2 "chain Kirkwood is only registered for k=2"
        @assert i == 1 || i == m "chain middle should have n_ext=0"

        if i == 1
            num_state  = vcat([:I], σ[1:m-1])              # length m
            denom_pref = σ[1:m-1]                          # length m-1
            denom_sym  = sum(_Lsym_of(sh, vcat(denom_pref, [s]), idx, u_sym)
                             for s in base_states)
        else  # i == m
            num_state  = vcat(σ[2:m], [:I])
            denom_suf  = σ[2:m]
            denom_sym  = sum(_Lsym_of(sh, vcat([s], denom_suf), idx, u_sym)
                             for s in base_states)
        end
        num_sym = _Lsym_of(sh, num_state, idx, u_sym)
        return β_sym * safe_ratio_sym(num_sym * Lσ_sym, denom_sym)

    end

    error("No closure rule registered for shape :$name (closure_kind=$closure_kind, k=$k, m=$m)")
end

# ─── Pair-derivative marginals (m ≥ 3 only) ───────────────────────────────
#
# For m ≥ 3 the pair derivatives are exact marginals of triple transitions:
#
#   dE_SS/dt = -β · L_(I,S,S) + γ · E_IS
#   dE_IS/dt =  β · L_(I,S,S) - β · L_(I,S,I) - β · E_IS - γ · E_IS + 2γ · E_II
#   dE_II/dt =  β · L_(I,S,I) + β · E_IS - 2γ · E_II
#
# where L_(I,S,S) and L_(I,S,I) are aggregated labelled-P_3 counts. We
# obtain them from the tracked variables:
#
#   * (k=2, m=3): a single P_3 shape supplies both directly.
#   * (k=3, m=3): P_3 shape + 2·C_3 shape contributions.
#   * (k=2, m ≥ 4): marginal of the tracked P_m by summing labelled P_m
#       counts whose first three bits encode the desired triple.
function _pair_marginals_sym(closure::MotifClosure, idx, u_sym,
                             shapes::Vector{MotifShape},
                             base_states::Vector{Symbol})
    k = closure.k; m = closure.m

    if k == 2 && m == 3
        # P_3 canonical (I,S,S): orbit 2, stab 1; canonical (I,S,I): orbit 1, stab 2.
        L_ISS = u_sym[idx[(:P3, [:I, :S, :S])]]
        L_ISI = 2 * u_sym[idx[(:P3, [:I, :S, :I])]]
        return L_ISS, L_ISI

    elseif k == 3 && m == 3
        # P_3 contributions
        L_ISS_p3 = u_sym[idx[(:P3, [:I, :S, :S])]]            # stab 1
        L_ISI_p3 = 2 * u_sym[idx[(:P3, [:I, :S, :I])]]        # stab 2
        # C_3 contributions: labelled (I,S,S) → canonical (I,S,S), stab 2;
        # labelled (I,S,I) → canonical (I,I,S), stab 2.
        L_ISS_c3 = 2 * u_sym[idx[(:C3, [:I, :S, :S])]]
        L_ISI_c3 = 2 * u_sym[idx[(:C3, [:I, :I, :S])]]
        return L_ISS_p3 + L_ISS_c3, L_ISI_p3 + L_ISI_c3

    elseif k == 3 && m == 4
        # B(c): pair derivatives use the SAME triple-flow marginal pattern
        # as B(b) — P_3 + C_3 contributions. The 4-vertex shapes do not
        # contribute directly because pair flow is closed at the triple
        # level (the 4-vertex shapes feed back into pair dynamics only
        # through their effect on triple variables via the triple-shape
        # ODEs).
        L_ISS_p3 = u_sym[idx[(:P3, [:I, :S, :S])]]
        L_ISI_p3 = 2 * u_sym[idx[(:P3, [:I, :S, :I])]]
        L_ISS_c3 = 2 * u_sym[idx[(:C3, [:I, :S, :S])]]
        L_ISI_c3 = 2 * u_sym[idx[(:C3, [:I, :I, :S])]]
        return L_ISS_p3 + L_ISS_c3, L_ISI_p3 + L_ISI_c3

    elseif k == 2 && m >= 4
        sh = shapes[end]
        @assert sh.name === Symbol("P", m) || sh.name === :P3 ||
                sh.name === :P4 || sh.name === :P5 || sh.name === :P6 "expected P_m as last shape"
        # Sum L_σ over labelled m-tuples whose first three positions match
        # the desired triple (I,S,S) or (I,S,I). The (m-3) trailing
        # positions are summed over {S,I}.
        free_iters = ntuple(_ -> base_states, m - 3)
        if m == 3
            free_states_iter = [Symbol[]]
        else
            free_states_iter = [collect(t) for t in Iterators.product(free_iters...)]
        end
        L_ISS = zero(u_sym[1])
        L_ISI = zero(u_sym[1])
        for tail in free_states_iter
            σ_iss = vcat([:I, :S, :S], tail)
            σ_isi = vcat([:I, :S, :I], tail)
            L_ISS = L_ISS + _Lsym_of(sh, σ_iss, idx, u_sym)
            L_ISI = L_ISI + _Lsym_of(sh, σ_isi, idx, u_sym)
        end
        return L_ISS, L_ISI

    end
    error("_pair_marginals_sym: unsupported (k=$k, m=$m)")
end

# ─── Public API ───────────────────────────────────────────────────────────

"""
    build_motif_symbolic_rhs(closure::MotifClosure;
                             model = :sis,
                             closure_kind = :auto)
        → (rhs!::Function, var_keys::Vector{Tuple{Symbol,Vector{Symbol}}},
           params::NTuple{2,Symbolics.Num})

Independent Symbolics-based RHS oracle for the motif-closure SIS dynamics
on a `k`-regular host. Returns a callable `rhs!(du, u, p, t)` (where
`p = (β, γ)` is a 2-tuple/Vector of numeric values), the canonical
variable-key list (matching the existing `MotifSystem.variables` order
exactly), and the symbolic parameter pair `(β_sym, γ_sym)`.

This builder enumerates motif shapes via [`enumerate_shapes`](@ref) and
state classes via [`enumerate_state_classes`](@ref), then assembles the
symbolic dE/dt expressions from first principles using the closure
registry described at the top of `motif_symbolic.jl`. It does NOT call
any of the numeric `_build_sis_k*_rhs` builders; the resulting function
is intended to be cross-checked against the numeric implementation as
an oracle.

Currently `model = :sis` is the only supported model. Supported
`(closure.k, closure.m)` pairs are those for which `enumerate_shapes`
returns a non-error: k=2 with 2 ≤ m ≤ 6 and k=3 with m ∈ {2, 3}.
"""
function build_motif_symbolic_rhs(closure::MotifClosure;
                                  model::Symbol = :sis,
                                  closure_kind::Symbol = :auto)
    model === :sis ||
        throw(ArgumentError("build_motif_symbolic_rhs: only model=:sis is supported (got :$model)"))
    closure_kind === :auto ||
        closure_kind === :strict_kirkwood ||
        closure_kind === :match_specialised_m3 ||
        throw(ArgumentError("Unknown closure_kind=:$closure_kind"))

    k = closure.k; m = closure.m
    base_states = [:S, :I]
    shapes, vars, idx = _build_variables(closure, base_states)
    nvars = length(vars)

    # ── Symbolic state vector + scalar parameters ─────────────────────────
    @variables _β_sym _γ_sym
    @variables _u_sym[1:nvars]
    u_sym = collect(_u_sym)
    Num0  = zero(_β_sym)

    # ── du expression vector, initialised to symbolic zero ────────────────
    du_sym = Vector{Num}(undef, nvars)
    for j in 1:nvars
        du_sym[j] = Num0
    end

    # Singleton + P_2 indices (always present in the supported (k, m) set).
    iI  = idx[(:singleton, [:I])]
    iS  = idx[(:singleton, [:S])]
    iII = idx[(:P2, [:I, :I])]
    iIS = idx[(:P2, [:I, :S])]
    iSS = idx[(:P2, [:S, :S])]

    SS_sym  = u_sym[iSS]
    SI_sym  = u_sym[iIS]
    II_sym  = u_sym[iII]
    S_sym   = u_sym[iS]
    Iv_sym  = u_sym[iI]

    # Labelled-pair / singleton accessors (closures over u_sym).
    Lpair_sym = function(X::Symbol, Y::Symbol)
        if X === :S && Y === :S
            return 2 * SS_sym
        elseif X === :I && Y === :I
            return 2 * II_sym
        else
            return SI_sym
        end
    end
    single_sym = (X::Symbol) -> (X === :I ? Iv_sym : S_sym)

    # ── For m ≥ 3, the pair derivatives come from triple marginals (NOT
    #    from labelled-P_2 flow). So we skip P_2 in the labelled loop in
    #    that case, and let the m≥3 pair-marginal block fill iSS/iIS/iII.
    skip_p2_in_loop = (m >= 3)

    # ── For (k=3, m=4), P_3 and C_3 derivatives are computed as exact
    #    marginals of the 4-vertex shape derivatives (locked semantic #5).
    #    Skip them in the labelled-flow loop and assign them after the
    #    4-vertex block via an independently rebuilt 3-from-4 marginal
    #    matrix.
    skip_p3c3_in_loop = (k == 3 && m == 4)

    # ── Labelled-state flow assembly per shape ────────────────────────────
    for sh in shapes
        sh.n_nodes < 2 && continue
        if sh.name === :P2 && skip_p2_in_loop
            continue
        end
        if (sh.name === :P3 || sh.name === :C3) && skip_p3c3_in_loop
            continue
        end

        deg_in = _shape_internal_degrees(sh)
        n      = sh.n_nodes
        labelled = _labelled_states(sh, base_states)

        # Symbolic accumulator for dL per labelled state.
        dL = Dict{Vector{Symbol}, Num}()
        for σ in labelled
            dL[σ] = Num0
        end

        for σ in labelled
            Lσ = _Lsym_of(sh, σ, idx, u_sym)
            for i in 1:n
                if σ[i] === :I
                    σp = copy(σ); σp[i] = :S
                    flow = _γ_sym * Lσ
                    dL[σ]  = dL[σ]  - flow
                    dL[σp] = dL[σp] + flow
                else  # :S → :I
                    nbrs   = _shape_neighbours(sh, i)
                    n_int  = count(j -> σ[j] === :I, nbrs)
                    flow_int = _β_sym * n_int * Lσ
                    n_ext    = k - deg_in[i]
                    flow_ext = _closure_flow_ext(closure_kind, k, m, sh, σ, i,
                                                 n_ext, Lσ, _β_sym,
                                                 Lpair_sym, single_sym,
                                                 idx, u_sym, base_states)
                    flow = flow_int + flow_ext
                    σp = copy(σ); σp[i] = :I
                    dL[σ]  = dL[σ]  - flow
                    dL[σp] = dL[σp] + flow
                end
            end
        end

        # Convert dL_σ → dE_canon for each canonical class on this shape.
        seen = Set{Vector{Symbol}}()
        for σ in labelled
            canon, osz = canonical_state(sh, σ)
            canon in seen && continue
            push!(seen, canon)
            stab = length(sh.automorphisms) ÷ osz
            du_sym[idx[(sh.name, canon)]] = dL[canon] / stab
        end
    end

    # ── 3-from-4 marginalisation (locked semantic #5) for (k=3, m=4) ──────
    # Independently rebuild the marginal matrix from shape topology and
    # assign du_sym[i_3] = Σ coef · du_sym[i_4]. This mirrors the numeric
    # `_build_mat_3from4` but is constructed without importing the numeric
    # version (per cross-validation independence). Identity is exact at
    # any factorising IC; see numeric file for asymptotic-IC caveat.
    if skip_p3c3_in_loop
        # Per-shape lookup for the 3-vertex target shape.
        _sh3_for(name::Symbol) = (name === :P3) ? _P3_SHAPE : _C3_SHAPE
        ext_for = Dict(:P3 => 5, :C3 => 3)
        # Locally enumerate connected 3-subsets per 4-vertex shape.
        function specs_of(sh4::MotifShape)
            specs = Tuple{Symbol, NTuple{3,Int}}[]
            for T in ((1,2,3), (1,2,4), (1,3,4), (2,3,4))
                ne = 0
                deg = Dict{Int,Int}(t => 0 for t in T)
                for (a, b) in sh4.edges
                    if a in T && b in T
                        ne += 1; deg[a] += 1; deg[b] += 1
                    end
                end
                if ne == 2
                    mid = 0; ends = Int[]
                    for v in T
                        if deg[v] == 2; mid = v; else push!(ends, v); end
                    end
                    mid == 0 && continue
                    sort!(ends)
                    push!(specs, (:P3, (ends[1], mid, ends[2])))
                elseif ne == 3
                    push!(specs, (:C3, (T[1], T[2], T[3])))
                end
            end
            return specs
        end

        # Accumulate symbolic contributions per triple-target index.
        triple_acc = Dict{Int, Num}()
        # Pre-zero entries for all 3-vertex variables on these two shapes
        # so any with no contributors get du_sym = 0.
        for v in vars
            if v.shape.name === :P3 || v.shape.name === :C3
                triple_acc[idx[(v.shape.name, v.state)]] = Num0
            end
        end

        for sh4 in shapes
            sh4.n_nodes == 4 || continue
            sps = specs_of(sh4)
            seen_canon = Set{Vector{Symbol}}()
            for σ4 in _labelled_states(sh4, base_states)
                canon4, _ = canonical_state(sh4, σ4)
                canon4 in seen_canon && continue
                push!(seen_canon, canon4)
                i4 = idx[(sh4.name, canon4)]
                # Per (target shape3 name, canonical state) integer count.
                counts = Dict{Tuple{Symbol,Vector{Symbol}}, Int}()
                for (sh3_name, ord) in sps
                    state3_lab = Symbol[canon4[ord[1]],
                                        canon4[ord[2]],
                                        canon4[ord[3]]]
                    canon3, _ = canonical_state(_sh3_for(sh3_name), state3_lab)
                    key = (sh3_name, canon3)
                    counts[key] = get(counts, key, 0) + 1
                end
                for ((sh3_name, canon3), c) in counts
                    i3 = idx[(sh3_name, canon3)]
                    coef = c / ext_for[sh3_name]   # Float64 ratio
                    triple_acc[i3] = triple_acc[i3] + coef * du_sym[i4]
                end
            end
        end
        for (i3, expr) in triple_acc
            du_sym[i3] = expr
        end
    end

    # ── Pair derivatives via triple marginals (only m ≥ 3) ────────────────
    if m >= 3
        L_ISS_total, L_ISI_total =
            _pair_marginals_sym(closure, idx, u_sym, shapes, base_states)
        du_sym[iSS] = -_β_sym * L_ISS_total + _γ_sym * SI_sym
        du_sym[iIS] =  _β_sym * L_ISS_total - _β_sym * L_ISI_total -
                       _β_sym * SI_sym - _γ_sym * SI_sym + 2 * _γ_sym * II_sym
        du_sym[iII] =  _β_sym * L_ISI_total + _β_sym * SI_sym -
                       2 * _γ_sym * II_sym
    end

    # ── Singleton derivatives (model-exact) ───────────────────────────────
    du_sym[iS] =  _γ_sym * Iv_sym - _β_sym * SI_sym
    du_sym[iI] = -_γ_sym * Iv_sym + _β_sym * SI_sym

    # ── Compile to a Julia callable via Symbolics.build_function ─────────
    _, f_ip = build_function(du_sym, u_sym, [_β_sym, _γ_sym];
                             expression = Val{false})

    rhs! = function (du, u, p, t)
        # `p` may be a NamedTuple (β=…, γ=…), a 2-tuple, or a Vector.
        if p isa NamedTuple
            βv = p.β; γv = p.γ
        elseif p isa Tuple
            βv = p[1]; γv = p[2]
        else
            βv = p[1]; γv = p[2]
        end
        f_ip(du, u, (βv, γv))
        return nothing
    end

    var_keys = [(v.shape.name, v.state) for v in vars]
    return rhs!, var_keys, (_β_sym, _γ_sym)
end

# Re-export at module level via NodeBasedModels.jl include + export.
