# Owner: V-NBM-infra. The definitions of the shared vignette setup; included once per page by `setup.jl`
# (never include this file directly). See `setup.jl` for the conventions.

using NetworkEpiCore
using NodeBasedModels
using NetworkOutbreaks
using Plots
using Printf
using Statistics
import TOML

# ── Plot defaults (one style for every page; representation colours come from the NEC recipes) ─────────────
gr()
default(; size = (720, 460), dpi = 150, framestyle = :box, linewidth = 2, legendfontsize = 8,
        guidefontsize = 10, tickfontsize = 8, titlefontsize = 11, fontfamily = "Computer Modern")

# ── Canonical anchors (vignette rule 7; DESIGN §E.2) ─────────────────────────────────────────────────────────
"""
    CANONICAL_ANCHORS

The canonical parameter anchors of the shared scenarios (DESIGN §E.2): R₀ = 2, recovery rate γ = 1/4 and the
per-contact (per-edge) transmission rate τ = 1/6, so T = τ/(τ + γ) = 0.4 and, with excess degree 5, R₀ = T·κ_ex = 2.
A scenario may override them; `anchors(sc)` prints what a scenario actually uses.
"""
const CANONICAL_ANCHORS = (R0 = 2.0, γ = 1 / 4, τ = 1 / 6)

# ── Lean citations (vignette rule 4) ─────────────────────────────────────────────────────────────────────────
const CITABLE_PATH = joinpath(pkgdir(NetworkEpiCore), "proofs", "CITABLE.txt")

"""
    citable_names() -> Set{String}

The fully qualified Lean theorem names that vignettes may cite: the non-comment lines of
`NetworkEpiCore.jl/proofs/CITABLE.txt` (DESIGN §L.4). The legacy EdgeBasedModels Lean tree is never cited.
"""
function citable_names()
    isfile(CITABLE_PATH) || error("citable_names: $(CITABLE_PATH) not found")
    return Set(strip(l) for l in eachline(CITABLE_PATH) if !isempty(strip(l)) && !startswith(strip(l), "#"))
end

"""
    lean(name) -> String

The Markdown citation of the Lean theorem `name` (for example `lean("NEP.eb_to_pws_pt_any")`), for inline code
`` `{julia} lean("…")` `` or a printed table. An error if `name` is not in `CITABLE.txt`, so a page cannot
render while it cites an unlisted (ungated) name. The text says "Lean theorem", never "certified".
"""
function lean(name::AbstractString)
    n = String(strip(name))
    n in citable_names() || error("lean: $(repr(n)) is not listed in $(CITABLE_PATH); vignettes may cite only " *
                                  "the names listed there (DESIGN §L.4)")
    return "Lean theorem `$(n)` (NetworkEpiCore.jl/proofs)"
end

# ── Reference summaries (vignette rule 2) ────────────────────────────────────────────────────────────────────
"""
    require_summaries(ids)

Check, before anything else runs, that every scenario in `ids` has a valid committed NetworkOutbreaks summary (the
current scenario hash, `ALGORITHM_REVISION` and `SUMMARY_REVISION`); an error listing the missing or stale ones
otherwise. Pages call it in their setup cell, so a page with a missing summary fails to render instead of warning.
"""
function require_summaries(ids)
    ids = collect(Symbol, ids)
    missing = missing_scenario_summaries(ids; companions = false)
    isempty(missing) || error("require_summaries: no valid committed NetworkOutbreaks summary for " *
                              join(("$(repr(k)) ($(v))" for (k, v) in missing), "; ") *
                              ". Regenerate with NetworkOutbreaks.jl/scripts/regenerate_scenarios.jl.")
    return nothing
end

_graphs_text(g::Symbol) = g === :per_run ? "a fresh graph per run" :
                          g === :fixed ? "one fixed (quenched) graph for every run" : string(g)
_graphs_text(g::Tuple) = "a pool of $(g[2]) graphs shared by the runs"

_condition_text(c::MajorOutbreak) =
    @sprintf("major outbreaks only (cumulative incidence excluding seeds ≥ %.3g·N by t_end)", c.threshold)
_condition_text(::Survival) = "surviving runs only (prevalence at t_end > 0)"
_condition_text(::Unconditioned) = "all runs (unconditioned)"

_align_text(::NoAlignment) = "no time alignment"
_align_text(a::CumulativeCrossing) =
    @sprintf("time-aligned at the %.3g cumulative-incidence crossing (CumulativeCrossing)", a.level)

"""
    kept_label(c) -> (runs, probability)

How to name the runs that satisfy the conditioning rule `c` (`sc.sim.condition`) and the fraction
`p_major = n_major/nsims` of a committed summary: `("major runs", "P(major)")` for `MajorOutbreak`,
`("surviving runs", "P(survival)")` for `Survival()` and `("runs kept", "P(kept)")` for `Unconditioned()`.
An endemic SIS/SIRS ensemble conditioned on survival keeps the runs still infected at t_end, so its fraction is
a survival probability, not a major-outbreak probability. The same labels as `kept_label` of the
EdgeBasedModels.jl vignettes, so mirrored pages name the fraction identically.
"""
kept_label(::MajorOutbreak) = ("major runs", "P(major)")
kept_label(::Survival) = ("surviving runs", "P(survival)")
kept_label(::Unconditioned) = ("runs kept", "P(kept)")

"""
    describe_reference(ref::EnsembleSummary; io = stdout) -> String

Print (and return) the one-paragraph description of a committed reference summary that every simulation
comparison must state (vignette rule 2): the scenario id and hash, N, the number of runs, how graphs are drawn, the
algorithm, the conditioning rule, the number of runs kept and the kept fraction with its 95% Wilson interval, and
the time alignment. The fraction is named by [`kept_label`](@ref): P(major) for `MajorOutbreak`, P(survival) for
`Survival()`.
"""
function describe_reference(ref::EnsembleSummary; io::IO = stdout)
    sc = scenario(ref.id)
    runs, prob = kept_label(sc.sim.condition)
    s = @sprintf("NetworkOutbreaks reference :%s (hash %s): N = %d nodes, %d runs on %s, algorithm :%s; conditioning: %s; %d of %d runs kept (%s), %s = %.3f (95%% CI %.3f–%.3f); %s.",
                 ref.id, first(ref.scenario_hash, 8), ref.N, ref.nsims, _graphs_text(sc.sim.graphs),
                 sc.sim.algorithm, _condition_text(sc.sim.condition), ref.n_major, ref.nsims, runs, prob,
                 ref.p_major, ref.p_major_ci..., _align_text(sc.sim.align))
    println(io, s)
    return s
end

"""
    reference_table(refs; io = stdout)

Print a Markdown table (use in a cell with `#| output: asis`) with one row per reference summary: id, N, runs,
graphs, conditioning, runs kept, and the kept fraction with its 95% CI, named per row by [`kept_label`](@ref)
(P(major) for `MajorOutbreak`, P(survival) for `Survival()`).
"""
function reference_table(refs; io::IO = stdout)
    println(io, "| scenario | N | runs | graphs | conditioning | kept | kept fraction (95% CI) |")
    println(io, "|---|---:|---:|---|---|---:|---|")
    for ref in refs
        sc = scenario(ref.id)
        cond = sc.sim.condition isa MajorOutbreak ? "major (≥ $(sc.sim.condition.threshold)·N)" :
               sc.sim.condition isa Survival ? "survival" : "none"
        _, prob = kept_label(sc.sim.condition)
        @printf(io, "| `:%s` | %d | %d | %s | %s | %d | %s = %.3f (%.3f–%.3f) |\n", ref.id, ref.N, ref.nsims,
                sc.sim.graphs isa Symbol ? string(sc.sim.graphs) : "pool", cond, ref.n_major, prob, ref.p_major,
                ref.p_major_ci...)
    end
    println(io)
    return nothing
end

# ── Scenario anchors (vignette rule 7) ───────────────────────────────────────────────────────────────────────
# Only the documented "not computed for this model / descriptor" errors are caught; anything else (a MethodError
# after a refactor, a BoundsError, ...) is a bug and is rethrown, so it fails the render instead of printing as an
# undefined quantity.
#
# - `AdmissibilityError` (NetworkEpiCore): the model is outside T_EB (SIS, SIRS: a transition back to S), so the
#   edge-based R₀ = T·κ_ex is not *defined* for it;
# - `ArgumentError`: the analysis does not cover this network descriptor (NEC's edge-based R₀ is not computed for
#   clustered or dormant/fleeting descriptors, although an R₀ exists mathematically there) or these dynamics (the
#   pairwise threshold needs strictly SIR or SIS dynamics).
function _try_quantity(f)
    try
        return f(), nothing
    catch e
        e isa ArgumentError && return nothing, :not_computed
        nameof(typeof(e)) === :AdmissibilityError && return nothing, :not_defined
        rethrow()
    end
end

"""
    anchors(sc; io = stdout) -> NamedTuple

Print the rate parameters of scenario `sc`, its initial seeds and its time grid, then its epidemic threshold
quantities, and flag every departure from [`CANONICAL_ANCHORS`](@ref):

- R₀ from NetworkEpiCore, `basic_reproduction_number(sc.model, sc.network, sc.params)` (the generation-based
  R₀ = T·κ_ex of the edge-based model on a configuration network), when it is defined. It is not for models
  outside T_EB (SIS, SIRS: a transition back to the susceptible state), where the line says "edge-based R₀ not
  defined (model outside T_EB)". For descriptors whose R₀ NEC does not compute (clustered, dormant/fleeting
  networks: the quantity exists, the edge-based formula does not apply) the line says "R₀ not computed by
  NetworkEpiCore for this network descriptor";
- the pairwise threshold τ_c from NodeBasedModels, `epidemic_threshold(sc.model, sc.network,
  default_closure(sc.network); p = sc.params)` (τ_c = γ/(k − 1) for SIS on a k-regular network, γ/(q − 1) for
  SIR with q = ⟨k(k − 1)⟩/⟨k⟩), and the ratio τ/τ_c, when the model is strictly SIR or SIS with a parameter `τ`.
  For SIS and SIRS pages it is τ/τ_c (not R₀) that says whether the scenario is above threshold. Whether NBM
  has such a threshold for the network descriptor is checked with `applicable` (none for `WellMixed` or
  `MultitypeNetwork`; the line says so), never by catching a `MethodError`.

Returns `(R0, τc, params)`, with `nothing` for a quantity that is not defined or not computed. Any other error of
the two analyses is rethrown.
"""
function anchors(sc::Scenario; io::IO = stdout)
    R0, why_R0 = _try_quantity(() -> basic_reproduction_number(sc.model, sc.network, sc.params))
    # The pairwise threshold needs a default closure (an ArgumentError of `default_closure`: NBM has no
    # population-level model for this descriptor) and an `epidemic_threshold` method for the descriptor (checked
    # with `applicable`: none for `WellMixed` or `MultitypeNetwork`), so a MethodError inside the analysis is never
    # mistaken for "no such quantity".
    closure, _ = _try_quantity(() -> default_closure(sc.network))
    τc_applies = closure !== nothing && applicable(epidemic_threshold, sc.model, sc.network, closure)
    τc, _ = τc_applies ?
        _try_quantity(() -> epidemic_threshold(sc.model, sc.network, closure; p = sc.params)) : (nothing, nothing)
    ps = join((@sprintf("%s = %.6g", k, sc.params[k]) for k in sort!(collect(keys(sc.params)); by = string)), ", ")
    seeds = join((@sprintf("%s %.4g", X, ρ) for (X, ρ) in sc.initial.fractions), ", ")
    @printf(io, ":%s: %s; seeds %s; t = %g:%g:%g", sc.id, ps, seeds, first(sc.tgrid), step(sc.tgrid),
            last(sc.tgrid))
    if R0 === nothing
        print(io, why_R0 === :not_defined ? "; edge-based R₀ not defined (model outside T_EB)" :
                                            "; R₀ not computed by NetworkEpiCore for this network descriptor")
    else
        @printf(io, "; R₀ = %.6g", R0)
    end
    τc_applies || print(io, "; no NodeBasedModels pairwise threshold for a ", nameof(typeof(sc.network)))
    if τc !== nothing
        @printf(io, "; pairwise threshold τ_c = %.6g", τc)
        haskey(sc.params, :τ) && @printf(io, ", τ/τ_c = %.6g", sc.params[:τ] / τc)
    end
    notes = String[]
    R0 === nothing ? push!(notes, why_R0 === :not_defined ? "no edge-based R₀" : "R₀ not computed") :
        !isapprox(R0, CANONICAL_ANCHORS.R0; rtol = 1e-6) && push!(notes, "R₀ ≠ 2")
    haskey(sc.params, :γ) && !isapprox(sc.params[:γ], CANONICAL_ANCHORS.γ; rtol = 1e-9) && push!(notes, "γ ≠ 1/4")
    haskey(sc.params, :τ) && !isapprox(sc.params[:τ], CANONICAL_ANCHORS.τ; rtol = 1e-9) && push!(notes, "τ ≠ 1/6")
    println(io, isempty(notes) ? " (canonical anchors)" : " (differs from the canonical anchors: " *
                                                           join(notes, ", ") * ")")
    return (R0 = R0, τc = τc, params = copy(sc.params))
end

# ── Formatting ───────────────────────────────────────────────────────────────────────────────────────────────
"""
    fmt(x; digits = 4) -> String

`x` rounded to `digits` significant digits, for inline code (`` `{julia} fmt(R0)` ``), so that every number in the
prose is printed by code (vignette rule 3).
"""
fmt(x::Real; digits::Integer = 4) = Printf.format(Printf.Format("%.$(Int(digits))g"), x)
fmt(x::Integer; digits::Integer = 4) = string(x)

"""
    md_table(header, rows; io = stdout, align = nothing)

Print a Markdown table (use in a cell with `#| output: asis`). `header` is a vector of column titles, `rows` a
vector of rows (vectors or tuples); numbers are printed with [`fmt`](@ref). `align` is a string of `l`/`r`/`c`,
one per column (default: right for numeric columns of the first row, left otherwise).
"""
function md_table(header, rows; io::IO = stdout, align = nothing)
    cell(x) = x isa AbstractFloat ? fmt(x) : string(x)
    n = length(header)
    al = align === nothing ?
         [(!isempty(rows) && first(rows)[j] isa Real) ? 'r' : 'l' for j in 1:n] : collect(align)
    sep = Dict('l' => "---", 'r' => "---:", 'c' => ":---:")
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", join((sep[a] for a in al), "|"), "|")
    for r in rows
        println(io, "| ", join((cell(x) for x in r), " | "), " |")
    end
    println(io)
    return nothing
end

# ── Distinguishable curves ───────────────────────────────────────────────────────────────────────────────────
const _FALLBACK_STYLES = [(:crimson, :dot), (:olive, :dashdotdot), (:deeppink, :solid), (:goldenrod, :dashdot),
                          (:navy, :dash), (:darkcyan, :dot), (:slateblue, :dashdotdot), (:chocolate, :solid)]

_isblack(c) = (r = Plots.RGBA(c); r.r == 0 && r.g == 0 && r.b == 0)
_is_curve_series(s) = s[:seriestype] === :path && s[:fillrange] === nothing && !_isblack(s[:linecolor]) &&
                      something(s[:linealpha], 1) > 0
_style_key(s) = (Plots.RGBA(s[:linecolor]), s[:linestyle])

"""
    distinct_styles!(plt) -> plt

Make every deterministic curve of every subplot of `plt` distinguishable. NetworkEpiCore's recipes
(`comparisonplot`, `plot!(p, c::ModelCurves, X)`) give every curve of one representation the same colour and line
style (all `:pairwise` curves are dark-orange dashes, for instance), so two pairwise closures on one figure would
be drawn alike. In each subplot the second and later curves whose (colour, line style) repeats an earlier curve's
get the next unused style of a fixed list; the first curve of each representation keeps the recipe style. The
ensemble band and mean and the black zero line are left alone. Curves are drawn in the same order in every subplot,
so a curve gets the same new style in every panel, and the legend follows. A no-op when no style repeats.
"""
function distinct_styles!(plt::Plots.Plot)
    for sp in plt.subplots
        curves = [s for s in sp.series_list if _is_curve_series(s)]
        used = Set(_style_key(s) for s in curves)
        seen = Set{Any}()
        for s in curves
            k = _style_key(s)
            if k in seen
                colours = Set(first(u) for u in used)
                i = findfirst(st -> !(Plots.RGBA(Plots.plot_color(st[1])) in colours), _FALLBACK_STYLES)
                i === nothing && error("distinct_styles!: more repeated curve styles than fallback styles")
                c, ls = _FALLBACK_STYLES[i]
                s[:linecolor] = Plots.RGBA(Plots.plot_color(c))
                s[:linestyle] = ls
                k = _style_key(s)
                push!(used, k)
            end
            push!(seen, k)
        end
    end
    return plt
end

const NBM_VIGNETTE_SETUP_LOADED = true
