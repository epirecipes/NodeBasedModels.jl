# Owner: V-NBM-infra (DESIGN_NetworkEpiCore.md §F, §F.2, §E.2, §E.5; amendments §J–§M).
#
# Shared setup of the NodeBasedModels.jl vignettes N01–N14. Every page includes it in its first code cell,
# *before* the shared model cell of §E.5:
#
#     ```{julia}
#     #| label: setup
#     #| output: false
#     include(joinpath(@__DIR__, "..", "_shared", "setup.jl"))
#     require_summaries([:sir_pois5, :sir_reg6])     # the committed NO summaries the page uses
#     ```
#
# The file only loads packages, sets the plotting defaults and the strict-cache policy, and defines small
# formatting and checking helpers. It never builds a model, never simulates and never computes a reference
# ensemble: every comparison goes through `scenario_summary(sc)` (the committed NetworkOutbreaks summaries),
# `model_curves`, `compare` and `comparisonplot` (NetworkEpiCore recipes), which the pages call visibly.
#
# The EdgeBasedModels.jl vignettes have their own copy of the same conventions; the helpers below are
# deliberately free of back-end code. They are NOT the EdgeBasedModels.jl helpers: EBM's setup.jl has its own names
# and wording (reference_note, mdtable, lean_cite versus describe_reference, md_table, lean here; only kept_label is
# common), so mirrored pages print the same numbers but not necessarily the same sentences. Only the `shared-cell`
# of a mirrored pair is required to be textually identical (rule 5; `_shared/check_pages.jl`).

# ── Strict cache (vignette rule 8): committed summaries only, never a user cache, never a simulation ─────────
# `scenario_summary(sc)` defaults to `policy = :committed` (a missing or stale summary is an error); the
# environment variable also makes any `policy = :auto` call strict. `_shared/render.sh` sets it as well.
ENV["NETEPI_STRICT_CACHE"] = "1"
# GR's null workstation: never open a plot window (a `display(plot)` in a worker would otherwise wait on gksqt).
ENV["GKSwstype"] = "100"

# ── Load once per page ───────────────────────────────────────────────────────────────────────────────────────
# A page that mirrors several EdgeBasedModels pages (N02) or whose shared cell is not its first cell (N04) includes
# this file again in its shared cell. The definitions live in `setup_body.jl` and are loaded only the first time,
# so a second include neither replaces docstrings nor redefines constants (and so emits no warning; a cell never
# needs `warning: false` for it).
if !isdefined(@__MODULE__, :NBM_VIGNETTE_SETUP_LOADED)
    include(joinpath(@__DIR__, "setup_body.jl"))
end
