# Shared vignette infrastructure (NodeBasedModels.jl)

Owner: V-NBM-infra. For page authors of N01–N14 (DESIGN_NetworkEpiCore.md §F.2; N14 is the WP36e stretch page).

| File | Purpose |
|---|---|
| `pages.toml` | The page catalogue: id, title, scenario ids, layers, EBM mirrors, whether the page compares with simulation. |
| `setup.jl` | Included by the first code cell of every page (and again by a shared cell that is not the first cell). Sets `NETEPI_STRICT_CACHE=1` and `GKSwstype`, then includes `setup_body.jl` once per page, so a second include emits no "replacing docs" or constant-redefinition warning and no cell needs `warning: false`. |
| `setup_body.jl` | The definitions: loads NEC, NBM, NO, Plots, Printf and Statistics, sets the plot defaults, and defines `require_summaries`, `describe_reference`, `reference_table`, `kept_label`, `anchors`, `lean`, `citable_names`, `fmt`, `md_table`, `distinct_styles!` and `CANONICAL_ANCHORS`. |
| `pdf-header.tex` | pdf preamble (`_quarto.yml`): code and code output wrap at the margin (fvextra) in a smaller type. The pdf fonts and their fallbacks (∞, ⟨ ⟩, … are not in STIX Two Text or Fira Code) are set once in `_quarto.yml`, not per page. |
| `page_template.qmd` | A working N01-style page: setup cell, `shared-cell`, reference description, `comparisonplot`, `compare`, an inline number. Copy it to `N<nn>_<slug>/index.qmd`. |
| `qmd-links.lua` | Quarto filter (listed in `_quarto.yml`): rewrites `.qmd` links between pages to `index.md` (gfm) or `index.html`. |
| `check_pages.jl` | Static checks of every existing page against the catalogue and the vignette rules. `--rendered` also checks the outputs, `_freeze`, that every image an `index.md` links exists (captions may hold citations and `[a, b]` intervals), that no `index.md` links a `.qmd`, that no output says "certif…", and that each mirrored EBM page can be compared. `--self-test` runs the checker's regression cases. |
| `render.sh` | `quarto render` under `NETEPI_STRICT_CACHE=1`, one pass per format: html, pdf, then gfm last. It fails on errors, warnings or missing summaries, then runs `check_pages.jl --rendered`. |

Conventions:

- **Where a page lives.** Each page is `N<nn>_<slug>/index.qmd`, with exactly one directory per id. Its front-matter
  `title` is the catalogue title. The index links a page as soon as it exists.
- **First cell.** It is always
  `include(joinpath(@__DIR__, "..", "_shared", "setup.jl")); require_summaries([...])`. A missing or stale committed
  summary is then a render error, not a warning.
- **Shared cell.** A mirrored page (see `mirrors` in the catalogue) shares one cell with each mirror E: the cell
  labelled `shared-cell-E` if the page has one, else `shared-cell` (N02 mirrors E03 and E06, whose cells differ, so
  it has `shared-cell-E03` and `shared-cell-E06`). On the EBM side the cell is `shared-cell-<Nxx>` or
  `shared-cell`. Apart from back-end lines the two cells must be identical and use the same scenario ids, which
  must be catalogue scenarios of the page; the rest of the pages may differ (N04 and E10 share `:sir_sbm2` only).
  N14 has no mirror (WP36e names none). Back-end lines are:
  - the sections that start at `# --- NBM vignette` or `# --- EBM vignette` and run to the next `# --- `;
  - any line ending in `# back end`.
- **Links between pages.** Link other pages by their source, `N<nn>_<slug>/index.qmd` or `../index.qmd`. The
  filter `_shared/qmd-links.lua` (in `_quarto.yml`) rewrites them to `index.md` in gfm and `index.html` otherwise,
  because a default-type Quarto project does not; `check_pages.jl --rendered` fails on any `.qmd` link left in an
  `index.md`.
- **Factory form.** The factory counterpart of `node_based(model, net)` is
  `generate_pairwise(sir_model(), net, default_closure(net); cumulative = true)`. `node_based` adds the
  `:cumulative` accumulator, so without `cumulative = true` the two vector fields differ (9 states vs 10).
- **Distinguishable curves.** NetworkEpiCore's recipes style a curve by its representation, so two `:pairwise`
  curves (two closures, say) are drawn alike. A figure with several curves is wrapped in `distinct_styles!(…)`
  (`distinct_styles!(comparisonplot(ref, c1, c2, …))`, or `distinct_styles!(p)` after `plot!(p, c, X)` calls),
  which gives every repeated (colour, line style) the next unused style, the same in every panel and the legend.
- **z∞.** `compare`'s z∞ is max_t |x_det − x̄|/max(se, 10⁻⁴), the largest standardised gap over the grid. It is
  not D∞/SE∞ and need not occur at t(D∞); a page that quotes it says so (N01 prints both times).
- **Numbers.** Every number in the prose comes from `` `{julia} fmt(x)` `` or from a printed table.
- **Lean.** Cite Lean only through `lean("NEP.…")`, which errors unless the name is in
  `NetworkEpiCore.jl/proofs/CITABLE.txt`.
- **No local simulation.** Never simulate, never call NBM's Gillespie code, and never compute a reference ensemble.
  Use `scenario_summary(sc)` only.
- **Render with `render.sh`, not a bare `quarto render`.** One `quarto render` of html (`embed-resources: true`),
  gfm and pdf together deletes `N<nn>_*/index_files/` when it finishes. That leaves `index.md` linking figures that
  no longer exist (they survive only in `_freeze/`). `render.sh` renders gfm in a separate, last pass, and
  `check_pages.jl --rendered` fails on any broken image link.
- **Reference text.** `describe_reference(ref)` and `reference_table(refs)` name the kept fraction by the
  scenario's conditioning (`kept_label`): P(major) for `MajorOutbreak`, P(survival) for `Survival()` (the SIS/SIRS
  scenarios `:sis_reg3` and `:sirs_pois5`), and P(kept) for `Unconditioned()`. These are the labels the
  EdgeBasedModels.jl vignettes use.
- **Anchors.** `anchors(sc)` prints the rates, seeds and time grid, and then:
  - R₀ from NEC (`basic_reproduction_number`, the edge-based T·κ_ex) where it is defined;
  - the pairwise threshold τ_c (`epidemic_threshold(sc.model, sc.network, default_closure(sc.network); p)`) and
    τ/τ_c for strictly SIR/SIS models.

  For SIS and SIRS it says "edge-based R₀ not defined (model outside T_EB)"; for SIS (`:sis_reg3`:
  τ_c = γ/(k − 1) = 0.125, τ/τ_c = 4) τ/τ_c is the threshold quantity. For clustered and dormant descriptors it says
  "R₀ not computed by NetworkEpiCore for this network descriptor" (R₀ exists there; NEC's edge-based formula does
  not apply). Where NBM has no pairwise threshold for the descriptor (`WellMixed`, `MultitypeNetwork`, descriptors
  without a default closure) it says so. Only NEC's `AdmissibilityError` and `ArgumentError` are read as "no such
  quantity", and applicability is checked with `applicable`; any other error (a `MethodError` after a refactor, say)
  is rethrown and fails the render. For `:sirs_pois5` it prints the rates only, since neither quantity is defined
  there. A page that needs a threshold for SIRS derives it on the page.
- **Helpers are not EBM's.** `describe_reference`, `md_table` and `lean` here correspond to EBM's
  `reference_note`, `mdtable` and `lean_cite`, with different wording; only `kept_label` is shared. Mirrored pages
  therefore print the same numbers, but only their shared cells are textually identical.
- **Where the shared cell comes from.** The `shared-cell` of a mirrored page follows DESIGN §E.5 line for line, with
  `model_curves` in place of `curves` (§J.1). It has one deviation. The last two lines are
  `tab = compare(ref, det)` and then `comparisonplot(…)`, so the figure is the cell's value: a Quarto cell shows
  only its last value, and `display(plot)` inside a cell is not captured (Plots opens a GR window, and the render
  hangs). The same lines must appear in the EdgeBasedModels.jl mirror; this is
  requested from V-EBM-infra. Until the mirror has a `shared-cell`, `check_pages.jl` warns, and `--rendered`
  fails.
- **`index.qmd` is never frozen.** It sets `freeze: false`, a deliberate exception to the project's
  `freeze: auto`: it reads the catalogue and the package versions, which change without `index.qmd` changing.
- **The template's rules block** is a `::: {.content-hidden}` div. No format renders it and the checker ignores
  it, so a page may keep it.
