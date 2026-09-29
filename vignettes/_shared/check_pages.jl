# Owner: V-NBM-infra (DESIGN_NetworkEpiCore.md §F rules, §E.5; amendments §J–§M).
#
# Static checks of the NodeBasedModels.jl vignette pages against the catalogue `_shared/pages.toml` and the
# vignette rules. Run from `NodeBasedModels.jl/vignettes`:
#
#     julia --project=. _shared/check_pages.jl              # source checks of every existing page
#     julia --project=. _shared/check_pages.jl --rendered   # also the rendered outputs and _freeze entries
#     julia --project=. _shared/check_pages.jl N01 N02      # only these pages
#     julia --project=. _shared/check_pages.jl --self-test  # regression cases of the checker's own parsers
#
# It exits with status 1 if any check fails, printing one line per failure. A page that does not exist yet is
# reported as "in preparation" and is not an error. Checks per page (the rule numbers are those of the vignette
# rules in DESIGN §F and the phase-4 task text):
#
#  - the page is `<id>_<slug>/index.qmd`, exactly one directory per id, and its front-matter title is the
#    catalogue title;
#  - its first code cell includes `_shared/setup.jl`;
#  - (1) a page with layers L and F asserts equality of the two forms (`vector_fields_equal` or `isequivalent`);
#  - (2) a page with `summaries = true` calls `scenario_summary`, `comparisonplot` and `compare`, states the
#    reference (`describe_reference` or `reference_table`) and calls `require_summaries`;
#  - (2) every catalogue scenario of the page is used in the code outside the `require_summaries([...])` list (a
#    listed scenario must actually be used), and every `scenario(:id)` / `scenario_summary(:id)` of the page is a
#    registered id with a valid committed summary (when it is compared);
#  - (4) every `NEP.*` name is listed in NetworkEpiCore.jl/proofs/CITABLE.txt, and the page never says
#    "certified";
#  - (5) for each mirror E of the page, one shared cell: the cell labelled `shared-cell-E` if there is one (a page
#    mirroring several EBM pages whose cells differ, such as N02 with E03 and E06), else `shared-cell`. Its
#    `scenario(:id)`s must be catalogue scenarios of the page. When the mirrored EdgeBasedModels.jl page exists, its
#    cell for this page (`shared-cell-<Nxx>`, else `shared-cell`) must be identical apart from the back-end lines,
#    with the same scenario ids. A mirror page that exists but has no such cell cannot be compared: a warning in
#    source mode, a FAILURE with `--rendered` (so the final render never passes an unenforced rule 5);
#  - (6) no min–max bands;
#  - DESIGN §F rule 5: no use of NodeBasedModels' own Gillespie code and no `_validation.jl`;
#  - with `--rendered`: `index.html` and `index.md` (gfm) exist and are newer than `index.qmd`, `_freeze/<dir>`
#    exists, the rendered Markdown contains no error, warning or missing-summary text and never says "certif…"
#    (prose or code output), links no `.qmd` source (nor does the top-level `index.md`), and every image target of
#    `index.md` (Markdown `![](…)`, parsed with balanced brackets so that citations and intervals in captions are
#    allowed, and HTML `<img src="…">`) is an existing file (a single-pass render of html + gfm deletes
#    `index_files/`; see `_shared/render.sh`).
#
# Text inside HTML comments and `::: {.content-hidden}` blocks (for example the template's rules block) is not
# page text and is ignored.
#
# Back-end lines of a shared cell (§E.5) are (a) the lines of a section that starts with a comment line
# `# --- EBM vignette` or `# --- NBM vignette` and runs to the next `# --- ` line, and (b) any line ending in the
# comment `# back end`. Cell option lines (`#| …`) are ignored.

using TOML
using NetworkEpiCore
using NetworkOutbreaks

const VIGNETTES = normpath(joinpath(@__DIR__, ".."))
const EBM_VIGNETTES = normpath(joinpath(VIGNETTES, "..", "..", "EdgeBasedModels.jl", "vignettes"))
const CATALOGUE = TOML.parsefile(joinpath(@__DIR__, "pages.toml"))
const CITABLE = Set(strip(l) for l in eachline(joinpath(pkgdir(NetworkEpiCore), "proofs", "CITABLE.txt"))
                    if !isempty(strip(l)) && !startswith(strip(l), "#"))

"""
    page_dirs(root, id) -> Vector{String}

The directories `<id>_*` of `root` (for example `N01_reaction_network_to_pairwise`) that hold an `index.qmd`.
"""
page_dirs(root::AbstractString, id::AbstractString) =
    isdir(root) ? sort!([d for d in readdir(root) if startswith(d, id * "_") && isfile(joinpath(root, d, "index.qmd"))]) :
    String[]

# The code cells of a .qmd file: (label or nothing, lines).
function code_cells(src::AbstractString)
    cells = Tuple{Union{Nothing,String},Vector{String}}[]
    lines = split(src, '\n')
    i = 1
    while i <= length(lines)
        if occursin(r"^```\{julia", lines[i])
            j = i + 1
            body = String[]
            while j <= length(lines) && !startswith(lines[j], "```")
                push!(body, lines[j])
                j += 1
            end
            label = nothing
            for l in body
                m = match(r"^#\|\s*label:\s*(\S+)", l)
                m === nothing || (label = String(m.captures[1]))
            end
            push!(cells, (label, body))
            i = j + 1
        else
            i += 1
        end
    end
    return cells
end

# The shared-cell lines without cell options and back-end lines (§E.5).
function shared_core(body::Vector{String})
    out = String[]
    backend = false
    for l in body
        startswith(l, "#|") && continue
        if occursin(r"^\s*# ---", l)
            backend = occursin(r"^\s*# --- (EBM|NBM) vignette", l)
            backend && continue
        end
        backend && continue
        occursin(r"#\s*back end\s*$", l) && continue
        push!(out, rstrip(l))
    end
    while !isempty(out) && isempty(last(out))
        pop!(out)
    end
    return out
end

# The page text without HTML comments and `::: {.content-hidden}` blocks (not rendered, so not page text).
strip_hidden(src::AbstractString) =
    replace(replace(src, r"(?s)<!--.*?-->" => ""), r"(?ms)^:::+\s*\{\.content-hidden[^}]*\}\s*$.*?^:::+\s*$" => "")

# Index just after the `]` that closes the `[` at `i` (brackets balanced, backslash escapes and inline code spans
# skipped, as in CommonMark link text), or `nothing`.
function _close_bracket(md::String, i::Int)
    depth = 0
    j = i
    while j <= ncodeunits(md)
        c = md[j]
        if c == '\\'
            j = nextind(md, j)                         # skip the escaped character too
            j <= ncodeunits(md) && (j = nextind(md, j))
            continue
        elseif c == '`'
            k = j
            while k <= ncodeunits(md) && md[k] == '`'
                k += 1
            end
            run = md[j:k-1]
            e = findnext(run, md, k)
            if e === nothing
                j = k                    # an unmatched backtick run is literal text
            else
                j = last(e) + 1
            end
            continue
        elseif c == '['
            depth += 1
        elseif c == ']'
            depth -= 1
            depth == 0 && return nextind(md, j)
        elseif c == '\n' && j + 1 <= ncodeunits(md) && md[j+1] == '\n'
            return nothing               # link text never spans a blank line
        end
        j = nextind(md, j)
    end
    return nothing
end

# The destination of an inline link `(dest "title")` starting at `md[i] == '('`: `<...>`, or a run of non-space
# characters with balanced parentheses. `nothing` if there is none.
function _destination(md::String, i::Int)
    (i <= ncodeunits(md) && md[i] == '(') || return nothing
    j = nextind(md, i)
    while j <= ncodeunits(md) && md[j] in (' ', '\t', '\n')
        j = nextind(md, j)
    end
    j <= ncodeunits(md) || return nothing
    if md[j] == '<'
        e = findnext('>', md, j)
        e === nothing && return nothing
        return md[nextind(md, j):prevind(md, e)]
    end
    depth = 0
    k = j
    while k <= ncodeunits(md)
        c = md[k]
        if c == '\\'
            k = nextind(md, k)
            k <= ncodeunits(md) && (k = nextind(md, k))
            continue
        elseif c == '('
            depth += 1
        elseif c == ')'
            depth == 0 && break
            depth -= 1
        elseif isspace(c)
            break
        end
        k = nextind(md, k)
    end
    k > j || return nothing
    return md[j:prevind(md, k)]
end

"""
    md_links(md) -> Vector{Tuple{Bool,String}}

The inline links of Markdown text as `(is_image, target)` pairs: `[text](target)` and `![alt](target)`, with an
optional `"title"`. The link text is matched with balanced brackets (escapes and code spans skipped), so a caption
holding a citation `[Keeling 1999](#ref-keeling1999)` or an interval `[0, 60]` does not hide the image target; a
nested `[![a](img)](url)` yields both. Fenced code blocks are skipped (code output is not a link).
"""
function md_links(md::AbstractString)
    txt = String(replace(md, r"(?ms)^(```+|~~~+)[^\n]*\n.*?^\1\s*$" => ""))
    out = Tuple{Bool,String}[]
    i = 1
    while (i = findnext('[', txt, i)) !== nothing
        after = _close_bracket(txt, i)
        if after !== nothing
            dest = _destination(txt, after)
            dest === nothing || push!(out, (i > 1 && txt[prevind(txt, i)] == '!', dest))
        end
        i = nextind(txt, i)
    end
    return out
end

_local_target(t) = !occursin(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|#)", t)   # not a URL (http:, data:, mailto:) or #anchor
_strip_frag(t) = String(first(split(t, r"[#?]")))

"""
    image_targets(md) -> Vector{String}

The local image targets of a rendered Markdown file: `![alt](target)` / `![alt](target "title")` (parsed by
[`md_links`](@ref), so brackets and links inside the caption are allowed) and `<img src="target">`, without URLs
(`http:`, `https:`, `data:`) and with any `#fragment` or `?query` removed.
"""
function image_targets(md::AbstractString)
    ts = [t for (img, t) in md_links(md) if img]
    for m in eachmatch(r"<img\b[^>]*\bsrc\s*=\s*\"([^\"]+)\"", md)
        push!(ts, m.captures[1])
    end
    ts = [_strip_frag(t) for t in ts if _local_target(t)]
    return unique!(filter!(!isempty, ts))
end

"""
    qmd_links(md) -> Vector{String}

The local (non-image) link targets of rendered Markdown that still point at a `.qmd` source: the gfm output
should link `index.md` (the `_shared/qmd-links.lua` filter rewrites them; a default-type Quarto project does not).
"""
qmd_links(md::AbstractString) =
    unique!([t for (img, t) in md_links(md) if !img && _local_target(t) && endswith(_strip_frag(t), ".qmd")])

# The `title:` of the YAML front matter, without quotes (nothing if absent).
function front_title(src)
    fm = match(r"(?s)\A---\s*\n(.*?)\n---", src)
    fm === nothing && return nothing
    m = match(r"(?m)^title:\s*(.*?)\s*$", fm.captures[1])
    m === nothing && return nothing
    return String(strip(m.captures[1], ['"', '\'']))
end

# The shared cell(s) of a page for its mirror `other`: those labelled `shared-cell-<other>`, else `shared-cell`.
function shared_cells(cells, other::AbstractString)
    own = [b for (l, b) in cells if l == "shared-cell-" * other]
    return isempty(own) ? [b for (l, b) in cells if l == "shared-cell"] : own
end

# The scenario ids a piece of code builds or looks up: `scenario(:id)` and `scenario_summary(:id)`, and, when the code
# looks a scenario up through a variable (`scenario(id)` in `Dict(id => scenario(id) for id in ids)`), the symbols of
# every `name = [:a, :b, …]` list literal that `name` is then iterated over (`for … in name`) in the same code.
function scenario_refs(code::AbstractString)
    ids = Set(Symbol(m.captures[1]) for m in eachmatch(r"scenario(?:_summary)?\(\s*:([A-Za-z0-9_]+)", code))
    if occursin(r"scenario(?:_summary)?\(\s*[A-Za-z_]", code)
        for m in eachmatch(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[((?:\s*:[A-Za-z0-9_]+\s*,?)+)\]", code)
            occursin(Regex("\\bin\\s+" * m.captures[1] * "\\b"), code) || continue
            union!(ids, Symbol(x.captures[1]) for x in eachmatch(r":([A-Za-z0-9_]+)", m.captures[2]))
        end
    end
    return ids
end

function check_page(id::String, entry::Dict, rendered::Bool, errs::Vector{String}, registered::Set{Symbol})
    err(msg) = push!(errs, "$(id): $(msg)")
    dirs = page_dirs(VIGNETTES, id)
    if isempty(dirs)
        println("  $(id)  in preparation (no $(id)_*/index.qmd)")
        return
    end
    length(dirs) == 1 || (err("several page directories: $(join(dirs, ", "))"); return)
    dir = only(dirs)
    path = joinpath(VIGNETTES, dir, "index.qmd")
    src = strip_hidden(read(path, String))
    
    cells = code_cells(src)
    code = join((join(b, '\n') for (_, b) in cells), '\n')
    prose = replace(src, r"(?s)```\{julia.*?```" => "")

    t = front_title(src)
    t == entry["title"] || err("front-matter title $(repr(t)) is not the catalogue title $(repr(entry["title"]))")
    (!isempty(cells) && occursin("_shared", join(cells[1][2], '\n')) && occursin("setup.jl", join(cells[1][2], '\n'))) ||
        err("the first code cell does not include _shared/setup.jl")

    layers = String.(entry["layers"])
    if "L" in layers && "F" in layers
        occursin(r"vector_fields_equal|isequivalent", code) ||
            err("layers L and F but no vector_fields_equal / isequivalent assertion (rule 1)")
    end
    usecode = replace(code, r"(?s)require_summaries\(\s*\[.*?\]\s*\)" => "")   # the list itself is not a use
    for s in String.(entry["scenarios"])
        occursin(Regex(":" * s * "\\b"), usecode) ||
            err("catalogue scenario :$(s) is never used in the code (only listed in require_summaries, or absent)")
    end
    used = scenario_refs(code)
    for s in used
        s in registered || err("scenario(:$(s)) is not a registered scenario id")
    end
    if entry["summaries"]
        for f in ("scenario_summary", "comparisonplot", "compare(", "require_summaries")
            occursin(f, code) || err("compares with simulation but never calls $(rstrip(f, '(')) (rule 2)")
        end
        occursin(r"describe_reference|reference_table", code) ||
            err("does not state N, runs, conditioning and P(major) (describe_reference / reference_table; rule 2)")
        ids = [s for s in used if s in registered]
        for (k, v) in missing_scenario_summaries(ids; companions = false)
            err("no valid committed summary for :$(k): $(v)")
        end
    end
    for m in eachmatch(r"NEP\.[A-Za-z_][A-Za-z0-9_.']*", src)
        n = rstrip(m.match, '.')
        n in CITABLE || err("cites $(n), which is not in CITABLE.txt (rule 4)")
    end
    occursin(r"(?i)certif", prose) && err("the prose says \"certified\" (rule 4)")
    occursin(r"(?i)min[–-]max|minimum[–-]maximum", src) && err("mentions a min–max band (rule 6)")
    occursin(r"\bgillespie\w*\s*\(", code) && err("calls NodeBasedModels' own Gillespie code (DESIGN §F rule 5)")
    occursin("_validation.jl", src) && err("uses _validation.jl (deleted)")

    mirrors = String.(entry["mirrors"])
    for e in mirrors
        # One shared cell per mirrored pair: `shared-cell-<E>` if the page has one for this mirror (needed when a page
        # mirrors several EBM pages whose shared cells differ, e.g. N02 with E03 and E06), else `shared-cell`.
        mine = shared_cells(cells, e)
        if length(mine) != 1
            err("mirrors $(e) but has $(length(mine)) cells labelled shared-cell-$(e) / shared-cell (rule 5)")
            continue
        end
        # Rule 5/7: the pair shares scenario ids. The shared cell's scenarios must be catalogue scenarios of this page
        # (identity with the mirror's cell then gives the same ids on both sides).
        cell_ids = scenario_refs(join(only(mine), '\n'))
        isempty(cell_ids) && err("shared cell for $(e) uses no scenario(:id) (rule 5/7: same scenario ids)")
        for s in cell_ids
            string(s) in entry["scenarios"] || err("shared cell for $(e) uses :$(s), not a catalogue scenario of $(id)")
        end
        edirs = page_dirs(EBM_VIGNETTES, e)
        if length(edirs) != 1
            println("  $(id)  mirror $(e): EdgeBasedModels page not present yet; shared cell not compared")
            continue
        end
        theirs = shared_cells(code_cells(read(joinpath(EBM_VIGNETTES, only(edirs), "index.qmd"), String)), id)
        if length(theirs) != 1
            msg = "mirror $(e): $(only(edirs)) exists but has $(length(theirs)) cells labelled " *
                  "shared-cell-$(id) / shared-cell, so rule 5 cannot be checked"
            rendered ? err(msg * " (rule 5)") : println("  $(id)  WARNING $(msg); FAILS under --rendered")
            continue
        end
        a, b = shared_core(only(mine)), shared_core(only(theirs))
        if a != b
            k = findfirst(i -> i > min(length(a), length(b)) || a[i] != b[i], 1:max(length(a), length(b)))
            err("shared cell differs from $(e) ($(only(edirs))) apart from back-end lines, first at line " *
                "$(k): $(repr(get(a, k, "<none>"))) vs $(repr(get(b, k, "<none>"))) (rule 5)")
        end
        ebm_ids = scenario_refs(join(only(theirs), '\n'))
        ebm_ids == cell_ids || err("shared cell scenario ids $(sort!(collect(cell_ids))) differ from $(e)'s " *
                                   "$(sort!(collect(ebm_ids))) (rule 5/7)")
    end

    if rendered
        for f in ("index.html", "index.md")
            p = joinpath(VIGNETTES, dir, f)
            if !isfile(p)
                err("not rendered: $(f) missing")
            elseif mtime(p) < mtime(path)
                err("$(f) is older than index.qmd (re-render)")
            end
        end
        isdir(joinpath(VIGNETTES, "_freeze", dir)) || err("no _freeze/$(dir) entry")
        md = joinpath(VIGNETTES, dir, "index.md")
        if isfile(md)
            out = read(md, String)
            for pat in (r"no valid committed summary", r"(?m)^\s*(?:┌ )?Warning:", r"(?m)^\s*(?:┌ )?Error:",
                        r"(?m)^\s*ERROR:", r"LoadError", r"MethodError", r"UndefVarError")
                occursin(pat, out) && err("rendered output contains $(pat.pattern)")
            end
            occursin(r"(?i)certif", out) && err("the rendered page (prose or code output) says \"certif…\" (rule 4)")
            for t in qmd_links(out)
                err("index.md links $(t), a .qmd source (is _shared/qmd-links.lua in _quarto.yml filters?)")
            end
            for t in image_targets(out)
                isfile(joinpath(VIGNETTES, dir, t)) ||
                    err("index.md links the image $(t), which does not exist (render gfm last: _shared/render.sh)")
            end
        end
    end
    println("  $(id)  $(dir): checked")
    return
end

"""
    self_test() -> Bool

Regression cases of the checker's own parsers (run with `--self-test`; `_shared/render.sh` runs it first). Each case
prints PASS/FAIL; returns whether all passed.
"""
function self_test()
    cases = Pair{String,Any}[
        # the reviewer's case: a gfm caption with a citation link and an interval must not hide the image target
        "citation + interval in caption" =>
            (image_targets("![Pairwise vs NO ([Keeling 1999](#ref-keeling1999)) on t ∈ [0, 60].](index_files/" *
                           "figure-commonmark/x.svg)\n\n![plain](index_files/y.svg)"),
             ["index_files/figure-commonmark/x.svg", "index_files/y.svg"]),
        "interval only" => (image_targets("![I(t), t ∈ [0, 60]](a.png)"), ["a.png"]),
        "title, fragment, query" => (image_targets("![a](b.png \"t\") ![c](d.svg#f) ![e](g.png?x=1)"),
                                     ["b.png", "d.svg", "g.png"]),
        "angle-bracket target" => (image_targets("![a](<dir/my fig.png>)"), ["dir/my fig.png"]),
        "parentheses in target" => (image_targets("![a](f(1).png)"), ["f(1).png"]),
        "code span with ] in caption" => (image_targets("![a `x]` b](c.png)"), ["c.png"]),
        "escaped ] in caption" => (image_targets("![a \\] b](e.png)"), ["e.png"]),
        "image inside a link" => (image_targets("[![a](i.png)](https://x.org)"), ["i.png"]),
        "URLs and data dropped" => (image_targets("![a](https://x/y.png) ![b](data:image/png;base64,AA)"), String[]),
        "html img" => (image_targets("<img src=\"index_files/z.png\" width=\"50%\">"), ["index_files/z.png"]),
        "no image across a blank line" => (image_targets("![a\n\nb](c.png)"), String[]),
        "fenced code is not a link" => (image_targets("```\n![a](code.png)\n```\n![b](real.png)"), ["real.png"]),
        "a link is not an image" => (image_targets("[a](not_an_image.png)"), String[]),
        "qmd links found" => (qmd_links("[p](N01_x/index.qmd) [q](N02_y/index.md) [r](../index.qmd#top) " *
                                        "![s](fig.qmd) [u](https://h/x.qmd)"),
                              ["N01_x/index.qmd", "../index.qmd#top"]),
        "qmd link with cited caption" => (qmd_links("[Page ([Keeling 1999](#ref-k)) [0, 1]](N03_z/index.qmd)"),
                                          ["N03_z/index.qmd"]),
        "shared_core drops back-end lines" =>
            (shared_core(["#| label: shared-cell", "a = 1", "# --- NBM vignette ---", "nbm()", "# --- both ---",
                          "b = 2   # back end", "c = 3", ""]), ["a = 1", "# --- both ---", "c = 3"]),
        "shared_cells prefers the per-mirror label" =>
            (shared_cells([("shared-cell", ["x"]), ("shared-cell-E06", ["y"])], "E06"), [["y"]]),
        "shared_cells falls back to shared-cell" =>
            (shared_cells([("shared-cell", ["x"]), ("shared-cell-E06", ["y"])], "E03"), [["x"]]),
        "scenario_refs" => (scenario_refs("sc = scenario(:sir_pois5); r = scenario_summary( :sir_reg6)"),
                            Set([:sir_pois5, :sir_reg6])),
        "scenario_refs through an id list" =>
            (scenario_refs("ids = [:sir_reg6, :sir_pl]\nscs = Dict(id => scenario(id) for id in ids)\nobs = [:I, :R]"),
             Set([:sir_reg6, :sir_pl])),
        "scenario_refs ignores a list never iterated" =>
            (scenario_refs("obs = [:I, :R]\nsc = scenario(:sir_pois5)\nx = scenario(s)"), Set([:sir_pois5])),
    ]
    ok = true
    for (name, (got, want)) in cases
        pass = got == want
        ok &= pass
        println(pass ? "  PASS " : "  FAIL ", name, pass ? "" : ": got $(repr(got)), want $(repr(want))")
    end
    println(ok ? "self-test: all $(length(cases)) cases passed" : "self-test: FAILED")
    return ok
end

function main(args)
    "--self-test" in args && return self_test()
    rendered = "--rendered" in args
    only_ids = [a for a in args if !startswith(a, "--")]
    ids = isempty(only_ids) ? sort!(collect(keys(CATALOGUE))) : only_ids
    registered = Set(scenario_ids())
    errs = String[]
    for id in ids
        haskey(CATALOGUE, id) || (push!(errs, "$(id): not in _shared/pages.toml"); continue)
        check_page(id, CATALOGUE[id], rendered, errs, registered)
    end
    stray = [d for d in readdir(VIGNETTES) if isdir(joinpath(VIGNETTES, d)) && occursin(r"^\d\d_", d)]
    isempty(stray) || push!(errs, "old vignette directories still present: $(join(stray, ", "))")
    isfile(joinpath(VIGNETTES, "_validation.jl")) && push!(errs, "_validation.jl still present")
    if rendered   # the landing page links every page: its gfm output must link index.md, not the .qmd sources
        top = joinpath(VIGNETTES, "index.md")
        isfile(top) ? foreach(t -> push!(errs, "index.md (top level) links $(t), a .qmd source"), qmd_links(read(top, String))) :
                      push!(errs, "index.md (top level) not rendered")
    end
    if isempty(errs)
        println("check_pages: all checks passed ($(length(ids)) catalogue entries)")
    else
        println("check_pages: $(length(errs)) failure(s)")
        foreach(e -> println("  FAIL ", e), errs)
    end
    return isempty(errs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS) || exit(1)
end
