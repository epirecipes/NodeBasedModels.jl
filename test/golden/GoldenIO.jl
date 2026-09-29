# GoldenIO.jl — golden-file I/O and comparison for the NodeBasedModels tests.
#
# A golden freezes the numbers the package produces *today* (known defects
# included) so that refactors can prove they change nothing. Layout:
#
#   test/golden/<area>/cases.jl            case definitions (single source of truth)
#   test/golden/<area>/<case>.toml         metadata, scalars, structure, table index
#   test/golden/<area>/<case>.<table>.csv  numeric tables (trajectories, RHS probes)
#
# Only stdlib I/O is used (TOML, plain CSV with shortest round-trip floats), so the
# files are exact: reading a golden back gives bit-identical Float64 values.
#
# A golden is replaced only by the work package that owns its area, in a
# bug-fix change accompanied by a literature or simulation test
# (DESIGN_NetworkEpiCore.md §G.1). Regenerate with `test/golden/generate.jl`.

module GoldenIO

using TOML
using Test

export GoldenTable, GoldenRecord, GoldenCase
export trajectory_table, probe_table, probe_states
export load_cases, write_record, read_record, compare_records, check_area, generate_area

"""Golden file-format version, written to every TOML file."""
const FORMAT_VERSION = 1

"""Root directory of the golden files (`test/golden`)."""
const GOLDEN_ROOT = @__DIR__

"""Default relative tolerance of every golden comparison (design §G.2: rtol 1e-8)."""
const DEFAULT_RTOL = 1e-8

# ─── Types ────────────────────────────────────────────────────────────────────

"""
    GoldenTable(header, data)

A numeric table: `header[j]` names column `j` of `data` (rows × columns).
Columns are compared by name, so their order may change between versions.
"""
struct GoldenTable
    header::Vector{String}
    data::Matrix{Float64}
    function GoldenTable(header::AbstractVector{<:AbstractString}, data::AbstractMatrix{<:Real})
        length(header) == size(data, 2) || throw(ArgumentError(
            "GoldenTable: $(length(header)) column names for $(size(data, 2)) columns"))
        allunique(header) || throw(ArgumentError("GoldenTable: duplicate column names"))
        for h in header
            occursin(r"[,\n\r\"]", h) && throw(ArgumentError("GoldenTable: bad column name $(repr(h))"))
        end
        new(String.(header), Matrix{Float64}(data))
    end
end

"""
    GoldenRecord(name; meta, scalars, structure, tables, rtol, atol, known_defects, description)

Everything frozen for one golden case.

- `meta` — free-form provenance and inputs (network, closure, parameters, seeding).
  Informational only: it is written but never compared.
- `scalars` — `String => Real | Bool | String` values that are compared
  (numbers with `rtol`/`atol`, others exactly). A thrown error is frozen as the
  string `"throws ArgumentError"` (see [`catch_scalar`](@ref)).
- `structure` — `String => Vector{String}` name sets (unknowns, parameters, ...)
  compared exactly as sorted sets.
- `tables` — `String => GoldenTable` compared column by column.
- `rtol`, `atol` — elementwise tolerance `|a - b| ≤ atol + rtol·max(|a|, |b|)`.
- `known_defects` — IDs from `VERIFIED_ISSUES.md` whose buggy numbers this golden
  deliberately freezes (e.g. `["B01"]`).
"""
struct GoldenRecord
    name::String
    description::String
    known_defects::Vector{String}
    rtol::Float64
    atol::Float64
    meta::Dict{String,Any}
    scalars::Dict{String,Any}
    structure::Dict{String,Vector{String}}
    tables::Dict{String,GoldenTable}
end

function GoldenRecord(name::AbstractString;
                      description::AbstractString = "",
                      known_defects = String[],
                      rtol::Real = DEFAULT_RTOL,
                      atol::Real = 1e-12,
                      meta = Dict{String,Any}(),
                      scalars = Dict{String,Any}(),
                      structure = Dict{String,Vector{String}}(),
                      tables = Dict{String,GoldenTable}())
    occursin(r"^[A-Za-z0-9_\-]+$", name) || throw(ArgumentError("bad golden case name $(repr(name))"))
    GoldenRecord(String(name), String(description), String.(collect(known_defects)),
                 Float64(rtol), Float64(atol),
                 Dict{String,Any}(meta), Dict{String,Any}(scalars),
                 Dict{String,Vector{String}}(k => sort!(String.(collect(v))) for (k, v) in structure),
                 Dict{String,GoldenTable}(tables))
end

"""
    GoldenCase(name, build)

A named golden case; `build()` computes its [`GoldenRecord`](@ref) from the
current package code.
"""
struct GoldenCase
    name::String
    build::Function
end

# ─── Helpers used by the case files ──────────────────────────────────────────

"""
    trajectory_table(t, columns::AbstractDict{<:AbstractString,<:AbstractVector})

Build a table with first column `"t"` followed by the named series (sorted by name).
"""
function trajectory_table(t::AbstractVector, columns::AbstractDict)
    names = sort!(collect(String.(keys(columns))))
    "t" in names && throw(ArgumentError("column name \"t\" is reserved"))
    data = Matrix{Float64}(undef, length(t), length(names) + 1)
    data[:, 1] .= t
    for (j, nm) in enumerate(names)
        v = columns[nm]
        length(v) == length(t) || throw(ArgumentError(
            "series $nm has length $(length(v)), expected $(length(t))"))
        data[:, j + 1] .= v
    end
    return GoldenTable(vcat("t", names), data)
end

"""
    probe_table(states, derivatives)

Build a table of vector-field probes. `states[k]` and `derivatives[k]` are
`name => value` dictionaries for probe `k`; the columns are `"x:<name>"` (the
state) and `"f:<name>"` (the right-hand side there), one row per probe.
"""
function probe_table(states::AbstractVector, derivatives::AbstractVector)
    length(states) == length(derivatives) || throw(ArgumentError("probe count mismatch"))
    names = sort!(collect(String.(keys(first(states)))))
    header = vcat(["x:" * n for n in names], ["f:" * n for n in names])
    data = Matrix{Float64}(undef, length(states), length(header))
    for k in eachindex(states)
        Set(String.(keys(states[k]))) == Set(names) || throw(ArgumentError("probe $k: state names differ"))
        Set(String.(keys(derivatives[k]))) == Set(names) || throw(ArgumentError("probe $k: derivative names differ"))
        for (j, n) in enumerate(names)
            data[k, j] = states[k][n]
            data[k, length(names) + j] = derivatives[k][n]
        end
    end
    return GoldenTable(header, data)
end

"""
    probe_states(names, scale, nprobes; salt = 0) -> Vector{Dict{String,Float64}}

Deterministic, version-independent probe states: every coordinate lies in
`scale · (0.05, 1.0)`. The values come from a 64-bit integer hash of
`(name, probe, salt)`, not from an RNG, so they never change across Julia or
package versions.
"""
function probe_states(names::AbstractVector, scale::Real, nprobes::Integer; salt::Integer = 0)
    out = Vector{Dict{String,Float64}}(undef, nprobes)
    for k in 1:nprobes
        d = Dict{String,Float64}()
        for n in names
            u = _unit_hash(String(n), k, salt)
            d[String(n)] = Float64(scale) * (0.05 + 0.95 * u)
        end
        out[k] = d
    end
    return out
end

# FNV-1a over the bytes of the name, then a splitmix64 finaliser; returns a
# Float64 in [0, 1) built from the top 53 bits.
function _unit_hash(name::String, k::Integer, salt::Integer)
    h = 0xcbf29ce484222325
    for b in codeunits(name)
        h = (h ⊻ UInt64(b)) * 0x00000100000001b3
    end
    h ⊻= UInt64(k) * 0x9e3779b97f4a7c15
    h ⊻= UInt64(salt) * 0xbf58476d1ce4e5b9
    h = (h ⊻ (h >> 30)) * 0xbf58476d1ce4e5b9
    h = (h ⊻ (h >> 27)) * 0x94d049bb133111eb
    h ⊻= h >> 31
    return Float64(h >> 11) / 9.007199254740992e15
end

"""
    catch_scalar(f)

Evaluate `f()` and return its value, or the string `"throws <ExceptionType>"` when
it throws. Lets a golden freeze "this call errors" alongside numbers.
"""
function catch_scalar(f)
    try
        return f()
    catch err
        return "throws " * string(nameof(typeof(err)))
    end
end
export catch_scalar

# ─── CSV / TOML I/O ──────────────────────────────────────────────────────────

_fmt(x::Float64) = isnan(x) ? "NaN" : string(x)       # shortest round-trip repr

function _write_csv(path::AbstractString, tab::GoldenTable)
    open(path, "w") do io
        println(io, join(tab.header, ','))
        for i in axes(tab.data, 1)
            println(io, join((_fmt(tab.data[i, j]) for j in axes(tab.data, 2)), ','))
        end
    end
end

function _read_csv(path::AbstractString)
    lines = filter!(!isempty, readlines(path))
    header = split(lines[1], ',')
    data = Matrix{Float64}(undef, length(lines) - 1, length(header))
    for (i, ln) in enumerate(@view lines[2:end])
        fields = split(ln, ',')
        length(fields) == length(header) || error("$path line $(i + 1): $(length(fields)) fields, expected $(length(header))")
        for j in eachindex(fields)
            data[i, j] = parse(Float64, fields[j])
        end
    end
    return GoldenTable(String.(header), data)
end

# TOML cannot hold `nothing`; Inf/NaN are written as TOML `inf`/`nan`.
_tomlable(x::Real) = x isa Bool ? x : (x isa Integer ? Int(x) : Float64(x))
_tomlable(x::AbstractString) = String(x)
_tomlable(x::Symbol) = String(x)
_tomlable(x::AbstractVector) = [_tomlable(v) for v in x]
_tomlable(x::Tuple) = [_tomlable(v) for v in x]
_tomlable(x::AbstractDict) = Dict{String,Any}(string(k) => _tomlable(v) for (k, v) in x)
_tomlable(x::Nothing) = "nothing"
_tomlable(x) = string(x)

area_dir(area::AbstractString; root::AbstractString = GOLDEN_ROOT) = joinpath(root, area)

"""
    write_record(area, rec; provenance = Dict(), root = GOLDEN_ROOT) -> Vector{String}

Write `rec` as `<root>/<area>/<name>.toml` plus one CSV per table; returns the paths.
"""
function write_record(area::AbstractString, rec::GoldenRecord; provenance = Dict{String,Any}(),
                      root::AbstractString = GOLDEN_ROOT)
    dir = area_dir(area; root = root)
    mkpath(dir)
    # drop this case's old tables first, so a removed table leaves no stray CSV
    for f in readdir(dir)
        if startswith(f, rec.name * ".") && endswith(f, ".csv") && count(==('.'), f) == 2
            rm(joinpath(dir, f))
        end
    end
    paths = String[]
    tables_index = Dict{String,Any}()
    for (tname, tab) in rec.tables
        file = "$(rec.name).$(tname).csv"
        _write_csv(joinpath(dir, file), tab)
        push!(paths, joinpath(dir, file))
        tables_index[tname] = Dict{String,Any}("file" => file, "rows" => size(tab.data, 1),
                                               "columns" => tab.header)
    end
    doc = Dict{String,Any}(
        "format" => FORMAT_VERSION,
        "area" => String(area),
        "name" => rec.name,
        "description" => rec.description,
        "known_defects" => rec.known_defects,
        "rtol" => rec.rtol,
        "atol" => rec.atol,
        "meta" => _tomlable(rec.meta),
        "scalars" => _tomlable(rec.scalars),
        "structure" => _tomlable(rec.structure),
        "tables" => tables_index,
        "provenance" => _tomlable(provenance),
    )
    tpath = joinpath(dir, "$(rec.name).toml")
    open(tpath, "w") do io
        println(io, "# Golden for NodeBasedModels test area \"$area\". Generated by test/golden/generate.jl;")
        println(io, "# do not edit by hand. Replace only in a bug-fix change of the owning work package.")
        TOML.print(io, doc; sorted = true)
    end
    push!(paths, tpath)
    return paths
end

"""
    read_record(area, name; root = GOLDEN_ROOT) -> GoldenRecord

Read a golden written by [`write_record`](@ref).
"""
function read_record(area::AbstractString, name::AbstractString; root::AbstractString = GOLDEN_ROOT)
    dir = area_dir(area; root = root)
    doc = TOML.parsefile(joinpath(dir, "$name.toml"))
    doc["format"] == FORMAT_VERSION || error("golden $area/$name: format $(doc["format"]) ≠ $FORMAT_VERSION")
    tables = Dict{String,GoldenTable}()
    for (tname, entry) in get(doc, "tables", Dict{String,Any}())
        tab = _read_csv(joinpath(dir, entry["file"]))
        tab.header == entry["columns"] || error("golden $area/$name: CSV header of $tname disagrees with TOML index")
        size(tab.data, 1) == entry["rows"] || error("golden $area/$name: $tname has $(size(tab.data, 1)) rows, index says $(entry["rows"])")
        tables[tname] = tab
    end
    return GoldenRecord(doc["name"];
                        description = get(doc, "description", ""),
                        known_defects = get(doc, "known_defects", String[]),
                        rtol = doc["rtol"], atol = doc["atol"],
                        meta = get(doc, "meta", Dict{String,Any}()),
                        scalars = get(doc, "scalars", Dict{String,Any}()),
                        structure = get(doc, "structure", Dict{String,Any}()),
                        tables = tables)
end

# ─── Comparison ───────────────────────────────────────────────────────────────

function _close(a::Real, b::Real, rtol, atol)
    (isnan(a) || isnan(b)) && return isnan(a) && isnan(b)
    (isinf(a) || isinf(b)) && return a == b
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))
end

function _scalar_equal(ref, new, rtol, atol)
    if ref isa Real && !(ref isa Bool) && new isa Real && !(new isa Bool)
        return _close(Float64(ref), Float64(new), rtol, atol)
    end
    return _tomlable(ref) == _tomlable(new)
end

"""
    compare_records(ref, new) -> Vector{String}

Compare a stored golden `ref` with a freshly computed `new` and return a list of
human-readable mismatches (empty when they agree). Tolerances come from `ref`.
"""
function compare_records(ref::GoldenRecord, new::GoldenRecord)
    problems = String[]
    rtol, atol = ref.rtol, ref.atol
    # structure: exact sorted name sets
    for key in sort!(collect(union(keys(ref.structure), keys(new.structure))))
        r = get(ref.structure, key, nothing)
        n = get(new.structure, key, nothing)
        if r === nothing || n === nothing
            push!(problems, "structure[$key] present only in $(r === nothing ? "new" : "golden")")
        elseif r != n
            push!(problems, "structure[$key]: golden-only $(setdiff(r, n)), new-only $(setdiff(n, r))")
        end
    end
    # scalars
    for key in sort!(collect(union(keys(ref.scalars), keys(new.scalars))))
        if !haskey(ref.scalars, key) || !haskey(new.scalars, key)
            push!(problems, "scalar $key present only in $(haskey(ref.scalars, key) ? "golden" : "new")")
        elseif !_scalar_equal(ref.scalars[key], new.scalars[key], rtol, atol)
            push!(problems, "scalar $key: golden $(repr(ref.scalars[key])), new $(repr(new.scalars[key]))")
        end
    end
    # tables: by column name, elementwise
    for tname in sort!(collect(union(keys(ref.tables), keys(new.tables))))
        if !haskey(ref.tables, tname) || !haskey(new.tables, tname)
            push!(problems, "table $tname present only in $(haskey(ref.tables, tname) ? "golden" : "new")")
            continue
        end
        rt, nt = ref.tables[tname], new.tables[tname]
        if Set(rt.header) != Set(nt.header)
            push!(problems, "table $tname columns: golden-only $(setdiff(rt.header, nt.header)), new-only $(setdiff(nt.header, rt.header))")
            continue
        end
        if size(rt.data, 1) != size(nt.data, 1)
            push!(problems, "table $tname: $(size(rt.data, 1)) rows in golden, $(size(nt.data, 1)) new")
            continue
        end
        ncol = Dict(h => j for (j, h) in enumerate(nt.header))
        nbad = 0
        worst = (0.0, "", 0, 0.0, 0.0)
        for (j, h) in enumerate(rt.header)
            a = view(rt.data, :, j)
            b = view(nt.data, :, ncol[h])
            for i in eachindex(a, b)
                if !_close(a[i], b[i], rtol, atol)
                    nbad += 1
                    excess = abs(a[i] - b[i]) / (atol + rtol * max(abs(a[i]), abs(b[i])))
                    excess = isnan(excess) ? Inf : excess
                    excess > worst[1] && (worst = (excess, h, i, a[i], b[i]))
                end
            end
        end
        if nbad > 0
            push!(problems, "table $tname: $nbad entries outside tolerance; worst column $(worst[2]) row $(worst[3]): " *
                            "golden $(worst[4]), new $(worst[5])")
        end
    end
    return problems
end

# ─── Area drivers ─────────────────────────────────────────────────────────────

"""
    load_cases(area) -> Vector{GoldenCase}

Include `<area>/cases.jl` into a fresh module (with `GoldenIO` in scope) and
return its `golden_cases()`.
"""
function load_cases(area::AbstractString)
    mod = Module(Symbol("GoldenCases_", area))
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    Core.eval(mod, :(const GoldenIO = $(@__MODULE__)))
    Core.eval(mod, :(using .GoldenIO))
    Base.include(mod, joinpath(area_dir(area), "cases.jl"))
    golden_cases = Base.invokelatest(getglobal, mod, :golden_cases)
    cases = Base.invokelatest(golden_cases)
    names = [c.name for c in cases]
    allunique(names) || error("golden area $area: duplicate case names")
    return cases
end

_case_filter(only) = only === nothing ? (_ -> true) : (n -> n in only)

"""
    check_area(area; only = nothing)

Inside a `@testset`, recompute every golden case of `area` and compare it with the
stored files: one test per case, plus a test that no stored golden is orphaned.
`only` restricts the check to the listed case names.
"""
function check_area(area::AbstractString; only = nothing)
    cases = load_cases(area)
    keep = _case_filter(only)
    @testset "golden $area" begin
        for c in cases
            keep(c.name) || continue
            @testset "$(c.name)" begin
                path = joinpath(area_dir(area), "$(c.name).toml")
                @test isfile(path)
                if isfile(path)
                    ref = read_record(area, c.name)
                    new = Base.invokelatest(c.build)
                    @test new.name == c.name
                    problems = compare_records(ref, new)
                    isempty(problems) ||
                        @error "golden $area/$(c.name) differs from the stored file" problems
                    @test isempty(problems)
                end
            end
        end
        if only === nothing
            stored = [splitext(f)[1] for f in readdir(area_dir(area)) if endswith(f, ".toml")]
            orphans = setdiff(stored, [c.name for c in cases])
            isempty(orphans) || @error "golden $area: stored goldens without a case" orphans
            @test isempty(orphans)
            indexed = Set{String}()
            for n in stored
                doc = TOML.parsefile(joinpath(area_dir(area), "$n.toml"))
                foreach(e -> push!(indexed, e["file"]), values(get(doc, "tables", Dict{String,Any}())))
            end
            stray = setdiff([f for f in readdir(area_dir(area)) if endswith(f, ".csv")], indexed)
            isempty(stray) || @error "golden $area: CSV files not indexed by any golden" stray
            @test isempty(stray)
        end
    end
end

"""
    generate_area(area; only = nothing, provenance = Dict()) -> Vector{String}

Recompute the golden cases of `area` (optionally only the names in `only`) and
write them to disk. Returns the written paths.
"""
function generate_area(area::AbstractString; only = nothing, provenance = Dict{String,Any}())
    cases = load_cases(area)
    keep = _case_filter(only)
    written = String[]
    for c in cases
        keep(c.name) || continue
        rec = Base.invokelatest(c.build)
        rec.name == c.name || error("case $(c.name) built a record named $(rec.name)")
        append!(written, write_record(area, rec; provenance = provenance))
    end
    return written
end

end # module GoldenIO
