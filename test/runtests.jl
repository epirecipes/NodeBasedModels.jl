# NodeBasedModels test harness.
#
# Runs the golden self-test, then every `test/suites/*.jl` in sorted order. Each
# suite is included into its own fresh module, so suites cannot see each other's
# globals and each file lists the packages it uses. A work package owns its suite
# files (DESIGN_NetworkEpiCore.md §G.1); `00_legacy_<area>.jl` holds the pre-0.2
# tests of an area together with that area's golden check (test/golden/<area>/).
#
# Selecting suites: a selector matches every suite whose file name (without `.jl`)
# contains it; "golden_io" selects the golden self-test.
#
#   julia --project=. -e 'using Pkg; Pkg.test(test_args = ["pairwise", "analysis"])'
#   NBM_TEST_SUITES=graph_level,motif julia --project=. -e 'using Pkg; Pkg.test()'
#
# `--list` (as a test argument) prints the suite names and runs nothing.

using Test

const SUITE_DIR = joinpath(@__DIR__, "suites")
const GOLDEN_SELFTEST = joinpath(@__DIR__, "golden", "selftest.jl")

suite_names() = sort!([splitext(f)[1] for f in readdir(SUITE_DIR) if endswith(f, ".jl")])

function selectors(args = ARGS)
    sel = String[strip(a) for a in args if !isempty(strip(a)) && !startswith(a, "--")]
    for s in split(get(ENV, "NBM_TEST_SUITES", ""), ',')
        isempty(strip(s)) || push!(sel, strip(s))
    end
    return sel
end

function selected(sel)
    names = vcat("golden_io", suite_names())
    isempty(sel) && return names
    chosen = filter(n -> any(s -> occursin(s, n), sel), names)
    isempty(chosen) && error("no test suite matches $(sel); available: $(join(names, ", "))")
    return chosen
end

"""Include `path` into a fresh module inside a testset named `name`."""
function run_isolated(name::AbstractString, path::AbstractString)
    mod = Module(Symbol("Suite_", name))
    # Module() does not define `eval`/`include` for the new module on Julia 1.12
    Core.eval(mod, :(eval(x) = Core.eval($mod, x)))
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    t0 = time()
    @testset "$name" begin
        Base.include(mod, path)
    end
    @info "test suite $name finished in $(round(time() - t0; digits = 1)) s"
    return nothing
end

if "--list" in ARGS
    foreach(println, selected(selectors()))
else
    @testset "NodeBasedModels" begin
        for name in selected(selectors())
            path = name == "golden_io" ? GOLDEN_SELFTEST : joinpath(SUITE_DIR, name * ".jl")
            run_isolated(name, path)
        end
    end
end
