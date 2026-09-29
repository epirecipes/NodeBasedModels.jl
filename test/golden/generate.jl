# generate.jl — (re)write the NodeBasedModels golden files.
#
# Usage, from the package root (the package environment has every dependency the
# case files need; TOML and Test come from the standard library):
#
#   julia --project=. test/golden/generate.jl                     # every area
#   julia --project=. test/golden/generate.jl pairwise analysis   # whole areas
#   julia --project=. test/golden/generate.jl pairwise:seir_hom4_bernoulli,seir_het39_bernoulli
#
# A golden freezes current behaviour, bugs included. Replace one only in the
# bug-fix change of the work package that owns its area, together with the
# literature or simulation test that justifies the new numbers
# (DESIGN_NetworkEpiCore.md §G.1), and say in the commit which goldens changed.

using NodeBasedModels
using ModelingToolkit
using OrdinaryDiffEqDefault
using Symbolics
using Graphs

include(joinpath(@__DIR__, "GoldenIO.jl"))
using .GoldenIO

const AREAS = sort!([d for d in readdir(@__DIR__) if isfile(joinpath(@__DIR__, d, "cases.jl"))])

function parse_targets(args)
    isempty(args) && return [(a, nothing) for a in AREAS]
    targets = Tuple{String,Union{Nothing,Vector{String}}}[]
    for arg in args
        area, cases = occursin(':', arg) ? split(arg, ':'; limit = 2) : (arg, nothing)
        area in AREAS || error("unknown golden area $(repr(area)); known: $(join(AREAS, ", "))")
        push!(targets, (String(area), cases === nothing ? nothing : String.(split(cases, ','))))
    end
    return targets
end

function provenance()
    Dict{String,Any}(
        "julia" => string(VERSION),
        "NodeBasedModels" => string(pkgversion(NodeBasedModels)),
        "ModelingToolkit" => string(pkgversion(ModelingToolkit)),
        "OrdinaryDiffEqDefault" => string(pkgversion(OrdinaryDiffEqDefault)),
        "Symbolics" => string(pkgversion(Symbolics)),
        "Graphs" => string(pkgversion(Graphs)),
        "generator" => "test/golden/generate.jl",
    )
end

function main(args)
    prov = provenance()
    for (area, only) in parse_targets(args)
        if only !== nothing
            known = [c.name for c in load_cases(area)]
            bad = setdiff(only, known)
            isempty(bad) || error("unknown case(s) $(bad) in area $area")
        end
        t0 = time()
        written = generate_area(area; only = only, provenance = prov)
        println("golden/$area: wrote $(length(written)) files in $(round(time() - t0; digits = 1)) s")
    end
end

main(ARGS)
