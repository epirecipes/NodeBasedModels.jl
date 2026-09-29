# 00_legacy_hygiene.jl — package hygiene checks added in Phase 0 (WP2):
# the unused NetworkDynamics dependency is gone and stays gone.
#
# Owned by the work package that owns NodeBasedModels.jl/Project.toml
# (WP15 in Phase 2, then WP24); see DESIGN_NetworkEpiCore.md §G.1.

using NodeBasedModels
using Test
using TOML

@testset "Dependency hygiene" begin
    project = TOML.parsefile(joinpath(pkgdir(NodeBasedModels), "Project.toml"))
    for section in ("deps", "weakdeps", "extras", "compat")
        @test !haskey(get(project, section, Dict{String,Any}()), "NetworkDynamics")
    end
    # NetworkDynamics is never loaded by `using NodeBasedModels`
    @test !any(id -> id.name == "NetworkDynamics", keys(Base.loaded_modules))
    # every [compat] entry names a dependency, an extra or julia itself
    known = union(keys(get(project, "deps", Dict())), keys(get(project, "weakdeps", Dict())),
                  keys(get(project, "extras", Dict())), ["julia"])
    @test issubset(keys(get(project, "compat", Dict())), known)
end
