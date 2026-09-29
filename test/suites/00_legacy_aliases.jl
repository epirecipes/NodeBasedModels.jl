# 00_legacy_aliases.jl — legacy NodeBasedModels tests:
# build_* / generate_* and node_*_model naming aliases.
#
# Moved verbatim (dedented one level) from the pre-0.2 test/runtests.jl by WP2
# (DESIGN_NetworkEpiCore.md §G.2). Owned by the work package that replaces this
# area; see §G.1. The imports are the original file's, so name resolution is unchanged.
#
# WP15 (NodeBasedModels 0.2): `sir_model` is NetworkEpiCore's (a ContactModel) and the
# `node_*_model` names are deprecated constructors of the 0.1 CompartmentalModels (DESIGN
# §A.7), no longer aliases of `sir_model` etc.; the alias identities are replaced by the new
# contract.

using NodeBasedModels
using Test
using OrdinaryDiffEqDefault
using ModelingToolkit
using Graphs
using Random
using Catalyst
using Symbolics

@testset "build_* / generate_* parity aliases" begin
    m = sir_model()
    net = regular_network(4)
    # Verify alias produces equivalent output to canonical fn
    psys_b = build_pairwise(m, net, BernoulliClosure())
    psys_g = generate_pairwise(m, net, BernoulliClosure())
    @test typeof(psys_b) === typeof(psys_g)
    @test length(ModelingToolkit.equations(psys_b.system)) ==
          length(ModelingToolkit.equations(psys_g.system))
    # Function aliases for individual / pair based
    @test build_individual_based === generate_individual_based
    @test build_pair_based === generate_pair_based
end

@testset "node_*_model: deprecated 0.1 CompartmentalModel constructors" begin
    same(a, b) = [(c.name, c.infectious) for c in a.compartments] ==
                 [(c.name, c.infectious) for c in b.compartments] &&
                 [(t.from, t.to, t.rate, t.type, t.via) for t in a.transitions] ==
                 [(t.from, t.to, t.rate, t.type, t.via) for t in b.transitions]
    for (old, new, name) in ((node_sir_model, sir_model, :SIR), (node_sis_model, sis_model, :SIS),
                             (node_seir_model, seir_model, :SEIR),
                             (node_sirs_model, sirs_model, :SIRS))
        m = @test_deprecated old()
        @test m isa CompartmentalModel
        @test m.name == name
        @test new() isa ContactModel
        @test same(m, CompartmentalModel(new()))
    end
    @test sir_model === NodeBasedModels.NetworkEpiCore.sir_model
end
