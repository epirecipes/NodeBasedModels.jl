# 00_legacy_compartments.jl — legacy NodeBasedModels tests:
# CompartmentalModel construction, built-in models, Catalyst conversion and validation.
#
# Moved verbatim (dedented one level) from the pre-0.2 test/runtests.jl by WP2
# (DESIGN_NetworkEpiCore.md §G.2). Owned by the work package that replaces this
# area; see §G.1. The imports are the original file's, so name resolution is unchanged.
#
# WP15 (NodeBasedModels 0.2): `sir_model()` and friends now return a NetworkEpiCore
# ContactModel (DESIGN §A.7). The 0.1 CompartmentalModel constructors are the deprecated
# `node_*_model()`; every 0.1 assertion below is kept on them, and the converter
# `CompartmentalModel(sir_model())` is checked to give the same model (named :sir).

using NodeBasedModels
using Test
using OrdinaryDiffEqDefault
using ModelingToolkit
using Graphs
using Random
using Catalyst
using Symbolics

# Same compartments, infectious flags, susceptible compartments and transitions (names aside).
same_model(a, b) =
    [(c.name, c.infectious) for c in a.compartments] == [(c.name, c.infectious) for c in b.compartments] &&
    a.infectious_compartments == b.infectious_compartments &&
    a.susceptible_compartments == b.susceptible_compartments &&
    [(t.from, t.to, t.rate, t.type, t.via) for t in a.transitions] ==
    [(t.from, t.to, t.rate, t.type, t.via) for t in b.transitions]

# ─── Compartmental Models ─────────────────────────────────────────────
@testset "Compartmental Models" begin
    @testset "SIR construction" begin
        m = @test_deprecated node_sir_model()
        @test m.name == :SIR
        @test length(m.compartments) == 3
        @test length(m.transitions) == 2
        @test m.infectious_compartments == [:I]
        @test m.susceptible_compartments == [:S]
        @test m.compartment_names == [:S, :I, :R]
        @test sir_model() isa ContactModel
        @test CompartmentalModel(sir_model()).name == :sir
        @test same_model(CompartmentalModel(sir_model()), m)
    end

    @testset "SIS construction" begin
        m = @test_deprecated node_sis_model()
        @test m.name == :SIS
        @test length(m.compartments) == 2
        @test m.infectious_compartments == [:I]
        @test same_model(CompartmentalModel(sis_model()), m)
    end

    @testset "SEIR construction" begin
        m = @test_deprecated node_seir_model()
        @test m.name == :SEIR
        @test length(m.compartments) == 4
        @test length(m.transitions) == 3
        @test same_model(CompartmentalModel(seir_model()), m)
    end

    @testset "SIRS construction" begin
        m = @test_deprecated node_sirs_model()
        @test m.name == :SIRS
        @test length(m.transitions) == 3
        @test same_model(CompartmentalModel(sirs_model()), m)
    end

    @testset "Custom model" begin
        m = CompartmentalModel(
            [Compartment(:S), Compartment(:E),
             Compartment(:I; infectious=true), Compartment(:R)],
            [Transition(:S, :E, :τ, :infection),
             Transition(:E, :I, :σ, :spontaneous),
             Transition(:I, :R, :γ, :spontaneous)];
            name = :custom_SEIR
        )
        @test m.name == :custom_SEIR
        @test length(m.infectious_compartments) == 1
    end

    @testset "Catalyst conversion" begin
        rn = @reaction_network begin
            τ, S + I --> 2I
            γ, I --> R
        end
        m = @test_deprecated model_from_catalyst(rn)
        @test m.compartment_names == [:S, :I, :R]
        @test length(m.transitions) == 2
        @test m.transitions[1].from == :S
        @test m.transitions[1].to == :I
        @test m.transitions[1].type == :infection
        @test m.transitions[2].from == :I
        @test m.transitions[2].to == :R
        @test m.transitions[2].type == :spontaneous
    end

    @testset "Validation" begin
        @test_throws ArgumentError CompartmentalModel(
            [Compartment(:S), Compartment(:I)],
            [Transition(:S, :I, :τ, :infection)];
            name = :no_infectious
        )
    end
end
