# 00_legacy_eon.jl — legacy NodeBasedModels tests:
# cross-validation against EoN 1.2 (Python) reference values.
#
# The pre-0.2 test/runtests.jl ended with `include("test_eon_crossval.jl")`; WP2
# keeps that file (and its eon_reference.json) where it is and runs it as a suite.
# Owned by the work package that replaces this area; see DESIGN_NetworkEpiCore.md §G.1.

include(joinpath(dirname(@__DIR__), "test_eon_crossval.jl"))
