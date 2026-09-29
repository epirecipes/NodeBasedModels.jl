# selftest.jl — tests of the golden machinery itself (GoldenIO.jl), run by
# test/runtests.jl before the suites. Uses a temporary directory; never touches
# the stored goldens.

using Test

include(joinpath(@__DIR__, "GoldenIO.jl"))
using .GoldenIO

@testset "GoldenIO self-test" begin
    x = [1 / 3, 1e-300, -2.5e17, Inf, -Inf, NaN, 0.0, -0.0, nextfloat(1.0), 5e-324]
    tab = GoldenTable(["t", "a"], hcat(collect(1.0:10.0), x))
    rec = GoldenRecord("demo";
        scalars = Dict("r" => 1 / 7, "inf" => Inf, "nan" => NaN, "err" => "throws ArgumentError",
                       "flag" => true, "count" => 3),
        structure = Dict("names" => ["b", "a"]),
        tables = Dict("traj" => tab), known_defects = ["B01"],
        meta = Dict("net" => Dict("n" => 4), "sym" => :x))

    mktempdir() do tmp
        area = "selftest"
        paths = write_record(area, rec; provenance = Dict("julia" => string(VERSION)), root = tmp)
        @test all(startswith(tmp), paths)
        @test sort!(basename.(paths)) == ["demo.toml", "demo.traj.csv"]
        back = read_record(area, "demo"; root = tmp)
        @test isequal(back.tables["traj"].data, tab.data)          # bit-exact, NaN included
        @test back.tables["traj"].header == ["t", "a"]
        @test back.scalars["r"] === 1 / 7
        @test back.structure["names"] == ["a", "b"]
        @test back.known_defects == ["B01"]
        @test isempty(compare_records(back, rec))

        # a perturbation just above rtol = 1e-8 is caught, one below is not
        small = GoldenRecord("demo"; scalars = merge(rec.scalars, Dict("r" => (1 / 7) * (1 + 1e-9))),
                             structure = rec.structure,
                             tables = Dict("traj" => GoldenTable(["a", "t"], hcat(x .* (1 + 1e-9), 1.0:10.0))))
        @test isempty(compare_records(back, small))                 # column order is irrelevant
        big = GoldenRecord("demo"; scalars = merge(rec.scalars, Dict("r" => (1 / 7) * (1 + 1e-7))),
                           structure = Dict("names" => ["a", "c"]),
                           tables = Dict("traj" => GoldenTable(["t", "a"], hcat(1.0:10.0, x .* (1 + 1e-7)))))
        problems = compare_records(back, big)
        @test length(problems) == 3
        @test any(p -> startswith(p, "structure[names]"), problems)
        @test any(p -> startswith(p, "scalar r"), problems)
        @test any(p -> startswith(p, "table traj: 3 entries"), problems)   # 1/3, -2.5e17, nextfloat(1)
        # missing columns and changed row counts are reported
        @test !isempty(compare_records(back, GoldenRecord("demo"; scalars = rec.scalars,
            structure = rec.structure, tables = Dict("traj" => GoldenTable(["t"], reshape(1.0:10.0, :, 1))))))
        @test !isempty(compare_records(back, GoldenRecord("demo"; scalars = rec.scalars,
            structure = rec.structure, tables = Dict("traj" => GoldenTable(["t", "a"], tab.data[1:9, :])))))
    end

    # probe states are deterministic, in range, and independent of RNG state
    p1 = probe_states(["S", "I", "[S|I]"], 2.0, 3; salt = 5)
    p2 = probe_states(["S", "I", "[S|I]"], 2.0, 3; salt = 5)
    @test p1 == p2
    @test all(0.1 <= v <= 2.0 for d in p1 for v in values(d))
    @test p1[1] != p1[2]
    @test probe_states(["S"], 1.0, 1; salt = 5) != probe_states(["S"], 1.0, 1; salt = 6)

    @test catch_scalar(() -> 1.5) == 1.5
    @test catch_scalar(() -> throw(ArgumentError("x"))) == "throws ArgumentError"
    @test_throws ArgumentError GoldenTable(["a", "a"], zeros(2, 2))
    @test_throws ArgumentError GoldenTable(["a,b"], zeros(2, 1))
    @test_throws ArgumentError GoldenRecord("bad name")
end
