# Golden cases, area "pairwise": population-level pairwise ODEs from
# `generate_pairwise` (homogeneous and heterogeneous networks; Bernoulli, Keeling,
# Barnard and Power closures; SIR, SIS, SEIR, SIRS and multi-compartment models).
#
# Each case freezes (i) the trajectory of every single [X] and pair [X|Y] on
# t = 0:1:T, solved at reltol 1e-10; (ii) the right-hand side at three
# deterministic probe states; (iii) the unknown and parameter name sets.
#
# Seeding (WP15, NodeBasedModels 0.2): the default initial condition seeds the ENTRY
# STATE of the infections (E for SEIR; DESIGN §E.2), so the `seir_*` goldens were replaced
# deliberately in WP15. The new numbers are justified by the testsets in test/suites/adopt.jl
# "SEIR seeding: the entry state E, initial condition = π^PW(EB)" and "SEIR goldens: the
# entry-seeded pairwise model is the edge-based model (k-regular)", which compares against the
# Miller–Volz edge-based SEIR model to 1e-8.
# The `seir_*_seedI` cases keep the 0.1 numbers (seeding the first infectious compartment, I)
# through `seed_state = :first_infectious`; their tables are identical to the 0.1 goldens.
#
# B03 (fixed in WP24, verified issue B03): 0.1's KeelingClosure used `network.N` in the
# clustering term while the state is scaled by the `N` keyword. It now takes N = Σ[X] from the
# state. The rhs_probe tables of every Keeling case were regenerated (the probe states lie off
# the invariant set Σ[X] = N, so they see the change; the trajectories of the consistent-N cases
# are unchanged), and the two former `*_B03` cases, renamed without the suffix, now equal the
# consistent case per capita. Justified by test/suites/00_legacy_closures.jl ("Keeling closure
# independent of the population scale (B03)": scale invariance and an independent
# transcription of Keeling's 1999 equations).
#
# Owner: the WP that owns NodeBasedModels pairwise generation (WP15 in Phase 2).

include(joinpath(@__DIR__, "..", "common.jl"))

include(joinpath(@__DIR__, "record.jl"))

# Custom models that exercise the loops over infectious compartments and
# infection transitions (the code that WP15's `via` field replaces).
two_stage_sir() = CompartmentalModel(
    [Compartment(:S), Compartment(:I1; infectious = true), Compartment(:I2; infectious = true),
     Compartment(:R)],
    [Transition(:S, :I1, :τ, :infection), Transition(:I1, :I2, :σ, :spontaneous),
     Transition(:I2, :R, :γ, :spontaneous)]; name = :SI1I2R)

sir_vaccination() = CompartmentalModel(
    [Compartment(:S), Compartment(:I; infectious = true), Compartment(:R), Compartment(:V)],
    [Transition(:S, :I, :τ, :infection), Transition(:I, :R, :γ, :spontaneous),
     Transition(:S, :V, :ν, :spontaneous)]; name = :SIRV)

sir_reinfection() = CompartmentalModel(
    [Compartment(:S), Compartment(:I; infectious = true), Compartment(:R)],
    [Transition(:S, :I, :τ, :infection), Transition(:I, :R, :γ, :spontaneous),
     Transition(:R, :I, :τR, :infection)]; name = :SIRI)

const SEED_NOTE = "entry seeding (0.2): the entry state E holds seed_fraction; the initial pairs are the π^PW image of the edge-based initial condition"
const SEEDI_NOTE = "0.1 seeding via seed_state = :first_infectious: the first infectious compartment I (not the entry state E) holds seed_fraction"
const B03_NOTE = "B03 fixed (WP24): KeelingClosure takes N = Σ[X] from the state, so the network N no longer matters; equal per capita to sir_hom6_keeling_phi02"

function golden_cases()
    hom4 = regular_network(4)
    hom6c = regular_network(6; ϕ = 0.2)
    het = degree_distribution_network(HET39)
    hetc = degree_distribution_network(HET39; ϕ = 0.1)
    er5 = erdos_renyi_network(5.0)
    B, K = BernoulliClosure(), KeelingClosure()
    sirp = Dict(:τ => 0.3, :γ => 0.2)
    sirp6 = Dict(:τ => 0.2, :γ => 0.25)
    sirph = Dict(:τ => 0.1, :γ => 0.25)
    sisp = Dict(:τ => 0.3, :γ => 0.25)
    sisph = Dict(:τ => 0.1, :γ => 0.3)
    seirp = Dict(:τ => 0.3, :σ => 0.5, :γ => 0.2)
    seirp6 = Dict(:τ => 0.2, :σ => 0.5, :γ => 0.25)
    seirph = Dict(:τ => 0.1, :σ => 0.5, :γ => 0.25)
    pc(name; kw...) = GoldenCase(name, () -> pairwise_record(name; kw...))
    return [
        # ─── SIR ───
        pc("sir_hom4_bernoulli"; model = sir_model(), network = hom4, closure = B, params = sirp,
           description = "SIR, 4-regular, Bernoulli closure"),
        pc("sir_hom6_keeling_phi02"; model = sir_model(), network = hom6c, closure = K, params = sirp6,
           description = "SIR, 6-regular with ϕ = 0.2, Keeling closure (consistent N = 1)"),
        pc("sir_het39_bernoulli"; model = sir_model(), network = het, closure = B, params = sirph,
           description = "SIR, degrees {3, 9} equiprobable, heterogeneous Bernoulli closure"),
        pc("sir_het39_keeling_phi01"; model = sir_model(), network = hetc, closure = K, params = sirph,
           description = "SIR, degrees {3, 9} with ϕ = 0.1, heterogeneous Keeling closure"),
        pc("sir_er5_bernoulli"; model = sir_model(), network = er5, closure = B, params = sirph,
           description = "SIR, erdos_renyi_network(5.0) (truncated Poisson), heterogeneous Bernoulli"),
        pc("sir_hom6_barnard_phi02"; model = sir_model(), network = hom6c, closure = BarnardClosure(),
           params = sirp6, description = "SIR, 6-regular with ϕ = 0.2, Barnard closure"),
        pc("sir_hom4_power15"; model = sir_model(), network = hom4, closure = PowerClosure(1.5),
           params = sirp, description = "SIR, 4-regular, PowerClosure(1.5)"),
        # ─── SIS ───
        pc("sis_hom4_bernoulli"; model = sis_model(), network = hom4, closure = B, params = sisp,
           description = "SIS, 4-regular, Bernoulli closure"),
        pc("sis_hom6_keeling_phi02"; model = sis_model(), network = hom6c, closure = K, params = sisp,
           description = "SIS, 6-regular with ϕ = 0.2, Keeling closure"),
        pc("sis_het39_bernoulli"; model = sis_model(), network = het, closure = B, params = sisph,
           description = "SIS, degrees {3, 9}, heterogeneous Bernoulli closure"),
        pc("sis_het39_keeling_phi01"; model = sis_model(), network = hetc, closure = K, params = sisph,
           description = "SIS, degrees {3, 9} with ϕ = 0.1, heterogeneous Keeling closure"),
        # ─── SEIR (entry seeding in E, replaced in WP15) ───
        pc("seir_hom4_bernoulli"; model = seir_model(), network = hom4, closure = B, params = seirp,
           description = "SEIR, 4-regular, Bernoulli closure; seeds E", notes = [SEED_NOTE]),
        pc("seir_hom6_keeling_phi02"; model = seir_model(), network = hom6c, closure = K, params = seirp6,
           description = "SEIR, 6-regular with ϕ = 0.2, Keeling closure; seeds E", notes = [SEED_NOTE]),
        pc("seir_het39_bernoulli"; model = seir_model(), network = het, closure = B, params = seirph,
           description = "SEIR, degrees {3, 9}, heterogeneous Bernoulli; seeds E", notes = [SEED_NOTE]),
        pc("seir_het39_keeling_phi01"; model = seir_model(), network = hetc, closure = K, params = seirph,
           description = "SEIR, degrees {3, 9} with ϕ = 0.1, heterogeneous Keeling; seeds E",
           notes = [SEED_NOTE]),
        # ─── SEIR with the 0.1 seeding in I (tables identical to the 0.1 seir_* goldens) ───
        pc("seir_hom4_bernoulli_seedI"; model = seir_model(), network = hom4, closure = B,
           params = seirp, seed_state = :first_infectious,
           description = "SEIR, 4-regular, Bernoulli closure; seeds I (0.1 default)",
           notes = [SEEDI_NOTE]),
        pc("seir_hom6_keeling_phi02_seedI"; model = seir_model(), network = hom6c, closure = K,
           params = seirp6, seed_state = :first_infectious,
           description = "SEIR, 6-regular with ϕ = 0.2, Keeling closure; seeds I (0.1 default)",
           notes = [SEEDI_NOTE]),
        pc("seir_het39_bernoulli_seedI"; model = seir_model(), network = het, closure = B,
           params = seirph, seed_state = :first_infectious,
           description = "SEIR, degrees {3, 9}, heterogeneous Bernoulli; seeds I (0.1 default)",
           notes = [SEEDI_NOTE]),
        pc("seir_het39_keeling_phi01_seedI"; model = seir_model(), network = hetc, closure = K,
           params = seirph, seed_state = :first_infectious,
           description = "SEIR, degrees {3, 9} with ϕ = 0.1, heterogeneous Keeling; seeds I (0.1 default)",
           notes = [SEEDI_NOTE]),
        # ─── other models ───
        pc("sirs_hom4_bernoulli"; model = sirs_model(), network = hom4, closure = B,
           params = Dict(:τ => 0.3, :γ => 0.2, :ε => 0.05), T = 60.0,
           description = "SIRS, 4-regular, Bernoulli closure"),
        pc("si1i2r_hom4_bernoulli"; model = two_stage_sir(), network = hom4, closure = B,
           params = Dict(:τ => 0.3, :σ => 0.4, :γ => 0.4),
           description = "two infectious stages S → I1 → I2 → R, both infectious at the same τ"),
        pc("si1i2r_het39_keeling_phi01"; model = two_stage_sir(), network = hetc, closure = K,
           params = Dict(:τ => 0.1, :σ => 0.5, :γ => 0.5),
           description = "two infectious stages on degrees {3, 9} with ϕ = 0.1, Keeling closure"),
        pc("sirv_hom4_bernoulli"; model = sir_vaccination(), network = hom4, closure = B,
           params = Dict(:τ => 0.3, :γ => 0.2, :ν => 0.02),
           description = "SIR with a spontaneous exit S → V (vaccination) at rate ν"),
        pc("siri_hom4_bernoulli"; model = sir_reinfection(), network = hom4, closure = B,
           params = Dict(:τ => 0.3, :γ => 0.2, :τR => 0.1), T = 60.0,
           description = "SIR with reinfection R → I at a second per-contact rate τR"),
        # ─── population scale and B03 ───
        pc("sir_hom6_keeling_phi02_N1000"; model = sir_model(), network = regular_network(6; ϕ = 0.2, N = 1000),
           closure = K, params = sirp6, N = 1000.0,
           description = "as sir_hom6_keeling_phi02 but in counts: network N = keyword N = 1000 (consistent)"),
        pc("sir_hom6_keeling_phi02_Nkw1000"; model = sir_model(), network = hom6c, closure = K,
           params = sirp6, N = 1000.0, notes = [B03_NOTE],
           description = "network N = 1, keyword N = 1000 (0.1: the clustering term was 1000× too small, B03)"),
        pc("sir_hom6_keeling_phi02_Nnet1000"; model = sir_model(),
           network = regular_network(6; ϕ = 0.2, N = 1000), closure = K, params = sirp6, N = 1.0,
           notes = [B03_NOTE],
           description = "network N = 1000, keyword N = 1 (0.1: the clustering term was 1000× too large, B03)"),
    ]
end
