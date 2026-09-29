# NodeBasedModelsJumpProcessesExt — the JumpProcesses kernel of the deprecated `gillespie_sir`
# (owner: WP24; DESIGN §A.4). NodeBasedModels 0.2 keeps `gillespie_sir` for one release with a
# deprecation warning; this extension, loaded whenever JumpProcesses is, holds the JumpProcesses
# code so that the rest of the package does not use JumpProcesses. It is removed together with
# `gillespie_sir` in NodeBasedModels 0.3.
module NodeBasedModelsJumpProcessesExt

using NodeBasedModels
using NodeBasedModels: GraphNetwork, GillespieResult, _effective_transmission_matrix
using NodeBasedModels.Graphs: nv, edges, src, dst, is_directed
using JumpProcesses: ConstantRateJump, DiscreteProblem, JumpProblem, Direct, SSAStepper, solve

# The 0.1 gillespie_sir: one ConstantRateJump per transmission direction of every edge and one
# recovery jump per node, simulated with the Direct aggregator.
function NodeBasedModels._gillespie_sir(net::GraphNetwork;
                                        infection_rate::Real = 0.5,
                                        recovery_rate::Real = 0.1,
                                        initial_infected::AbstractVector{<:Integer} = [1],
                                        tmax::Real = 100.0,
                                        seed::Union{Integer, Nothing} = nothing)
    g = net.graph
    N = nv(g)
    T = _effective_transmission_matrix(net, infection_rate)
    all(i -> 1 <= i <= N, initial_infected) || throw(ArgumentError(
        "gillespie_sir: initial_infected lists nodes outside 1:$N"))

    # State: u[1:N] = S indicators, u[N+1:2N] = I indicators
    u0 = zeros(Int, 2N)
    for i in 1:N
        if i in initial_infected
            u0[N+i] = 1
        else
            u0[i] = 1
        end
    end

    jumps = ConstantRateJump[]

    # Infection jumps: one jump per transmission direction.
    for e in edges(g)
        let s = src(e), d = dst(e), n = N
            rate_sd = T[d, s]
            if rate_sd > 0
                push!(jumps, ConstantRateJump(
                    (u, p, t) -> rate_sd * u[d] * u[n+s],
                    integrator -> begin
                        integrator.u[d] -= 1
                        integrator.u[n+d] += 1
                    end
                ))
            end
            if !is_directed(g)
                rate_ds = T[s, d]
                if rate_ds > 0
                    push!(jumps, ConstantRateJump(
                        (u, p, t) -> rate_ds * u[s] * u[n+d],
                        integrator -> begin
                            integrator.u[s] -= 1
                            integrator.u[n+s] += 1
                        end
                    ))
                end
            end
        end
    end

    # Recovery jumps: for each node
    for i in 1:N
        let node = i, n = N
            push!(jumps, ConstantRateJump(
                (u, p, t) -> p[2] * u[n+node],
                integrator -> begin
                    integrator.u[n+node] -= 1
                end
            ))
        end
    end

    p = [0.0, Float64(recovery_rate)]
    dprob = DiscreteProblem(u0, (0.0, Float64(tmax)), p)
    jprob = JumpProblem(dprob, Direct(), jumps...)

    sol = isnothing(seed) ? solve(jprob, SSAStepper()) : solve(jprob, SSAStepper(); seed = seed)
    return GillespieResult(sol, g, N)
end

end # module
