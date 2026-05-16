using LowLevelParticleFiltersMTK
using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using ModelingToolkit
using SeeToDee
using Test
using Plots
using StaticArrays
using LinearAlgebra

# @testset "LowLevelParticleFiltersMTK.jl" begin
    t = ModelingToolkit.t_nounits
D = ModelingToolkit.D_nounits

@component function SimpleSys(; name)
    pars = @parameters begin
    end

    vars = @variables begin
        x(t) = 0
        u(t) = 0
        y(t)
        w(t), [disturbance = true, input = true]
    end

    equations = [
        D(x) ~ -x + u + w # Explicitly encode where dynamics noise enters the system with w
        y ~ x
    ]

    return ODESystem(equations, t; name)
end

@named model = SimpleSys()
cmodel = complete(model)
inputs = [cmodel.u]
outputs = [cmodel.y]
disturbance_inputs = [cmodel.w]


nw = length(disturbance_inputs)
ny = length(outputs)
R1 = SMatrix{nw,nw}(0.1I(nw))
R2 = SMatrix{ny,ny}(0.1I(ny))

df = SimpleMvNormal(R1)
dg = SimpleMvNormal(R2)

Ts = 0.1
discretization = (f,Ts,x_inds,a_inds,nu)->SeeToDee.Rk4(f, Ts)

prob = StateEstimationProblem(cmodel, inputs, outputs; disturbance_inputs, df, dg, discretization, Ts)
ekf = get_filter(prob, ExtendedKalmanFilter)
ukf = get_filter(prob, UnscentedKalmanFilter)

u = [randn(1) for _ in 1:10]
x,u,y = simulate(ekf, u, dynamics_noise=true, measurement_noise=true)


fsole = forward_trajectory(ekf, u, y)
fsolu = forward_trajectory(ukf, u, y)
sole = StateEstimationSolution(fsole, prob)
solu = StateEstimationSolution(fsolu, prob)

@test fsole.xt ≈ fsolu.xt
@test fsole.Rt ≈ fsolu.Rt

using Plots
plot(fsole, size=(1000, 1000))
plot!(fsole.t, reduce(hcat, x)')

plot(fsolu, size=(1000, 1000))
plot!(fsolu.t, reduce(hcat, x)')


plot(sole)
plot!(solu)


plot(sole, idxs=cmodel.y^2 + 0.1*sin(cmodel.u))
plot!(solu, idxs=cmodel.y^2 + 0.1*sin(cmodel.u))



##
@test sole[cmodel.x] == sole[cmodel.y]

## Test static keyword argument
@testset "static keyword argument" begin
    # Test default behavior (static=true)
    prob_static = StateEstimationProblem(cmodel, inputs, outputs; disturbance_inputs, df, dg, discretization, Ts, static=true)
    @test prob_static.d0.μ isa SVector
    @test prob_static.d0.Σ isa SMatrix

    # Test with static=false
    prob_dynamic = StateEstimationProblem(cmodel, inputs, outputs; disturbance_inputs, df, dg, discretization, Ts, static=false)
    @test prob_dynamic.d0.μ isa Vector
    @test prob_dynamic.d0.Σ isa Matrix

    # Test backward compatibility (default should be static=true)
    prob_default = StateEstimationProblem(cmodel, inputs, outputs; disturbance_inputs, df, dg, discretization, Ts)
    @test prob_default.d0.μ isa SVector
    @test prob_default.d0.Σ isa SMatrix

    # Verify filters work with both array types
    ekf_static = get_filter(prob_static, ExtendedKalmanFilter)
    ekf_dynamic = get_filter(prob_dynamic, ExtendedKalmanFilter)

    # Test that both filters can process the same data
    fsole_static = forward_trajectory(ekf_static, u, y)
    fsole_dynamic = forward_trajectory(ekf_dynamic, u, y)

    # Results should be approximately equal regardless of array type
    @test fsole_static.xt[end] ≈ fsole_dynamic.xt[end]
    @test fsole_static.Rt[end] ≈ fsole_dynamic.Rt[end]
end


@testset "propagate_distribution uses kf.state_mean / kf.state_cov (UKF)" begin
    # Build a UKF where state_mean / state_cov are sentinel wrappers around the
    # defaults that record they were called. propagate_distribution must invoke
    # the kf's own functions, not hard-coded weighted_mean/weighted_cov.
    state_mean_called = Ref(0)
    state_cov_called  = Ref(0)
    function tagged_mean(xs, W)
        state_mean_called[] += 1
        LowLevelParticleFilters.weighted_mean(xs, W)
    end
    function tagged_cov(xs, m, W)
        state_cov_called[] += 1
        LowLevelParticleFilters.weighted_cov(xs, m, W)
    end

    prob_p = StateEstimationProblem(cmodel, inputs, outputs; disturbance_inputs, df, dg, discretization, Ts)
    ukf_tagged = UnscentedKalmanFilter{false,false,true,false}(
        prob_p.f, prob_p.g, prob_p.df.Σ, prob_p.dg.Σ, prob_p.d0;
        prob_p.Ts, prob_p.nu, prob_p.ny, prob_p.nx, prob_p.p,
        state_mean = tagged_mean, state_cov = tagged_cov,
    )

    f = x -> 2.0 .* x
    d_in = SimpleMvNormal(SVector(0.5), SMatrix{1,1}(0.25))
    propagate_distribution(f, ukf_tagged, d_in)

    @test state_mean_called[] >= 1
    @test state_cov_called[]  >= 1
end


@testset "EstimatedOutput(kf,...) defaults t from passed kf" begin
    # Build a fresh prob/EKF and advance the filter so its index > 0
    prob_t = StateEstimationProblem(cmodel, inputs, outputs;
        disturbance_inputs, df, dg, discretization, Ts)
    ekf_t = get_filter(prob_t, ExtendedKalmanFilter)
    u0 = [0.0]
    y0 = [0.0]
    for _ = 1:3
        LowLevelParticleFilters.predict!(ekf_t, u0)
        LowLevelParticleFilters.correct!(ekf_t, u0, y0)
    end
    @assert ekf_t.t > 0

    # Output expression that depends on time: 2*t (using the global iv)
    eo = EstimatedOutput(ekf_t, prob_t, 2 * t)
    # default t should be LowLevelParticleFilters.index(kf)*kf.Ts
    expected_t = 2 * (LowLevelParticleFilters.index(ekf_t) * ekf_t.Ts)
    out = eo(ekf_t, u0)
    @test out.μ[1] ≈ expected_t

    # explicit t override
    out_override = eo(ekf_t, u0, ekf_t.p, 5.0)
    @test out_override.μ[1] ≈ 10.0
end


@testset "linear" begin
    @info "Testing linear"
    include("test_linear.jl")
end

# end
