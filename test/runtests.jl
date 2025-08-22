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

@mtkmodel SimpleSys begin
    @variables begin
        x(t) = 0
        u(t) = 0
        y(t)
        w(t), [disturbance = true, input = true]
    end
    @equations begin
        D(x) ~ -x + u + w # Explicitly encode where dynamics noise enters the system with w
        y ~ x
    end
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


# end
