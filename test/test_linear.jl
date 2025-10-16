using LowLevelParticleFiltersMTK
using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using ModelingToolkit
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
    @parameters begin
        a = 1.0
    end
    @equations begin
        D(x) ~ -a*x + u + w # Explicitly encode where dynamics noise enters the system with w
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


Ts = 0.1

kf = KalmanFilter(cmodel, inputs, outputs; disturbance_inputs, R1, R2, Ts, split=false)

u = [randn(1) for _ in 1:100]
x,u,y = simulate(kf, u, dynamics_noise=true, measurement_noise=true)


fsole = forward_trajectory(kf, u, y)

using Plots
plot(fsole, size=(1000, 1000))
plot!(fsole.t, reduce(hcat, x)')



norm(reduce(hcat, x .- fsole.xt))
