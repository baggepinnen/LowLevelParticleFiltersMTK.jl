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

@component function SimpleSys(; name)
    pars = @parameters begin
        a = 1.0
    end

    vars = @variables begin
        x(t) = 0
        u(t) = 0
        y(t)
        w(t), [disturbance = true, input = true]
    end

    equations = [
        D(x) ~ -a*x + u + w # Explicitly encode where dynamics noise enters the system with w
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


Ts = 0.1

for parametricA = (true, ),
    parametricB = (true, false),
    parametricC = (true, false),
    parametricD = (true, false),
    parametricR1 = (true, false),
    parametricR2 = (true, false),
    discretize = (true, false), split = (false)
    
    kf, x_sym, ps, iosys = KalmanFilter(cmodel, inputs, outputs; disturbance_inputs, R1, R2, Ts, split,
    parametricA, parametricB, parametricC, parametricD, parametricR1, parametricR2, discretize)
    @test kf.Ts == Ts

    u = [randn(1) for _ in 1:100]
    x,u,y = simulate(kf, u, dynamics_noise=true, measurement_noise=true, sample_initial=true)


    fsole = forward_trajectory(kf, u, y)
    if discretize # result is nonsense if not discretized, so we just test that it runs
        @test sum(abs2, reduce(hcat, x .- fsole.xt)) / 100 < 0.1
    end
end



using Plots
plot(fsole, size=(1000, 1000))
plot!(fsole.t, reduce(hcat, x)')



