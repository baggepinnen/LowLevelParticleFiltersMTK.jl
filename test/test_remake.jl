using LowLevelParticleFiltersMTK
using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using ModelingToolkit
using SeeToDee
using StaticArrays
using LinearAlgebra
using ForwardDiff
using Random
using LeastSquaresOptim
using Test

t = ModelingToolkit.t_nounits
D = ModelingToolkit.D_nounits

rk4 = (f, Ts, x_inds, a_inds, nu) -> SeeToDee.Rk4(f, Ts)

# The parameter d is bound to an expression of a and is therefore not part of the parameter object
@component function OrderingSys(; name)
    pars = @parameters begin
        a = 1.0
        b = 2.0
        c = 3.0
        d = 2a
    end
    vars = @variables begin
        x(t) = 0.5
        v(t) = 0.0
        u(t) = 0.0
        y(t)
        w(t), [disturbance = true, input = true]
    end
    eqs = [
        D(x) ~ v
        D(v) ~ -a*x - b*v + c*u + d + w
        y ~ x
    ]
    System(eqs, t, vars, pars; name)
end

# The disturbance force is divided by the mass m, such that Bw depends on a parameter
@component function MassSpringDamper(; name)
    pars = @parameters begin
        m = 1.0
        c = 0.5
        k = 2.0
    end
    vars = @variables begin
        x(t) = 0.0
        v(t) = 0.0
        u(t) = 0.0
        y(t)
        w(t), [disturbance = true, input = true]
    end
    eqs = [
        D(x) ~ v
        m*D(v) ~ -k*x - c*v + u + w
        y ~ x
    ]
    System(eqs, t, vars, pars; name)
end

@component function ArraySys(; name)
    pars = @parameters begin
        a = 1.0
        q[1:2] = [2.0, 3.0]
        c = 4.0
    end
    vars = @variables begin
        x(t) = 0.5
        u(t) = 0.0
        y(t)
        w(t), [disturbance = true, input = true]
    end
    eqs = [
        D(x) ~ -a*x + q[1]*u + q[2] + c + w
        y ~ x
    ]
    System(eqs, t, vars, pars; name)
end

df1 = SimpleMvNormal(SMatrix{1,1}(0.1))
dg1 = SimpleMvNormal(SMatrix{1,1}(0.01))

@named ordering_model = OrderingSys()
cord = complete(ordering_model)
ordvals = Dict(cord.a => 1.1, cord.b => 2.2, cord.c => 3.3)

ordprob(init) = StateEstimationProblem(cord, [cord.u], [cord.y]; disturbance_inputs = [cord.w],
    df = df1, dg = dg1, discretization = rk4, Ts = 0.1, pmap = ordvals, init, warn_initialize_determined = false)

@testset "parameter ordering and bound parameters" begin
    @test any(isequal(ModelingToolkit.unwrap(cord.d)), ModelingToolkit.bound_parameters(cord))
    for init in (false, true)
        prob = ordprob(init)
        @test prob.p isa Tuple
        @test length(prob.p) >= length(prob.ps)
        @test collect(prob.p[1:length(prob.ps)]) == [ordvals[s] for s in prob.ps]
        @test !any(isequal(cord.d), prob.ps)
        # The bound parameter d = 2a is computed by the generated code
        x0 = SA[0.5, 0.0]
        @test prob.f_cont(x0, [0.0], prob.p, 0.0) ≈ [0.0, -1.1*0.5 + 2*1.1]
    end
end

@testset "remake" begin
    prob = ordprob(false)
    setter = parameter_setter(prob, [cord.a, cord.c])

    prob2 = remake(prob)
    @test prob2 isa StateEstimationProblem
    @test prob2.f === prob.f
    @test prob2.f_cont === prob.f_cont
    @test prob2.g === prob.g
    @test prob2.iosys === prob.iosys
    @test prob2.ps === prob.ps
    @test prob2.Ts == prob.Ts
    @test prob2.p === prob.p
    @test prob2.d0 === prob.d0

    # d0 is reused when the numeric type is unchanged
    prob3 = remake(prob; p = setter([2.0, 4.0]))
    @test prob3.d0 === prob.d0
    @test prob3.p[1] == 2.0
    @test prob3.f === prob.f

    # d0 is promoted when p contains dual numbers
    θd = ForwardDiff.Dual.([2.0, 4.0], 1.0)
    prob4 = remake(prob; p = setter(θd))
    @test eltype(prob4.d0.μ) <: ForwardDiff.Dual
    @test eltype(prob4.d0.Σ) <: ForwardDiff.Dual
    @test prob4.d0.μ isa SVector
    @test prob4.d0.Σ isa SMatrix
    @test ForwardDiff.value.(prob4.d0.μ) == prob.d0.μ
    @test ForwardDiff.value.(prob4.d0.Σ) == prob.d0.Σ

    # d0 is promoted when the noise covariances contain dual numbers
    dfd = SimpleMvNormal(SMatrix{1,1}(ForwardDiff.Dual(0.2, 1.0)))
    @test eltype(remake(prob; df = dfd).d0.μ) <: ForwardDiff.Dual

    # An explicit d0 is used as given
    d0x = SimpleMvNormal(SA[1.0, 2.0], SMatrix{2,2}(0.5I))
    @test remake(prob; d0 = d0x).d0 === d0x
    @test remake(prob; p = setter(θd), d0 = d0x).d0 === d0x

    # Substitution of df and dg is reflected in the filters
    df2 = SimpleMvNormal(SMatrix{1,1}(0.3))
    dg2 = SimpleMvNormal(SMatrix{1,1}(0.04))
    prob5 = remake(prob; df = df2, dg = dg2)
    ukf5 = get_filter(prob5, UnscentedKalmanFilter)
    @test ukf5.R1 == df2.Σ
    @test ukf5.R2 == dg2.Σ
    @test ukf5 isa UnscentedKalmanFilter{false,false,true,false}
    ekf5 = get_filter(prob5, ExtendedKalmanFilter)
    ekf1 = get_filter(prob, ExtendedKalmanFilter)
    @test ekf5.R1 ≈ 3 * ekf1.R1
    @test ekf5.R2 == dg2.Σ
end

@testset "parameter_setter" begin
    for init in (false, true)
        prob = ordprob(init)
        setter = parameter_setter(prob, [cord.c, cord.a])
        p = setter([30.0, 10.0])
        @test p isa Tuple
        @test length(p) == length(prob.p)
        @test p[1] == 10.0 && p[3] == 30.0
        @test all(p[i] === prob.p[i] for i in eachindex(p) if i ∉ (1, 3))
        @inferred setter([30.0, 10.0])
        @inferred setter(SA[30.0, 10.0])

        # Mixed types: dual numbers in the replaced entries, Float64 elsewhere
        θd = ForwardDiff.Dual.([30.0, 10.0], 1.0)
        pd = @inferred setter(θd)
        @test pd[1] isa ForwardDiff.Dual
        @test pd[3] isa ForwardDiff.Dual
        @test pd[2] === prob.p[2]
        @test all(pd[i] === prob.p[i] for i in eachindex(pd) if i ∉ (1, 3))

        # The bound parameter d = 2a is recomputed from the new value of a
        x0 = SA[0.5, 0.25]
        u0 = [2.0]
        a, b, c = 10.0, 2.2, 30.0
        @test prob.f_cont(x0, u0, p, 0.0) ≈ [0.25, -a*0.5 - b*0.25 + c*2.0 + 2a]

        @test get_parameters(prob, [cord.c, cord.a]) == [3.3, 1.1]
        @test get_parameters(prob, [cord.a, cord.b, cord.c]) == [1.1, 2.2, 3.3]
    end

    prob = ordprob(false)

    # Vector-valued parameter object
    probv = remake(prob; p = collect(prob.p))
    setterv = parameter_setter(probv, [cord.b])
    pv = setterv([5.0])
    @test pv isa Vector{Float64}
    @test pv[2] == 5.0
    @test pv[[1; 3:end]] == collect(prob.p)[[1; 3:end]]
    @test probv.p[2] == 2.2 # The original parameter object is not mutated
    pvd = setterv([ForwardDiff.Dual(5.0, 1.0)])
    @test eltype(pvd) <: ForwardDiff.Dual
    @test ForwardDiff.value.(pvd) == pv

    # Error cases
    @test_throws "bound parameter" parameter_setter(prob, [cord.d])
    @test_throws "is an input" parameter_setter(prob, [cord.u])
    @test_throws "disturbance input" parameter_setter(prob, [cord.w])
    @test_throws "state variable" parameter_setter(prob, [cord.x])
    @test_throws "observed variable" parameter_setter(prob, [cord.y])
    @test_throws "not completed" parameter_setter(prob, [ordering_model.a])
    @variables notinmodel
    @test_throws "was not found" parameter_setter(prob, [notinmodel])
    @test_throws "more than once" parameter_setter(prob, [cord.a, cord.b, cord.a])
    @test_throws "length 3" parameter_setter(prob, [cord.a, cord.b])([1.0, 2.0, 3.0])
    @test_throws "`Tuple` or an `AbstractVector`" parameter_setter(remake(prob; p = (; a = 1.0)), [cord.a])
    @test_throws ArgumentError parameter_setter(prob, [cord.d])
    @test_throws ArgumentError get_parameters(prob, [cord.x])

    # Array-valued parameters
    @named array_model = ArraySys()
    carr = complete(array_model)
    proba = StateEstimationProblem(carr, [carr.u], [carr.y]; disturbance_inputs = [carr.w],
        df = df1, dg = dg1, discretization = rk4, Ts = 0.1, init = true, warn_initialize_determined = false)
    @test_throws "array-valued" parameter_setter(proba, [carr.q])
    @test_throws "element of an array-valued" parameter_setter(proba, [carr.q[1]])
    settera = parameter_setter(proba, [carr.c])
    @test proba.f_cont(SA[0.5], [1.0], settera([5.0]), 0.0) ≈ [-0.5 + 2.0 + 3.0 + 5.0]
end

@named msd_model = MassSpringDamper()
cmsd = complete(msd_model)
Ts_msd = 0.05
θtrue = Dict(cmsd.m => 1.0, cmsd.c => 0.5, cmsd.k => 2.0)
θnom = Dict(cmsd.m => 1.0, cmsd.c => 0.8, cmsd.k => 1.5)
msdprob(pmap; df = SimpleMvNormal(SMatrix{1,1}(0.5)), dg = SimpleMvNormal(SMatrix{1,1}(0.01))) =
    StateEstimationProblem(cmsd, [cmsd.u], [cmsd.y]; disturbance_inputs = [cmsd.w], df, dg,
        discretization = rk4, Ts = Ts_msd, pmap)

prob_true = msdprob(θtrue)
Random.seed!(0)
u_msd = [[sin(0.5k*Ts_msd) + sign(sin(0.13k*Ts_msd))] for k in 1:200]
x_msd, u_msd, y_msd = simulate(get_filter(prob_true, UnscentedKalmanFilter), u_msd)
prob_msd = msdprob(θnom)

@testset "EKF p keyword" begin
    prob = prob_msd
    setter = parameter_setter(prob, [cmsd.m])
    p2 = setter([2.5])
    ekf_kw = get_filter(prob, ExtendedKalmanFilter; p = p2)
    ekf_remake = get_filter(remake(prob; p = p2), ExtendedKalmanFilter)
    @test ekf_kw.p === p2
    @test ekf_kw.R1 ≈ ekf_remake.R1
    @test !(ekf_kw.R1 ≈ get_filter(prob, ExtendedKalmanFilter).R1)
    ukf_kw = get_filter(prob, UnscentedKalmanFilter; p = p2)
    @test ukf_kw.p === p2
end

@testset "gradient of log-likelihood w.r.t. parameters" begin
    prob = prob_msd
    setter = parameter_setter(prob, [cmsd.c, cmsd.k])
    θ0 = get_parameters(prob, [cmsd.c, cmsd.k])
    for F in (UnscentedKalmanFilter, ExtendedKalmanFilter)
        cost = θ -> loglik(get_filter(remake(prob; p = setter(θ)), F), u_msd, y_msd)
        g = ForwardDiff.gradient(cost, θ0)
        h = 1e-5
        gfd = map(eachindex(θ0)) do i
            e = zeros(length(θ0)); e[i] = h
            (cost(θ0 + e) - cost(θ0 - e)) / 2h
        end
        @test all(isfinite, g)
        @test g ≈ gfd rtol = 1e-4
    end
end

@testset "autotune_covariances round trip" begin
    prob = msdprob(θtrue; df = SimpleMvNormal(SMatrix{1,1}(2.0)), dg = SimpleMvNormal(SMatrix{1,1}(0.1)))
    ukf = get_filter(prob, UnscentedKalmanFilter)
    sol0 = forward_trajectory(ukf, u_msd, y_msd)
    # The offset keeps the log-determinant residuals positive for the small innovation covariance
    res = autotune_covariances(sol0; show_trace = false, offset = 10.0)
    @test res.sol_opt.ll > sol0.ll
    @test size(res.R1) == (prob.nw, prob.nw)
    @test size(res.R2) == (prob.ny, prob.ny)
    @test res.filter isa UnscentedKalmanFilter{false,false,true,false}
    prob_tuned = remake(prob; df = SimpleMvNormal(res.R1), dg = SimpleMvNormal(res.R2))
    sol_tuned = forward_trajectory(get_filter(prob_tuned, UnscentedKalmanFilter), u_msd, y_msd)
    @test sol_tuned.ll ≈ res.sol_opt.ll
end
