module LowLevelParticleFiltersMTK


using ModelingToolkit
using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal, AbstractKalmanFilter
using MonteCarloMeasurements
using Distributions
using ForwardDiff
using LinearAlgebra, Statistics
using StaticArrays
using RecipesBase
using ModelingToolkit: generate_control_function, build_explicit_observed_function

export StateEstimationProblem, StateEstimationSolution, get_filter, propagate_distribution, EstimatedOutput

ModelingToolkit.parameters(f::LowLevelParticleFilters.AbstractFilter) = LowLevelParticleFilters.parameters(f)

struct StateEstimationProblem
    model
    iosys
    inputs
    outputs
    disturbance_inputs
    state
    nx::Int
    nu::Int
    ny::Int
    nw::Int
    na::Int
    f
    g
    ps
    p
    df
    dg
    d0
    Ts
    names
end



"""
    StateEstimationProblem(model, inputs, outputs; disturbance_inputs, discretization, Ts, df, dg, x0map=[], pmap=[], init=false)

A structure representing a state-estimation problem.

## Arguments:
- `model`: An MTK ODESystem model, this model must not have undergone structural simplification.
- `inputs`: The inputs to the dynamical system, a vector of symbolic variables that must be of type `@variables`.
- `outputs`: The outputs of the dynamical system, a vector of symbolic variables that must be of type `@variables`.
- `disturbance_inputs`: The disturbance inputs to the dynamical system, a vector of symbolic variables that must be of type `@variables`. These disturbance inputs indicate where dynamics noise ``w`` enters the system. The probability distribution ``d_f`` is defined over these variables.
- `discretization`: A function `discretization(f_cont, Ts, x_inds, alg_inds, nu) = f_disc` that takes a continuous-time dynamics function `f_cont(x,u,p,t)` and returns a discrete-time dynamics function `f_disc(x,u,p,t)`. `x_inds` is the indices of differential state variables, `alg_inds` is the indices of algebraic variables, and `nu` is the number of inputs.
- `Ts`: The discretization time step.
- `df`: The probability distribution of the dynamics noise ``w``. When using Kalman-type estimators, this must be a `MvNormal` or `SimpleMvNormal` distribution.
- `dg`: The probability distribution of the measurement noise ``e``. When using Kalman-type estimators, this must be a `MvNormal` or `SimpleMvNormal` distribution.
- `x0map`: A dictionary mapping symbolic variables to their initial values. If a variable is not provided, it is assumed to be initialized to zero. The value can be a scalar number, in which case the covariance of the initial state is set to `σ0^2*I(nx)`, and the value can be a `Distributions.Normal`, in which case the provided distributions are used as the distribution of the initial state. When passing distributions, all state variables must be provided values.
- `σ0`: The standard deviation of the initial state. This is used when `x0map` is not provided or when the values in `x0map` are scalars.
- `pmap`: A dictionary mapping symbolic variables to their values. If a variable is not provided, it is assumed to be initialized to zero.
- `init`: If `true`, the initial state is computed using an initialization problem. If `false`, the initial state is computed using the `get_u0` function.
- `xscalemap`: A dictionary mapping state variables to scaling factors. This is used to scale the state variables during integration to improve numerical stability. If a variable is not provided, it is assumed to have a scaling factor of 1.0. If provided, `discretization` is a function with signature `discretization(f_cont, Ts, x_inds, alg_inds, nu, scale_x)` where `scale_x` is a vector of scaling factors for the state variables.

## Usage:
Pseudocode
```julia
prob      = StateEstimationProblem(...)
kf        = get_filter(prob, ExtendedKalmanFilter)      # or UnscentedKalmanFilter
filtersol = forward_trajectory(kf, u, y)
sol       = StateEstimationSolution(filtersol, prob)   # Package into higher-level solution object
plot(sol, idxs=[prob.state; prob.outputs; prob.inputs]) # Plot the solution
```
"""
function StateEstimationProblem(model, inputs, outputs; disturbance_inputs, discretization, Ts, df, dg, x0map=[], pmap=[], σ0 = 1e-4, init=false, static=true, split = false, simplify=false, force_SA=true, xscalemap = nothing, kwargs...)

    # We always generate two versions of the dynamics function, the difference between them is that one has a signature augmented with disturbance inputs w, f(x,u,p,t,w), and the other does not, f(x,u,p,t).
    # The particular filter used for estimation dictates which version of the dynamics function will be used.
    if !ModelingToolkit.isscheduled(model)
        model = mtkcompile(model; inputs, outputs, disturbance_inputs, simplify, split, kwargs...)
    end
    f, x_sym, ps, iosys = generate_control_function(model, inputs, disturbance_inputs; simplify, split, force_SA, disturbance_argument=false, kwargs...);
    f_aug, _ = generate_control_function(model, inputs, disturbance_inputs; simplify, split, force_SA, disturbance_argument=true, kwargs...);

    nx = length(x_sym)
    ny = length(outputs)
    nu = length(inputs)
    nw = length(disturbance_inputs)
    na = count(ModelingToolkit.is_alg_equation, equations(iosys))
    x_inds = findall(ModelingToolkit.isdiffeq, equations(iosys))
    a_inds = findall(!ModelingToolkit.isdiffeq, equations(iosys))

    f_cont = FCont(f[1].f_oop, f_aug[1].f_oop)


    g = build_explicit_observed_function(iosys, outputs; inputs, ps, disturbance_inputs, disturbance_argument=false, checkbounds=true)

    # Handle initial distribution
    pmap = merge(Dict(disturbance_inputs .=> 0.0), Dict(pmap)) # Make sure disturbance inputs are initialized to zero if they are not explicitly provided in pmap
    x0map = collect(x0map)
    if !isempty(x0map) && (x0map[1][2] isa Distribution)
        stdmap = map(collect(x0map)) do (sym, dist)
            sym => dist.σ
        end |> Dict
        x0map = map(collect(x0map)) do (sym, dist)
            sym => dist.μ
        end
        dists = true
    else
        dists = false
    end

    # Merge x0map and pmap into a single op mapping for InitializationProblem
    op = Dict(x0map)
    inputmap = Dict([inputs .=> 0.0; disturbance_inputs .=> 0.0]) # Ensure inputs are initialized to zero if not provided
    op = merge(inputmap, op, pmap)
    if init
        initprob = ModelingToolkit.InitializationProblem(iosys, 0.0, op)
        initsol = solve(initprob)
        p = Tuple(initprob.ps[p] for p in ps) # Use tuple instead of heterogeneously typed array for better performance
        x0 = SVector{nx}(initsol[x_sym])
    else
        prob = ModelingToolkit.ODEProblem(iosys, op, (0.0, Ts))
        x0 = SVector{nx}(prob.u0)
        p0 = prob.p
        # x0 = SVector{nx}(ModelingToolkit.get_u0(iosys, op))
        # p0 = ModelingToolkit.get_p(iosys, op)
        p = p0 isa AbstractVector ? tuple(p0...) : p0
    end

    if dists
        R_mat = diagm([stdmap[Num(sym)]^2 for sym in x_sym])
        d0 = SimpleMvNormal(static ? x0 : Vector(x0), static ? SMatrix{nx,nx}(R_mat) : R_mat)
    else
        R_mat = σ0^2*I(nx)
        d0 = SimpleMvNormal(static ? x0 : Vector(x0), static ? SMatrix{nx,nx}(R_mat) : Matrix(R_mat))
    end

    if xscalemap !== nothing
        scale_x = map(unknowns(iosys)) do sym
            abs(get(xscalemap, sym, 1.0))
        end
        f_disc = discretization(f_cont, Ts, x_inds, a_inds, nu, scale_x)
    else
        f_disc = discretization(f_cont, Ts, x_inds, a_inds, nu)

    end

    names = SignalNames(x = string.(x_sym), u = string.(inputs), y = string.(outputs), name = "")

    StateEstimationProblem(model, iosys, inputs, outputs, disturbance_inputs, x_sym, nx, nu, ny, nw, na, f_disc, g.f_oop, ps, p, df, dg, d0, Ts, names)
end

struct FCont{F,FA}
    f::F
    fa::FA
end
(f::FCont)(x,u,p,t) = f.f(x,u,p,t)
(f::FCont)(x,u,p,t,w) = f.fa(x,u,p,t,w)


"""
    get_filter(prob::StateEstimationProblem, ::Type{ExtendedKalmanFilter}; constant_R1=true, kwargs)
    get_filter(prob::StateEstimationProblem, ::Type{UnscentedKalmanFilter}; kwargs)

Instantiate a filter from a state-estimation problem. `kwargs` are sent to the filter constructor.

If `constant_R1=true`, the dynamics noise covariance matrix `R1` is assumed to be constant and is computed at the initial state. Otherwise, `R1` is computed at each time step throug repeated linearization w.r.t. the disturbance inputs `w`.
"""
function get_filter(prob::StateEstimationProblem, ::Type{ExtendedKalmanFilter}; constant_R1=true, kwargs...)
    R1mat = prob.df.Σ
    w0 = SVector(zeros(prob.nw)...) # type instability here
    R1 = function (x,u,p,t)
        Bw = ForwardDiff.jacobian(w->prob.f(eltype(w).(x), u, p, t, w), w0)
        Bw * R1mat * Bw'
    end
    if constant_R1
        R1c = R1(mean(prob.d0), zeros(prob.nu), prob.p, 0.0)
        R1 = R1c
    end
    ExtendedKalmanFilter(prob.f, prob.g, R1, prob.dg.Σ, prob.d0; prob.Ts, prob.nu, prob.ny, prob.nx, prob.p, names = SignalNames(prob.names, "EKF"), kwargs...)
end

function get_filter(prob::StateEstimationProblem, ::Type{UnscentedKalmanFilter}; kwargs...)
    UnscentedKalmanFilter{false,false,true,false}(prob.f, prob.g, prob.df.Σ, prob.dg.Σ, prob.d0; prob.Ts, prob.nu, prob.ny, prob.nx, prob.p, names = SignalNames(prob.names, "UKF"), kwargs...)
end


"""
    StateEstimationSolution(kfsol, prob)

A solution object that provides symbolic indexing to a `KalmanFilteringSolution` object.

## Fields:
- `sol`:  a `KalmanFilteringSolution` object.
- `prob`: a `StateEstimationProblem` object.

## Example
```julia
sol = StateEstimationSolution(kfsol, prob)
sol[model.x]                 # Index with a variable
sol[model.y^2]               # Index with an expression
sol[model.y^2, dist=true]    # Obtain the posterior probability distribution of the provided expression
sol[model.y^2, Nsamples=100] # Draw 100 samples from the posterior distribution of the provided expression
```
"""
struct StateEstimationSolution # <: ModelingToolkit.SciMLBase.AbstractTimeseriesSolution
    sol
    prob
end

function Base.getproperty(sol::StateEstimationSolution, s::Symbol)
    s ∈ fieldnames(typeof(sol)) && return getfield(sol, s)
    innersol = getfield(sol, :sol)
    if s ∈ propertynames(innersol)
        return getproperty(innersol, s)
    else
        throw(ArgumentError("$(typeof(sol)) has no property named $s"))
    end
end

Base.propertynames(sol::StateEstimationSolution) = (fieldnames(typeof(sol))..., propertynames(getfield(sol, :sol))...)

"""
    EstimatedOutput(kf, prob, sym)

Create an output function that can be called like
```
g(x::Vector,    u, p, t)     # Compute an output
g(xR::MvNormal, u, p, t)     # Compute an output distribution given input distribution xR
g(kf,           u, p, t)     # Compute an output distribution given the current state of an AbstractKalmanFilter
```

## Arguments:
- `kf`: A Kalman type filter
- `prob`: A `StateEstimationProblem` object
- `sym`: A symbolic expression or vector of symbolic expressions that the function should output.
"""
struct EstimatedOutput{KF, P, G}
    kf::KF
    prob::P
    g::G
    function EstimatedOutput(kf, prob, sym)
        g = ModelingToolkit.build_explicit_observed_function(prob.iosys, sym; prob.inputs, prob.disturbance_inputs)
        new{typeof(kf), typeof(prob), typeof(g)}(kf, prob, g)
    end
end

(gg::EstimatedOutput)(x::AbstractVector, u, p=gg.kf.p, t=gg.kf.t) = gg.g(x, u, p, t)
function (gg::EstimatedOutput)(kf::AbstractKalmanFilter, u, args...; kwargs...)
    x = kf.x
    R = kf.R
    gg(SimpleMvNormal(x,R),u,args...)
end

function (gg::EstimatedOutput)(xR::SimpleMvNormal, u, p = gg.kf.p, t = gg.kf.t, args...; kwargs...)
    propagate_distribution(gg.g, gg.kf, xR, u, p, t, args...; kwargs...)
end

function Base.getindex(osol::StateEstimationSolution, sym; dist=false, Nsamples::Int = 1, inds=eachindex(osol.sol.xt))
    prob = osol.prob
    sol = osol.sol
    f = sol.f
    smoothing = osol.sol isa LowLevelParticleFilters.KalmanSmoothingSolution
    xt = smoothing ? sol.xT : sol.xt
    Rt = smoothing ? sol.RT : sol.Rt
    if !dist
        i = findfirst(isequal(sym), prob.state)
        if i !== nothing
            if Nsamples <= 1
                return getindex.(xt, i)
            else
                return [Particles(Nsamples, Normal(xt[t][i], sqrt(Rt[t][i,i])))  for t in inds]
            end
        end
        i = findfirst(isequal(sym), prob.inputs)
        if i !== nothing
            return getindex.(sol.u, i)
        end
    end

    if dist
        g = EstimatedOutput(f, prob, vcat(sym))# ModelingToolkit.build_explicit_observed_function(prob.iosys, vcat(sym); prob.inputs, prob.disturbance_inputs)
        timevec = range(0, step=f.Ts, length=length(xt))
        return [g(SimpleMvNormal(xt[i], Rt[i]), sol.u[i], f.p, timevec[i]) for i in inds]
    end
    g = EstimatedOutput(f, prob, sym) # ModelingToolkit.build_explicit_observed_function(prob.iosys, sym; prob.inputs, prob.disturbance_inputs)
    timevec = range(0, step=f.Ts, length=length(xt))
    if Nsamples <= 1
        [g(xt[i], sol.u[i], f.p, timevec[i]) for i in inds]
    else
        [g(Particles(Nsamples, MvNormal(xt[i], Rt[i])), sol.u[i], f.p, timevec[i]) for i in inds]
    end
end

@recipe function solplot(timevec, osol::StateEstimationSolution; prob = osol.prob, idxs=[prob.state; prob.outputs; prob.inputs], Nsamples=1, plotRt=true, σ=1.96, plotRT=true, plotxT=true)
    n = length(idxs)
    layout --> n
    smoothing = osol.sol isa LowLevelParticleFilters.KalmanSmoothingSolution

    plotconf = plotRt || (smoothing && plotRT)
    if plotconf
        dists = osol[vcat(idxs), dist=true]
        y = [mean(d) for d in dists]
        R = [cov(d) for d in dists]
        twoσ = σ .* sqrt.(reduce(hcat, diag.(R))')
    else
        y = osol[vcat(idxs), Nsamples=Nsamples]
    end
    for (i,sym) in enumerate(idxs)
        @series begin
            label --> string(sym)
            sp --> i
            N --> 0 # This turns off particle paths for MCM plots
            if plotconf
                ribbon := twoσ[:,i]
            end
            timevec, getindex.(y, i)
        end
    end
end

@recipe function solplot(osol::StateEstimationSolution)
    sol = osol.sol
    timevec = sol.t
    @series timevec, osol
end


"""
    propagate_distribution(f, kf, dist, args...; kwargs...)

Propagate a probability distribution `dist` through a nonlinear function `f` using the covariance-propagation method of filter `kf`.

## Arguments:
- `f`: A nonlinear function `f(x, args...; kwargs...)` that takes a vector `x` and returns a vector.
- `kf`: A state estimator, such as an `ExtendedKalmanFilter` or `UnscentedKalmanFilter`.
- `dist`: A probability distribution, such as a `MvNormal` or `SimpleMvNormal`.
- `args`: Additional arguments to `f`.
- `kwargs`: Additional keyword arguments to `f`.
"""
function propagate_distribution(f, kf::ExtendedKalmanFilter, x, args...; kwargs...)
    hasproperty(x, :μ) || error("Expected x to be a MvNormal or SimpleMvNormal")
    m,S = mean(x), cov(x)
    C = ForwardDiff.jacobian(m->vcat(f(m, args...; kwargs...)), m) # vcat is a hack to ensure array output
    my = f(m, args...; kwargs...)
    Sy = C * S * C'
    return SimpleMvNormal(my, Sy)
end

function propagate_distribution(f, kf::UnscentedKalmanFilter, x, args...; kwargs...)
    hasproperty(x, :μ) || error("Expected x to be a MvNormal or SimpleMvNormal")
    m,S = mean(x), cov(x)
    xs = LowLevelParticleFilters.sigmapoints(m, S, kf.weight_params; cholesky! = kf.cholesky!)
    ys = [f(x, args...; kwargs...) for x in xs]
    my = LowLevelParticleFilters.mean_with_weights(weighted_mean, ys, kf.weight_params)
    Sy = LowLevelParticleFilters.cov_with_weights(weighted_cov, ys, my, kf.weight_params)
    return SimpleMvNormal(my, Sy)
end


function has_vars(exprs)
    for expr in exprs
        if !isempty(ModelingToolkit.get_variables(expr))
            return true
        end
    end
    return false
end


# ==============================================================================
## Linear systems
# ==============================================================================
"""
    kf, x_sym, ps, iosys, mats, prob = KalmanFilter(model::System, inputs, outputs; disturbance_inputs, Ts, R1, R2, x0map=[], pmap=[], σ0 = 1e-4, init=false, static=true, split = true, simplify=true, discretize = true, parametric = false, kwargs...)

Construct a Kalman filter for a linear MTK ODESystem. No check is performed to verify that the system is truly linear, if it is nonlinear, it will be linearized.

## Returns:
- `kf`: A Kalman filter. If `parametric=true`, the `A,B,C,D,R1,R2` fields are all functions of `(x,u,p,t)`, otherwise they are matrices that are evaluated at the `x0map, pmap` values.
- `x_sym`: The symbolic state variables of the system.
- `ps`: The symbolic parameters of the system.
- `iosys`: The simplified MTK `System`
- `mats`: A named tuple containing the symbolic system matrices `(A,B,C,D,Bw,Dw)`, where `Bw` and `Dw` are the input matrices corresponding to the disturbance inputs.
- `prob`: A `StateEstimationProblem` object. This problem object does not play quite the same role as when using Unscented or Extended Kalman filters since the filter is created already by this constructor, but the problem object can still be useful for inspecting the simplified MTK system, to create `EstimatedOutput` objects and to make use of the symbolic indexing functionality.

## Arguments:
- `model`: An MTK System model, this model must not have undergone structural simplification.
- `inputs`: The inputs to the dynamical system, a vector of symbolic variables.
- `outputs`: The outputs of the dynamical system, a vector of symbolic variables.
- `disturbance_inputs`: The disturbance inputs to the dynamical system, a vector of symbolic variables. These disturbance inputs indicate where dynamics noise ``w`` enters the system. The probability distribution ``R1`` is defined over these variables.
- `Ts`: The discretization time step.
- `R1`: The covariance matrix of the dynamics noise ``w``.
- `R2`: The covariance matrix of the measurement noise ``e``.
- `x0map`: A dictionary mapping symbolic variables to their initial values. If a variable is not provided, it is assumed to be initialized to zero.  The value can be a scalar number, in which case the covariance of the initial state is set to `σ0^2*I(nx)`, and the value can be a `Distributions.Normal`, in which case the provided distributions are used as the distribution of the initial state. When passing distributions, all state variables must be provided values.
- `pmap`: A dictionary mapping symbolic variables to their values.
- `σ0`: The standard deviation of the initial state. This is used when `x0map` is not provided.
- `init`: If `true`, the initial state is computed using an initialization problem. If `false`, the initial state is computed using the `get_u0` function.
- `static`: If `true`, static arrays are used for the state and covariance matrix. This can improve performance for small systems.
- `split`: Passed to `mtkcompile`, see the documentation there.
- `simplify`: Passed to `mtkcompile`, see the documentation there.
- `discretize`: If `true`, the system is discretized using zero-order hold. If `false`, matrices/functions are generated for the continuous-time system, in which case the user must handle discretization themselves (filtering with a continuous-time system without discretization will yield nonsensical results).
- `parametric_A`: If `true`, the `A` field of the returned filter is a function of `(x,u,p,t)`, otherwise it is a matrix that is evaluated at the `x0map, pmap` values.
- `parametric_B`: If `true`, the `B` field of the returned filter is a function of `(x,u,p,t)`, otherwise it is a matrix that is evaluated at the `x0map, pmap` values.
- `parametric_C`: If `true`, the `C` field of the returned filter is a function of `(x,u,p,t)`, otherwise it is a matrix that is evaluated at the `x0map, pmap` values.
- `parametric_D`: If `true`, the `D` field of the returned filter is a function of `(x,u,p,t)`, otherwise it is a matrix that is evaluated at the `x0map, pmap` values.
- `parametric_R1`: If `true`, the `R1` field of the returned filter is a function of `(x,u,p,t)`, otherwise it is a matrix that is evaluated at the `x0map, pmap` values.
- `parametric_R2`: If `true`, the `R2` field of the returned filter is a function of `(x,u,p,t)`, otherwise it is a matrix that is evaluated at the `x0map, pmap` values.
- `tuplify`: If `true`, the parameter vector `p` is returned as a tuple instead of an array. This can improve performance for filters with a small number of parameters of heterogeneous types.
- `kwargs`: Additional keyword arguments passed to `mtkcompile`.
"""
function LowLevelParticleFilters.KalmanFilter(model::System, inputs, outputs; disturbance_inputs, Ts, R1, R2, x0map=[], pmap=[], σ0 = 1e-4, init=false, static=true, split = true, simplify=true, discretize = true, tuplify = true,
    parametricA = false,
    parametricB = false,
    parametricC = false,
    parametricD = false,
    parametricR1 = false,
    parametricR2 = false,
    kwargs...)

    # We always generate two versions of the dynamics function, the difference between them is that one has a signature augmented with disturbance inputs w, f(x,u,p,t,w), and the other does not, f(x,u,p,t).
    # The particular filter used for estimation dictates which version of the dynamics function will be used.
    all_inputs = [inputs; disturbance_inputs]

    (; A, B, C, D), iosys = ModelingToolkit.linearize_symbolic(model, all_inputs, outputs; simplify = false, allow_input_derivatives = false, split, kwargs...)
    for mat in (A,B,C,D)
        # This step should not be needed for a linear model that is affine in the disturbance inputs, but for nonlinear models we need to do this to not have the disturbance inputs appear in the matrices
        subs = Dict(disturbance_inputs .=> 0)
        mat .= ModelingToolkit.substitute.(mat, Ref(subs))
    end
    BBw = B
    DDw = D
    B = BBw[:, 1:length(inputs)]
    Bw = BBw[:, length(inputs)+1:end]
    D = DDw[:, 1:length(inputs)]
    Dw = DDw[:, length(inputs)+1:end]
    ps = setdiff(parameters(iosys), all_inputs)
    x_sym = unknowns(iosys)
    function really_unwrap(x)
        r = Symbolics.unwrap(x)
        r isa Number ? r : r.val
    end

    mats = (; A, B, C, D, Bw, Dw)
    constantA = !has_vars(A)
    constantB = !has_vars(B)
    constantC = !has_vars(C)
    constantD = !has_vars(D)
    constantBw = !has_vars(Bw)
    constantDw = !has_vars(Dw)
    parametricA || constantA || error("parametricA=false but A depends on (x,u,p,t), A = $(A)")
    parametricB || constantB || error("parametricB=false but B depends on (x,u,p,t), B = $(B)")
    parametricC || constantC || error("parametricC=false but C depends on (x,u,p,t), C = $(C)")
    parametricD || constantD || error("parametricD=false but D depends on (x,u,p,t), D = $(D)")
    parametricR1 || constantBw || error("parametricR1=false but Bw depends on (x,u,p,t), Bw = $(Bw)")
    parametricR2 || constantDw || error("parametricR2=false but Dw depends on (x,u,p,t), Dw = $(Dw)")

    nx = length(x_sym)
    ny = length(outputs)
    nu = length(inputs)
    nw = length(disturbance_inputs)
    na = count(ModelingToolkit.is_alg_equation, equations(iosys))
    na == 0 || error("KalmanFilter only supports ODE systems, found $na algebraic equations. \n", equations(iosys))

    # Handle initial distribution
    pmap = merge(Dict(all_inputs .=> 0.0), Dict(pmap)) # Make sure disturbance inputs are initialized to zero if they are not explicitly provided in pmap
    x0map = collect(x0map)
    if !isempty(x0map) && (x0map[1][2] isa Distribution)
        stdmap = map(collect(x0map)) do (sym, dist)
            sym => dist.σ
        end |> Dict
        x0map = map(collect(x0map)) do (sym, dist)
            sym => dist.μ
        end
        dists = true
    else
        dists = false
    end

    # Merge x0map and pmap into a single op mapping for InitializationProblem
    op = Dict(x0map)
    inputmap = Dict(all_inputs .=> 0.0) # Ensure inputs are initialized to zero if not provided
    op = merge(inputmap, op, pmap)
    if init
        initprob = ModelingToolkit.InitializationProblem(iosys, 0.0, op)
        initsol = solve(initprob)
        p = Tuple(initprob.ps[p] for p in ps) # Use tuple instead of heterogeneously typed array for better performance
        x0 = SVector{nx}(initsol[x_sym])
    else
        x0 = SVector{nx}(ModelingToolkit.get_u0(iosys, op))
        p0 = ModelingToolkit.get_p(iosys, op; split)
        p = tuplify && p0 isa AbstractVector ? tuple((p0)...) : p0
    end

    if dists
        R_mat = diagm([stdmap[Num(sym)]^2 for sym in x_sym])
        d0 = SimpleMvNormal(static ? x0 : Vector(x0), static ? SMatrix{nx,nx}(R_mat) : R_mat)
    else
        R_mat = σ0^2*I(nx)
        d0 = SimpleMvNormal(static ? x0 : Vector(x0), static ? SMatrix{nx,nx}(R_mat) : Matrix(R_mat))
    end

    names = SignalNames(x = string.(x_sym), u = string.(inputs), y = string.(outputs), name = string(nameof(model)))

    # build functions
    force_SA = static
    expression=Val{false}
    t = ModelingToolkit.get_iv(iosys)
    Afun  = Symbolics.build_function(A,  x_sym, inputs, ps, t; force_SA, expression)[1]
    Bfun = Symbolics.build_function(B, x_sym, inputs, ps, t; force_SA, expression)[1]
    discABfun = Symbolics.build_function(Base.exp(Ts*Num.([A B; zeros(nu, nx+nu)])),  x_sym, inputs, ps, t; force_SA, expression)[1]

    if parametricA
        if discretize
            A = function (x,u,p,t)
                ABd = (discABfun(x,u,p,t))
                ABd[1:nx, 1:nx]
            end
        else
            A = Afun
        end
    else
        A = really_unwrap.(A)
    end
    if parametricB
        if discretize
            B = function (x,u,p,t)
                ABd = (discABfun(x,u,p,t))
                ABd[1:nx, nx+1:end]
            end
        else
            B = Bfun
        end
    else
        B = really_unwrap.(B)
    end
    if parametricC
        Cfun = Symbolics.build_function(C, x_sym, inputs, ps, t; force_SA, expression)[1]
        C = Cfun
    else
        C = really_unwrap.(C)
    end
    if parametricD
        Dfun = Symbolics.build_function(D, x_sym, inputs, ps, t; force_SA, expression)[1]
        D = Dfun
    else
        D = really_unwrap.(D)
    end
    if parametricR1
        Bwfun = Symbolics.build_function(Bw, x_sym, inputs, ps, t; force_SA, expression)[1]
        R1 = let R1=R1
            function (x,u,p,t)
                Bwi = Bwfun(x,u,p,t)
                Bwi * R1 * Bwi'
            end
        end
    else
        Bw = really_unwrap.(Bw)
        R1 = Bw * R1 * Bw'
    end
    if parametricR2
        Dwfun = Symbolics.build_function(Dw, x_sym, inputs, ps, t; force_SA, expression)[1]
        R2 = let R1=R1, R2=R2
            if parametricR1
                function (x,u,p,t)
                    Dwi = Dwfun(x,u,p,t)
                    R2 + Dwi * R1(x,u,p,t) * Dwi'
                end
            else
                function (x,u,p,t)
                    Dwi = Dwfun(x,u,p,t)
                    R2 + Dwi * R1 * Dwi'
                end
            end
        end
    else
        Dw = really_unwrap.(Dw)
        if iszero(Dw)
            # R2 = R2
        elseif !parametricR1
            R2 = R2 + Dw * R1 * Dw'
        else
            error("Constant R2 with parametric R1 and nonzero Dw is not yet supported.")
        end
    end

    prob = StateEstimationProblem(model, inputs, outputs; disturbance_inputs, discretization = (f_cont, Ts, x_inds, a_inds, nu)->f_cont, Ts, df = SimpleMvNormal(zeros(nw), I(nw)), dg = SimpleMvNormal(zeros(ny), I(ny)), x0map, pmap, σ0, init, static, split, simplify, force_SA, kwargs...)

    (; kf=KalmanFilter(A, B, C, D, R1, R2, d0; Ts, nu, ny, nx, p, names), x_sym, ps, iosys, mats, prob)


end
    

end
