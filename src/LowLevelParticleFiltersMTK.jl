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
    StateEstimationProblem(model, inputs, outputs; disturbance_inputs, discretization, Ts, df, dg, x0map=[], pmap=[])

A structure representing a state-estimation problem.

## Arguments:
- `model`: An MTK ODESystem model, this model must not have undergone structural simplification.
- `inputs`: The inputs to the dynamical system, a vector of symbolic variables that must be of type `@variables`.
- `outputs`: The outputs of the dynamical system, a vector of symbolic variables that must be of type `@variables`.
- `disturbance_inputs`: The disturbance inputs to the dynamical system, a vector of symbolic variables that must be of type `@variables`. These disturbance inputs indicate where dynamics noise ``w`` enters the system. The probability distribution ``d_f`` is defined over these variables.
- `discretization`: A function `discretization(f_cont, Ts, ndiff, nalg, nu) = f_disc` that takes a continuous-tiem dynamics function `f_cont(x,u,p,t)` and returns a discrete-time dynamics function `f_disc(x,u,p,t)`. `ndiff` is the number of differential state variables, `nalg` is the number of algebraic variables, and `nu` is the number of inputs.
- `Ts`: The discretization time step.
- `df`: The probability distribution of the dynamics noise ``w``. When using Kalman-type estimators, this must be a `MvNormal` or `SimpleMvNormal` distribution.
- `dg`: The probability distribution of the measurement noise ``e``. When using Kalman-type estimators, this must be a `MvNormal` or `SimpleMvNormal` distribution.
- `x0map`: A dictionary mapping symbolic variables to their initial values. If a variable is not provided, it is assumed to be initialized to zero. The value can be a scalar number, in which case the covariance of the initial state is set to `σ0^2*I(nx)`, and the value can be a `Distributions.Normal`, in which case the provided distributions are used as the distribution of the initial state. When passing distributions, all state variables must be provided values.
- `σ0`: The standard deviation of the initial state. This is used when `x0map` is not provided or when the values in `x0map` are scalars.
- `pmap`: A dictionary mapping symbolic variables to their values. If a variable is not provided, it is assumed to be initialized to zero.

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
function StateEstimationProblem(model, inputs, outputs; disturbance_inputs, discretization, Ts, df, dg, x0map=[], pmap=[], σ0 = 1e-4, kwargs...)

    # We always generate two versions of the dynamics function, the difference between them is that one has a signature augmented with disturbance inputs w, f(x,u,p,t,w), and the other does not, f(x,u,p,t).
    # The particular filter used for estimation dictates which version of the dynamics function will be used.
    model = mtkcompile(model; inputs, outputs, disturbance_inputs, simplify = true, split = false)
    f, x_sym, ps, iosys = generate_control_function(model, inputs, disturbance_inputs; simplify = true, split=false, force_SA=true, disturbance_argument=false, kwargs...);
    f_aug, _ = generate_control_function(model, inputs, disturbance_inputs; simplify = true, split=false, force_SA=true, disturbance_argument=true, kwargs...);

    nx = length(x_sym)
    ny = length(outputs)
    nu = length(inputs)
    nw = length(disturbance_inputs)
    na = count(ModelingToolkit.is_alg_equation, equations(iosys))

    function f_cont(x,u,p,t)
        f[1].f_oop(x,u,p,t)
    end

    function f_cont(x,u,p,t,w)
        f_aug[1].f_oop(x,u,p,t,w)
    end

    f_disc = discretization(f_cont, Ts, nx-na, na, nu)

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
    merge!(op, pmap)
    initprob = ModelingToolkit.InitializationProblem(iosys, 0.0, op)
    initsol = solve(initprob, ModelingToolkit.FastShortcutNonlinearPolyalg())

    p = Tuple(initprob.ps[p] for p in ps) # Use tuple instead of heterogeneously typed array for better performance
    x0 = SVector{nx}(initsol[x_sym])
    if dists
        d0 = SimpleMvNormal(x0, diagm([stdmap[Num(sym)]^2 for sym in x_sym]))
    else
        d0 = SimpleMvNormal(x0, σ0^2*I(nx))
    end

    names = SignalNames(x = string.(x_sym), u = string.(inputs), y = string.(outputs), name = "")

    StateEstimationProblem(model, iosys, inputs, outputs, disturbance_inputs, x_sym, nx, nu, ny, nw, na, f_disc, g.f_oop, ps, p, df, dg, d0, Ts, names)
end




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

# Arguments:
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
    if !dist
        i = findfirst(isequal(sym), prob.state)
        if i !== nothing
            if Nsamples <= 1
                return getindex.(sol.xt, i)
            else
                return [Particles(Nsamples, Normal(sol.xt[t][i], sqrt(sol.Rt[t][i,i])))  for t in inds]
            end
        end
        i = findfirst(isequal(sym), prob.inputs)
        if i !== nothing
            return getindex.(sol.u, i)
        end
    end

    if dist
        g = EstimatedOutput(f, prob, vcat(sym))# ModelingToolkit.build_explicit_observed_function(prob.iosys, vcat(sym); prob.inputs, prob.disturbance_inputs)
        timevec = range(0, step=f.Ts, length=length(sol.xt))
        return [g(SimpleMvNormal(sol.xt[i], sol.Rt[i]), sol.u[i], f.p, timevec[i]) for i in inds]
    end
    g = EstimatedOutput(f, prob, sym) # ModelingToolkit.build_explicit_observed_function(prob.iosys, sym; prob.inputs, prob.disturbance_inputs)
    timevec = range(0, step=f.Ts, length=length(sol.xt))
    if Nsamples <= 1
        [g(sol.xt[i], sol.u[i], f.p, timevec[i]) for i in inds]
    else
        [g(Particles(Nsamples, MvNormal(sol.xt[i], sol.Rt[i])), sol.u[i], f.p, timevec[i]) for i in inds]
    end
end

@recipe function solplot(timevec, osol::StateEstimationSolution; prob = osol.prob, idxs=[prob.state; prob.outputs; prob.inputs], Nsamples=1, plotRt=true, σ=1.96)
    n = length(idxs)
    layout --> n
    if plotRt
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
            if plotRt
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



end
