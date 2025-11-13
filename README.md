# LowLevelParticleFiltersMTK

[![Build Status](https://github.com/baggepinnen/LowLevelParticleFiltersMTK.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/baggepinnen/LowLevelParticleFiltersMTK.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/baggepinnen/LowLevelParticleFiltersMTK.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/baggepinnen/LowLevelParticleFiltersMTK.jl)

A helper package for state-estimation workflows using [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl) with [ModelingToolkit](https://github.com/SciML/ModelingToolkit.jl/) models.

# Installation
The package is registered, you can install it using:
```julia
import Pkg; Pkg.add("LowLevelParticleFiltersMTK")
```

# Challenges with performing state estimation with ModelingToolkit models

Consider a discrete-time dynamical system for which we want to perform state estimation:
```math
\begin{aligned}
x(t+1) &= f(x(t), u(t), p, t, w(t))\\
y(t) &= g(x(t), u(t), p, t, e(t))
\end{aligned}
```

Getting a ModelingToolkit model into this form requires several steps that are non trivial, such as generating the dynamics and measurement functions, $f$ and $g$, on the form required by the filter and discretizing a continuous-time model.

Workflows involving ModelingToolkit also demand symbolic indexing rather than indexing with integers, this need arises due to the fact that **the state realization for the system is chosen by MTK** rather than by the user, and this realization may change between different versions of MTK. One cannot normally specify the required initial state distribution and the dynamics noise distribution without having knowledge of the state realization. To work around this issue, this package requires the user to explicitly model how the disturbance inputs $w$ are affecting the dynamics, such that the realization of the dynamics noise becomes independent on the chosen state realization. This results in a dynamical model where the dynamics disturbance $w$ is an input to the model
```math
\dot{x} = f(x, u, p, t, w)
```
Some state estimators handle this kind of dynamics natively, like the [`UnscentedKalmanFilter`](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/api/#LowLevelParticleFilters.UnscentedKalmanFilter-Union{Tuple{AUGM},%20Tuple{AUGD},%20Tuple{IPM},%20Tuple{IPD},%20Tuple{Any,%20LowLevelParticleFilters.AbstractMeasurementModel,%20Any},%20Tuple{Any,%20LowLevelParticleFilters.AbstractMeasurementModel,%20Any,%20Any}}%20where%20{IPD,%20IPM,%20AUGD,%20AUGM}) with `AUGD = true`, while others, like the [`ExtendedKalmanFilter`](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/api/#LowLevelParticleFilters.ExtendedKalmanFilter) require manipulation of this model to work. This package handles such manipulation automatically, e.g., by continuously linearizing $f$ w.r.t. $w$ to obtain $B_w(t)$ and providing the `ExtendedKalmanFilter` with the time-varying dynamics covariance matrix $R_1(x, u, p, t) = B_w(t) R_w B_w(t)^T$.

Finally, this package provides symbolic indexing of the solution object, such that one can easily obtain the estimated posterior distribution over any arbitrary variable in the model, including "observed" variables that are not part of the state vector being estimated by the estimator.



# Workflow

> [!TIP]
> It is assumed that the reader is familiar with the basics of LowLevelParticleFilters.jl. Consult [the documentation](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/) and the video lectures liked therein to obtain such familiarity.

The workflow can be summarized as follows
1. Define a model using ModelingToolkit
2. Create an instance of `prob = StateEstimationProblem(...)`. This problem contains the model as well as specifications of inputs, outputs, disturbance inputs, noise probability distributions and discretization method.
3. Instantiate a state estimator using `filt = get_filter(prob, FilterConstructor)`. This calls the filter constructor with the appropriate dynamics functions depending on what type of filter is used.
4. Perform state estimation using the filter object as you would normally do with LowLevelParticleFilters.jl. Obtain a `fsol::KalmanFilteringSolution` object, either from calling `LowLevelParticleFilters.forward_trajectory` or by creating one manually after having performed custom filtering.
5. Wrap the `fsol` object in a `sol = StateEstimationSolution(fsol, prob)` object. This will provide symbolic indexing capabilities similar to how solution objects work in ModelingToolkit.
6. Analyze the solution object using, e.g., `sol[var], plot(sol), plot(sol, idxs=[var1, var2])` etc.
7. Profit from your newly derived insight.

As you can see, the workflow is similar to the standard MTK workflow, but contains a few more manual steps, notably the instantiation of the filter in step 3. and the manual wrapping of the solution object in step 5. The design is made this way since state estimation does not fit neatly into a problem->solve framework, in particular, one may have measurements arriving at irregular intervals, partial measurements, custom modifications of the covariance of the estimator etc. For simple cases where batch filtering (offline) is applicable, the function [`LowLevelParticleFilters.forward_trajectory`](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/api/#LowLevelParticleFilters.forward_trajectory) produces the required `KalmanFilteringSolution` object that can be wrapped in a `StateEstimationSolution` object. Situations that demand more flexibility instead require the user to manually construct this solution object, in which case inspecting the implementation of `LowLevelParticleFilters.forward_trajectory` and modifying it to suit your needs is a good starting point. An example of this is demonstrated in the tutorial [fault detection](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/fault_detection/).

# Example
The example below demonstrates a complete workflow, annotating the code with comments to point out things that are perhaps non-obvious.
```julia
using LowLevelParticleFiltersMTK
using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using ModelingToolkit
using SeeToDee # used to discretize the dynamics
using Plots
using StaticArrays
using LinearAlgebra

t = ModelingToolkit.t_nounits
D = ModelingToolkit.D_nounits

@component function SimpleSys(; name)
    pars = @parameters begin
    end

    vars = @variables begin
        x(t) = 2.0
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

@named model = SimpleSys()  # Do not use @mtkbuild here
cmodel  = complete(model)   # complete is required for variable indexing since we did not use @mtkbuild above
inputs  = [cmodel.u]        # The (unbound) inputs to the system
outputs = [cmodel.y]        # The outputs for which we obtain measurements
disturbance_inputs = [cmodel.w] # The dynamics disturbance inputs to the system

nu = length(inputs)             # Number of inputs
nw = length(disturbance_inputs) # Number of disturbance inputs
ny = length(outputs)            # Number of measured outputs
R1 = SMatrix{nw,nw}(0.01I(nw))  # Dynamics noise covariance
R2 = SMatrix{ny,ny}(0.1I(ny))   # Measurement noise covariance

df = SimpleMvNormal(R1)         # Dynamics noise distribution. This has to be a Gaussian if using a Kalman-type filter
dg = SimpleMvNormal(R2)         # Measurement noise distribution. This has to be a Gaussian if using a Kalman-type filter

Ts = 0.1                        # Sampling interval
discretization = function (f,Ts,x_inds,alg_inds,nu)
  isempty(alg_inds) || error("Rk4 only handles differential equations, consider `Trapezoidal` instead")
  SeeToDee.Rk4(f, Ts) # Discretization method
end

prob = StateEstimationProblem(model, inputs, outputs; disturbance_inputs, df, dg, discretization, Ts)

# We instantiate two different filters for comparison
ekf = get_filter(prob, ExtendedKalmanFilter)
ukf = get_filter(prob, UnscentedKalmanFilter)

# Simulate some data from the trajectory distribution implied by the model
u     = [randn(nu) for _ in 1:30] # A random input sequence
x,u,y = simulate(ekf, u, dynamics_noise=true, measurement_noise=true)

# Perform the filtering in batch since the entire input-output sequence is available
fsole = forward_trajectory(ekf, u, y)
fsolu = forward_trajectory(ukf, u, y)

# Wrap the filter solution objects in a StateEstimationSolution object
sole = StateEstimationSolution(fsole, prob)
solu = StateEstimationSolution(fsolu, prob)

## We can access the solution to any variable in the model easily
sole[cmodel.x] == sole[cmodel.y]

## We can also obtain the solution as a trajectory of probability distributions
sole[cmodel.x, dist=true]

## We can plot the filter solution object using the plot recipe from LowLevelParticleFilters
using Plots
plot(fsole, size=(1000, 1000))
plot!(fsole.t, reduce(hcat, x)', lab="True x")
##
plot(fsolu, size=(1000, 1000))
plot!(fsolu.t, reduce(hcat, x)', lab="True x")

## We can also plot the wrapped solution object
plot(sole)
plot!(solu)

## The wrapped solution object allows for symbolic indexing,
# note how we can easily plot the posterior distribution over y^2 + 0.1*sin(u) 
plot(sole, idxs=cmodel.y^2 + 0.1*sin(cmodel.u))
plot!(solu, idxs=cmodel.y^2 + 0.1*sin(cmodel.u))
```

# API
The following is a summary of the exported functions, followed by their docstrings
## Summary
- `StateEstimationProblem`: A structure representing a state-estimation problem.
- `StateEstimationSolution`: A solution object that provides symbolic indexing to a `KalmanFilteringSolution` object.
- `get_filter`: Instantiate a filter from a state-estimation problem.
- `propagate_distribution`: Propagate a probability distribution `dist` through a nonlinear function `f` using the covariance-propagation method of filter `kf`.


# `EstimatedOutput`
```
EstimatedOutput(kf, prob, sym)
```

Create an output function that can be called like

```julia
g(x::Vector,    u, p, t)     # Compute an output
g(xR::MvNormal, u, p, t)     # Compute an output distribution given input distribution xR
g(kf,           u, p, t)     # Compute an output distribution given the current state of an AbstractKalmanFilter
```

## Arguments:

  * `kf`: A Kalman type filter
  * `prob`: A `StateEstimationProblem` object
  * `sym`: A symbolic expression or vector of symbolic expressions that the function should output.

# `StateEstimationProblem`
```julia
StateEstimationProblem(model, inputs, outputs; disturbance_inputs, discretization, Ts, df, dg, x0map=[], pmap=[], init=false)
```

A structure representing a state-estimation problem.

## Arguments:

  * `model`: An MTK ODESystem model, this model must not have undergone structural simplification.
  * `inputs`: The inputs to the dynamical system, a vector of symbolic variables that must be of type `@variables`.
  * `outputs`: The outputs of the dynamical system, a vector of symbolic variables that must be of type `@variables`.
  * `disturbance_inputs`: The disturbance inputs to the dynamical system, a vector of symbolic variables that must be of type `@variables`. These disturbance inputs indicate where dynamics noise $w$ enters the system. The probability distribution $d_f$ is defined over these variables.
  * `discretization`: A function `discretization(f_cont, Ts, x_inds, alg_inds, nu) = f_disc` that takes a continuous-time dynamics function `f_cont(x,u,p,t)` and returns a discrete-time dynamics function `f_disc(x,u,p,t)`. `x_inds` is the indices of differential state variables, `alg_inds` is the indices of algebraic variables, and `nu` is the number of inputs.
  * `Ts`: The discretization time step.
  * `df`: The probability distribution of the dynamics noise $w$. When using Kalman-type estimators, this must be a `MvNormal` or `SimpleMvNormal` distribution.
  * `dg`: The probability distribution of the measurement noise $e$. When using Kalman-type estimators, this must be a `MvNormal` or `SimpleMvNormal` distribution.
  * `x0map`: A dictionary mapping symbolic variables to their initial values. If a variable is not provided, it is assumed to be initialized to zero. The value can be a scalar number, in which case the covariance of the initial state is set to `σ0^2*I(nx)`, and the value can be a `Distributions.Normal`, in which case the provided distributions are used as the distribution of the initial state. When passing distributions, all state variables must be provided values.
  * `σ0`: The standard deviation of the initial state. This is used when `x0map` is not provided or when the values in `x0map` are scalars.
  * `pmap`: A dictionary mapping symbolic variables to their values. If a variable is not provided, it is assumed to be initialized to zero.
  * `init`: If `true`, the initial state is computed using an initialization problem. If `false`, the initial state is computed using the `get_u0` function.
  * `xscalemap`: A dictionary mapping state variables to scaling factors. This is used to scale the state variables during integration to improve numerical stability. If a variable is not provided, it is assumed to have a scaling factor of 1.0. If provided, `discretization` is a function with signature `discretization(f_cont, Ts, x_inds, alg_inds, nu, scale_x)` where `scale_x` is a vector of scaling factors for the state variables.

## Usage:

Pseudocode

```julia
prob      = StateEstimationProblem(...)
kf        = get_filter(prob, ExtendedKalmanFilter)      # or UnscentedKalmanFilter
filtersol = forward_trajectory(kf, u, y)
sol       = StateEstimationSolution(filtersol, prob)   # Package into higher-level solution object
plot(sol, idxs=[prob.state; prob.outputs; prob.inputs]) # Plot the solution
```

# `StateEstimationSolution`
```julia
StateEstimationSolution(kfsol, prob)
```

A solution object that provides symbolic indexing to a `KalmanFilteringSolution` object.

## Fields:

  * `sol`:  a `KalmanFilteringSolution` object.
  * `prob`: a `StateEstimationProblem` object.

## Example

```julia
sol = StateEstimationSolution(kfsol, prob)
sol[model.x]                 # Index with a variable
sol[model.y^2]               # Index with an expression
sol[model.y^2, dist=true]    # Obtain the posterior probability distribution of the provided expression
sol[model.y^2, Nsamples=100] # Draw 100 samples from the posterior distribution of the provided expression
```

# `get_filter`
```julia
get_filter(prob::StateEstimationProblem, ::Type{ExtendedKalmanFilter}; constant_R1=true, kwargs)
get_filter(prob::StateEstimationProblem, ::Type{UnscentedKalmanFilter}; kwargs)
```

Instantiate a filter from a state-estimation problem. `kwargs` are sent to the filter constructor.

If `constant_R1=true`, the dynamics noise covariance matrix `R1` is assumed to be constant and is computed at the initial state. Otherwise, `R1` is computed at each time step throug repeated linearization w.r.t. the disturbance inputs `w`.

# `propagate_distribution`
```julia
propagate_distribution(f, kf, dist, args...; kwargs...)
```

Propagate a probability distribution `dist` through a nonlinear function `f` using the covariance-propagation method of filter `kf`.

## Arguments:

  * `f`: A nonlinear function `f(x, args...; kwargs...)` that takes a vector `x` and returns a vector.
  * `kf`: A state estimator, such as an `ExtendedKalmanFilter` or `UnscentedKalmanFilter`.
  * `dist`: A probability distribution, such as a `MvNormal` or `SimpleMvNormal`.
  * `args`: Additional arguments to `f`.
  * `kwargs`: Additional keyword arguments to `f`.



# `KalmanFilter`
```julia
kf, x_sym, ps, iosys, mats, prob = KalmanFilter(model::System, inputs, outputs; disturbance_inputs, Ts, R1, R2, x0map=[], pmap=[], σ0 = 1e-4, init=false, static=true, split = true, simplify=true, discretize = true, parametric = false, kwargs...)
```

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


# Generate docs
```julia
io = IOBuffer()
for n in names(LowLevelParticleFiltersMTK)
    n === :LowLevelParticleFiltersMTK && continue
    println(io, "# `", n, "`")
    println(io, Base.Docs.doc(getfield(LowLevelParticleFiltersMTK, n)))
end
s = String(take!(io))
clipboard(s)
```
