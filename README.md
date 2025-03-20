# LowLevelParticleFiltersMTK

[![Build Status](https://github.com/baggepinnen/LowLevelParticleFiltersMTK.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/baggepinnen/LowLevelParticleFiltersMTK.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/baggepinnen/LowLevelParticleFiltersMTK.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/baggepinnen/LowLevelParticleFiltersMTK.jl)

A helper package for state-estimation workflows using [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl) with [ModelingToolkit](https://github.com/SciML/ModelingToolkit.jl/) models.

# Installation
The package is not registered, you need to install it using the URL:
```julia
import Pkg; Pkg.add(url="https://github.com/baggepinnen/LowLevelParticleFiltersMTK.jl")
```

# Challenges with performing state estimation with ModelingToolkit models

Consider a discrete-time dynamical system for which we want to perform state estimation:
```math
\begin{aligned}
x(t+1) &= f(x(t), u(t), p, t, w(t))\\
y(t) &= g(x(t), u(t), p, t, e(t))
\end{aligned}
```

Setting up a state estimator with a model from ModelingToolkit requires several steps that are non trivial, such as generating the dynamics and measurement functions on the form required by the filter and discretizing a continuous-time model. Workflows involving ModelingToolkit also demand symbolic indexing rather than indexing with integers, this need arises due to the fact that **the state realization for the system being chosen by MTK** rather than by the user, and this realization may change between different versions of MTK. One cannot normally specify the required initial state distribution and the dynamics noise distribution without having knowledge of the state realization. To work around this issue, this package requires the user to explicitly model the disturbance inputs affecting the dynamics, such that the realization of the dynamics noise becomes independent on the chosen state realization. This results in a dynamical model where the dynamics disturbance ``w`` is an input to the model
```math
\dot{x} = f(x, u, p, t, w)
```
Some state estimators handle this kind of model natively, like the [`UnscentedKalmanFilter`](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/api/#LowLevelParticleFilters.UnscentedKalmanFilter-Union{Tuple{AUGM},%20Tuple{AUGD},%20Tuple{IPM},%20Tuple{IPD},%20Tuple{Any,%20LowLevelParticleFilters.AbstractMeasurementModel,%20Any},%20Tuple{Any,%20LowLevelParticleFilters.AbstractMeasurementModel,%20Any,%20Any}}%20where%20{IPD,%20IPM,%20AUGD,%20AUGM}) with `AUGD = true`, others, like the [`ExtendedKalmanFilter`](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/api/#LowLevelParticleFilters.ExtendedKalmanFilter) require manipulation of this model to work. This package handles such manipulation automatically, e.g., by continuously linearizing ``f`` w.r.t. ``w`` to obtain ``B_w(k)`` and providing the `ExtendedKalmanFilter` with the time-varying dynamics covariance matrix ``R_1(x, u, p, t) = B_w(k) R_w B_w(k)^T``.

Finally, this package provides symbolic indexing of the solution object, such that one can easily obtain the estimated posterior distribution over any arbitrary variable in the model, including "observed" variables that are not part of the state vector being estimated by the estimator.



# Workflow

> [!TIP]
> It is assumed that the reader is familiar with the basics of LowLevelParticleFilters.jl. consult [the documentation](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/) and the video lectures liked therein to obtain such familiarity.

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

@mtkmodel SimpleSys begin
    @variables begin
        x(t) = 2.0
        u(t) = 0
        y(t)
        w(t), [disturbance = true, input = true]
    end
    @equations begin
        D(x) ~ -x + u + w # Explicitly encode where dynamics noise enters the system with w
        y ~ x
    end
end

@named model = SimpleSys() # Do not use @mtkbuild here
cmodel = complete(model) # complete is required for variable indexing since we did not use @mtkbuild above
inputs  = [cmodel.u]     # The (unbound) inputs to the system
outputs = [cmodel.y]     # The outputs for which we obtain measurements
disturbance_inputs = [cmodel.w] # The dynamics disturbance inputs to the system

nu = length(inputs)             # Number of inputs
nw = length(disturbance_inputs) # Number of disturbance inputs
ny = length(outputs)            # Number of measured outputs
R1 = SMatrix{nw,nw}(0.01I(nw))   # Dynamics noise covariance
R2 = SMatrix{ny,ny}(0.1I(ny))   # Measurement noise covariance

df = SimpleMvNormal(R1)         # Dynamics noise distribution. This has to be a Gaussian if using a Kalman-type filter
dg = SimpleMvNormal(R2)         # Measurement noise distribution. This has to be a Gaussian if using a Kalman-type filter

Ts = 0.1                        # Sampling interval
discretization = (f,Ts,ndiff,nalg,nu)->SeeToDee.Rk4(f, Ts) # Discretization method

prob = StateEstimationProblem(model, inputs, outputs; disturbance_inputs, df, dg, discretization, Ts)

# We instantiate two different filters for comparison
ekf = get_filter(prob, ExtendedKalmanFilter)
ukf = get_filter(prob, UnscentedKalmanFilter)

# Simulate some data from the trajectory distribution implied by the model
u = [randn(nu) for _ in 1:30] # A random input sequence
x,u,y = simulate(ekf, u, dynamics_noise=true, measurement_noise=true)

# Perform the filtering in batch since the entire input-output sequence is available
fsole = forward_trajectory(ekf, u, y)
fsolu = forward_trajectory(ukf, u, y)

# Wrap the filter solution objects in a StateEstimationSolution object
sole = StateEstimationSolution(fsole, prob)
solu = StateEstimationSolution(fsolu, prob)

# We can access the solution to any variable in the model easily
sole[cmodel.x] == sole[cmodel.y]

# We can also obtain the solution as a trajectory of probability distributions
sole[cmodel.x, dist=true]

# We can plot the filter solution object using the plot recipe from LowLevelParticleFilters
using Plots
plot(fsole, size=(1000, 1000))
plot!(fsole.t, reduce(hcat, x)', lab="True x")

plot(fsolu, size=(1000, 1000))
plot!(fsolu.t, reduce(hcat, x)', lab="True x")

# We can also plot the wrapped solution object
plot(sole)
plot!(solu)

# The wrapped solution object allows for symbolic indexing,
# note how we can easily plot the posterior distribution over y^2 + 0.1*sin(u) 
plot(sole, idxs=cmodel.y^2 + 0.1*sin(cmodel.u))
plot!(solu, idxs=cmodel.y^2 + 0.1*sin(cmodel.u))
```