## DAE UKF on a constrained pendulum — INDEX-1 Baumgarte form. The position
## constraint x² + y² = 1 is replaced by its twice-differentiated,
## dynamics-substituted (acceleration-level) form augmented with Baumgarte
## stabilization. The unstabilized acceleration-level constraint is neutrally
## stable in (x²+y²−1): discretization error drifts the state off the unit
## circle with no restoring force. Adding 2α·(x·vx + y·vy) + (β²/2)·(x²+y²−1)
## gives the constraint a critically-damped restoring law (α = β = 50), so
## violations decay exponentially with timescale 1/α. With this reformulation
## MTK does not need to do index reduction and the full predict + correct +
## `forward_trajectory` pipeline keeps the covariance positive-definite.
@testset "DAE UKF" begin
    using SimpleNonlinearSolve
    using SciMLBase

    @component function Pendulum(; name)
        pars = @parameters begin
            g_pend = 9.82
            α_bg   = 50.0
            β_bg   = 50.0
        end
        vars = @variables begin
            x(t)  = 1.0
            y(t)  = 0.0
            vx(t) = 0.0
            vy(t) = 0.0
            λ(t)  = 0.0
            f1(t) = 0.0
            f2(t) = 0.0
            meas_x(t)
            meas_λ(t)
            w1(t), [disturbance = true, input = true]
            w2(t), [disturbance = true, input = true]
            w3(t), [disturbance = true, input = true]
            w4(t), [disturbance = true, input = true]
        end
        eqs = Equation[
            D(x)  ~ vx + w1
            D(y)  ~ vy + w2
            D(vx) ~ -λ*x + f1 + w3
            D(vy) ~ -λ*y - g_pend + f2 + w4
            0     ~ vx^2 + vy^2 + x*(-λ*x + f1 + w3) + y*(-λ*y - g_pend + f2 + w4) +
                2*α_bg*(x*vx + y*vy) +
                (β_bg^2 / 2)*(x^2 + y^2 - 1)
            meas_x ~ x
            meas_λ ~ λ
        ]
        return System(eqs, t, vars, pars; name)
    end

    @named pendulum = Pendulum()
    cpend = complete(pendulum)
    pend_inputs   = [cpend.f1, cpend.f2]
    pend_outputs  = [cpend.meas_x, cpend.meas_λ]
    pend_dist_in  = [cpend.w1, cpend.w2, cpend.w3, cpend.w4]

    Ts_dae = 0.01
    R1_dae = SMatrix{4,4}(Diagonal([1e-3, 1e-3, 1e-3, 1e-3]))
    R2_dae = SMatrix{2,2}(1e-2*I)
    df_dae = SimpleMvNormal(R1_dae)
    dg_dae = SimpleMvNormal(R2_dae)

    dae_disc = (f, Ts, xi, ai, nu) ->
        SeeToDee.Trapezoidal(f, Ts, xi, ai, nu; inplace=false)

    x0map_pend = [cpend.x => 1.0, cpend.y => 0.0,
                  cpend.vx => 0.0, cpend.vy => 0.0,
                  cpend.λ => 0.0]

    prob_pend = StateEstimationProblem(cpend, pend_inputs, pend_outputs;
                                       disturbance_inputs = pend_dist_in,
                                       df = df_dae, dg = dg_dae,
                                       discretization = dae_disc,
                                       Ts = Ts_dae,
                                       x0map = x0map_pend, σ0 = 0.1)

    # Index-1 form: 4 differential states + 1 algebraic equation, nw == nx_diff.
    @test prob_pend.na == 1
    @test length(prob_pend.x_inds) == 4
    @test prob_pend.nw == length(prob_pend.x_inds)

    csolver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol=1e-12)
    daeukf  = get_filter(prob_pend, DAEUnscentedKalmanFilter; constraint_solver = csolver)

    @test daeukf isa DAEUnscentedKalmanFilter
    @test length(daeukf.x) == 4
    @test length(daeukf.xz) == 5
    @test size(daeukf.R) == (4, 4)
    @test daeukf.nu == prob_pend.nu
    @test daeukf.ny == prob_pend.ny

    # Auto-generated descriptor split round-trips, and the residual at the
    # initial on-manifold state vanishes.
    let
        xz0 = daeukf.xz
        x_part, z_part = daeukf.get_x_z(xz0)
        @test daeukf.build_xz(x_part, z_part) ≈ xz0
        r = daeukf.residual(x_part, z_part, SA[0.0, 0.0], daeukf.p, 0.0)
        @test maximum(abs, r) < 1e-8
    end

    # End-to-end forward_trajectory under a driven input: the filter completes
    # the trajectory and the covariance stays positive-definite throughout.
    T_steps = 500
    t_vec   = (0:T_steps-1) .* Ts_dae
    u_drive = [SA[sin(tt^2), sin(tt^2 + 1.0)] for tt in t_vec]
    _, u_sim, y_sim = simulate(daeukf, u_drive)

    daeukf2 = get_filter(prob_pend, DAEUnscentedKalmanFilter; constraint_solver = csolver)
    fsol = forward_trajectory(daeukf2, u_sim, y_sim)

    @test length(fsol.xt) == T_steps
    @test all(R -> minimum(eigen(Symmetric(Matrix(R))).values) > 0, fsol.Rt)
    @test all(xt -> all(isfinite, xt), fsol.xt)

    # ---- StateEstimationSolution symbolic indexing --------------------------
    # The DAE-UKF stores only the differential sub-state (length nx_diff). The
    # solution object must still index the algebraic states (reconstructed from
    # the constraint) and arbitrary expressions, not just the differential
    # slice — here x,y,vx,vy are differential and λ is algebraic.
    @testset "StateEstimationSolution indexing" begin
        sesol = StateEstimationSolution(fsol, prob_pend)

        # A differential state round-trips against the stored estimate.
        i_x       = findfirst(isequal(cpend.x), prob_pend.state)
        x_diffpos = findfirst(==(i_x), prob_pend.x_inds)
        @test sesol[cpend.x] ≈ [xt[x_diffpos] for xt in fsol.xt]

        # An algebraic state (λ) — previously a BoundsError — is reconstructed.
        λest = sesol[cpend.λ]
        @test length(λest) == length(fsol.xt)
        @test all(isfinite, λest)

        # The reconstructed full state lies on the constraint manifold: the
        # algebraic residual vanishes at every step.
        fullstates = sesol[collect(prob_pend.state)]
        @test all(eachindex(fullstates)) do k
            xp, zp = daeukf2.get_x_z(SVector{prob_pend.nx}(fullstates[k]))
            maximum(abs, daeukf2.residual(xp, zp, u_sim[k], daeukf2.p, (k-1)*Ts_dae)) < 1e-6
        end

        # Distribution + sampling paths work for an algebraic state too
        # (previously a MethodError: propagate_distribution lacked a DAE method).
        dλ = sesol[cpend.λ, dist=true]
        @test length(dλ) == length(fsol.xt)
        @test all(d -> isfinite(d.μ[1]) && d.Σ[1,1] ≥ 0, dλ)
        @test length(sesol[cpend.λ, Nsamples=10]) == length(fsol.xt)

        # An expression sampled with Nsamples>1 routes the (rank-deficient,
        # regularized) reconstructed covariance through MvNormal sampling.
        psx = sesol[cpend.x^2, Nsamples=5]
        @test length(psx) == length(fsol.xt)
        @test all(p -> all(isfinite, p.particles), psx)

        # The plotting recipe routes through the dist=true path; must not error.
        @test (plot(sesol); true)
    end
end

## DAE-UKF with nw < nx_diff. Bead on the cubic surface  z³ + 3z = x² + y²,
## under gravity. Index-3 form; MTK can't symbolically invert the cubic in z,
## so the position constraint stays as a live algebraic equation (na = 3) and
## the dummy-derivative chart is globally fold-free (∂F/∂z = 3z² + 3 ≥ 3).
##
## Crucially, only the three force equations D(vx), D(vy), D(vz) can carry
## disturbance noise without breaking Pantelides, while the reduction leaves
## 4 differential states. This testset exercises the new path where the
## wrapper builds R1_diff = Bw·prob.df.Σ·Bwᵀ instead of requiring nw==nx_diff.
@testset "DAE UKF — nw < nx_diff (cubic surface)" begin
    using SimpleNonlinearSolve
    using SciMLBase

    @component function CubicSurface(; name)
        pars = @parameters begin
            g_c = 9.82
        end
        vars = @variables begin
            x(t)  = 0.4,    [state_priority=100]
            y(t)  = -0.3,   [state_priority=100]
            z(t)  = 0.0830, [state_priority=-1]
            vx(t) = 0.0,    [state_priority=100]
            vy(t) = 0.0,    [state_priority=100]
            vz(t) = 0.0,    [state_priority=-1]
            λ(t)  = 0.0
            f1(t) = 0.0
            f2(t) = 0.0
            f3(t) = 0.0
            meas_x(t)
            meas_λ(t)
            w3(t), [disturbance = true, input = true]
            w4(t), [disturbance = true, input = true]
            w5(t), [disturbance = true, input = true]
        end
        eqs = Equation[
            D(x)  ~ vx
            D(y)  ~ vy
            D(z)  ~ vz
            D(vx) ~ -2*x*λ          + f1 + w3
            D(vy) ~ -2*y*λ          + f2 + w4
            D(vz) ~  (3*z^2 + 3)*λ  + f3 - g_c + w5
            0     ~ z^3 + 3*z - (x^2 + y^2)
            meas_x ~ x
            meas_λ ~ λ
        ]
        return System(eqs, t, vars, pars; name)
    end

    @named cubic_surf = CubicSurface()
    ccs = complete(cubic_surf)
    cubic_inputs  = [ccs.f1, ccs.f2, ccs.f3]
    cubic_outputs = [ccs.meas_x, ccs.meas_λ]
    cubic_dist_in = [ccs.w3, ccs.w4, ccs.w5]

    Ts_c = 0.01
    R1_c = SMatrix{3,3}(Diagonal([1e-3, 1e-3, 1e-3]))
    R2_c = SMatrix{2,2}(1e-2*I)
    df_c = SimpleMvNormal(R1_c)
    dg_c = SimpleMvNormal(R2_c)
    dae_disc_c = (f, Ts, xi, ai, nu) ->
        SeeToDee.Trapezoidal(f, Ts, xi, ai, nu; inplace=false)
    x0map_c = [ccs.x => 0.4, ccs.y => -0.3, ccs.z => 0.0830,
               ccs.vx => 0.0, ccs.vy => 0.0, ccs.vz => 0.0, ccs.λ => 0.0]

    prob_c = StateEstimationProblem(ccs, cubic_inputs, cubic_outputs;
                                    disturbance_inputs = cubic_dist_in,
                                    df = df_c, dg = dg_c,
                                    discretization = dae_disc_c,
                                    Ts = Ts_c, x0map = x0map_c, σ0 = 0.05,
                                    init = true)

    # Sanity check: we are genuinely exercising nw < nx_diff.
    @test prob_c.nw == 3
    @test length(prob_c.x_inds) == 4
    @test prob_c.na == 3   # z, zˍt, λ all kept as algebraic

    csolver_c = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol=1e-12)
    daeukf_c  = get_filter(prob_c, DAEUnscentedKalmanFilter; constraint_solver = csolver_c)

    @test daeukf_c isa DAEUnscentedKalmanFilter
    @test size(daeukf_c.R) == (4, 4)
    # Bw·R1·Bwᵀ is rank-deficient (rank ≤ nw = 3) but still PSD.
    R1_eff = daeukf_c.R1
    @test issymmetric(R1_eff) || R1_eff ≈ R1_eff'
    @test minimum(eigvals(Symmetric(Matrix(R1_eff)))) >= -eps(Float64) * maximum(abs.(R1_eff))

    # End-to-end with moderate forcing. Gravity-bound bowl; chart is
    # globally fold-free so the trajectory cannot escape.
    T_c = 200
    t_vec_c = (0:T_c-1) .* Ts_c
    u_drive_c = [SA[1.0*sin(tt^2), 1.0*sin(tt^2 + 1.0), 1.0*sin(tt^2 + 2.0)] for tt in t_vec_c]
    _, u_sim_c, y_sim_c = simulate(daeukf_c, u_drive_c)

    daeukf_c2 = get_filter(prob_c, DAEUnscentedKalmanFilter; constraint_solver = csolver_c)
    fsol_c = forward_trajectory(daeukf_c2, u_sim_c, y_sim_c)

    @test length(fsol_c.xt) == T_c
    @test all(R -> minimum(eigen(Symmetric(Matrix(R))).values) > 0, fsol_c.Rt)
    @test all(xt -> all(isfinite, xt), fsol_c.xt)

    # ---- StateEstimationSolution indexing, multi-dimensional algebraic block --
    # Exercises the full-state reconstruction in the harder regime: na = 3
    # (z, zˍt, λ are algebraic), so the warm-started constraint solve and the
    # rank-deficient covariance regularizer run on a genuinely nonlinear
    # (cubic-in-z) constraint rather than the scalar index-1 case.
    @testset "StateEstimationSolution indexing (na=3)" begin
        sesol = StateEstimationSolution(fsol_c, prob_c)

        # A differential state round-trips against the stored estimate.
        i_x       = findfirst(isequal(ccs.x), prob_c.state)
        x_diffpos = findfirst(==(i_x), prob_c.x_inds)
        @test sesol[ccs.x] ≈ [xt[x_diffpos] for xt in fsol_c.xt]

        # An algebraic state (z) is reconstructed: finite, right length.
        zest = sesol[ccs.z]
        @test length(zest) == length(fsol_c.xt)
        @test all(isfinite, zest)

        # The full reconstructed state satisfies the cubic constraint everywhere.
        fullstates = sesol[collect(prob_c.state)]
        @test all(eachindex(fullstates)) do k
            xp, zp = daeukf_c2.get_x_z(SVector{prob_c.nx}(fullstates[k]))
            maximum(abs, daeukf_c2.residual(xp, zp, u_sim_c[k], daeukf_c2.p, (k-1)*Ts_c)) < 1e-6
        end

        # A nonlinear expression mixing differential (x, y) and algebraic (z)
        # states: the constraint residual must be ≈ 0 on the manifold (covers the
        # observed-function path for an arbitrary expression).
        cres = sesol[ccs.z^3 + 3*ccs.z - ccs.x^2 - ccs.y^2]
        @test maximum(abs, cres) < 1e-6

        # Distribution path on an algebraic state (multi-dim propagate_distribution).
        dz = sesol[ccs.z, dist=true]
        @test length(dz) == length(fsol_c.xt)
        @test all(d -> isfinite(d.μ[1]) && d.Σ[1,1] ≥ 0, dz)

        # An input symbol still resolves through the reconstruction gate.
        @test sesol[ccs.f1] == getindex.(fsol_c.u, 1)
    end
end

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
