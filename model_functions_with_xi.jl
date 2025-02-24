function normalisation(ω_c, interval_time)
    return 2 * sinint(ω_c * interval_time / 2) / pi
end

function G_weight(t, p)
    return ifelse(t != p.t_star, sin(p.ω_c * (p.t_star - t)) / (pi * (p.t_star - t)), p.ω_c / pi)
end

function int_G_weight(t, p)
    sol = sinint(p.ω_c * p.interval_time / 2) - sinint(p.ω_c * (p.t_star - t))
    return sol / pi
end

function model_setup(;checkpoint_name="checkpoint", Nx=250, Nz=500, H=3kilometers, Fr=0.25, N = 1e-3, k = 0.005, h₀=25meters, f = 1e-4, stop_time=5days)
    # set other constants
    Nsqr = N^2
    x_range = 3 * pi / k
    U_const = N * h₀ / Fr

    # create grid
    underlying_grid = RectilinearGrid(
        size = (Nx, Nz),
        x = (-x_range, x_range),
        z = (-H, 0),
        halo = (4, 4),
        topology = (Periodic, Flat, Bounded)
    )

    hill(x) = h₀ * (1 + sin(k * x))
    bottom(x) = -H + hill(x)

    grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

    # v forcing
    coriolis = FPlane(f = f)
    v_forcing_eq(x, z, t, p) = p.U_const * p.coriolis.f
    v_forcing = Forcing(v_forcing_eq, parameters = (; U_const, coriolis))

    ν_diff = κ_diff = 1
    horizontal_diffusivity = HorizontalScalarDiffusivity(ν = ν_diff, κ = κ_diff)

    model = NonhydrostaticModel(;
        grid,
        coriolis = coriolis,
        buoyancy = BuoyancyTracer(),
        tracers = (:b),
        closure = horizontal_diffusivity,
        forcing = (; v=v_forcing)
    )

    # field constants and timestepping
    uᵢ(x, z) = U_const
    vᵢ(x, z) = 0

    bᵢ(x, z) = Nsqr * z

    set!(model, u=uᵢ, b = bᵢ, v = vᵢ)    

    dx = max(H / Nz, 2 * x_range / Nx)
    Δt = dx / (2 * U_const)

    simulation = Simulation(model; Δt, stop_time)
    wizard = TimeStepWizard(max_change=1.1, cfl=0.7)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))
    simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(1days), prefix=checkpoint_name)

    wall_clock = Ref(time_ns())

    function progress(sim)
        print("progress\n")
        elapsed = 1e-9 * (time_ns() - wall_clock[])
    
        msg = @sprintf(
            "iteration: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹\n",
            iteration(sim),
            prettytime(sim),
            prettytime(elapsed),
            maximum(abs, sim.model.velocities.w)
        )
    
        wall_clock[] = time_ns()
    
        @info msg
    
        return nothing
    end

    add_callback!(simulation, progress, name = :progress, IterationInterval(1))

    run!(simulation)
    nothing #hide
end

function model_pickup(nc_save_name; Nx=250, Nz=500, H=3kilometers, Fr=0.25, N = 1e-3, k = 0.005, h₀=25meters, f = 1e-4, initial_time=5days, interval_time=1days, checkpoint_file_path=raw"checkpoint_files\model_checkpoint_weight_iteration8608.jld2")
    Nsqr = N^2
    x_range = 3 * pi / k
    U_const = N * h₀ / Fr

    underlying_grid = RectilinearGrid(
        size = (Nx, Nz),
        x = (-x_range, x_range),
        z = (-H, 0),
        halo = (4, 4),
        topology = (Periodic, Flat, Bounded)
    )

    hill(x) = h₀ * (1 + sin(k * x))
    bottom(x) = -H + hill(x)

    grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

    coriolis = FPlane(f = f)
    v_forcing_eq(x, z, t, p) = p.U_const * p.coriolis.f
    v_forcing = Forcing(v_forcing_eq, parameters = (; U_const, coriolis))

    # ω_c = U_const * k / 2
    ω_c = coriolis.f
    t_star = initial_time + (interval_time/2)
    normalisation_const = normalisation(ω_c, interval_time)

    # b lagrangian mean forcing
    b_LM_forcing_eq(x, z, t, b, p) = b * G_weight(t, p) / p.normalisation_const
    b_LM_forcing = Forcing(b_LM_forcing_eq, parameters = (; ω_c, t_star, normalisation_const), field_dependencies=:b)
    
    # u lagrangian mean forcing
    u_LM_forcing_eq(x, z, t, u, p) = u * G_weight(t, p) / p.normalisation_const
    u_LM_forcing = Forcing(u_LM_forcing_eq, parameters = (; ω_c, t_star, U_const, normalisation_const), field_dependencies=:u)

    # xi maps
    xix_forcing_eq(x, z, t, u, p) = -u * int_G_weight(t, p) / p.normalisation_const
    xix_forcing = Forcing(xix_forcing_eq, parameters = (; ω_c, t_star, U_const, interval_time, normalisation_const), field_dependencies=:u)
    
    xiz_forcing_eq(x, z, t, w, p) = -w * int_G_weight(t, p) / p.normalisation_const
    xiz_forcing = Forcing(xiz_forcing_eq, parameters = (; ω_c, t_star, U_const, interval_time, normalisation_const), field_dependencies=:w)
    # include 1 to 3 map using heaviside function (3.16)

    ν_diff = κ_diff = 1
    horizontal_diffusivity = HorizontalScalarDiffusivity(ν = ν_diff, κ = κ_diff)

    forcing_params = (;
        v=v_forcing,
        b_LM=b_LM_forcing,
        u_LM=u_LM_forcing,
        xix=xix_forcing,
        xiz=xiz_forcing
    )

    model = NonhydrostaticModel(;
        grid,
        coriolis = coriolis,
        buoyancy = BuoyancyTracer(),
        tracers = (:b, :b_LM, :u_LM, :xix, :xiz),
        closure = horizontal_diffusivity,
        forcing = forcing_params
    )

    b_LMᵢ(x, z) = 0
    u_LMᵢ(x, z) = 0
    xixᵢ(x, z) = 0
    xizᵢ(x, z) = 0

    set!(model, b_LM=b_LMᵢ, u_LM=u_LMᵢ, xix=xixᵢ, xiz=xizᵢ)
    set!(model, checkpoint_file_path)
    dx = max(H / Nz, 2 * x_range / Nx)

    Δt = dx / (2 * U_const)

    stop_time = initial_time + interval_time

    simulation = Simulation(model; Δt, stop_time)
    wizard = TimeStepWizard(max_change=1.1, cfl=0.7)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

    wall_clock = Ref(time_ns())

    function progress(sim)
        elapsed = 1e-9 * (time_ns() - wall_clock[])
    
        msg = @sprintf(
            "iteration: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹\n",
            iteration(sim),
            prettytime(sim),
            prettytime(elapsed),
            maximum(abs, sim.model.velocities.w)
        )
    
        wall_clock[] = time_ns()
    
        @info msg
    
        return nothing
    end

    add_callback!(simulation, progress, name = :progress, IterationInterval(10))
    nothing #hide

    b = model.tracers.b
    b_LM = model.tracers.b_LM
    u_LM = model.tracers.u_LM
    xix = model.tracers.xix
    xiz = model.tracers.xiz
    u, v, w = model.velocities

    u′ = u - U_const
    # b′ = b - Nsqr * z

    N² = ∂z(b)

    filename = nc_save_name
    save_fields_interval = 5minutes

    model_dicts = Dict(
        "u" => u, 
        "u_pert" => u′, 
        "w" => w,
        "b" => b,
        "N2" => N²,
        "b_LM" => b_LM,
        "u_LM" => u_LM,
        "xix" => xix,
        "xiz" => xiz
    )

    simulation.output_writers[:fields] = NetCDFOutputWriter(
        model,
        model_dicts,
        filename=filename,
        schedule = TimeInterval(save_fields_interval),
        overwrite_existing = true
    )

    run!(simulation)
end
