# ROLL NO: 6446426 --> RKF4(5) integrator
# tolerance: [10e-12, 10e-10, 10e-8, 10e-6]
# initial step size: 10 s
# maximum step size: infinity
# min step size: 10e-12


import os
import numpy as np
import plotly.graph_objects as go
from integrator_analysis_helper_functions_Q2 import *
import tudatpy.kernel.astro.element_conversion as ec
from scipy.stats import linregress

current_directory = os.getcwd()

spice.load_standard_kernels()
spice.load_kernel(current_directory + "/juice_mat_crema_5_1_150lb_v01.bsp")

bodies = create_bodies()

# Define list of integrator tolerances
integration_tolerances = [1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6]


error_histories_benchmark =     {0: {}, 1: {}}
max_errors_benchmark =          {0: {}, 1: {}}
mean_step_sizes =               {0: {}, 1: {}}
reg_case_step_sizes =           {0: {}, 1: {}}
blockwise_step_sizes =          {0: {}, 1: {}}
blockwise_error_histories =     {0: {}, 1: {}}


# Iterate over the mission phases
for current_phase in range(len(central_bodies_per_phase)):

    # Create initial state and time
    current_phase_start_time = initial_times_per_phase[current_phase]
    current_phase_end_time = current_phase_start_time + propagation_times_per_phase[current_phase]

    # Define central body of propagation
    current_central_body = central_bodies_per_phase[current_phase]


    initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="JUICE",
        observer_body_name=current_central_body,
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=current_phase_start_time,
    )



    # Define termination conditions (enforce exact termination time)
    termination_condition = propagation_setup.propagator.time_termination(
        current_phase_end_time, terminate_exactly_on_final_condition=True
    )

    # Create acceleration models for perturbed case
    perturbed_acceleration_models = get_perturbed_accelerations(
        current_central_body, bodies
    )

# ......................................................................................................................
#                                   BENCHMARK ROUTINE
# ......................................................................................................................

    # Define integrator settings for benchmark
    benchmark_step_size = 10.0 if current_central_body == 'Callisto' else 20.0
    benchmark_integrator_settings = get_fixed_step_size_integrator_settings(benchmark_step_size)

    # Create integrator settings


    # Create propagator settings for perturbed case
    perturbed_propagator_settings = propagation_setup.propagator.translational(
        central_bodies=[current_central_body],
        acceleration_models=perturbed_acceleration_models,         bodies_to_integrate=['JUICE'],
        initial_states=initial_state,
        initial_time=current_phase_start_time,
        integrator_settings=benchmark_integrator_settings,  #
        termination_settings=termination_condition,
    )

    # Propagate benchmark dynamics
    benchmark_dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, perturbed_propagator_settings)

    # Create interpolator for benchmark results
    interpolator_settings = interpolators.lagrange_interpolation(8)
    benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        benchmark_dynamics_simulator.propagation_results.state_history,
        interpolator_settings,
    )


# ......................................................................................................................
#                                   REGULAR ROUTINE
#                             (Variable step-size routine)
# ......................................................................................................................

    # Perform integration of dynamics with different tolerances
    for current_tolerance in integration_tolerances:

        # Define integrator step settings
        initial_time_step = 10.0
        minimum_step_size = 1.0e-12
        maximum_step_size = np.inf

        # Retrieve coefficient set
        coefficient_set = propagation_setup.integrator.rkf_45

        step_size_control_settings = (
            propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(current_tolerance,
                                                                                        current_tolerance))

        step_size_validation_settings = propagation_setup.integrator.step_size_validation(minimum_step_size,
                                                                                          maximum_step_size)

        # Create variable step-size integrator settings
        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(initial_time_step, coefficient_set,
                                                                                     step_size_control_settings, step_size_validation_settings)

        # Define output file name
        file_output_identifier = (
            "Q3_tolerance_index_"
            + str(integration_tolerances.index(current_tolerance))
            + "_phase_index"
            + str(current_phase)
        )

        # Propagate dynamics for perturbed and unperturbed case
        perturbed_propagator_settings = propagation_setup.propagator.translational(
            central_bodies=[current_central_body],
            acceleration_models=perturbed_acceleration_models,
            bodies_to_integrate=['JUICE'],
            initial_states=initial_state,
            initial_time=current_phase_start_time,
            integrator_settings=integrator_settings,
            termination_settings=termination_condition,
        )

        perturbed_dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, perturbed_propagator_settings)

        write_propagation_results_and_benchmark_difference_to_file(
            perturbed_dynamics_simulator, file_output_identifier, benchmark_interpolator
        )



# ......................................................................................................................
#                                      BENCHMARK ERRORS
# ......................................................................................................................

        state_history = perturbed_dynamics_simulator.propagation_results.state_history

        benchmark_difference = get_difference_wrt_benchmarks(state_history, benchmark_interpolator)
        epochs_Q3a = np.array(list(benchmark_difference.keys()))
        step_sizes_used = np.diff(epochs_Q3a)
        mean_step_size = np.mean(step_sizes_used)
        mean_step_sizes[current_phase][current_tolerance] = mean_step_size
        reg_case_step_sizes[current_phase][current_tolerance] = step_sizes_used
        times_hours_Q3a = (epochs_Q3a - current_phase_start_time) / 3600.0

        state_difference_Q3a = np.vstack(list(benchmark_difference.values()))
        position_error_norm_Q3a = np.linalg.norm(state_difference_Q3a[:,:3], axis=1)

        error_histories_benchmark[current_phase][current_tolerance] = (times_hours_Q3a, position_error_norm_Q3a)
        max_errors_benchmark[current_phase][current_tolerance] = np.max(position_error_norm_Q3a)

        # Write results to files
        file_output_identifier = (
                "Q3_tolerance_size_" + str(current_tolerance) + "_phase_index" + str(current_phase)
        )
        write_propagation_results_and_benchmark_difference_to_file(
            perturbed_dynamics_simulator,
            file_output_identifier,
            benchmark_interpolator
        )


    # ......................................................................................................................
    #                                                    Q3d
    # ......................................................................................................................
    # ALTERNATIVE STEP SIZE METHOD USED: step_size_control_blockwise_scalar_tolerance

    for current_tolerance in integration_tolerances:
        block_indices = propagation_setup.integrator.standard_cartesian_state_element_blocks(6,1)

        step_size_control_settings_Q3d = (
            propagation_setup.integrator.step_size_control_blockwise_scalar_tolerance(block_indices, current_tolerance, current_tolerance))
        step_size_validation_settings_Q3d = propagation_setup.integrator.step_size_validation(1.0e-12, np.inf)
        integrator_settings_Q3d = propagation_setup.integrator.runge_kutta_variable_step(10.0,
                                                                                         propagation_setup.integrator.rkf_45,
                                                                                     step_size_control_settings_Q3d,
                                                                                     step_size_validation_settings_Q3d)
        perturbed_propagator_settings_Q3d = propagation_setup.propagator.translational(
            central_bodies=[current_central_body],
            acceleration_models=perturbed_acceleration_models,
            bodies_to_integrate=['JUICE'],
            initial_states=initial_state,
            initial_time=current_phase_start_time,
            integrator_settings=integrator_settings_Q3d,
            termination_settings=termination_condition,
        )

        perturbed_dynamics_simulator_Q3d = numerical_simulation.create_dynamics_simulator(bodies,
                                                                                          perturbed_propagator_settings_Q3d)

        state_history_Q3d = perturbed_dynamics_simulator_Q3d.propagation_results.state_history
        benchmark_difference_Q3d = get_difference_wrt_benchmarks(state_history_Q3d, benchmark_interpolator)
        epochs_Q3d = np.array(list(benchmark_difference_Q3d.keys()))
        step_sizes_Q3d = np.diff(epochs_Q3d)
        times_hours_Q3d = (epochs_Q3d - current_phase_start_time) / 3600.0

        state_diff_Q3d = np.vstack(list(benchmark_difference_Q3d.values()))
        position_error_norm_Q3d = np.linalg.norm(state_diff_Q3d[:, :3], axis=1)

        blockwise_step_sizes[current_phase][current_tolerance] = step_sizes_Q3d
        blockwise_error_histories[current_phase][current_tolerance] = (times_hours_Q3d, position_error_norm_Q3d)

# ......................................................................................................................
#                                                    PLOTS
# ......................................................................................................................

phase_names = {
    0: 'Callisto Flyby',
    1: 'GCO500 Orbit'
}

p_dir = 'plots'
os.makedirs(p_dir, exist_ok=True)

# ......................................................................................................................
#                                                    Q3a
# ......................................................................................................................

# .... PLOT 3.1: ERROR vs TIME PLOT ....

for current_phase in range(len(central_bodies_per_phase)):
    fig = go.Figure()

    for tol in integration_tolerances:
        times_hours, errors = error_histories_benchmark[current_phase][tol]
        fig.add_trace(go.Scatter(x=times_hours, y=errors, mode="lines", name=f'tol = {tol} s'))

    fig.update_layout(
        title=f'PLOT 3.1: Position Error in {phase_names[current_phase]} Phase',

        xaxis=dict(
            title='Time (hours)',
            showline=True,
            linecolor="white",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickcolor="white",
        ),
        yaxis=dict(
            title='Position Error (m)',
            showline=True,
            linecolor="white",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickcolor="white",
        ),
        yaxis_type='log',
        template="plotly_dark",
        font=dict(
            family="Times New Roman",
            size=16,
            color="white"
        ),
        title_font=dict(
            family="Times New Roman",
            size=18,
            color="white"
        )
    )

    fig.show()
    fig.write_image(os.path.join(p_dir, f'Q3a_position_error_vs_time_{current_phase}.png'), width=1200, height=800)


# ......................................................................................................................
#                                                    Q3b
# ......................................................................................................................

# .... PLOT 3.2: MAXIMUM ERROR vs TOLERANCE PLOT ....
fig = go.Figure()

for current_phase in range(len(central_bodies_per_phase)):
    dt = np.array(integration_tolerances)
    max_vals = np.array([max_errors_benchmark[current_phase][tol] for tol in integration_tolerances])

    fig.add_trace(go.Scatter(x = dt, y=max_vals, mode="lines+markers", name=f'{phase_names[current_phase]} max error'))

fig.update_layout(
    title = f'PLOT 3.2: Maximum Position Error vs Tolerance',

    xaxis=dict(
        title='Integrator Tolerance',
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="white",
    ),
    yaxis=dict(
        title='Maximum Position Error (m)',
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="white",
    ),
    xaxis_type = 'log', # x-axis type is log as step sizes vary exponentially
    yaxis_type = 'log',
    template = "plotly_dark",
    font=dict(
        family="Times New Roman",
        size=14,
        color="white"
    ),
    title_font=dict(
        family="Times New Roman",
        size=16,
        color="white"
    )

)

fig.show()
fig.write_image(os.path.join(p_dir, f'Q3b_max_error_vs_tolerance.png'), width=1200, height=800)


# .... PLOT 3.3: STEP SIZE vs TIME ....

for current_phase in range(len(central_bodies_per_phase)):

    fig = go.Figure()
    for tol in integration_tolerances:
        times_hours, _ = error_histories_benchmark[current_phase][tol]
        dt = reg_case_step_sizes[current_phase][tol]
        fig.add_trace(go.Scatter(x = times_hours[:-1], y=dt, mode="lines", name=f'tol = {tol}'))

    fig.update_layout(
        title = f'PLOT 3.3: Step Size vs Time in {phase_names[current_phase]}',

        xaxis=dict(
            title='Time (hours)',
            showline=True,
            linecolor="white",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickcolor="white",
        ),
        yaxis=dict(
            title='Step Size Δt (s)',
            showline=True,
            linecolor="white",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickcolor="white",
        ),
        yaxis_type = 'log',
        template = "plotly_dark",
        font=dict(
            family="Times New Roman",
            size=14,
            color="white"
        ),
        title_font=dict(
            family="Times New Roman",
            size=16,
            color="white"
        )

    )

    fig.show()
    fig.write_image(os.path.join(p_dir, f'Q3b_step_time_vs_time_{current_phase}.png'), width=1200, height=800)

# ......................................................................................................................
#                                                    Q3d
# ......................................................................................................................

fig = go.Figure()
for tol in integration_tolerances:
    times_ref, _ = error_histories_benchmark[1][tol]
    dt_ref = reg_case_step_sizes[1][tol]
    fig.add_trace(go.Scatter(x=times_ref[:-1], y=dt_ref, mode="lines",line=dict(dash='dash'), name=f'elementwise tol = {tol}'))

    times_bw, _ = blockwise_error_histories[1][tol]
    dt_bw = blockwise_step_sizes[1][tol]
    fig.add_trace(go.Scatter(x=times_bw[:-1], y=dt_bw, mode="lines",line=dict(dash='solid'), name=f'blockwise tol = {tol}'))
fig.update_layout(
    title=f'PLOT 3.4: Blockwise and Elementwise Step Size vs Time in {phase_names[current_phase]}',

    xaxis=dict(
        title='Time (hours)',
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="white",
    ),
    yaxis=dict(
        title='Step Size Δt (s)',
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="white",
    ),
    yaxis_type='log',
    template="plotly_dark",
    font=dict(
        family="Times New Roman",
        size=14,
        color="white"
    ),
    title_font=dict(
        family="Times New Roman",
        size=16,
        color="white"
    )

)

fig.show()
fig.write_image(os.path.join(p_dir, f'Q3d_step_time_vs_time_{current_phase}.png'), width=1200, height=800)

# .... PLOT 3.5: POSITION ERROR vs TIME ....
fig = go.Figure()
for tol in integration_tolerances:
    times_bw, errors_bw = blockwise_error_histories[1][tol]
    fig.add_trace(go.Scatter(x=times_bw, y=errors_bw, mode="lines",line=dict(dash='solid'), name=f'blockwise tol = {tol}'))
fig.update_layout(
    title=f'PLOT 3.4: Blockwise Position Error vs Time in {phase_names[current_phase]}',

    xaxis=dict(
        title='Time (hours)',
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="white",
    ),
    yaxis=dict(
        title='Position Error (m)',
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="white",
    ),
    yaxis_type='log',
    template="plotly_dark",
    font=dict(
        family="Times New Roman",
        size=14,
        color="white"
    ),
    title_font=dict(
        family="Times New Roman",
        size=16,
        color="white"
    )
)


fig.show()
fig.write_image(os.path.join(p_dir, f'Q3d_pos_error_vs_time_{current_phase}_bw.png'), width=1200, height=800)



# Get initial state for GCO500
initial_state_gco500 = spice.get_body_cartesian_state_at_epoch(
    target_body_name="JUICE",
    observer_body_name="Ganymede",
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=initial_times_per_phase[1],  # GCO500 phase
)

# Convert to Keplerian elements
mu_ganymede = bodies.get("Ganymede").gravitational_parameter
keplerian_state = ec.cartesian_to_keplerian(initial_state_gco500[:6], mu_ganymede)

print(f"Semi-major axis:  {keplerian_state[0]/1000:.2f} km")
print(f"Eccentricity:     {keplerian_state[1]:.6f}")
print(f"Inclination:      {np.degrees(keplerian_state[2]):.2f} deg")
print(f"Orbital period:   {2*np.pi*np.sqrt(keplerian_state[0]**3/mu_ganymede)/3600:.2f} hours")
print(f"Orbital period: {2*np.pi*np.sqrt(keplerian_state[0]**3/mu_ganymede)/3600:.4f} hours")
print(f"Orbital period: {2*np.pi*np.sqrt(keplerian_state[0]**3/mu_ganymede):.1f} seconds")



T = 11013.2  # orbital period in seconds
t0 = initial_times_per_phase[1]  # GCO500 start time

# Approximate dip times from your plot (in hours, convert to seconds)
dip_times_hours = [0.5, 1.35, 2.1, 2.9]
dip_times_seconds = [t * 3600 for t in dip_times_hours]

print(f"Orbital period: {T:.1f} s = {T / 3600:.4f} h\n")

for i, dt in enumerate(dip_times_seconds):
    epoch = t0 + dt

    # Get cartesian state at this epoch
    state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="JUICE",
        observer_body_name="Ganymede",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=epoch,
    )

    # Convert to Keplerian
    kep = ec.cartesian_to_keplerian(state[:6], mu_ganymede)

    sma = kep[0] / 1000  # km
    ecc = kep[1]
    inc = np.degrees(kep[2])
    raan = np.degrees(kep[3])
    aop = np.degrees(kep[4])
    ta = np.degrees(kep[5]) % 360  # true anomaly 0-360
    r = np.linalg.norm(state[:3]) / 1000  # km from Ganymede centre
    lat = np.degrees(np.arcsin(state[2] / np.linalg.norm(state[:3])))  # latitude

    print(f"Dip {i + 1} at t = {dip_times_hours[i]} h:")
    print(f"  r (distance)    = {r:.1f} km")
    print(f"  true anomaly    = {ta:.1f} deg")
    print(f"  latitude        = {lat:.1f} deg")
    print(f"  Expected:       ", end="")
    if ta < 30 or ta > 330:
        print("periapsis (TA ~ 0)")
    elif 150 < ta < 210:
        print("apoapsis (TA ~ 180)")
    elif lat > 60:
        print("north pole pass")
    elif lat < -60:
        print("south pole pass")
    else:
        print("equatorial region")
    print()


for current_phase in range(2):
    tols = np.array(integration_tolerances)
    max_vals = np.array([max_errors_benchmark[current_phase][tol] for tol in integration_tolerances])
    slope, intercept, r, p, se = linregress(np.log10(tols), np.log10(max_vals))
    print(f"{phase_names[current_phase]}: slope = {slope:.2f}")

for tol in integration_tolerances:
    _, errors_ref = error_histories_benchmark[1][tol]
    times_bw, errors_bw = blockwise_error_histories[1][tol]
    print(f"tol={tol:.0e}: elementwise max={np.max(errors_ref):.4e} m, "
          f"blockwise max={np.max(errors_bw):.4e} m, "
          f"ratio={np.max(errors_bw)/np.max(errors_ref):.2f}x")