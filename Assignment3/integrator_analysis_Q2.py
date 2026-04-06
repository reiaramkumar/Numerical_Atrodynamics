import os

import numpy as np
import time
from integrator_analysis_helper_functions_Q2 import *
import plotly.graph_objects as go
import json
current_directory = os.getcwd()

# Load spice kernels.
spice.load_standard_kernels()
spice.load_kernel(current_directory + "/juice_mat_crema_5_1_150lb_v01.bsp")

# Create the bodies for the numerical simulation
bodies = create_bodies()

# Define list of step size for integrator to take
step_sizes = [2 ** n for n in range(3, 10)]


phase_names = {
    0: 'Callisto Flyby',
    1: 'GCO500 Orbit'
}

# ....................................................
#                    Storage
state_histories_per_phase =     {0: {}, 1: {}}

error_histories_step_half =     {0: {}, 1: {}}
max_errors_step_half =          {0: {}, 1: {}}

error_histories_benchmark =     {0: {}, 1: {}}
max_errors_benchmark =          {0: {}, 1: {}}

cpu_times =                     {0: {}, 1: {}}
# ....................................................





# Iterate over phases
for current_phase in range(len(central_bodies_per_phase)):

    # Create initial state and time
    current_phase_start_time = initial_times_per_phase[current_phase]
    current_phase_end_time = (
            current_phase_start_time + propagation_times_per_phase[current_phase]
    )

    termination_settings = propagation_setup.propagator.time_termination(
        current_phase_end_time
    )

    current_central_body = central_bodies_per_phase[current_phase]

    # Retrieve JUICE initial state
    initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="JUICE",
        observer_body_name=current_central_body,
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=current_phase_start_time,
    )

    # Retrieve acceleration settings without perturbations
    acceleration_models = get_perturbed_accelerations(current_central_body, bodies)

# ......................................................................................................................
#                                   BENCHMARK ROUTINE
# ......................................................................................................................

    benchmark_step_size = 10.0 if current_central_body == 'Callisto' else 20.0
    benchmark_integrator_settings = get_fixed_step_size_integrator_settings(benchmark_step_size)

    benchmark_propagator_settings = propagation_setup.propagator.translational(
        central_bodies=[current_central_body],
        acceleration_models=acceleration_models,
        bodies_to_integrate=['JUICE'],
        initial_states=initial_state,
        initial_time=current_phase_start_time,
        integrator_settings=benchmark_integrator_settings,
        termination_settings=termination_settings,
    )

    benchmark_dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, benchmark_propagator_settings)
    benchmark_state_history = benchmark_dynamics_simulator.propagation_results.state_history
    interpolator_settings = interpolators.lagrange_interpolation(8,
                                                                 boundary_interpolation=interpolators.extrapolate_at_boundary)
    benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(benchmark_state_history,
                                                                                      interpolator_settings)
# ......................................................................................................................
# ......................................................................................................................

# ......................................................................................................................
#                                    REGULAR ROUTINE
# ......................................................................................................................

    # Save propagation results for each time step into a list, for analysis after all propagations are done
    propagation_results_per_step_size = list()

    # Iterate over step size
    for step_size in step_sizes:
        # Define integrator settings
        integrator_settings = get_fixed_step_size_integrator_settings(step_size)

        propagator_settings = propagation_setup.propagator.translational(central_bodies=[current_central_body],
                                                                         acceleration_models=acceleration_models,
                                                                         bodies_to_integrate=['JUICE'],
                                                                         initial_states=initial_state,
                                                                         initial_time=current_phase_start_time,
                                                                         integrator_settings=integrator_settings,
                                                                         termination_settings=termination_settings)

        # Propagate dynamics
        start = time.perf_counter()
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
        end = time.perf_counter()
        cpu_times[current_phase][step_size] = end - start

        state_history = dynamics_simulator.propagation_results.state_history
        state_histories_per_phase[current_phase][step_size] = state_history
        propagation_results_per_step_size.append(state_history)


# ......................................................................................................................
#                                      Q2b: BENCHMARK ERRORS
# ......................................................................................................................

        # Q2b
        benchmark_difference = get_difference_wrt_benchmarks(state_history, benchmark_interpolator)
        epochs_all = np.array(list(benchmark_difference.keys()))
        benchmark_epochs = np.array(list(benchmark_state_history.keys()))
        t_min = benchmark_epochs[4]
        t_max = benchmark_epochs[-5]
        valid_m = (epochs_all > t_min) & (epochs_all < t_max)
        valid_e = epochs_all[valid_m]
        benchmark_difference = {epoch: benchmark_difference[epoch] for epoch in valid_e if epoch in benchmark_difference}

        epochs_Q2b = np.array(list(benchmark_difference.keys()))
        times_hours_Q2b = (epochs_Q2b - current_phase_start_time) / 3600.0

        state_difference_Q2b = np.vstack(list(benchmark_difference.values()))
        position_error_norm_Q2b = np.linalg.norm(state_difference_Q2b[:,:3], axis=1)
        error_histories_benchmark[current_phase][step_size] = (times_hours_Q2b, position_error_norm_Q2b)
        max_errors_benchmark[current_phase][step_size] = np.max(position_error_norm_Q2b)

        # Write results to files
        file_output_identifier = (
                "Q2_step_size_" + str(step_size) + "_phase_index" + str(current_phase)
        )
        write_propagation_results_and_benchmark_difference_to_file(
            dynamics_simulator,
            file_output_identifier,
            benchmark_interpolator
        )

# ......................................................................................................................
#                                      Q2a: STEP HALVING ERRORS
# ......................................................................................................................
for cp in range(len(central_bodies_per_phase)):
    for step_size in step_sizes[1:]:

        coarse_history = state_histories_per_phase[cp][step_size]
        fine_history = state_histories_per_phase[cp][step_size // 2]

        difference_dict_Q2a = {}
        for epoch in coarse_history.keys():
            if epoch in fine_history:
                difference_dict_Q2a[epoch] = coarse_history[epoch] - fine_history[epoch]



        epochs_Q2a = np.array(list(difference_dict_Q2a.keys()))
        times_hours_Q2a = (epochs_Q2a - initial_times_per_phase[cp]) / 3600.0

        state_difference_Q2a = np.vstack(list(difference_dict_Q2a.values()))
        position_error_norm_Q2a = np.linalg.norm(state_difference_Q2a[:, :3], axis=1)
        error_histories_step_half[cp][step_size] = (times_hours_Q2a, position_error_norm_Q2a)
        max_errors_step_half[cp][step_size] = np.max(position_error_norm_Q2a)


# ......................................................................................................................
#                                                   PLOTS
# ......................................................................................................................


p_dir = 'plots'
os.makedirs(p_dir, exist_ok=True)

phase_names = {
    0: 'Callisto Flyby',
    1: 'GCO500 Orbit'
}

# ......................................................................................................................
#                                                    Q2a
# ......................................................................................................................

# .... PLOT 2.1: ERROR vs TIME PLOT ....

for current_phase in range(len(central_bodies_per_phase)):
    fig = go.Figure()

    for step_size in step_sizes[1:]:
        times_hours, errors = error_histories_step_half[current_phase][step_size]
        fig.add_trace(go.Scatter(x=times_hours, y=errors, mode="lines", name=f'Δt = {step_size} s vs {step_size // 2} s'))

    fig.update_layout(
        title=f'PLOT 2.1: Position Error in {phase_names[current_phase]} Phase',

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
    fig.write_image(os.path.join(p_dir, f'Q2a_position_error_vs_time_{current_phase}.png'), width=1200, height=800)

# .... PLOT 2.2: MAXIMUM ERROR vs STEP SIZE PLOT ....
fig = go.Figure()

for current_phase in range(len(central_bodies_per_phase)):
    dt = np.array(step_sizes[1:])
    max_vals = np.array([max_errors_step_half[current_phase][step_size] for step_size in step_sizes[1:]])

    fig.add_trace(go.Scatter(x=dt, y=max_vals, mode="lines+markers", name=f'{phase_names[current_phase]} max error'))

fig.update_layout(
    title=f'PLOT 2.2: Maximum Position Error vs Step Size',

    xaxis=dict(
        title='Step Size Δt (s)',
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
    xaxis_type='log',  # x-axis type is log as step sizes vary exponentially
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
fig.write_image(os.path.join(p_dir, f'Q2a_max_error_vs_step_size.png'), width=1200, height=800)



# ......................................................................................................................
#                                                    Q2b
# ......................................................................................................................

# .... PLOT 2.3: ERROR vs TIME PLOT ....

for current_phase in range(len(central_bodies_per_phase)):
    fig = go.Figure()

    for step_size in step_sizes:
        times_hours, errors = error_histories_benchmark[current_phase][step_size]
        fig.add_trace(go.Scatter(x=times_hours, y=errors, mode="lines", name=f'Δt = {step_size} s'))

    fig.update_layout(
        title=f'PLOT 2.3: Position Error in {phase_names[current_phase]} Phase',

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
    fig.write_image(os.path.join(p_dir, f'Q2b_position_error_vs_time_{current_phase}.png'), width=1200, height=800)



# .... PLOT 2.4: MAXIMUM ERROR vs STEP SIZE PLOT ....
fig = go.Figure()

for current_phase in range(len(central_bodies_per_phase)):
    dt = np.array(step_sizes)
    max_vals = np.array([max_errors_benchmark[current_phase][step_size] for step_size in step_sizes])

    fig.add_trace(go.Scatter(x = dt, y=max_vals, mode="lines+markers", name=f'{phase_names[current_phase]} max error'))

fig.update_layout(
    title = f'PLOT 2.4: Maximum Position Error vs Step Size',

    xaxis=dict(
        title='Step Size Δt (s)',
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
fig.write_image(os.path.join(p_dir, f'Q2b_max_error_vs_step_size.png'), width=1200, height=800)


# ......................................................................................................................
#                                                    Q2c
# ......................................................................................................................

fig = go.Figure()
for current_phase in range(len(central_bodies_per_phase)):
    dt = np.array(step_sizes[1:])
    errors = np.array([max_errors_step_half[current_phase][step_size] for step_size in step_sizes[1:]])
    slope = np.polyfit(np.log(dt), np.log(errors), 1)
    errors_bench = np.array([max_errors_benchmark[current_phase][step_size] for step_size in step_sizes[1:]])
    slope_bench = np.polyfit(np.log(dt), np.log(errors_bench), 1)
    y = slope[0] * np.log(dt) + slope[1]
    y_bench = slope_bench[0] * np.log(dt) + slope_bench[1]
    fig.add_trace(go.Scatter(x=dt, y=errors, mode="markers", name=f'Half-step {phase_names[current_phase]} data'))
    fig.add_trace(go.Scatter(x=dt, y=np.exp(y), mode='lines', name=f'Half-step fit  (slope={slope[0]:.2f})'))
    fig.add_trace(go.Scatter(x=dt, y=errors_bench, mode="markers", name=f'Benchmark {phase_names[current_phase]} data'))
    fig.add_trace(go.Scatter(x=dt, y=np.exp(y_bench), mode='lines', name=f'Benchmark fit  (slope={slope[0]:.2f})'))

fig.update_layout(
    title=f'PLOT 2.5: Maximum Position Error vs Step Size (with Log–Log Best-Fit Line)',

    xaxis=dict(
        title='Step Size Δt (s)',
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
    xaxis_type='log',  # x-axis type is log as step sizes vary exponentially
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
fig.write_image(os.path.join(p_dir, f'Q2c_max_pos_error_vs_step_size_w_slope.png'), width=1200, height=800)


# ......................................................................................................................
#                                                Q2d: CPU TIMES
# ......................................................................................................................
fig = go.Figure()
for current_phase in range(len(central_bodies_per_phase)):
    dt = np.array(step_sizes)
    cpu = np.array([cpu_times[current_phase][step_size] for step_size in step_sizes])
    fig.add_trace(go.Scatter(x=dt, y=cpu, mode='lines+markers', name=f'{phase_names[current_phase]} cpu time' ))


fig.update_layout(
        title=f'PLOT 2.6: CPU Time vs Step Size',

    xaxis = dict(
        title='Step Size Δt (s)',
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="white",
    ),
    yaxis = dict(
        title='CPU Time(s)',
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="white",
    ),
    xaxis_type = 'log',  # x-axis type is log as step sizes vary exponentially
    yaxis_type = 'log',
    template = "plotly_dark",
    font = dict(
        family="Times New Roman",
        size=14,
        color="white"
    ),
    title_font = dict(
        family="Times New Roman",
        size=16,
        color="white"
    )

)
fig.show()
fig.write_image(os.path.join(p_dir, f'Q2d_cpu_time_vs_step_size.png'), width=1200, height=800)

with open('q1_max_errors.json', 'r') as f:
    q1_loaded = json.load(f)

q1_max_errors_gco500= {int(k): v for k, v in q1_loaded['1'].items()}
fig = go.Figure()
dt = np.array(step_sizes)
q1_vals = np.array([q1_max_errors_gco500[s] for s in step_sizes])
q2b_vals = np.array([max_errors_benchmark[1][s] for s in step_sizes])

fig.add_trace(go.Scatter(x=dt, y=q1_vals, mode = 'lines+markers', line = dict(color = 'magenta'), name = 'Q1: Unperturbed (analytical)'))
fig.add_trace(go.Scatter(x=dt, y=q2b_vals, mode = 'lines+markers', line = dict(color = 'purple'), name = 'Q2B: Perturbed (benchmark)'))

fig.update_layout(
        title=f'PLOT 2.6: Perturbed and Unperturbed Max Position Error',

    xaxis = dict(
        title='Step Size Δt (s)',
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="white",
    ),
    yaxis = dict(
        title='Max Position Error (m)',
        showline=True,
        linecolor="white",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="white",
    ),
    xaxis_type = 'log',  # x-axis type is log as step sizes vary exponentially
    yaxis_type = 'log',
    template = "plotly_dark",
    font = dict(
        family="Times New Roman",
        size=14,
        color="white"
    ),
    title_font = dict(
        family="Times New Roman",
        size=16,
        color="white"
    )

)
fig.show()
fig.write_image(os.path.join(p_dir, 'Q2d_Q1_vs_Q2b_GCO500.png'), width=1200, height=800)

print("Q2d comparison:GCO500 truncation-dominated regime:")
print(f"{'Step size':<12} {'Q1 error (m)':<20} {'Q2b error (m)':<20} {'Ratio'}")
for s in step_sizes:
    q1 = q1_max_errors_gco500[s]
    q2b = max_errors_benchmark[1][s]
    print(f"{s:<12} {q1:<20.4e} {q2b:<20.4e} {q2b/q1:.2f}")