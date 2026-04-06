
import os
from integrator_analysis_helper_functions_Q1 import *
import plotly.graph_objects as go
import json
current_directory = os.getcwd()

# Load spice kernels.
spice.load_standard_kernels()
spice.load_kernel(current_directory + "/juice_mat_crema_5_1_150lb_v01.bsp")

# Create the bodies for the numerical simulation
bodies = create_bodies()

# Define list of step size for integrator to take
step_sizes = [2**n for n in range(3,10)]


# histories stored as {phase (0/1), {step size, (times, errors)}}
# here phase --> 0: flyby phase     --> 1: orbit phase
error_histories = {0: {}, 1: {}}
max_errors = {0: {}, 1: {}}




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

    # Define current centra
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
    acceleration_models = get_unperturbed_accelerations(current_central_body, bodies)


    # Iterate over step size
    for step_size in step_sizes:

        # Define integrator settings
        # integrator_settings = get_fixed_step_size_integrator_settings(current_phase_start_time, step_size)
        integrator_settings = get_fixed_step_size_integrator_settings(step_size)
        # Define propagator settings
        propagator_settings = propagation_setup.propagator.translational(central_bodies = [current_central_body],
                                                                       acceleration_models = acceleration_models,
                                                                       bodies_to_integrate = ['JUICE'],
                                                                       initial_states = initial_state,
                                                                       initial_time = current_phase_start_time,
                                                                       integrator_settings = integrator_settings,
                                                                       termination_settings = termination_settings)



        # Propagate dynamics
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
        state_history = dynamics_simulator.propagation_results.state_history

        # Compute difference w.r.t. analytical solution to file
        central_body_gravitational_parameter = bodies.get_body(
            current_central_body
        ).gravitational_parameter
        keplerian_solution_difference = get_difference_wrt_kepler_orbit(
            state_history, central_body_gravitational_parameter
        )

        # keplerian_solution_difference stores the epoch value n the state differrence which we extract here
        epochs = np.array(list(keplerian_solution_difference.keys()))
        times_hours = (epochs - current_phase_start_time) / 3600.0

        state_difference = np.vstack(list(keplerian_solution_difference.values()))
        position_error_norm = np.linalg.norm(state_difference[:, :3], axis = 1)
        error_histories[current_phase][step_size] = (times_hours, position_error_norm)
        max_errors[current_phase][step_size] = np.max(position_error_norm)


        # Write results to files
        file_output_identifier = (
            "Q1_step_size_" + str(step_size) + "_phase_index" + str(current_phase)
        )
        write_propagation_results_and_analytical_difference_to_file(
            dynamics_simulator,
            file_output_identifier,
            bodies.get_body(current_central_body).gravitational_parameter,
        )


# .................... PLOTS ......................
p_dir = 'plots'
os.makedirs(p_dir, exist_ok = True)

phase_names = {
    0: 'Callisto Flyby',
    1: 'GCO500 Orbit'
}

# .... PLOT 1.1: ERROR vs TIME PLOT ....

for current_phase in range(len(central_bodies_per_phase)):
    fig = go.Figure()

    for step_size in step_sizes:
        print(error_histories)
        times_hours, errors = error_histories[current_phase][step_size]
        fig.add_trace(go.Scatter(x=times_hours, y=errors, mode="lines", name=f'Δt = {step_size} s'))

    fig.update_layout(
        title = f'PLOT 1.1: Position Error in {phase_names[current_phase]} Phase',

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
        yaxis_type = 'log',
        template = "plotly_dark",
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
    fig.write_image(os.path.join(p_dir ,f'Q1_position_error_vs_time_{current_phase}.png'), width=1200, height=800)


# .... PLOT 1.2: MAXIMUM ERROR vs STEP SIZE PLOT ....
fig = go.Figure()

for current_phase in range(len(central_bodies_per_phase)):
    dt = np.array(step_sizes)
    max_vals = np.array([max_errors[current_phase][step_size] for step_size in step_sizes])

    fig.add_trace(go.Scatter(x = dt, y=max_vals, mode="lines+markers", name=f'{phase_names[current_phase]} max error'))

fig.update_layout(
    title = f'PLOT 1.2: Maximum Position Error vs Step Size',

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
fig.write_image(os.path.join(p_dir, f'Q1_max_error_vs_step_size.png'), width=1200, height=800)



# Save Q1 max errors to file
q1_max_errors_save = {
    str(phase): {str(step): float(max_errors[phase][step]) for step in step_sizes}
    for phase in range(2)
}
with open('q1_max_errors.json', 'w') as f:
    json.dump(q1_max_errors_save, f)
print("Q1 max errors saved to q1_max_errors.json")

