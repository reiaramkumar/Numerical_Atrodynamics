import os
import numpy as np
import plotly.graph_objects as go
from integrator_analysis_helper_functions_Q2 import *

current_directory = os.getcwd()

spice.load_standard_kernels()
spice.load_kernel(current_directory + "/juice_mat_crema_5_1_150lb_v01.bsp")

bodies = create_bodies()

# ......................................................................................................................
#                                               INITIAL SETUP
# ......................................................................................................................

fixed_time_steps = [2 ** n for n in range(3, 10)]
variable_tolerances = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]

no_of_fn_evals_per_step = {
    'rk4':             4,
    'rk6':             8,
    'rk8':            13,
    'rkf45':           6,
    'rkf56':           8,
    'rkf78':          13,
    'rkf78 - chosen': 13
}

phase_names = {
    0: 'Callisto Flyby',
    1: 'GCO500 Orbit'
}

p_dir = 'plots'
os.makedirs(p_dir, exist_ok=True)

results =       {0: {}, 1: {}}
results_all =   {0: {}, 1: {}}


for current_phase in range(len(central_bodies_per_phase)):

    current_phase_start_time = initial_times_per_phase[current_phase]
    current_phase_end_time = current_phase_start_time + propagation_times_per_phase[current_phase]

    current_central_body = central_bodies_per_phase[current_phase]

    initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="JUICE",
        observer_body_name=current_central_body,
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=current_phase_start_time,
    )

    termination_condition = propagation_setup.propagator.time_termination(
        current_phase_end_time, terminate_exactly_on_final_condition=True
    )

    perturbed_acceleration_models = get_perturbed_accelerations(
        current_central_body, bodies
    )

# ......................................................................................................................
#                                               BENCHMARK ROUTINE
# ......................................................................................................................

    benchmark_step_size = 10.0 if current_central_body == 'Callisto' else 20.0
    benchmark_integrator_settings = get_fixed_step_size_integrator_settings(benchmark_step_size)

    perturbed_propagator_settings = propagation_setup.propagator.translational(
        central_bodies=[current_central_body],
        acceleration_models=perturbed_acceleration_models,         bodies_to_integrate=['JUICE'],
        initial_states=initial_state,
        initial_time=current_phase_start_time,
        integrator_settings=benchmark_integrator_settings,  #
        termination_settings=termination_condition,
    )

    benchmark_dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, perturbed_propagator_settings)

    interpolator_settings = interpolators.lagrange_interpolation(8)
    benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        benchmark_dynamics_simulator.propagation_results.state_history,
        interpolator_settings,
    )

# ......................................................................................................................
#                                       FIXED STEP INTEGRATOR ROUTINE
# ......................................................................................................................

    fixed_integrator_configs = [
        ('rk4', propagation_setup.integrator.rkf_45, propagation_setup.integrator.lower),
        ('rk6', propagation_setup.integrator.rkf_56, propagation_setup.integrator.higher),
        ('rk8', propagation_setup.integrator.rkf_78, propagation_setup.integrator.higher),
    ]

    for label, coeff_set, order in fixed_integrator_configs:
        results[current_phase][label] = []
        for dt in fixed_time_steps:
            integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(dt, coeff_set, order)

            propagator_settings = propagation_setup.propagator.translational(central_bodies=[current_central_body],
                                                                             acceleration_models=perturbed_acceleration_models,
                                                                             bodies_to_integrate=['JUICE'],
                                                                             initial_states=initial_state,
                                                                             initial_time=current_phase_start_time,
                                                                             integrator_settings=integrator_settings,
                                                                             termination_settings= termination_condition)


# ......................................................................................................................
#                                               BENCHMARK ERRORS
# ......................................................................................................................

            simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
            state_history = simulator.propagation_results.state_history

            benchmark_difference = get_difference_wrt_benchmarks(state_history, benchmark_interpolator)
            state_difference = np.vstack(list(benchmark_difference.values()))
            position_error = np.linalg.norm(state_difference[:, :3], axis=1)
            max_position_error_norm = np.max(position_error)

            n_steps = len(state_history) - 1  # why? as no of steps correspond to +1 states
            n_fn_evals = n_steps * no_of_fn_evals_per_step[label]


            results[current_phase][label].append((max_position_error_norm, n_fn_evals))

# ......................................................................................................................
#                                       VARIABLE STEP INTEGRATOR ROUTINE
# ......................................................................................................................

    variable_integrator_configs = [
        ('rkf45', propagation_setup.integrator.rkf_45),
        ('rkf56', propagation_setup.integrator.rkf_56),
        ('rkf78', propagation_setup.integrator.rkf_78),
        ('rkf78 - chosen', propagation_setup.integrator.rkf_78),

    ]

    step_size_validation_settings = propagation_setup.integrator.step_size_validation(1.0e-12, np.inf)

    for label, coeff_set in variable_integrator_configs:
        results[current_phase][label] = []
        for tol in variable_tolerances:
            if label == 'rkf78 - chosen' and tol != 1e-10:
                continue

            if label == 'rkf78 - chosen' and tol == 1e-10:

                step_size_control_settings = propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(
                    1e-10, 1e-10)
            else:
                step_size_control_settings = propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(
                    tol, tol)

            integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(10.0,
                                                                                         coeff_set,
                                                                                         step_size_control_settings,
                                                                                         step_size_validation_settings)


            propagator_settings = propagation_setup.propagator.translational(
                central_bodies=[current_central_body],
                acceleration_models=perturbed_acceleration_models,
                bodies_to_integrate=['JUICE'],
                initial_states=initial_state,
                initial_time=current_phase_start_time,
                integrator_settings=integrator_settings,
                termination_settings=termination_condition,
            )

# ......................................................................................................................
#                                               BENCHMARK ERRORS
# ......................................................................................................................

            simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
            state_history = simulator.propagation_results.state_history

            benchmark_difference = get_difference_wrt_benchmarks(state_history, benchmark_interpolator)
            state_difference = np.vstack(list(benchmark_difference.values()))
            position_error = np.linalg.norm(state_difference[:, :3], axis=1)
            max_position_error_norm = np.max(position_error)

            n_steps = len(state_history) - 1
            n_fn_evals = n_steps * no_of_fn_evals_per_step[label]

            epochs = np.array(list(benchmark_difference.keys()))
            time_hours = (epochs - current_phase_start_time) / 3600.0

            results[current_phase][label].append((max_position_error_norm, n_fn_evals))
            if label == 'rkf78 - chosen':
                if label not in results_all[current_phase]:
                    results_all[current_phase][label] = []
                results_all[current_phase][label].append((time_hours, position_error))


# ......................................................................................................................
#                                               PLOTS
# ......................................................................................................................

integrator_styles = {
    'rk4':              dict(dash='dot',    color='red'),
    'rk6':              dict(dash='dot',    color='green'),
    'rk8':              dict(dash='dot',    color='blue'),
    'rkf45':            dict(dash='solid',  color='red'),
    'rkf56':            dict(dash='solid',  color='green'),
    'rkf78':            dict(dash='solid',  color='blue'),
    'rkf78 - chosen':   dict(dash='dot',    color='purple'),
    }

integrator_labels = {
    'rk4':   'Fixed RK4',
    'rk6':   'Fixed RK6',
    'rk8':   'Fixed RK8',
    'rkf45': 'Variable RKF4(5)',
    'rkf56': 'Variable RKF5(6)',
    'rkf78': 'Variable RKF7(8)',
    'rkf78 - chosen': 'Variable RKF7(8), tol=1e-10',
}

# .... PLOT 4.1: MAXIMUM ERROR vs FUNCTION EVALUATIONS PLOT ....

for current_phase in range(len(central_bodies_per_phase)):
    fig = go.Figure()
    for label in integrator_styles:
        errors = [r[0] for  r in results[current_phase][label]]
        max_vals = [r[1] for r in results[current_phase][label]]
        if label == 'rkf78 - chosen':
            fig.add_trace(go.Scatter(x = max_vals, y=errors, mode="lines+markers", marker=dict(symbol='star', size=14, color='gold'),
                                     line=integrator_styles[label], name=integrator_labels[label]))
        else:
            fig.add_trace(go.Scatter(x = max_vals, y=errors, mode="lines+markers", line=integrator_styles[label], name=integrator_labels[label]))

    fig.update_layout(
    title = f'PLOT 4.1: Maximum Position Error vs Function Evaluations in {phase_names[current_phase]}',

    xaxis=dict(
        title='Number of Function Evaluations',
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
    xaxis_type = 'log',
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
    fig.write_image(os.path.join(p_dir, f'Q4_max_pos_error_vs_fn_evals_{current_phase}.png'), width=1200, height=800)

if current_phase == 1:
    fig = go.Figure()
    for label in integrator_styles:
        if label == 'rkf45' or label == 'rkf56' or label == 'rkf78':
            errors = [r[0] for  r in results[current_phase][label]]
            max_vals = [r[1] for r in results[current_phase][label]]
            fig.add_trace(go.Scatter(x = max_vals, y=errors, mode="lines+markers", line=integrator_styles[label], name=integrator_labels[label]))

    fig.update_layout(
    title = f'PLOT 4.1: Maximum Position Error vs Function Evaluations in {phase_names[current_phase]}',

    xaxis=dict(
        title='Number of Function Evaluations',
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
    xaxis_type = 'log',
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
    fig.write_image(os.path.join(p_dir, f'Q4_max_pos_error_vs_fn_evals_{current_phase}_zoom.png'), width=1200, height=800)


# .... PLOT 4.2: ....
fig = go.Figure()
for current_phase in range(len(central_bodies_per_phase)):
    for label in integrator_styles:
        if label == 'rkf78 - chosen':
            times = results_all[current_phase]['rkf78 - chosen'][0][0]
            errors = results_all[current_phase]['rkf78 - chosen'][0][1]
            fig.add_trace(go.Scatter(x = times, y=errors, mode="lines+markers", marker=dict(symbol='star', size=12, color='gold'),
                                     line=integrator_styles[label], name=f'{phase_names[current_phase]}, Variable {integrator_labels[label]}, tol = 1e-10'))

fig.update_layout(
title = f'PLOT 4.2: Position Error vs Function Evaluations',

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
fig.write_image(os.path.join(p_dir, f'Q4b_pos_error_vs_time.png'), width=1200, height=800)
