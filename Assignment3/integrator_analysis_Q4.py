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
    'chosen': 13,

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
        ('chosen', propagation_setup.integrator.rkf_78),

    ]

    step_size_validation_settings = propagation_setup.integrator.step_size_validation(1.0e-12, np.inf)

    for label, coeff_set in variable_integrator_configs:
        results[current_phase][label] = []
        for tol in variable_tolerances:
            if label == 'chosen' and tol != 1e-10:
                continue

            if label == 'chosen' and tol == 1e-10:

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
            if label == 'chosen' and current_phase==0:
                if label not in results_all[current_phase]:
                    results_all[current_phase][label] = []
                results_all[current_phase][label].append((time_hours, position_error))
    if current_phase == 1:
        rk8_chosen_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            256.0, propagation_setup.integrator.rkf_78, propagation_setup.integrator.higher)
        rk8_propagator = propagation_setup.propagator.translational(
            central_bodies=[current_central_body],
            acceleration_models=perturbed_acceleration_models,
            bodies_to_integrate=['JUICE'],
            initial_states=initial_state,
            initial_time=current_phase_start_time,
            integrator_settings=rk8_chosen_settings,
            termination_settings=termination_condition,
        )
        rk8_simulator = numerical_simulation.create_dynamics_simulator(bodies, rk8_propagator)
        rk8_state_history = rk8_simulator.propagation_results.state_history

        rk8_benchmark_difference = get_difference_wrt_benchmarks(rk8_state_history, benchmark_interpolator)
        rk8_state_difference = np.vstack(list(rk8_benchmark_difference.values()))
        rk8_position_error = np.linalg.norm(rk8_state_difference[:, :3], axis=1)
        rk8_max_position_error_norm = np.max(rk8_position_error)

        rk8_n_steps = len(rk8_state_history) - 1  # why? as no of steps correspond to +1 states
        rk8_n_fn_evals = rk8_n_steps * no_of_fn_evals_per_step['rk8']
        rk8_epochs = np.array(list(rk8_benchmark_difference.keys()))
        rk8_times_hours = (rk8_epochs - current_phase_start_time) / 3600.0
        results_all[1]['rk8_chosen'] = [(rk8_times_hours, rk8_position_error)]

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
    'chosen':   dict(dash='dot',    color='purple'),
    }

integrator_labels = {
    'rk4':   'Fixed RK4',
    'rk6':   'Fixed RK6',
    'rk8':   'Fixed RK8',
    'rkf45': 'Variable RKF4(5)',
    'rkf56': 'Variable RKF5(6)',
    'rkf78': 'Variable RKF7(8)',
    'chosen': 'Variable RKF7(8), tol=1e-10',
}

# .... PLOT 4.1: MAXIMUM ERROR vs FUNCTION EVALUATIONS PLOT ....

for current_phase in range(len(central_bodies_per_phase)):
    fig = go.Figure()
    for label in integrator_styles:
        if label == 'chosen':
            continue
        errors = [r[0] for r in results[current_phase][label]]
        max_vals = [r[1] for r in results[current_phase][label]]
        fig.add_trace(go.Scatter(x = max_vals, y=errors, mode="lines+markers", line=integrator_styles[label], name=integrator_labels[label]))
    if current_phase == 0:
        chosen_err = results[0]['rkf78'][2][0]  # tol=1e-10 is index 2
        chosen_eval = results[0]['rkf78'][2][1]
        chosen_name = 'Variable RKF7(8) tol=1e-10 [CHOSEN]'
    else:
        chosen_err = results[1]['rk8'][5][0]  # dt=256s is index 5
        chosen_eval = results[1]['rk8'][5][1]
        chosen_name = 'Fixed RK8 dt=256s [CHOSEN]'

    fig.add_trace(go.Scatter(
        x=[chosen_eval], y=[chosen_err], mode='markers',
        marker=dict(symbol='star', size=16, color='gold'),
        name=chosen_name
    ))


    fig.add_hline(y=1.0, line_dash='dash', line_color='pink')
    fig.add_annotation(
        x=1,
        y=1.0,
        xref="paper",
        yref="y",
        text="Threshold line = 1 m",
        showarrow=False,
        font=dict(color="pink"),
        xanchor="right",
        yanchor="bottom"
    )
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

flyby_times = results_all[0]['chosen'][0][0]
rk8_times = results_all[1]['rk8_chosen'][0][0]
flyby_errors = results_all[0]['chosen'][0][1]
rk8_errors = results_all[1]['rk8_chosen'][0][1]
fig.add_trace(go.Scatter(x=flyby_times, y=flyby_errors, mode="lines",
                          line=dict(color='pink'),
                          name='Callisto Flyby:  Variable RKF7(8), tol = 1e-10'))
fig.add_trace(go.Scatter(x=rk8_times, y=rk8_errors, mode="lines",
                          line=dict(color='purple'),
                          name='GCO500 Orbit:  Fixed RK8, dt = 256s'))

fig.add_hline(y=1.0, line_dash='dash', line_color='pink')
fig.add_annotation(
    x=1,  # right side of plot (in paper coords)
    y=1.0,
    xref="paper",
    yref="y",
    text="Threshold line = 1 m",
    showarrow=False,
    font=dict(color="pink"),
    xanchor="right",
    yanchor="bottom"
)

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

print(f"\n{'*' * 85}")
print(f"{'INTEGRATOR PERFORMANCE SUMMARY':^85}")
print(f"{'*' * 85}")

for current_phase in range(2):
    print(f"\n{phase_names[current_phase].upper()}")
    print(f"{'.' * 85}")
    print(f"{'Integrator':<30} {'Setting':<15} {'Max Error (m)':<20} {'Fn Evals':<15} {'< 1m?'}")
    print(f"{'.' * 85}")

    # Fixed step integrators
    for label, dt_list in [('rk4', fixed_time_steps), ('rk6', fixed_time_steps), ('rk8', fixed_time_steps)]:
        for i, dt in enumerate(dt_list):
            err, evals = results[current_phase][label][i]
            print(f"{integrator_labels[label]:<30} {'dt=' + str(dt) + 's':<15} {err:<20.4e} {evals:<15} {'YES' if err < 1.0 else 'NO'}")

    print(f"{'-' * 85}")

    # Variable step integrators
    for label in ['rkf45', 'rkf56', 'rkf78']:
        for i, tol in enumerate(variable_tolerances):
            err, evals = results[current_phase][label][i]
            print(f"{integrator_labels[label]:<30} {'tol=' + str(tol):<15} {err:<20.4e} {evals:<15} {'YES' if err < 1.0 else 'NO'}")

    print(f"{'-' * 85}")

    # Chosen integrator
    if current_phase == 0:
        err, evals = results[0]['chosen'][0]
        print(
            f"{'Variable RKF7(8) CHOSEN':<30} {'tol=1e-10':<15} {err:<20.4e} {evals:<15} {'YES' if err < 1.0 else 'NO'}")
    else:
        print(
            f"{'Fixed RK8 CHOSEN':<30} {'dt=256s':<15} {rk8_max_position_error_norm:<20.4e} {rk8_n_fn_evals:<15} {'YES' if rk8_max_position_error_norm < 1.0 else 'NO'}")