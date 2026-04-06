"""
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from integrator_analysis_helper_functions_Q2 import *
import math
current_directory = os.getcwd()

spice.load_standard_kernels()
spice.load_kernel(current_directory + "/juice_mat_crema_5_1_150lb_v01.bsp")

bodies = create_bodies()

# ......................................................................................................................
#                                               INITIAL SETUP
# ......................................................................................................................

# Define settings for orbit only, unperturbed only
current_phase = 1
central_body = "Ganymede"
bodies_to_integrate = ["JUICE"]
central_bodies = [central_body]
current_phase_start_time = initial_times_per_phase[current_phase]
acceleration_models = get_unperturbed_accelerations(central_body, bodies)

state_differences_rkf = np.zeros((6, 20))
state_differences_euler = np.zeros((6, 20))
mu_ganymede = bodies.get('Ganymede').gravitational_parameter

step_per_run = 600.0
dt_rkf =       300.0
dt_euler =      10.0
p_rkf =            7
p_euler =          1


# Perform 20 individual steps
for i in range(20):

    # Compute initial time of current step
    current_start_time = current_phase_start_time + i * step_per_run

    # Compute initial state of current step
    initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="JUICE",
        observer_body_name=central_body,
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=current_start_time,
    )

# ......................................................................................................................
#                                               RKF
# ......................................................................................................................

    # Get fixed step RKF78 integrator settings
    integrator_settings_rkf = get_fixed_step_size_integrator_settings(dt_rkf)

    # Define propagator settings, terminate after 300 s.
    termination_time_rkf = current_start_time + dt_rkf
    termination_settings_rkf = propagation_setup.propagator.time_termination(
        termination_time_rkf, terminate_exactly_on_final_condition=True
    )
    propagator_settings_rkf = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_integrate,
        initial_state,
        current_start_time,
        integrator_settings_rkf,
        termination_settings_rkf,
    )

    # Propagate Dynamics
    dynamics_simulator_rkf = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_rkf)
    kepler_difference_rkf = get_difference_wrt_kepler_orbit(dynamics_simulator_rkf.propagation_results.state_history, mu_ganymede)
    local_truncation_error_rkf = list(kepler_difference_rkf.values())[-1]
    state_differences_rkf[:,i] = local_truncation_error_rkf

# ......................................................................................................................
#                                               EULER
# ......................................................................................................................

    integrator_settings_euler =  propagation_setup.integrator.runge_kutta_fixed_step_size(10.0, propagation_setup.integrator.euler_forward)
    termination_settings_euler = propagation_setup.propagator.time_termination(current_start_time+ 10.0, terminate_exactly_on_final_condition=True)
    propagator_settings_euler = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_integrate,
        initial_state,
        current_start_time,
        integrator_settings_euler,
        termination_settings_euler,
    )
    dynamics_simulator_euler = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_euler)
    kepler_difference_euler = get_difference_wrt_kepler_orbit(dynamics_simulator_euler.propagation_results.state_history, mu_ganymede)
    local_truncation_error_euler = list(kepler_difference_euler.values())[-1]
    state_differences_euler[:,i] = local_truncation_error_euler

# ......................................................................................................................
#                                               NUMERICAL K(t)
# ......................................................................................................................

k_rkf_num = state_differences_rkf / (dt_rkf ** (p_rkf + 1))
k_euler_num = state_differences_euler / (dt_euler ** (p_euler + 1))

Kr_rkf_num_norm = np.linalg.norm(k_rkf_num[:3, :], axis = 0)
Kr_euler_num_norm = np.linalg.norm(k_euler_num[:3, :], axis = 0)

# ......................................................................................................................
#                                               ANALYTICAL K(t)
# ......................................................................................................................

initial_state_ref = spice.get_body_cartesian_state_at_epoch(
    target_body_name="JUICE",
    observer_body_name=central_body,
    reference_frame_name=global_frame_orientation,
    aberration_corrections="NONE",
    ephemeris_time=current_phase_start_time,
)

R = np.linalg.norm(initial_state_ref[:3])
n = np.sqrt(mu_ganymede / R**3)
Kr_rkf_ana = R * n**(p_rkf + 1) / math.factorial(p_rkf + 1)
Kr_euler_ana = R * n**(p_euler + 1) / math.factorial(p_euler + 1)


# ......................................................................................................................
#                                                   PLOTS
# ......................................................................................................................

# .... PLOT 5.1: Numerical vs Analytical (RKF) ....
p_dir = 'plots'
os.makedirs(p_dir, exist_ok = True)
times_hours = np.array([i * step_per_run for i in range(20)]) / 3600.0
component_colors = ['red', 'green', 'blue']

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(x=times_hours, y=Kr_rkf_num_norm, mode="lines", line=dict(color="purple"), name = "Numerical ||Kr|| RKF7(8)"), secondary_y = False)
fig.add_trace(go.Scatter(x=times_hours, y=[Kr_rkf_ana] * 20, mode='lines', line=dict(color="magenta", dash='dash'), name = "Analytical ||Kr|| RKF7(8)"), secondary_y= False)

relative_error_rkf = np.abs(Kr_rkf_num_norm - Kr_rkf_ana)/ np.abs(Kr_rkf_num_norm)
fig.add_trace(go.Scatter(x=times_hours, y=relative_error_rkf, mode="lines", line=dict(color="pink", dash ='dot'), name = "Relative Error RKF7(8)"), secondary_y= True)

for idx, comp in enumerate(['x', 'y', 'z']):
    fig.add_trace(go.Scatter(
        x=times_hours, y=k_rkf_num[idx, :], mode='lines',
        line=dict(color=component_colors[idx], dash='dot'),
        name=f'Kr_{comp} RKF7(8)'), secondary_y=False)

fig.update_layout(
    title='PLOT 5.1: RKF ||Kr(t)|| vs Time for dt=300s, GCO500',
    xaxis=dict(
        title='Time (hours)',
        showline=True, linecolor='white', linewidth=1,
        mirror=True, ticks='outside', tickcolor='white',
    ),
    template='plotly_dark',
    font=dict(family='Times New Roman', size=14, color='white'),
    title_font=dict(family='Times New Roman', size=16, color='white'),
)

fig.update_yaxes(
        title='||Kr(t)||',type = 'log',
        showline=True, linecolor='white', linewidth=1,
        mirror=True, ticks='outside', tickcolor='white', secondary_y=False,
)

fig.update_yaxes(
        title='Relative Error',type = 'log',
        showline=True, linecolor='white', linewidth=1,
        mirror=True, ticks='outside', tickcolor='white', secondary_y=True,
)
fig.show()
fig.write_image(os.path.join(p_dir, f'Q5_Kr_vs_time_rkf.png'), width=1200, height=800)



# .... PLOT 5.2: Numerical vs Analytical (EULER) ....
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(x=times_hours, y=Kr_euler_num_norm, mode="lines",line=dict(color="purple"), name = "Numerical ||Kr|| Euler"), secondary_y= False)
fig.add_trace(go.Scatter(x=times_hours, y=[Kr_euler_ana] * 20, mode='lines', line=dict(color="magenta", dash='dash'), name = "Analytical ||Kr|| Euler"), secondary_y= False)

relative_error_euler = np.abs(Kr_euler_num_norm - Kr_euler_ana)/ np.abs(Kr_euler_num_norm)
fig.add_trace(go.Scatter(x=times_hours, y=relative_error_euler, mode='lines', line=dict(color="pink", dash='dot'), name = "Relative Error Euler" ), secondary_y= True)

for idx, comp in enumerate(['x', 'y', 'z']):
    fig.add_trace(go.Scatter(
        x=times_hours, y=k_euler_num[idx, :], mode='lines',
        line=dict(color=component_colors[idx], dash='dot'),
        name=f'Kr_{comp} Euler'), secondary_y=False)

fig.update_layout(
    title='PLOT 5.2: Euler ||Kr(t)|| vs Time for dt=300s, GCO500',
    xaxis=dict(
        title='Time (hours)',
        showline=True, linecolor='white', linewidth=1,
        mirror=True, ticks='outside', tickcolor='white',
    ),
    template='plotly_dark',
    font=dict(family='Times New Roman', size=14, color='white'),
    title_font=dict(family='Times New Roman', size=16, color='white'),
)

fig.update_yaxes(
        title='||Kr(t)||', type = 'log',
        showline=True, linecolor='white', linewidth=1,
        mirror=True, ticks='outside', tickcolor='white', secondary_y=False,
)

fig.update_yaxes(
        title='Relative Error', type = 'log',
        showline=True, linecolor='white', linewidth=1,
        mirror=True, ticks='outside', tickcolor='white', secondary_y=True,
)
fig.show()
fig.write_image(os.path.join(p_dir, f'Q5_Kr_vs_time_euler.png'), width=1200, height=800)

print(f"Analytical RKF:            {Kr_rkf_ana:.4e}")
print(f"Analytical Euler:          {Kr_euler_ana:.4e}")
print(f"Mean numerical RKF:        {np.mean(Kr_rkf_num_norm):.4e}")
print(f"Mean numerical Euler:      {np.mean(Kr_euler_num_norm):.4e}")
print(f"Max relative error RKF:    {np.max(relative_error_rkf)*100:.2f} %")
print(f"Max relative error Euler:  {np.max(relative_error_euler)*100:.2f} %")
print(f"Mean relative error RKF:   {np.mean(relative_error_rkf)*100:.2f} %")
print(f"Mean relative error Euler: {np.mean(relative_error_euler)*100:.2f} %")
# zoom
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=times_hours, y=Kr_rkf_num_norm, mode="lines",line=dict(color="purple"), name = "Numerical ||Kr|| RKF"), secondary_y= True)

for idx, comp in enumerate(['x', 'y', 'z']):
    fig.add_trace(go.Scatter(
        x=times_hours, y=k_rkf_num[idx, :], mode='lines',
        line=dict(color=component_colors[idx], dash='dot'),
        name=f'Kr_{comp} RKF7(8)'), secondary_y=False)

fig.update_layout(
    title='PLOT 5.3: RKF ||Kr(t)|| vs Time for dt=300s, GCO500',
    xaxis=dict(
        title='Time (hours)',
        showline=True, linecolor='white', linewidth=1,
        mirror=True, ticks='outside', tickcolor='white',
    ),
    template='plotly_dark',
    font=dict(family='Times New Roman', size=14, color='white'),
    title_font=dict(family='Times New Roman', size=16, color='white'),
)

fig.update_yaxes(
        title='Kr(t) Components',
        showline=True, linecolor='white', linewidth=1,
        mirror=True, ticks='outside', tickcolor='white', secondary_y=False,
)

fig.update_yaxes(
        title='||Kr(t)||',type = 'log',
        showline=True, linecolor='white', linewidth=1,
        mirror=True, ticks='outside', tickcolor='white', secondary_y=True,
)
fig.show()
fig.write_image(os.path.join(p_dir, f'Q5_Kr_vs_time_rkf_comp.png'), width=1200, height=800)



# zoom
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=times_hours, y=Kr_euler_num_norm, mode="lines",line=dict(color="purple"), name = "Numerical ||Kr|| Euler"), secondary_y= True)

for idx, comp in enumerate(['x', 'y', 'z']):
    fig.add_trace(go.Scatter(
        x=times_hours, y=k_euler_num[idx, :], mode='lines',
        line=dict(color=component_colors[idx], dash='dot'),
        name=f'Kr_{comp} Euler'), secondary_y=False)

fig.update_layout(
    title='PLOT 5.4: Euler ||Kr(t)|| Components vs Time for dt=300s, GCO500',
    xaxis=dict(
        title='Time (hours)',
        showline=True, linecolor='white', linewidth=1,
        mirror=True, ticks='outside', tickcolor='white',
    ),
    template='plotly_dark',
    font=dict(family='Times New Roman', size=14, color='white'),
    title_font=dict(family='Times New Roman', size=16, color='white'),
)

fig.update_yaxes(
        title='Kr(t) Components',
        showline=True, linecolor='white', linewidth=1,
        mirror=True, ticks='outside', tickcolor='white', secondary_y=False,
)

fig.update_yaxes(
        title='||Kr(t)||',type = 'log',
        showline=True, linecolor='white', linewidth=1,
        mirror=True, ticks='outside', tickcolor='white', secondary_y=True,
)
fig.show()
fig.write_image(os.path.join(p_dir, f'Q5_Kr_vs_time_euler_comp.png'), width=1200, height=800)
