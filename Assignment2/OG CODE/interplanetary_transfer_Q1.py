""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

from interplanetary_transfer_helper_functions_Q1 import *
import matplotlib.pyplot as plt

# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"


if __name__ == "__main__":

    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    target_body = 'Venus'

    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    # Create propagation settings and propagate dynamics
    termination_settings = propagation_setup.propagator.time_termination(arrival_epoch)
    dynamics_simulator = propagate_trajectory(
        departure_epoch,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=False,
    )

    # Write results to file
    write_propagation_results_to_file(
        dynamics_simulator, lambert_arc_ephemeris, "Q1", output_directory
    )

    # Extract state history from dynamics simulator
    state_history = dynamics_simulator.propagation_results.state_history

    # Evaluate the Lambert arc model at each of the epochs in the state_history
    lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)
    time = np.array(list(state_history.keys()))
    time_days = time / constants. JULIAN_DAYS
    x_t = np.array(list(state_history))
    x_bar_t = np.array(list(lambert_history))

    print(x_t)
    print(x_bar_t)

#.......................................................................................................................

# Saving :)
ROW_1 = np.hstack([time_days], x_t[0] )
ROW_2 = np.hstack([time_days], x_t[-1])


#........................................................................................................................
# PLOT 1 - 3D TRAJECTORY
combined_states = {}

for i in state_history.keys():
    combined_states[i] = np.concatenate((state_history[i], lambert_history[i]))

fig, ax = plotting.trajectory_3d(  vehicles_states = combined_states,
                vehicles_names = ['Spacecraft','Lambert'],
                central_body_name = 'Sun',
                spice_bodies = ['Earth', 'Venus'],
                frame_orientation = 'J2000',
                center_plot = True,
                colors = ['blue', 'red', 'green', 'orange'],
                linestyles = ['solid','dashed','solid', 'solid']
                )
plt.show()

#.......................................................................................................................
# PLOT 2 - LAMBERT TARGETER VS NUMERICAL PROPAGATION

time = np.array(list(state_history.keys()))
time_days = [
    t / constants.JULIAN_DAY - arrival_epoch / constants.JULIAN_DAY
    for t in time
]


residual = x_bar_t - x_t
fig,ax = plt.subplots(6,1, figsize=(15,10))
for i in range(6):
    ax[i].plot(time_days, residual[i, :])

plt.show()
#.......................................................................................................................
#.......................................................................................................................
