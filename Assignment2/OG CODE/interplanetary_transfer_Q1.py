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
    x_t = np.array(list(state_history.values()))
    x_bar_t = np.array(list(lambert_history.values()))

    print(x_t)
    print(x_bar_t)

#.......................................................................................................................

    # Saving :)
    ROW_1 = np.hstack([[time[0]], x_t[0]] )
    ROW_2 = np.hstack([[time[-1]], x_t[-1]])
    save_data = np.vstack([ROW_1, ROW_2])
    np.savetxt('CartesianResults_AE4868_2025_2_6446426.dat', save_data)
    #........................................................................................................................
    # PLOT 1 - 3D TRAJECTORY
    combined_states = {}

    for i in state_history.keys():
        combined_states[i] = np.concatenate((state_history[i], lambert_history[i]))

    fig, ax = plotting.trajectory_3d(  vehicles_states = combined_states,
                    vehicles_names = ['Spacecraft','Lambert'],
                    central_body_name = 'Sun',
                    spice_bodies = ['Earth', 'Venus'],
                    frame_orientation = 'ECLIPJ2000',
                    center_plot = True,
                    colors = ['#FF1493', '#9B59B6', '#00BFFF', '#FFA500'],
                    linestyles = ['solid','dashed','solid', 'solid']
                    )
    fig.set_size_inches(10, 8)

    ax3d = fig.axes[0]
    start = x_t[0]
    end = x_t[-1]
    ax3d.scatter(start[0], start[1], start[2], color='green', s=50, zorder=5, label='Departure')
    ax3d.scatter(end[0], end[1], end[2], color='red', s=50, zorder=5, label='Arrival')
    ax3d.legend()
    ax3d.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title('3D Trajectory: Earth to Venus Transfer', y = 1.05)
    plt.subplots_adjust(right=0.85)
    fig.savefig('Q1P1.png', dpi=300, bbox_inches='tight')
    plt.show()

    #.......................................................................................................................
    # PLOT 2 - LAMBERT TARGETER VS NUMERICAL PROPAGATION

    time_days = (time - departure_epoch) / constants.JULIAN_DAY

    residual = x_bar_t - x_t
    labels = ['Δx (m)', 'Δy (m)', 'Δz (m)']
    label = [r'$\Delta x = \bar{x}(t) - x(t)$ (m)', r'$\Delta y = \bar{y}(t) - y(t)$ (m)', r'$\Delta z = \bar{z}(t) - z(t)$ (m)']
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    for i in range(3):
        ax[i].plot(time_days, residual[:, i], color = '#FF1493', label = label[i])
        ax[i].set_ylabel(labels[i])
        ax[i].set_xlabel("Time (Days)")
        ax[i].legend(loc='upper right')
        ax[i].set_title(f'{labels[i]} Residual Plot')
    fig.suptitle('Position residuals between Lambert arc and numerical propagation')
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.93, hspace=0.4)
    fig.savefig('Q1P2.png', dpi=300, bbox_inches='tight')
    plt.show()
    #.......................................................................................................................
    #.......................................................................................................................
