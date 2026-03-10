""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

from interplanetary_transfer_helper_functions_Q2 import *
import matplotlib.pyplot as plt

# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 2 #################################################
###########################################################################

if __name__ == "__main__":

    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )


    """
    case_i: The initial and final propagation time equal to the initial and final times of the Lambert arc.
    case_ii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=1 hour.
    case_iii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t such that we start/end on the sphere of influence
    case_iv: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=1 hour. The propagation is started from the middle point in time of the Lambert arc and propagated forward and backward in time.

    """

    # case 1
    departure_epoch_ci = departure_epoch
    arrival_epoch_ci = arrival_epoch

    # case 2
    departure_epoch_cii = departure_epoch + 3600
    arrival_epoch_cii = arrival_epoch - 3600

    # case 3
    mu_sun = bodies.get_body('Sun').gravitational_parameter

    # first we need to fig out the soi for earth and venus
    mass_earth = bodies.get_body('Earth').gravitational_parameter / constants.GRAVITATIONAL_CONSTANT
    mass_venus = bodies.get_body('Venus').gravitational_parameter / constants.GRAVITATIONAL_CONSTANT
    mass_sun = bodies.get_body('Sun').gravitational_parameter / constants.GRAVITATIONAL_CONSTANT

    departure_epoch_ciii = None
    arrival_epoch_ciii = None
    t = departure_epoch
    while t <= arrival_epoch:
        r_cart_state_sc = lambert_arc_ephemeris.cartesian_state(t)
        r_cart_sc = r_cart_state_sc[:3]

        r_earth = spice.get_body_cartesian_state_at_epoch('Earth', 'Sun', 'J2000', 'NONE', t)
        r_venus = spice.get_body_cartesian_state_at_epoch('Venus', 'Sun', 'J2000', 'NONE', t)

        a_earth = element_conversion.cartesian_to_keplerian(r_earth, mu_sun)
        a_earth = a_earth[0]
        a_venus = element_conversion.cartesian_to_keplerian(r_venus, mu_sun)
        a_venus = a_venus[0]

        r_earth = r_earth[0:3]
        r_venus = r_venus[0:3]

        r_soi_earth = a_earth * (mass_earth / mass_sun) ** (2 / 5)
        r_soi_venus = a_venus * (mass_venus / mass_sun) ** (2 / 5)

        if np.linalg.norm(r_cart_sc - r_earth) >= r_soi_earth and departure_epoch_ciii is None:
            departure_epoch_ciii = t

        if np.linalg.norm(r_cart_sc - r_venus) <= r_soi_venus and departure_epoch_ciii is not None:
            arrival_epoch_ciii = t
            break
        t+=fixed_step_size


    # case 4
    t_mid = (departure_epoch + arrival_epoch)/2
    departure_epoch_civ_fwd = t_mid
    arrival_epoch_civ_fwd = arrival_epoch_cii - 3600
    departure_epoch_civ_bwd = t_mid
    arrival_epoch_civ_bwd  = departure_epoch + 3600



    # List cases to iterate over. STUDENT NOTE: feel free to modify if you see fit
    cases = {
        "case_i": (departure_epoch_ci, arrival_epoch_ci),
        "case_ii": (departure_epoch_cii, arrival_epoch_cii),
        "case_iii": (departure_epoch_ciii, arrival_epoch_ciii),
    }
    state_histories = {}
    lambert_histories = {}

    for case_name, (departure_epoch_with_buffer, arrival_epoch_with_buffer) in cases.items():

        # Perform propagation
        if case_name in ["case_i", "case_ii"]:
            termination_settings = propagation_setup.propagator.time_termination(arrival_epoch_with_buffer)
        elif case_name == "case_iii":
            time_termination = propagation_setup.propagator.time_termination(arrival_epoch_with_buffer)
            soi_termination = propagation_setup.propagator.dependent_variable_termination(dependent_variable_settings =
                                propagation_setup.dependent_variable.relative_distance("Spacecraft", "Venus"), limit_value = r_soi_venus, use_as_lower_limit = True)
            termination_settings = propagation_setup.propagator.hybrid_termination([soi_termination, time_termination], fulfill_single_condition = True)


        dynamics_simulator = propagate_trajectory(
            departure_epoch_with_buffer,
            termination_settings,
            bodies,
            lambert_arc_ephemeris,
            use_perturbations=True,
        )



        write_propagation_results_to_file(
            dynamics_simulator,
            lambert_arc_ephemeris,
            "Q2_" + case_name,
            output_directory,
        )


        state_histories[case_name] = dynamics_simulator.propagation_results.state_history
        lambert_histories[case_name] = get_lambert_arc_history(lambert_arc_ephemeris, state_histories[case_name])
        print(f'{case_name} done')

    # case 4
    forward_termination = propagation_setup.propagator.time_termination(arrival_epoch_civ_fwd)
    backward_termination = propagation_setup.propagator.time_termination(arrival_epoch_civ_bwd)

    termination_settings_iv = propagation_setup.propagator.hybrid_termination([forward_termination, backward_termination],
                                                                              fulfill_single_condition = True)
    dynamics_simulator = propagate_trajectory(
        departure_epoch_civ_fwd, # basically tm same for both fwd n bwd cases
        termination_settings_iv,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=True,
    )

    write_propagation_results_to_file(
        dynamics_simulator,
        lambert_arc_ephemeris,
        "Q2_" + "case_iv",
        output_directory,
    )
    state_histories['case_iv'] = dynamics_simulator.propagation_results.state_history
    lambert_histories['case_iv'] = get_lambert_arc_history(lambert_arc_ephemeris, state_histories[case_name])

    times_dict = {}
    delta_r_dict = {}
    delta_v_dict = {}
    delta_a_dict = {}

    for case_name in state_histories:
        state_history = state_histories[case_name]
        lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

        times = []
        delta_r = []
        delta_v = []
        delta_a = []

        for t in sorted(state_history.keys()):
            x_t = state_history[t]
            x_bar_t = lambert_history[t]

            r = x_t[:3]
            v = x_t[3:6]

            r_bar = x_bar_t[:3]
            v_bar= x_bar_t[3:6]

            delta_r.append(np.linalg.norm(r - r_bar))
            delta_v.append(np.linalg.norm(v - v_bar))

            a = - mu_sun * r / np.linalg.norm(r)**3
            a_bar = - mu_sun * r_bar / np.linalg.norm(r_bar)**3

            delta_a.append(np.linalg.norm(a - a_bar))

            times.append(t)

        delta_r_dict[case_name] = np.array(delta_r)
        delta_v_dict[case_name] = np.array(delta_v)
        delta_a_dict[case_name] = np.array(delta_a)
        times_dict[case_name] = np.array(times)

    # %%


    fig,ax = plt.subplots(3,4, figsize = (14,8))

    for i, case_name in enumerate(times_dict):
        time_days = times_dict[case_name] - times_dict[case_name][0]/86400

        ax[0,i].plot(time_days, delta_r_dict[case_name])
        ax[0,i].set_xlabel('Time (days)')
        ax[0,i].set_ylabel(r'$\Delta r$ (m)')
        ax[1,i].plot(time_days, delta_v_dict[case_name])
        ax[1,i].set_xlabel('Time (days)')
        ax[1,i].set_ylabel(r'$\Delta v$ (m/s)')
        ax[2,i].plot(time_days, delta_a_dict[case_name])
        ax[2,i].set_xlabel('Time (days)')
        ax[2,i].set_ylabel(r'$\Delta a$ (m/s$^2$)')

    ax[0, 0].set_ylabel(r'$\Delta r$ (m)')
    ax[1, 0].set_ylabel(r'$\Delta v$ (m/s)')
    ax[2, 0].set_ylabel(r'$\Delta a$ (m/s$^2$)')
    ax[0, 0].set_title('Case 1')
    ax[0, 1].set_title('Case 2')
    ax[0, 2].set_title('Case 3')
    ax[0, 3].set_title('Case 4')
    fig.suptitle('Deviation from Lambert Trajectory', fontsize=14)
    plt.tight_layout()
    plt.show()
#%%



