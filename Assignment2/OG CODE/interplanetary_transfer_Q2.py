"""
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""
from matplotlib.pyplot import yscale

from interplanetary_transfer_helper_functions_Q2 import *
import matplotlib.pyplot as plt

# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"


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
    departure_epoch = 2132.212895 * constants.JULIAN_DAY
    arrival_epoch = departure_epoch + 157.9635921 * constants.JULIAN_DAY
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


    earth_state_vec = spice.get_body_cartesian_state_at_epoch('Earth', 'Sun', 'ECLIPJ2000', 'NONE', departure_epoch)
    venus_state_vec = spice.get_body_cartesian_state_at_epoch('Venus', 'Sun', 'ECLIPJ2000', 'NONE', arrival_epoch)

    a_earth = element_conversion.cartesian_to_keplerian(earth_state_vec, mu_sun)
    a_earth = a_earth[0]
    a_venus = element_conversion.cartesian_to_keplerian(venus_state_vec, mu_sun)
    a_venus = a_venus[0]
    r_soi_earth = a_earth * (mass_earth / mass_sun) ** (2 / 5)
    r_soi_venus = a_venus * (mass_venus / mass_sun) ** (2 / 5)
    t_soi = departure_epoch

    while t_soi <= arrival_epoch:
        r_cart_sc = lambert_arc_ephemeris.cartesian_state(t_soi)[:3]


        r_earth = spice.get_body_cartesian_state_at_epoch('Earth', 'Sun', 'ECLIPJ2000', 'NONE', t_soi)[:3]
        r_venus = spice.get_body_cartesian_state_at_epoch('Venus', 'Sun', 'ECLIPJ2000', 'NONE', t_soi)[:3]

        if np.linalg.norm(r_cart_sc - r_earth) >= r_soi_earth and departure_epoch_ciii is None:
            departure_epoch_ciii = t_soi
            print(departure_epoch_ciii)

        if np.linalg.norm(r_cart_sc - r_venus) <= r_soi_venus and departure_epoch_ciii is not None:
            arrival_epoch_ciii = t_soi
            print(arrival_epoch_ciii)
            break
        t_soi+=fixed_step_size
        if arrival_epoch_ciii is None:
            arrival_epoch_ciii = arrival_epoch


    # case 4
    t_mid = (departure_epoch + arrival_epoch)/2
    departure_epoch_civ_fwd = t_mid
    arrival_epoch_civ_fwd = arrival_epoch - 3600
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
    dep_var_histories = {}


    for case_name, (departure_epoch_with_buffer, arrival_epoch_with_buffer) in cases.items():

        # Perform propagation
        if case_name in ["case_i", "case_ii"]:
            termination_settings = propagation_setup.propagator.time_termination(arrival_epoch_with_buffer)
        elif case_name == "case_iii":
            time_termination = propagation_setup.propagator.time_termination(arrival_epoch_with_buffer)
            soi_termination = propagation_setup.propagator.dependent_variable_termination(dependent_variable_settings =
                                propagation_setup.dependent_variable.relative_distance("Spacecraft", "Venus"),
                                limit_value = r_soi_venus, use_as_lower_limit = True)
            # false --> as terminates when the r_sc drops below the r_soi_venus values

            termination_settings = propagation_setup.propagator.hybrid_termination([soi_termination, time_termination],
                                                                                   fulfill_single_condition = True)


        dynamics_simulator = propagate_trajectory(
            departure_epoch_with_buffer,
            termination_settings,
            bodies,
            lambert_arc_ephemeris,
            use_perturbations=True,
        )

        dep_var_histories[case_name] = dynamics_simulator.propagation_results.dependent_variable_history



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

    termination_settings_iv = propagation_setup.propagator.non_sequential_termination(forward_termination,
                                                                                      backward_termination)
    dynamics_simulator_iv = propagate_trajectory(
        departure_epoch_civ_fwd, # basically tm same for both fwd n bwd cases
        termination_settings_iv,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=True,
    )

    write_propagation_results_to_file(
        dynamics_simulator_iv,
        lambert_arc_ephemeris,
        "Q2_" + "case_iv",
        output_directory,
    )
    state_histories['case_iv'] = dynamics_simulator_iv.propagation_results.state_history
    lambert_histories['case_iv'] = get_lambert_arc_history(lambert_arc_ephemeris, state_histories['case_iv'])
    dep_var_histories['case_iv'] = dynamics_simulator_iv.propagation_results.dependent_variable_history
    print('case iv done')

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
        dep_var_history = dep_var_histories[case_name]


        for t in sorted(state_history.keys()):
            x_t = state_history[t]
            x_bar_t = lambert_history[t]

            r = x_t[:3]
            v = x_t[3:6]

            r_bar = x_bar_t[:3]
            v_bar= x_bar_t[3:6]

            delta_r.append(np.linalg.norm(r - r_bar))
            delta_v.append(np.linalg.norm(v - v_bar))
            a = dep_var_history[t][30:33]
            a_bar = - mu_sun * r_bar / np.linalg.norm(r_bar)**3

            delta_a.append(np.linalg.norm(a - a_bar))

            times.append(t)

        delta_r_dict[case_name] = np.array(delta_r)
        delta_v_dict[case_name] = np.array(delta_v)
        delta_a_dict[case_name] = np.array(delta_a)
        times_dict[case_name] = np.array(times)

#.......................................................................................................................
    # SAVING :)
    sh_ci = state_histories['case_i']
    times_ci = np.array(list(sh_ci.keys()))
    states_ci = np.array(list(sh_ci.values()))
    sh_cii = state_histories['case_ii']
    times_cii = np.array(list(sh_cii.keys()))
    states_cii = np.array(list(sh_cii.values()))
    sh_ciii = state_histories['case_iii']
    times_ciii = np.array(list(sh_ciii.keys()))
    states_ciii = np.array(list(sh_ciii.values()))

    ROW_3 = np.hstack([[times_ci[0]], states_ci[0]])
    ROW_5= np.hstack([[times_cii[0]], states_cii[0]])
    ROW_7 = np.hstack([[times_ciii[0]], states_ciii[0]])

    ROW_4 = np.hstack([[times_ci[-1]], states_ci[-1]])
    ROW_6 = np.hstack([[times_cii[-1]], states_cii[-1]])
    ROW_8 = np.hstack([[times_ciii[-1]], states_ciii[-1]])
    save_data = np.vstack([ROW_3, ROW_4, ROW_5, ROW_6, ROW_7, ROW_8])
    with open('CartesianResults_AE4868_2025_2_6446426.dat', 'ab') as f:
        np.savetxt(f, save_data)


    fig,ax = plt.subplots(3,4, figsize = (15,10))
    colors = {'case_i': '#FF1493', 'case_ii': '#9B59B6', 'case_iii': '#00BFFF', 'case_iv': '#FFA500'}
    for i, case_name in enumerate(times_dict):
        time_days = (times_dict[case_name] - departure_epoch)/ 86400.0

        ax[0,i].plot(time_days, delta_r_dict[case_name], color = colors[case_name])
        ax[0,i].set_xlabel('Time (days)')
        ax[0,i].set_ylabel(r'$\Delta r$ (m)')
        ax[0,i].set_yscale('log')
        ax[1,i].plot(time_days, delta_v_dict[case_name], color = colors[case_name])
        ax[1,i].set_xlabel('Time (days)')
        ax[1,i].set_ylabel(r'$\Delta v$ (m/s)')
        ax[1, i].set_yscale('log')
        ax[2,i].plot(time_days, delta_a_dict[case_name], color = colors[case_name])
        ax[2,i].set_xlabel('Time (days)')
        ax[2,i].set_ylabel(r'$\Delta a$ (m/s$^2$)')
        ax[2, i].set_yscale('log')

    ax[0, 0].set_ylabel(r'$\Delta r$ (m)')
    ax[1, 0].set_ylabel(r'$\Delta v$ (m/s)')
    ax[2, 0].set_ylabel(r'$\Delta a$ (m/s$^2$)')
    ax[0, 0].set_title('Case I')
    ax[0, 1].set_title('Case II')
    ax[0, 2].set_title('Case III')
    ax[0, 3].set_title('Case IV')
    fig.suptitle('Deviation from Lambert arc for all propagation cases', fontsize=14)
    fig.savefig('Q2P1.png', dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))

    labels = {'case_i': 'Case I', 'case_ii': 'Case II', 'case_iii': 'Case III', 'case_iv': 'Case IV'}
    for case_name in times_dict:
        time_days = (times_dict[case_name] - departure_epoch)/ 86400.0
        ax[0].plot(time_days, delta_r_dict[case_name], color = colors[case_name], label = labels[case_name])
        ax[1].plot(time_days, delta_v_dict[case_name], color = colors[case_name], label = labels[case_name])
        ax[2].plot(time_days, delta_a_dict[case_name], color = colors[case_name], label = labels[case_name])

    ax[0].set_ylabel(r'$\Delta r$ (m)')
    ax[1].set_ylabel(r'$\Delta v$ (m/s)')
    ax[2].set_ylabel(r'$\Delta a$ (m/s)')
    for a in ax:
        a.set_xlabel('Time (days)')
        a.set_yscale('log')
        a.legend(loc = 'center right')
    fig.suptitle('Deviation from Lambert arc for all propagation cases', fontsize=14)
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.93)
    fig.savefig('Q2P2.png', dpi=300, bbox_inches='tight')
    plt.show()


    delta_r_mass = {}

    for mass in [1000, 500, 250]:
        bodies.get_body('Spacecraft').mass = mass
        dynamics_simulator_mass = propagate_trajectory(departure_epoch_civ_fwd, termination_settings_iv, bodies,
                                                       lambert_arc_ephemeris, use_perturbations= True)

        state_history_mass = dynamics_simulator_mass.propagation_results.state_history
        lambert_history_mass = get_lambert_arc_history(lambert_arc_ephemeris, state_history_mass)
        delta_r_arr = np.array([np.linalg.norm(state_history_mass[t][:3] - lambert_history_mass[t][:3]) for t in sorted(state_history_mass.keys())])
        times_mass = np.array(sorted(state_history_mass.keys()))
        delta_r_mass[mass] = (times_mass, delta_r_arr)

    colors_M = {'500': '#FF1493', '250': '#9B59B6'}
    fig,ax = plt.subplots(figsize = (10,8))
    time_1000, dr_1000 = delta_r_mass[1000]
    for mass in [500, 250]:
        t_m, dr_m = delta_r_mass[mass]
        dr_interp = np.interp(t_m, time_1000, dr_1000)
        time_days_mass = (t_m - departure_epoch)/86400.0
        ax.plot(time_days_mass, np.abs(dr_interp - dr_m), label = f'{mass:.2f} kg', color = colors_M[f'{mass}'])

    ax.set_xlabel('Time (days)')
    ax.set_ylabel(r'$||\Delta r_{mass} - \Delta r_{1000}||$ (m)')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Position difference w.r.t. case IV reference (1000 kg) for varying spacecraft mass')
    plt.tight_layout()
    fig.savefig('Q2P3.png', dpi=300, bbox_inches='tight')
    plt.show()

    # figure of merit
    max_dr_1000 = np.max(delta_r_mass[1000][1])
    max_dr_500 = np.max(delta_r_mass[500][1])
    max_dr_250 = np.max(delta_r_mass[250][1])

    L = np.abs((max_dr_1000 - max_dr_500) / (max_dr_500 - max_dr_1000))
    print(f'fig og merit L: {L:.4f}')