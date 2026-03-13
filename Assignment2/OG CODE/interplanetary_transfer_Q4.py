""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

from interplanetary_transfer_helper_functions_Q4 import *

# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"



# REQUIREMENT:
# - one arc model
# - one constant thrust vector over the full transfer





if __name__ == "__main__":

    #### 4b #### - one arc case over full transfer
    # pipeline:
    # lamberts arc --> initial state --> nominal_propagator --> sensitivity_parameter --> variational_equations

    # Create body objects
    bodies_1arc = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris_1arc = get_lambert_problem_result(bodies_1arc,
                                                            target_body,
                                                            departure_epoch,
                                                            arrival_epoch)

    initial_time_1arc = departure_epoch
    final_time_1arc = arrival_epoch

    initial_state_1arc = lambert_arc_ephemeris_1arc.cartesian_state(initial_time_1arc)

    termination_settings_1arc = propagation_setup.propagator.time_termination(final_time_1arc)

    # NOMINAL PROPAGATION  AND SENSITIVITY PARAMETER#
    nominal_propagator_settings_1arc = get_perturbed_propagator_settings(bodies_1arc,
                                                                         initial_state_1arc,
                                                                         initial_time_1arc,
                                                                         termination_settings_1arc,
                                                                         empirical_acceleration = np.zeros(3))

    sensitivity_parameters_1arc = get_sensitivity_parameter_set(nominal_propagator_settings_1arc,
                                                                bodies_1arc)


    variational_equations_1arc = numerical_simulation.create_variational_equations_solver(bodies_1arc,
                                                                                          nominal_propagator_settings_1arc,
                                                                                          sensitivity_parameters_1arc)



    nominal_state_history_1arc = variational_equations_1arc.state_history
    final_epoch_1arc = list(nominal_state_history_1arc.keys())[-1]
    x_nom_final_1arc = nominal_state_history_1arc[final_epoch_1arc]
    r_nom_final_1arc = x_nom_final_1arc[0:3]

    sensitivity_histories_1arc = variational_equations_1arc.sensitivity_matrix_history
    S_final_1arc = sensitivity_histories_1arc[final_epoch_1arc]
    S_r_1arc = S_final_1arc[0:3, 0:3]

    # final target position from lambert
    r_bar_target_1arc = lambert_arc_ephemeris_1arc.cartesian_state(arrival_epoch)[0:3]

    delta_r_1arc = r_bar_target_1arc - r_nom_final_1arc


    print("x_nom_final shape:", x_nom_final_1arc.shape)
    print("r_nom_final:", r_nom_final_1arc)
    print("r_bar_target:", r_bar_target_1arc)
    print("delta_r:", delta_r_1arc)
    print("S_final shape:", S_final_1arc.shape)
    print("S_r shape:", S_r_1arc.shape)
    print("S_r:\n", S_r_1arc)

    # PROPAGATING WITH THE THRUST VECTOR (p) #
    # computing thrust vector
    p_b = np.linalg.pinv(S_r_1arc).dot(delta_r_1arc)
    p_b = p_b.reshape(3)

    corrected_propagator_settings_1arc = get_perturbed_propagator_settings( bodies_1arc,
                                                                            initial_state_1arc,
                                                                            initial_time_1arc,
                                                                            termination_settings_1arc,
                                                                            empirical_acceleration = p_b)


    dynamics_simulator_1arc = numerical_simulation.create_dynamics_simulator(bodies_1arc,
                                                                             corrected_propagator_settings_1arc)



    state_history_corrected_1arc = dynamics_simulator_1arc.state_history

    final_epoch_corrected_1arc = list(state_history_corrected_1arc.keys())[-1]
    x_cor_final_1arc = state_history_corrected_1arc[final_epoch_corrected_1arc]
    r_cor_final_1arc = x_cor_final_1arc[0:3]

    residual_corrected_1arc = r_bar_target_1arc - r_cor_final_1arc

    row_13 = np.hstack([[final_epoch_corrected_1arc], x_cor_final_1arc])



















    #
    # # Set arc length
    # number_of_arcs = 10
    # arc_length = (arrival_epoch - departure_epoch) / number_of_arcs # by length they mean the delta t for each arc
    #


    #
    #
    #     # 4a pipeline:
    #     # predicted_x_t_f = x_t_f + S(tf) * delta_p
    #     # this can be written in terms of pos vectors rather than state
    #     # dp = p = [p_r, p_s, p_w]
