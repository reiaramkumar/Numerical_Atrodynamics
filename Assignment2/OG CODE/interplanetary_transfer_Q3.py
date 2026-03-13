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
# RUN CODE FOR QUESTION 3 #################################################
###########################################################################

if __name__ == "__main__":

    Q3a = False
    Q3c = False
    Q3d = True
    Q3e = False


    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    ##############################################################
    # Compute number of arcs and arc length
    q3_start_time = 184557994.12800002

    q3_end_time = 197737594.12800002
    number_of_arcs = 10
    arc_length = (q3_end_time - q3_start_time) / number_of_arcs # by length they mean the delta t for each arc

    ##############################################################
    arc_state_histories_3a = {}
    arc_lambert_histories_3a = {}
    iterations_per_arc_3d = {}
    dv_total_per_arc_3d = {}
    # arc_state_histories_3d = {}
    fig,ax = plt.subplots(figsize=(15,10))
    fig3c, ax3c = plt.subplots(figsize=(15,10))
    fig3d, ax3d = plt.subplots(figsize=(15,10))

    # Compute relevant parameters (dynamics, state transition matrix, Delta V) for each arc
    for arc_index in range(number_of_arcs):

        # Compute initial and final time for arc
        current_arc_initial_time = q3_start_time + arc_length * arc_index
        current_arc_final_time = q3_start_time + arc_length * (arc_index + 1)
        # ...................................................................................................................
        # 3A UNPERTURBED ARC-WISE PROPAGATION (10 ARCS)
        if Q3a == True:

            time_termination = propagation_setup.propagator.time_termination(current_arc_final_time)

            dynamics_simulator = propagate_trajectory(current_arc_initial_time, time_termination,
                                                      bodies, lambert_arc_ephemeris, use_perturbations=True)

            state_history_3a = dynamics_simulator.propagation_results.state_history
            lambert_history_3a = get_lambert_arc_history(lambert_arc_ephemeris, state_history_3a)

            arc_state_histories_3a[arc_index] = state_history_3a
            arc_lambert_histories_3a[arc_index] = lambert_history_3a

            delta_r_3a = []
            times_3a = []
            for t in sorted(state_history_3a.keys()):
                r = state_history_3a[t][:3]
                r_bar = lambert_history_3a[t][:3]
                dr = np.linalg.norm(r - r_bar)
                delta_r_3a.append(dr)
                times_3a.append(t)
            times_3a = np.array(times_3a)
            delta_r_3a = np.array(delta_r_3a)


            time_days_3a = (times_3a - q3_start_time) / constants.JULIAN_DAY
            ax.plot(time_days_3a, delta_r_3a, label = f'Arc {arc_index}')

            # Saving :)
            ROW_9 = state_history_3a[0][0,:6]
            ROW_10 = state_history_3a[0][-1,:6]
            ROW_11 = state_history_3a[4][0,:6]
            ROW_12 = state_history_3a[4][0,:6]
            print('3a done')
        # ...................................................................................................................
        # 3C SINGLE PASS CORRECTION
        if Q3c == True:
            #### 3b n 3c ####

            # 1. OBTAINING THE CORRECTION VALUE
            # Solve for state transition matrix on current arc
            termination_settings_3c = propagation_setup.propagator.time_termination(
                current_arc_final_time
            )
            variational_equations_solver_3c = propagate_variational_equations(
                current_arc_initial_time,
                termination_settings_3c,
                bodies,
                lambert_arc_ephemeris,
            )

            state_transition_matrix_history_3c = (
                variational_equations_solver_3c.state_transition_matrix_history
            )
            state_history_3c = variational_equations_solver_3c.state_history
            lambert_history_3c = get_lambert_arc_history(lambert_arc_ephemeris, state_history_3c)

            # Get final state transition matrix (and its inverse)
            final_epoch_3c = list(state_transition_matrix_history_3c.keys())[-1]
            final_state_transition_matrix_3c = state_transition_matrix_history_3c[final_epoch_3c]

            # Retrieve final state deviation
            final_state_deviation_3c = (
                lambert_history_3c[final_epoch_3c][:3] - state_history_3c[final_epoch_3c][:3]
            )



            phi_rv = final_state_transition_matrix_3c[0:3, 3:6]
            delta_v_3c_ini = np.linalg.solve(phi_rv, final_state_deviation_3c)

            initial_state_correction_3c = np.hstack((np.zeros(3), delta_v_3c_ini))

            # 2. PROPAGATING THE CORRECTED TRAJECTORY

            dynamic_simulator_3c = propagate_trajectory(current_arc_initial_time, termination_settings_3c,
                                                        bodies, lambert_arc_ephemeris, use_perturbations=True,
                                                        initial_state_correction = initial_state_correction_3c)

            write_propagation_results_to_file(dynamic_simulator_3c, lambert_arc_ephemeris,
                                              f'Q3_C_{arc_index}', output_directory)
            state_history_3c_corrected =dynamic_simulator_3c.propagation_results.state_history
            lambert_history_3c_corrected =get_lambert_arc_history(lambert_arc_ephemeris, state_history_3c_corrected)

            delta_r_3c = []
            times_3c = []

            for t in sorted(state_history_3c_corrected.keys()):
                r = state_history_3c_corrected[t][:3]
                r_bar = lambert_history_3c_corrected[t][:3]
                dr = np.linalg.norm(r - r_bar)
                delta_r_3c.append(dr)
                times_3c.append(t)
            delta_r_3c = np.array(delta_r_3c)
            times_3c = np.array(times_3c)
            time_days_3c = (times_3c - current_arc_initial_time)/ constants.JULIAN_DAY
            ax3c.plot(time_days_3c, delta_r_3c, label = f'Arc {arc_index}')

            print('3c done')





        #### 3d ###
        if Q3d == True:
            print('step 1 done')
            tolerance = 1.0
            max_iterations = 5
            iteration_no = 0
            dv_total_3d = np.zeros(3) # initial velocity correction is zero


            termination_settings_3d = propagation_setup.propagator.time_termination(
                current_arc_final_time)
            while True:
                print('step 2 done')
                # Solve for state transition matrix on current arc
                initial_state_correction_3d = np.hstack((np.zeros(3), dv_total_3d))

                variational_equations_solver_3d = propagate_variational_equations(
                    current_arc_initial_time,
                    termination_settings_3d,
                    bodies,
                    lambert_arc_ephemeris,
                    initial_state_correction = initial_state_correction_3d,
                )

                state_transition_matrix_history_3d = (
                    variational_equations_solver_3d.state_transition_matrix_history
                )
                state_history_3d = variational_equations_solver_3d.state_history
                lambert_history_3d = get_lambert_arc_history(lambert_arc_ephemeris, state_history_3d)

                # Get final state transition matrix (and its inverse)
                final_epoch_3d = list(state_transition_matrix_history_3d.keys())[-1]
                final_state_transition_matrix_3d = state_transition_matrix_history_3d[final_epoch_3d]

                # Retrieve final state deviation
                r_3d = state_history_3d[final_epoch_3d][:3]
                r_bar_3d = lambert_history_3d[final_epoch_3d][:3]
                final_state_deviation_3d = r_3d - r_bar_3d

                if np.linalg.norm(final_state_deviation_3d) < tolerance:
                    print('arc done')
                    break


                phi_rv_3d = final_state_transition_matrix_3d[0:3, 3:6]
                delta_v_correction_3d = np.linalg.solve(phi_rv_3d, final_state_deviation_3d)

                dv_total_3d += delta_v_correction_3d

                iteration_no += 1

                if iteration_no >= max_iterations:
                    break

            iterations_per_arc_3d[arc_index] = iteration_no
            dv_total_per_arc_3d[arc_index] = dv_total_3d
            # arc_state_histories_3d[arc_index] = state_history_3d


        print("Arc   Iterations")
        for arc in sorted(iterations_per_arc_3d.keys()):
            print(f"{arc:2d}    {iterations_per_arc_3d[arc]}")

        print('3d done')

    #... 3E ...
    if Q3 == True:
        print(f'total delta_v {dv_total_per_arc_3d}')
        print('3e done')






    # 3a plot
    if Q3a == True:
        ax.set_xlabel('Time since departure(days)')
        ax.set_ylabel(r'$\Delta r(t)=||r(t)-\bar{r}(t)||$ [m]')
        ax.set_title('Arcwise propagation deviation from Lambert arc')
        ax.set_yscale('log')
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        plt.show()



    # 3c plot
    if Q3c == True:
        ax3c.set_xlabel('Time since arc start (days)')
        ax3c.set_ylabel(r'$\Delta r_{corrected}(t)=||r_{corrected}(t)-\bar{r}(t)||$ [m]')
        ax3c.set_title('Deviation from Lambert arc after correction')
        ax3c.set_yscale('log')
        ax3c.legend(ncol=2, fontsize=9)
        fig3c.tight_layout()
        plt.show()

    # 3d plot
    if Q3d == True:
        arc_idx = sorted(iterations_per_arc_3d.keys())
        iterations = [iterations_per_arc_3d[arc] for arc in arc_idx]
        ax3d.bar(arc_idx, iterations)
        ax3d.set_xlabel('Arc Index')
        ax3d.set_ylabel('Iterations to convergence')
        ax3d.set_title('Iterations to convergence per arc')
        ax3d.set_xticks(arc_idx)
        fig3d.tight_layout()
        plt.show()


            #%%
        #
        # #### 3c ####
        #
        # # Note: for question 3e, part of the code below will be put into a loop
        # # for the requested iterations
        #
        #
        # # Compute required velocity change at beginning of arc to meet required final state
        # initial_state_correction = XXXX
        #
        # # Propagate with correction to initial state (use propagate_trajectory function),
        # # and its optional initial_state_correction input
        # dynamics_simulator = XXXX
