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

    Q3a = True
    Q3c = True
    Q3d = True
    Q3e = True


    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    ##############################################################
    # Compute number of arcs and arc length
    q3_start_time = 184557994.12800002  # depature_epoch_ciii
    q3_end_time = 197737594.12800002    # arrival_epoch_ciii
    number_of_arcs = 10
    arc_length = (q3_end_time - q3_start_time) / number_of_arcs # by length they mean the delta t for each arc

    ##############################################################
    arc_state_histories_3a = {}
    arc_lambert_histories_3a = {}
    iterations_per_arc_3d = {}
    dv_total_per_arc_3d = {}
    # arc_state_histories_3d = {}


    Q3a_data = {}
    Q3c_data = {}
    Q3d_data = {}


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
            Q3a_data[arc_index] = (time_days_3a, delta_r_3a)


            arc_idx = sorted(iterations_per_arc_3d.keys())
            times_3a_list = sorted(state_history_3a.keys())
            if arc_index == 0:
                ROW_9 = np.hstack([[times_3a_list[0]], state_history_3a[times_3a_list[0]]])
                ROW_10 = np.hstack([[times_3a_list[-1]], state_history_3a[times_3a_list[-1]]])
            if arc_index == 4:
                ROW_11 = np.hstack([[times_3a_list[0]], state_history_3a[times_3a_list[0]]])
                ROW_12 = np.hstack([[times_3a_list[-1]], state_history_3a[times_3a_list[-1]]])

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
            Q3c_data[arc_index] = (time_days_3c, delta_r_3c)

            print('3c done')





        #### 3d ###
        if Q3d == True:

            tolerance = 1.0
            max_iterations = 5
            iteration_no = 0
            dv_total_3d = np.zeros(3) # initial velocity correction is zero


            termination_settings_3d = propagation_setup.propagator.time_termination(
                current_arc_final_time)
            while True:
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
                final_state_deviation_3d =  r_bar_3d - r_3d

                if np.linalg.norm(final_state_deviation_3d) < tolerance:
                    print(f'arc {arc_index} done')
                    break


                phi_rv_3d = final_state_transition_matrix_3d[0:3, 3:6]
                delta_v_correction_3d = np.linalg.solve(phi_rv_3d, final_state_deviation_3d)

                dv_total_3d += delta_v_correction_3d

                iteration_no += 1

                if iteration_no >= max_iterations:
                    break

            iterations_per_arc_3d[arc_index] = iteration_no
            dv_total_per_arc_3d[arc_index] = dv_total_3d
            final_correction_3d = np.hstack((np.zeros(3), dv_total_3d))
            dynamics_simulator_3d_re = propagate_trajectory(current_arc_initial_time, termination_settings_3d,
                                                           bodies, lambert_arc_ephemeris, use_perturbations=True,
                                                           initial_state_correction=final_correction_3d)
            state_history_3d_re = dynamics_simulator_3d_re.propagation_results.state_history
            lambert_history_3d_re = get_lambert_arc_history(lambert_arc_ephemeris,state_history_3d_re)

            delta_r_3d_re = []
            times_3d_re = []

            for t in sorted(state_history_3d_re):
                r = state_history_3d_re[t][:3]
                r_bar = lambert_history_3d_re[t][:3]
                dr = np.linalg.norm(r - r_bar)
                delta_r_3d_re.append(dr)
                times_3d_re.append(t)

            delta_r_3d_re = np.array(delta_r_3d_re)
            times_3d_re = np.array(times_3d_re)
            times_3d_re_days = (times_3d_re - current_arc_initial_time)/ constants.JULIAN_DAY
            Q3d_data[arc_index] = (times_3d_re_days, delta_r_3d_re)

            # arc_state_histories_3d[arc_index] = state_history_3d

            print('3d done')

    #... 3E ...
    if Q3e == True:
        total_delta_v = sum(np.linalg.norm(dv) for dv in dv_total_per_arc_3d.values())
        print(f'Total Delta V: {total_delta_v:.4f} m/s')
        print('3e done')

    print("Arc   Iterations")
    for arc in sorted(iterations_per_arc_3d.keys()):
        print(f"{arc:2d}    {iterations_per_arc_3d[arc]}")

    colors = {0: '#FF0000', 1: '#FF7F00', 2: '#FFD700', 3: '#00CC00', 4: '#00BFFF', 5: '#0000FF', 6: '#8B00FF',
              7: '#FF1493', 8: '#8B4513', 9: '#20B2AA'}
    # 3a plot
    if Q3a == True:
        fig, ax = plt.subplots(figsize=(12, 7))
        for arc_index, (t, dr) in Q3a_data.items():
            ax.plot(t, dr, label=f'Arc {arc_index}', color = colors[arc_index])

        ax.set_xlabel('Time since departure(days)')
        ax.set_ylabel(r'$\Delta r(t)=||r(t)-\bar{r}(t)||$ [m]')
        ax.set_title('Arcwise propagation deviation from Lambert arc')
        ax.set_yscale('log')
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig('Q3P1.png', dpi = 300)
        plt.show()



    # 3c plot
    if Q3c == True:
        fig3c, ax3c = plt.subplots(figsize=(12, 7))
        for arc_index, (t, dr) in Q3c_data.items():
            ax3c.plot(t, dr, label=f'Arc {arc_index}', color = colors[arc_index])

        ax3c.set_xlabel('Time since arc start (days)')
        ax3c.set_ylabel(r'$\Delta r_{corrected}(t)=||r_{corrected}(t)-\bar{r}(t)||$ [m]')
        ax3c.set_title('Deviation from Lambert arc after correction')
        ax3c.set_yscale('log')
        ax3c.legend(ncol=2, fontsize=9)
        fig3c.tight_layout()
        fig3c.savefig('Q3P2.png', dpi = 300)
        plt.show()

    # 3d plot
    if Q3d == True:
        fig3d, ax3d = plt.subplots(figsize=(12, 7))
        for arc_index, (t, dr) in Q3d_data.items():
            ax3d.plot(t, dr, label=f'Arc {arc_index}',color = colors[arc_index])

        ax3d.set_xlabel('Time since arc start (days)')
        ax3d.set_ylabel(r'$\Delta r$ (m)')
        ax3d.set_title('Deviation from Lambert arc after iterative correction')
        ax3d.set_yscale('log')
        ax3d.legend(ncol=2, fontsize=9)
        fig3d.tight_layout()
        fig3d.savefig('Q3P3.png', dpi = 300)
        plt.show()

        arc_idx = sorted(iterations_per_arc_3d.keys())
        iterations = [iterations_per_arc_3d[arc] for arc in arc_idx]
        fig_iter, ax_iter = plt.subplots(figsize=(12, 7))
        for idx, arc_i in zip(arc_idx, iterations):
            ax_iter.bar(idx, arc_i, color = colors[idx])

        ax_iter.set_xlabel('Arc index')
        ax_iter.set_ylabel('Number of iterations')
        ax_iter.set_title('Iterations to convergence per arc')
        ax_iter.set_xticks(range(10))
        ax_iter.set_xticklabels([f'Arc {i}' for i in range(10)])
        fig_iter.tight_layout()
        fig_iter.savefig('Q3P4.png', dpi = 300)
        plt.show()

        # Saving :)
        if Q3a == True:
            save_data = np.vstack([ROW_9, ROW_10, ROW_11, ROW_12])
            with open('CartesianResults_AE4868_2025_2_6446426.dat', 'ab') as f:
                np.savetxt(f, save_data)

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
