import numpy as np
from interplanetary_transfer_helper_functions_Q4 import *
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd

# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

def single_run_mc(args):
    p1_mc, p_ref, initial_state, departure_epoch, t_mid, arrival_epoch, \
        r_bar_target, tolerance_mc, max_iterations_mc = args

    termination_settings_arc1 = propagation_setup.propagator.time_termination(t_mid)
    termination_settings_arc2 = propagation_setup.propagator.time_termination(arrival_epoch)

    bodies = create_simulation_bodies()

    p1_mc = np.array(p1_mc, dtype=np.float64).flatten()

    propagator_settings_arc1_mc = get_perturbed_propagator_settings(bodies, initial_state, departure_epoch,
                                                                    termination_settings_arc1,
                                                                    empirical_acceleration=p1_mc)

    simulate_arc1_mc = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_arc1_mc)

    state_history_arc1_mc = simulate_arc1_mc.propagation_results.state_history
    t_mid_mc = list(state_history_arc1_mc.keys())[-1]
    x_mid_mc = state_history_arc1_mc[t_mid_mc]

    p2_mc = np.zeros(3)

    for iteration in range(max_iterations_mc):
        print(f'iteration {iteration}')
        propagator_settings_arc2_mc = get_perturbed_propagator_settings(bodies, x_mid_mc, t_mid_mc,
                                                                        termination_settings_arc2,
                                                                        empirical_acceleration=p2_mc)

        sensitivity_parameters_arc2_mc = get_sensitivity_parameter_set(propagator_settings_arc2_mc, bodies)

        variational_solver_arc2_mc = numerical_simulation.create_variational_equations_solver(bodies,
                                                                                              propagator_settings_arc2_mc,
                                                                                              sensitivity_parameters_arc2_mc)

        state_history_arc2_mc = variational_solver_arc2_mc.state_history
        final_epoch_arc2_mc = list(state_history_arc2_mc.keys())[-1]
        x_arc2_final_mc = state_history_arc2_mc[final_epoch_arc2_mc]
        r_arc2_final_mc = x_arc2_final_mc[0:3]
        delta_r_arc2_mc = r_bar_target - r_arc2_final_mc

        sensitivity_history_arc2_mc = variational_solver_arc2_mc.sensitivity_matrix_history
        S_final_arc2_mc = sensitivity_history_arc2_mc[final_epoch_arc2_mc]
        S_r_final_arc2_mc = S_final_arc2_mc[0:3, 0:3]

        if np.linalg.norm(delta_r_arc2_mc) < tolerance_mc:
            print('monte carlo converged :)')
            break

        p2_mc += np.linalg.pinv(S_r_final_arc2_mc) @ delta_r_arc2_mc
        p2_mc = np.array(p2_mc, dtype=np.float64).flatten()

    avg_thrust = (np.linalg.norm(p1_mc) + np.linalg.norm(p2_mc)) / 2.0
    p1_deviation = np.linalg.norm(p1_mc - p_ref)

    return avg_thrust, p1_deviation, p1_mc.tolist(), p2_mc.tolist()


if __name__ == "__main__":

    departure_epoch = 184557994.12800002
    arrival_epoch = 197737594.12800002
    target_body = 'Venus'

    # .... 4 B - 1 ARC MODEL ....

    bodies = create_simulation_bodies()
    lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)

    initial_time = 184557994.12800002
    final_time = 197737594.12800002

    initial_state = lambert_arc_ephemeris.cartesian_state(initial_time)
    r_bar_target = lambert_arc_ephemeris.cartesian_state(final_time)[0:3]

    termination_settings = propagation_setup.propagator.time_termination(final_time)
    nominal_propagator_settings = get_perturbed_propagator_settings(bodies, initial_state, initial_time, termination_settings,
                                                           empirical_acceleration = np.zeros(3, dtype = np.float64))

    sensitivity_parameters = get_sensitivity_parameter_set(nominal_propagator_settings, bodies)

    variational_solver = numerical_simulation.create_variational_equations_solver(bodies, nominal_propagator_settings,
                                                                                  sensitivity_parameters)

    state_history = variational_solver.state_history
    sensitivity_history = variational_solver.sensitivity_matrix_history

    final_epoch = list(state_history.keys())[-1]
    x_nom_final = state_history[final_epoch]
    S_final = sensitivity_history[final_epoch]
    print(S_final)

    r_nom_final = x_nom_final[0:3]
    delta_r_nom = r_bar_target - r_nom_final
    S_r = S_final[0:3, 0:3]

    p = np.linalg.pinv(S_r) @ delta_r_nom
    p = np.array(p, dtype = np.float64).flatten()

    corrected_propagator_settings = get_perturbed_propagator_settings(bodies, initial_state, initial_time, termination_settings,
                                                             empirical_acceleration =  p)

    dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, corrected_propagator_settings)

    state_history_cor = dynamics_simulator.propagation_results.state_history
    lambert_history_cor = get_lambert_arc_history(lambert_arc_ephemeris, state_history_cor)
    times_cor = np.array(sorted(lambert_history_cor.keys()))
    time_days_cor = (times_cor - departure_epoch) / 86400.0
    final_epoch_cor = list(state_history_cor.keys())[-1]
    x_cor_final = state_history_cor[final_epoch_cor]

    delta_r_cor = np.array([np.linalg.norm(state_history_cor[t][:3] - lambert_history_cor[t][:3])for t in times_cor])

    residual_before = np.linalg.norm(delta_r_nom)
    residual_after = np.linalg.norm(delta_r_cor)
    print(f'Residual before correction: {residual_before:.4e} m')
    print(f'Residual after correction: {residual_after:.4e} m')
    print(f'Reduction factor: {residual_before / residual_after:.2f}x')

    ROW_13 = np.hstack([[final_epoch_cor], x_cor_final])
    with open('CartesianResults_AE4868_2025_2_6446426.dat', 'ab') as f:
        np.savetxt(f, ROW_13.reshape(1, -1))
    print('row 13 saved')

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(time_days_cor, delta_r_cor, color = '#FF1493', label = r'$\Delta r(t)$ corrected after single arc')
    ax.set_xlabel(r'Time since departure (days)')
    ax.set_ylabel(r'$\Delta r(t)$ (m)')
    ax.set_yscale('log')
    ax.set_title('Position deviation after single-arc thrust correction (4b)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('Q4P1.png', dpi = 300)
    plt.show()

    # .... 4 C - 2 ARC MODEL ....
    mc_rows = []

    for p1_percent in [0.25, 0.75]:
        t_mid = departure_epoch + p1_percent * (arrival_epoch - departure_epoch)
        print(f"\n{'='*60}\nRUNNING SPLIT = {p1_percent}\n{'='*60}")

        tolerance = 1e-3
        max_iterations = 20
        termination_settings_arc1 = propagation_setup.propagator.time_termination(t_mid)
        termination_settings_arc2 = propagation_setup.propagator.time_termination(arrival_epoch)

        p1 = p.copy()
        p2 = np.zeros(3)

        propagator_settings_arc1 = get_perturbed_propagator_settings(bodies, initial_state, departure_epoch,
                                                                     termination_settings_arc1, empirical_acceleration = p1)

        simulate_arc1 = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_arc1)

        state_history_arc1 = simulate_arc1.propagation_results.state_history
        t_mid_epoch = list(state_history_arc1.keys())[-1]
        x_mid = state_history_arc1[t_mid_epoch]

        for iteration in range(max_iterations):
            propagator_settings_arc2 = get_perturbed_propagator_settings(bodies, x_mid, t_mid_epoch,
                                                                     termination_settings_arc2, empirical_acceleration = p2)

            sensitivity_parameters_arc2 = get_sensitivity_parameter_set(propagator_settings_arc2, bodies)

            variational_solver_arc2 = numerical_simulation.create_variational_equations_solver(bodies,
                                                                                               propagator_settings_arc2,
                                                                                               sensitivity_parameters_arc2)

            state_history_arc2 = variational_solver_arc2.state_history
            sensitivity_history_arc2 = variational_solver_arc2.sensitivity_matrix_history

            final_epoch_arc2 = list(state_history_arc2.keys())[-1]
            x_arc2_final = state_history_arc2[final_epoch_arc2]
            S_final_arc2 = sensitivity_history_arc2[final_epoch_arc2]

            r_arc2_final = x_arc2_final[0:3]
            delta_r_arc2 = r_bar_target - r_arc2_final
            S_r_arc2 = S_final_arc2[0:3, 0:3]

            print(f'iteration {iteration}:  ||delta_r_arc2|| = {delta_r_arc2}')

            if np.linalg.norm(delta_r_arc2) < tolerance:
                print('arc 1 case converged :)')
                break

            p2 += np.linalg.pinv(S_r_arc2) @ delta_r_arc2
            p2 = np.array(p2, dtype = np.float64).flatten()

        times_arc1 = np.array(sorted(state_history_arc1.keys()), dtype = np.float64)
        times_arc2 = np.array(sorted(state_history_arc2.keys()), dtype = np.float64)
        time_days_arc1 = (times_arc1 - departure_epoch) / 86400.0
        time_days_arc2 = (times_arc2 - departure_epoch) / 86400.0

        lambert_history_arc1 = get_lambert_arc_history(lambert_arc_ephemeris, state_history_arc1)
        lambert_history_arc2 = get_lambert_arc_history(lambert_arc_ephemeris,state_history_arc2)

        delta_r_arc1 = np.array([np.linalg.norm(state_history_arc1[t][:3] - lambert_history_arc1[t][:3]) for t in times_arc1])
        delta_r_arc2_plot = np.array([np.linalg.norm(state_history_arc2[t][:3] - lambert_history_arc2[t][:3]) for t in times_arc2])

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(time_days_arc1, delta_r_arc1, color = '#9B59B6', label = 'Arc 1')
        ax.plot(time_days_arc2, delta_r_arc2_plot, color='#FF1493', label = 'Arc 2')
        ax.set_xlabel(r'Time since departure (days)')
        ax.set_ylabel(r'$\Delta r(t)$ (m)')
        ax.set_yscale('log')
        ax.set_title('Position deviation over full two-arc trajectory (4c)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'Q4P2{p1_percent}.png',dpi = 300)
        plt.show()

        ROW_14 = np.hstack([[final_epoch_arc2], x_arc2_final])
        with open('CartesianResults_AE4868_2025_2_6446426.dat', 'ab') as f:
            np.savetxt(f, ROW_14.reshape(1, -1))
        print('row 14 saved')

        # .... 4 D - 2 MONTE CARLO MODEL ON P1 ....

        np.random.seed(10)
        no_of_runs = 1000
        std_dev_mc  = 0.4 * np.linalg.norm(p)
        tolerance_mc = 1e-3
        max_iterations_mc = 20
        p1_random_values = np.random.normal(loc = p, scale = std_dev_mc, size = (no_of_runs, 3))

        avg_thrust_values = []
        p1_deviation_values = []
        p1_values = []
        p2_values = []

        args_list = [
            (p1_random_values[i], p, initial_state, departure_epoch, t_mid, arrival_epoch,
             r_bar_target, tolerance_mc, max_iterations_mc)
            for i in range(no_of_runs)
        ]

        with Pool(processes=12) as pool:
            results = pool.map(single_run_mc, args_list)

        avg_thrust_values = np.array([r[0] for r in results])
        p1_deviation_values = np.array([r[1] for r in results])
        p1_values = [np.array(r[2]) for r in results]
        p2_values = [np.array(r[3]) for r in results]

        avg_thrust_mc = (np.linalg.norm(p1) + np.linalg.norm(p2))/2.0
        fig, ax = plt.subplots(figsize = (15,10))
        ax.scatter(p1_deviation_values, avg_thrust_values, s = 6, alpha = 0.5, color = 'blue', label = 'Monte Carlo Runs')
        ax.scatter(0, avg_thrust_mc, color = 'red', zorder = 5, label = 'Reference')
        ax.set_xlabel(r'||p1 - p|| (m/s^2)')
        ax.set_ylabel(r'(||p1|| + ||p2||)/2 (m/s^2)')
        ax.set_title('Average thrust vs deviation from p')
        ax.legend()
        ax.grid(True, alpha = 0.3, color = 'gray', linestyle = '--', linewidth = 0.3)
        plt.tight_layout()
        plt.savefig(f'Q4P3{p1_percent}.png', dpi=300)
        plt.show()

        optimal_index = np.argmin(avg_thrust_values)
        p1_optimal_mc = p1_values[optimal_index]
        p2_optimal_mc = p2_values[optimal_index]

        mc_rows.append({
            "arc_split": p1_percent,
            "best_avg_thrust": avg_thrust_values[optimal_index],
            "best_p1_deviation": p1_deviation_values[optimal_index],
            "p1": p1_optimal_mc.tolist(),
            "p2": p2_optimal_mc.tolist(),
        })

        print(f'\n p1 optimal mc:       {p1_optimal_mc}'
              f'\n p2 optimal mc:       {p2_optimal_mc}'
              f'\n ||p1_optimal||:      {np.linalg.norm(p1_optimal_mc):.4e}'
              f'\n ||p2_optimal||:      {np.linalg.norm(p2_optimal_mc):.4e}'
              f'\n avg thrust optimal:  {avg_thrust_values[optimal_index]:.4e}')

        termination_settings_tt = propagation_setup.propagator.time_termination(arrival_epoch)
        propagator_settings_tt = get_perturbed_propagator_settings(bodies, initial_state, departure_epoch,
                                                                   termination_settings_tt, empirical_acceleration = p)

        simulate_tt = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_tt)
        times_tt = np.array(list(simulate_tt.propagation_results.state_history.keys()))
        p_R = np.full(len(times_tt), p[0])
        p_S = np.full(len(times_tt), p[1])
        p_W = np.full(len(times_tt), p[2])

        propagator_settings_tt_opt_arc1 =  get_perturbed_propagator_settings(bodies, initial_state, departure_epoch,
                                                                             termination_settings_arc1,
                                                                             empirical_acceleration = p1_optimal_mc)

        simulate_tt_opt_arc1 = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_tt_opt_arc1)
        times_tt_opt_arc1 = np.array(list(simulate_tt_opt_arc1.propagation_results.state_history.keys()))

        t_mid_tt  = list(simulate_tt_opt_arc1.propagation_results.state_history.keys())[-1]
        x_mid_tt = simulate_tt_opt_arc1.propagation_results.state_history[t_mid_tt]
        propagator_settings_tt_opt_arc2 = get_perturbed_propagator_settings(bodies, x_mid_tt, t_mid_tt,
                                                                            termination_settings_arc2,
                                                                            empirical_acceleration = p2_optimal_mc)

        simulate_tt_opt_arc2 = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_tt_opt_arc2)
        times_tt_opt_arc2 = np.array(list(simulate_tt_opt_arc2.propagation_results.state_history.keys()))

        times_tt_opt = np.concatenate((times_tt_opt_arc1, times_tt_opt_arc2))
        p_R_opt = np.concatenate([np.full(len(times_tt_opt_arc1), p1_optimal_mc[0]),
                                  np.full(len(times_tt_opt_arc2), p2_optimal_mc[0])])

        p_S_opt = np.concatenate([np.full(len(times_tt_opt_arc1), p1_optimal_mc[1]),
                                  np.full(len(times_tt_opt_arc2), p2_optimal_mc[1])])

        p_W_opt = np.concatenate([np.full(len(times_tt_opt_arc1), p1_optimal_mc[2]),
                                  np.full(len(times_tt_opt_arc2), p2_optimal_mc[2])])

        fig, axes = plt.subplots(3, 1, figsize = (15,10))

        components = ['R (Radial)', 'S (Along Track)', 'W (Cross-Track)']
        thrust_tt_vals = [p_R, p_S, p_W]
        thrust_tt_opt_vals = [p_R_opt, p_S_opt, p_W_opt]
        days_tt = (times_tt - departure_epoch) / 86400.0
        days_opt = (times_tt_opt - departure_epoch) / 86400.0

        for i ,ax in enumerate(axes):
            ax.plot(days_tt, thrust_tt_vals[i], label = '1 Arc Case Thrust', linewidth = 1.5, color = 'red')
            ax.plot(days_opt, thrust_tt_opt_vals[i], label = 'Optimal Case Thrust', linewidth = 1.5, color = 'blue')
            ax.axvline((t_mid - departure_epoch) / 86400.0, color = 'green', linestyle = '--', linewidth = 2.0, label = 'Arc Split')
            ax.set_xlabel(r'Days since departure epoch')
            ax.set_ylabel(f'{components[i]} acceleration (m/s²)')
            ax.grid(True, alpha = 0.5, color = 'gray', linestyle = '--', linewidth = 0.3)
            ax.legend()
        fig.suptitle('RSW Thrust Components')
        plt.tight_layout()
        plt.savefig(f'Q4P4{p1_percent}.png', dpi = 300)
        plt.show()
    best_df = pd.DataFrame(mc_rows)
    best_df.to_csv("Q2_MonteCarlo_BestPerSplit.csv", index=False)
    print(best_df)