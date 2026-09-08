# MONTE CARLO [0.25, 0.5, 0.75]
# imports
from interplanetary_transfer_helper_functions_Q4 import *
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd


# ......................................................................................................................
#                                        Single Run for Monte Carlo
# ......................................................................................................................
def single_run_mc(args):

    p1_mc, p_ref, initial_state, departure_epoch, t_mid, arrival_epoch, \
        r_bar_target, tolerance_mc, max_iterations_mc = args

    termination_settings_arc1 = propagation_setup.propagator.time_termination(t_mid)
    termination_settings_arc2 = propagation_setup.propagator.time_termination(arrival_epoch)
    bodies_mc, x_mid_mc, t_mid_mc = propagate_arc1(p1_mc, initial_state, departure_epoch, termination_settings_arc1)
    p1_mc = np.array(p1_mc, dtype=np.float64).flatten()
    p2_mc = np.zeros(3)
    S_r_arc2_mc = None
    delta_r_arc2_mc = None

    for iteration in range(max_iterations_mc):
        print(f'iteration {iteration}')
        propagator_settings_arc2_mc = get_perturbed_propagator_settings(bodies_mc, x_mid_mc, t_mid_mc,
                                                                        termination_settings_arc2,
                                                                        empirical_acceleration=p2_mc)

        sensitivity_parameters_arc2_mc = get_sensitivity_parameter_set(propagator_settings_arc2_mc, bodies_mc)

        variational_solver_arc2_mc = numerical_simulation.create_variational_equations_solver(bodies_mc,
                                                                                              propagator_settings_arc2_mc,
                                                                                              sensitivity_parameters_arc2_mc)

        state_history_arc2_mc = variational_solver_arc2_mc.state_history
        final_epoch_arc2_mc = list(state_history_arc2_mc.keys())[-1]

        x_arc2_final_mc = state_history_arc2_mc[final_epoch_arc2_mc]
        r_arc2_final_mc = x_arc2_final_mc[0:3]
        delta_r_arc2_mc = r_bar_target - r_arc2_final_mc

        sensitivity_history_arc2_mc = variational_solver_arc2_mc.sensitivity_matrix_history
        S_final_arc2_mc = sensitivity_history_arc2_mc[final_epoch_arc2_mc]
        S_r_arc2_mc = S_final_arc2_mc[0:3, 0:3]  # (3 x 3)

        if np.linalg.norm(delta_r_arc2_mc) < tolerance_mc:
            print('monte carlo converged :)')
            break
        # p2 updated after break as we require the p2 correction for the final run
        p2_mc += np.linalg.pinv(S_r_arc2_mc) @ delta_r_arc2_mc
        p2_mc = np.array(p2_mc, dtype = np.float64).flatten()

    avg_thrust = (np.linalg.norm(p1_mc) + np.linalg.norm(p2_mc)) / 2.0
    p1_deviation = np.linalg.norm(p1_mc - p_ref)
    S_r_norm = np.linalg.norm(S_r_arc2_mc)
    final_miss = np.linalg.norm(delta_r_arc2_mc)
    return avg_thrust, p1_deviation, p1_mc.tolist(), p2_mc.tolist(), S_r_norm, final_miss

# ......................................................................................................................
#                                       Parallel Processing Monte Carlo
# ......................................................................................................................

def init_worker():
    spice.load_standard_kernels()

def monte(p1_percent, t_mid, p, initial_state, departure_epoch, arrival_epoch,
          r_bar_target, std_deviation_mc, no_of_runs_mc, tolerance_mc, max_iterations_mc):

    np.random.seed(10)
    p1_random_values = np.random.normal(loc = p , scale = std_deviation_mc, size = (no_of_runs_mc, 3))

    args_list = [
        (p1_random_values[i], p, initial_state, departure_epoch, t_mid, arrival_epoch,
         r_bar_target, tolerance_mc, max_iterations_mc)
        for i in range(no_of_runs_mc)
    ]


    # parallel processing mc runs
    with Pool(processes=7, initializer= init_worker) as pool:
        results = pool.map(single_run_mc, args_list)

    avg_thrust_values = np.array([r[0] for r in results])
    p1_deviation_values = np.array([r[1] for r in results])
    p1_values = [np.array(r[2]) for r in results]
    p2_values = [np.array(r[3]) for r in results]
    S_r_norm_values = np.array([r[4] for r in results])
    final_miss_values = np.array([r[5] for r in results])

    # finding the optimal mc

    return avg_thrust_values, p1_deviation_values, p1_values, p2_values, S_r_norm_values, final_miss_values


# ......................................................................................................................
#                                            Arc Propagation
# ......................................................................................................................

def propagate_arc1(p1, initial_state, departure_epoch, termination_settings_arc1):
    bodies = create_simulation_bodies()
    propagator_settings_arc1_mc = get_perturbed_propagator_settings(bodies, initial_state, departure_epoch,
                                                                    termination_settings_arc1,
                                                                    empirical_acceleration=p1)
    simulate_arc1 = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_arc1_mc)
    state_history_arc1 = simulate_arc1.propagation_results.state_history
    t_mid = list(state_history_arc1.keys())[-1]
    x_mid = state_history_arc1[t_mid]
    return bodies, x_mid, t_mid

# ......................................................................................................................
#                                               Main
# ......................................................................................................................

if __name__ == "__main__":

    # ..... Assigned Values - from A2 Q2 ciii .....
    departure_epoch = 184557994.12800002
    arrival_epoch = 197737594.12800002
    target_body = 'Venus'

# ......................................................................................................................
#                                            Reference Thrust
# ......................................................................................................................

    # ..... Running The Reference Arc .....
    spice.load_standard_kernels()
    bodies = create_simulation_bodies()
    lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)
    initial_state = lambert_arc_ephemeris.cartesian_state(departure_epoch)
    r_bar_target = lambert_arc_ephemeris.cartesian_state(arrival_epoch)[0:3]

    # ..... Propagating The 1 Arc Case .....
    # this is done to obtain p or pref --> i.e, the ref thrust required to gen the random numbers for monte-carlo
    # optimisation process
    termination_settings = propagation_setup.propagator.time_termination(arrival_epoch)
    nominal_propagator_settings = get_perturbed_propagator_settings(bodies, initial_state, departure_epoch, termination_settings,
                                                           empirical_acceleration = np.zeros(3, dtype = np.float64))

    sensitivity_parameters = get_sensitivity_parameter_set(nominal_propagator_settings, bodies)
    variational_solver = numerical_simulation.create_variational_equations_solver(bodies, nominal_propagator_settings,
                                                                                  sensitivity_parameters)

    state_history = variational_solver.state_history
    sensitivity_history = variational_solver.sensitivity_matrix_history
    final_epoch = list(state_history.keys())[-1]
    x_nom_final = state_history[final_epoch]
    S_final = sensitivity_history[final_epoch]
    r_nom_final = x_nom_final[0:3]
    delta_r_nom = r_bar_target - r_nom_final
    S_r = S_final[0:3, 0:3]     # (3 x 3)
    # *NOTE* sup to be cols 0 - 5: initial state, cols 6 - 8: RSW empirical acceleration but since
    # δx₀ = 0 --> δx(tE) = Φ · δx₀ + S · δp = S · δp

    # ..... Reference Thrust .....
    p = np.linalg.pinv(S_r) @ delta_r_nom
    p = np.array(p, dtype = np.float64).flatten()

# ......................................................................................................................
#                                            Monte Carlo Cases
# ......................................................................................................................

    # ..... Initiating Monte Carlo .....
    no_of_runs_mc = 1000
    tolerance_mc = 1e-3
    max_iterations_mc = 20
    std_deviation_mc = 0.4 * np.linalg.norm(p)
    rows = []
    S_r_norm_all_splits = {}
    ref_avg_thrust = np.linalg.norm(p)
    fig_p1p2, ax_p1p2 = plt.subplots(figsize=(12, 8))

    # ..... PLOT: Average Thrust vs p1 Deviation .....
    fig, ax = plt.subplots(figsize=(15, 10))
    for i, p1_percent in enumerate([0.25, 0.5, 0.75]):
        t_mid = departure_epoch + p1_percent * (arrival_epoch - departure_epoch)
        (avg_thrust_values_mc, p1_deviation_values_mc, p1_values_mc, p2_values_mc, S_r_norm_values_mc, final_miss_values_mc) = monte(
            p1_percent, t_mid, p, initial_state, departure_epoch, arrival_epoch, r_bar_target,
            std_deviation_mc, no_of_runs_mc, tolerance_mc, max_iterations_mc)

        best_idx = np.argmin(avg_thrust_values_mc )
        p1_norms = [np.linalg.norm(p1v) for p1v in p1_values_mc]
        p2_norms = [np.linalg.norm(p2v) for p2v in p2_values_mc]
        ax_p1p2.scatter(p1_norms, p2_norms, s=6, alpha=0.4, label=f'split={p1_percent}')
        ax_p1p2.scatter(p1_norms[best_idx], p2_norms[best_idx],
                        marker='*', s=200, edgecolor='black', linewidth=0.5, zorder=5,
                        label=f'best for {p1_percent}')
        ax.scatter(p1_deviation_values_mc, avg_thrust_values_mc, s=6, alpha=0.5, label=f'split={p1_percent}')
        ax.scatter(p1_deviation_values_mc[best_idx], avg_thrust_values_mc[best_idx],
                   marker='*', s=200, color='gold', edgecolor='black', linewidth=0.5,
                   zorder=5, label=f'best for {p1_percent}')
        rows.append([p1_percent, avg_thrust_values_mc[best_idx], p1_deviation_values_mc[best_idx],
                     p1_values_mc[best_idx], p2_values_mc[best_idx], S_r_norm_values_mc[best_idx], final_miss_values_mc[best_idx]])
        if final_miss_values_mc[best_idx] >= tolerance_mc:
            print(f"[WARNING] best candidate for split={p1_percent} did not converge "
                  f"(final_miss={final_miss_values_mc[best_idx]:.4g}, tol={tolerance_mc})")
        S_r_norm_all_splits[p1_percent] = S_r_norm_values_mc
        print(f"S_final.shape = {S_final.shape}")
        print(f"|p| = {np.linalg.norm(p):.4e}")
    best_df = pd.DataFrame(rows, columns=["arc_split", "best_avg_thrust", "best_p1_deviation", "p1", "p2", "S_r_norm", "final_miss"])
    best_df.to_csv("Q2_MonteCarlo_BestPerSplit.csv", index=False)
    print(best_df)

    ax.set_ylabel("Average Thrust Values (||p1|| + ||p2||)/2 [m/s²]")
    ax.set_xlabel("p1 Deviation ||p1 - p_ref|| [m/s²]")
    fig.suptitle('Average thrust vs p1 Deviation')
    ax.legend()
    plt.tight_layout()
    plt.savefig('MC.png', dpi=300)
    plt.show()

    ax_p1p2.set_xlabel('||p1|| [m/s²]')
    ax_p1p2.set_ylabel('||p2|| [m/s²]')
    ax_p1p2.set_xscale('log')
    ax_p1p2.set_yscale('log')
    fig_p1p2.suptitle('p1 vs p2 Magnitude Across All Splits')
    ax_p1p2.legend()
    plt.tight_layout()
    plt.savefig('Q2_p1_vs_p2_magnitude.png', dpi=300)
    plt.show()

    print("\n--- Convergence check on picked best candidates ---")
    print(best_df[["arc_split", "final_miss", "S_r_norm"]])
    print(f"(tolerance_mc = {tolerance_mc}; final_miss should be well below this)\n")



# ..... PLOT:: RSW Components vs Days Since Departure .....
    best_overall = best_df.loc[best_df["best_avg_thrust"].idxmin()]
    t_mid_best = departure_epoch + best_overall["arc_split"] * (arrival_epoch - departure_epoch)
    p1_best = best_overall["p1"]
    p2_best = best_overall["p2"]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    components = ['R (Radial)', 'S (Along Track)', 'W (Cross-Track)']

    days_ref = [0, (arrival_epoch - departure_epoch) / 86400.0]
    days_arc1 = [0, (t_mid_best - departure_epoch) / 86400.0]
    days_arc2 = [(t_mid_best - departure_epoch) / 86400.0, (arrival_epoch - departure_epoch) / 86400.0]

    for i, ax in enumerate(axes):
        ax.plot(days_ref, [p[i], p[i]], label='1 Arc Case Thrust', linewidth=1.5, color='red')
        ax.plot(days_arc1, [p1_best[i], p1_best[i]], label='Optimal Case Thrust (arc 1)', linewidth=1.5, color='blue')
        ax.plot(days_arc2, [p2_best[i], p2_best[i]], linewidth=1.5, color='blue')
        ax.axvline((t_mid_best - departure_epoch) / 86400.0, color='green', linestyle='--', linewidth=2.0,
                   label='Arc Split')
        ax.set_xlabel(r'Days since departure epoch')
        ax.set_ylabel(f'{components[i]} acceleration (m/s²)')
        ax.grid(True, alpha=0.5, color='gray', linestyle='--', linewidth=0.3)
        ax.legend()

    fig.suptitle('RSW Thrust Components')
    plt.tight_layout()
    plt.savefig('Q2_RSW_thrust_components.png', dpi=300)
    plt.show()

    # ...... PLOT: Sensitivity Magnitude vs Arc Split .....
    fig, ax = plt.subplots(figsize=(15, 10))
    splits = [0.25, 0.5, 0.75]
    best_S_r = [best_df.loc[best_df["arc_split"] == s, "S_r_norm"].values[0] for s in splits]
    mean_S_r = np.array([np.mean(S_r_norm_all_splits[s]) for s in splits])
    std_S_r = np.array([np.std(S_r_norm_all_splits[s]) for s in splits])
    lower_err = np.minimum(std_S_r, mean_S_r * 0.999)
    ax.errorbar(splits, mean_S_r, yerr=[lower_err, std_S_r], fmt='o-', color='tab:gray', alpha=0.6,
                label='Mean ± std across all 1000 candidates', capsize=4)
    ax.plot(splits, best_S_r, 's--', color='tab:blue', label='S_r at converged best candidate')

    ax.set_xlabel("Arc 1 Split Fraction")
    ax.set_ylabel(r"$\|S_r\|$ Arc 2 Position Sensitivity to p2")
    ax.grid(True, alpha=0.5, color='gray', linestyle='--', linewidth=0.3)
    ax.set_yscale("log")
    ax.legend()
    fig.suptitle('Sensitivity Magnitude vs Arc Split Fraction')
    plt.tight_layout()
    plt.savefig('Q2_sensitivity_vs_split.png', dpi=300)
    plt.show()



    # ......................................................................................................................
    #                                           PROPAGATE BEST CASES INDIVIDUALLY
    # ......................................................................................................................
    bodies_best = create_simulation_bodies()
    termination_arc1_best = propagation_setup.propagator.time_termination(t_mid_best)
    termination_arc2_best = propagation_setup.propagator.time_termination(arrival_epoch)

    propagator_settings_arc1_best = get_perturbed_propagator_settings(bodies_best, initial_state,
                                                                 departure_epoch, termination_arc1_best,
                                                                 empirical_acceleration=np.array(p1_best))

    sim_arc1_best = numerical_simulation.create_dynamics_simulator(bodies_best, propagator_settings_arc1_best)
    dep_vars_arc1 = sim_arc1_best.propagation_results.dependent_variable_history
    state_arc1_best = sim_arc1_best.propagation_results.state_history
    t_mid_actual = list(state_arc1_best.keys())[-1]
    x_mid_actual = state_arc1_best[t_mid_actual]


    propagator_settings_arc2_best = get_perturbed_propagator_settings(bodies_best, x_mid_actual,
                                                                 t_mid_actual, termination_arc2_best,
                                                                 empirical_acceleration=np.array(p2_best))
    sim_arc2_best = numerical_simulation.create_dynamics_simulator(bodies_best, propagator_settings_arc2_best)
    dep_vars_arc2 = sim_arc2_best.propagation_results.dependent_variable_history
    # column layout of the 30-element dependent variable vector (see get_perturbed_propagator_settings):
    # 0:6 kepler state | 6:9 Mars | 9:12 Earth | 12:15 Moon | 15:18 Venus |
    # 18:21 Jupiter | 21:24 Saturn | 24:27 Sun gravity (central, not a perturbation) | 27:30 Sun SRP
    body_columns = {
        'Mars': slice(6, 9), 'Earth': slice(9, 12), 'Moon': slice(12, 15),
        'Venus': slice(15, 18), 'Jupiter': slice(18, 21), 'Saturn': slice(21, 24),
        'SRP (Sun)': slice(27, 30),
    }


    def extract_per_body(dep_vars):
        times = np.array(list(dep_vars.keys()))
        days = (times - departure_epoch) / 86400.0
        values = np.array(list(dep_vars.values()))  # shape (N, 30)
        per_body = {name: np.linalg.norm(values[:, cols], axis=1) for name, cols in body_columns.items()}
        return days, per_body


    days1, perbody1 = extract_per_body(dep_vars_arc1)
    days2, perbody2 = extract_per_body(dep_vars_arc2)

    fig, ax1 = plt.subplots(figsize=(15, 6))
    for name in body_columns:
        full_days = np.concatenate([days1, days2])
        full_vals = np.concatenate([perbody1[name], perbody2[name]])
        ax1.plot(full_days, full_vals, linewidth=1.2, label=name)

    ax1.set_xlabel('Days since departure epoch')
    ax1.set_ylabel('Perturbing acceleration (m/s²)')
    ax1.set_yscale('log')  # third-body accelerations often span orders of magnitude - log scale helps

    ax2 = ax1.twinx()
    days_arc1 = [0, (t_mid_best - departure_epoch) / 86400.0]
    days_arc2 = [(t_mid_best - departure_epoch) / 86400.0, (arrival_epoch - departure_epoch) / 86400.0]
    ax2.plot(days_arc1, [np.linalg.norm(p1_best), np.linalg.norm(p1_best)],
             color='black', linewidth=2.5, label='|p1| correction thrust')
    ax2.plot(days_arc2, [np.linalg.norm(p2_best), np.linalg.norm(p2_best)],
             color='black', linewidth=2.5, linestyle='--', label='|p2| correction thrust')
    ax2.axvline((t_mid_best - departure_epoch) / 86400.0, color='green', linestyle=':', linewidth=1.5)
    ax2.set_ylabel('Correction thrust magnitude (m/s²)')

    fig.suptitle('Per-Body Perturbation Intensity vs Optimal Correction Thrust')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig('Q2_perturbation_per_body_vs_thrust.png', dpi=300)
    plt.show()















