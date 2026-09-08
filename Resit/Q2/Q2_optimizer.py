# OPTIMIZER - PYGMO

# ..... IMPORTS .....
from interplanetary_transfer_helper_functions_Q4 import *
import pandas as pd
import pygmo as pg
import matplotlib.pyplot as plt

# ......................................................................................................................
#                                   Input Extraction from Monte Carlo
# ......................................................................................................................


# assigned i/p vals
departure_epoch = 184557994.12800002
arrival_epoch = 197737594.12800002
target_body = 'Venus'


spice.load_standard_kernels()
bodies = create_simulation_bodies()
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)
initial_state = lambert_arc_ephemeris.cartesian_state(departure_epoch)
r_bar_target = lambert_arc_ephemeris.cartesian_state(arrival_epoch)[0:3]

# ......................................................................................................................
#                                                Propagator
# ......................................................................................................................

def propagator(p1, p2, t_mid):

    # ..... propagate arc 1 n obtain p1 .....
    p1 = np.array(p1, dtype=np.float64).flatten()
    termination_settings_arc1 = propagation_setup.propagator.time_termination(t_mid)

    propagator_settings_arc1 = get_perturbed_propagator_settings(bodies, initial_state, departure_epoch,
                                                                     termination_settings_arc1,
                                                                     empirical_acceleration=p1)

    simulate_arc1 = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_arc1)
    state_history_arc1 = simulate_arc1.propagation_results.state_history
    t_mid = list(state_history_arc1.keys())[-1]
    x_mid = state_history_arc1[t_mid]

    # ..... propagate arc 1 n obtain p2 .....
    p2 = np.array(p2, dtype=np.float64).flatten()
    termination_settings_arc2 = propagation_setup.propagator.time_termination(arrival_epoch)

    propagator_settings_arc2 = get_perturbed_propagator_settings(bodies, x_mid, t_mid,
                                                                     termination_settings_arc2,
                                                                     empirical_acceleration=p2)

    simulate_arc2 = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_arc2)
    state_history_arc2 = simulate_arc2.propagation_results.state_history
    final_epoch = list(state_history_arc2.keys())[-1]
    x_final = state_history_arc2[final_epoch]

    return x_final
# ......................................................................................................................
#                                               Min_dv Class
# ......................................................................................................................

class Min_dv:
    def __init__ (self, t_mid, bound, lam = 1e6):
        self.t_mid = t_mid
        self.bound = bound
        self.lam = lam

# ......................................................................................................................
#                                             Penalty Score
# ......................................................................................................................

    def fitness(self, dv):

        p1 = np.array(dv[:3])
        p2 = np.array(dv[3:])
        x_final = propagator(p1, p2, self.t_mid)
        miss = np.linalg.norm(r_bar_target - x_final[0:3])
        delta_v = np.linalg.norm(p1) * (self.t_mid - departure_epoch) + np.linalg.norm(p2) * (arrival_epoch - self.t_mid)
        score = delta_v + self.lam * miss

        return [score]

    def get_bounds(self):
        return ([-self.bound] * 6, [self.bound] * 6)


# ......................................................................................................................
#                                               Run Optimizer
# ......................................................................................................................

def run_optimizer(t_mid, p_ref, pop_size = 12 , generations = 20):
    bound = 5 * np.linalg.norm(p_ref)
    prob = pg.problem(Min_dv(t_mid, bound))
    algo = pg.algorithm(pg.de(gen = generations))
    pop = algo.evolve(pg.population(prob, size = pop_size))
    best_dv = pop.champion_x
    best_score = pop.champion_f[0]
    best_p1 = best_dv[:3]
    best_p2 = best_dv[3:]
    return best_score, best_p1, best_p2


# ......................................................................................................................
#                                               Main
# ......................................................................................................................


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    best_df = pd.read_csv(os.path.join(script_dir, "Q2_MonteCarlo_BestPerSplit.csv"))
    best_row = best_df.loc[best_df["best_avg_thrust"].idxmin()]
    p1_percent_best = best_row["arc_split"]
    t_mid_best = departure_epoch + p1_percent_best * (arrival_epoch - departure_epoch)
    p_ref = best_row["p1"]
    p_ref = np.fromstring(best_row["p1"].strip("[]"), sep = " ")
    print(f"parsed p_ref = {p_ref}")

    mc_p1_best = p_ref
    mc_p2_best = np.fromstring(best_row["p2"].strip("[]"), sep=" ")
    pg.set_global_rng_seed(42)
    best_score_opt, best_p1_opt, best_p2_opt = run_optimizer(t_mid_best, p_ref)

    print(f"optimal thrust for arc 1 | {best_p1_opt}")
    print(f"optimal thrust for arc 2 | {best_p2_opt}")
    print(f"min total dv score | {best_score_opt}")

    # computing true time weightwd value of delta v instead of the avg thrust val
    mc_true_dv = (np.linalg.norm(mc_p1_best) * (t_mid_best - departure_epoch) +
                  np.linalg.norm(mc_p2_best) * (arrival_epoch - t_mid_best))

    x_final_op = propagator(best_p1_opt, best_p2_opt, t_mid_best)
    opt_final_miss = np.linalg.norm(r_bar_target-x_final_op[:3])
    op_true_dv = (np.linalg.norm(best_p1_opt) * (t_mid_best - departure_epoch) +
                   np.linalg.norm(best_p2_opt) * (arrival_epoch - t_mid_best))

    fig, axes = plt.subplots(3, 1, figsize = (15,10))
    components = ["R (Radial)", "S (Along Track)", "W (Cross-Track)" ]
    days_arc1 = [0, (t_mid_best - departure_epoch) / 86400.0]
    days_arc2 = [(t_mid_best - departure_epoch) / 86400.0, (arrival_epoch - departure_epoch) / 86400.0]
    for i, ax in enumerate(axes):
        ax.plot(days_arc1, [mc_p1_best[i], mc_p1_best[i]], color = "blue", linewidth = 1.5, label = "MC best (arc 1)")
        ax.plot(days_arc2, [mc_p2_best[i], mc_p2_best[i]], color = "blue", linewidth = 1.5, label = "MC best (arc 2)")
        ax.plot(days_arc1, [best_p1_opt[i], best_p1_opt[i]], color="darkorange", linewidth=1.5, label = "Optimizer best (arc1)")
        ax.plot(days_arc2, [best_p2_opt[i], best_p2_opt[i]], color="darkorange", linewidth=1.5, label = "Optimizer best (arc2)")
        ax.axvline((t_mid_best - departure_epoch) / 86400.0, color = "green", linestyle = "--", linewidth = 1, label = "Arc Split")
        ax.set_xlabel('Days since departure epoch')
        ax.set_ylabel(f'{components[i]} acceleration (m/s²)')
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend()
    fig.suptitle(f'Optimal Thrust Profile: MC vs Pygmo Optimizer (split = {p1_percent_best})')
    plt.tight_layout()
    plt.savefig('Q2_optimizer_vs_MC_thrust.png', dpi=300)
    plt.show()

    print(f"\n--- Comparison at split={p1_percent_best} ---")
    print(
        f"MC best:        DeltaV={mc_true_dv:.6e}  |p1|={np.linalg.norm(mc_p1_best):.4e}  |p2|={np.linalg.norm(mc_p2_best):.4e}")
    print(
        f"Optimizer best: DeltaV={op_true_dv:.6e}  |p1|={np.linalg.norm(best_p1_opt):.4e}  |p2|={np.linalg.norm(best_p2_opt):.4e}  final_miss={opt_final_miss:.4e}")
    print(f"Improvement: {100 * (mc_true_dv - op_true_dv) / mc_true_dv:.2f}% lower DeltaV than MC best")