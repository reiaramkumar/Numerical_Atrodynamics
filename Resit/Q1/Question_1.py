# importing modules
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from tudatpy import constants
from tudatpy.data import save2txt
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.astro import frame_conversion
from scipy.signal import find_peaks


# state extraction - not sure if it is req for this que
sum_file = 'cartesian_results_AE4868_2026_A1_6446426.txt'
def sum_table_gen(file_name, row_label, time, state):
    with open(file_name,'a') as f:
        state_str = "\t".join([f"{x:.15e}" for x in state])
        f.write(f'{row_label}\t{time:.15e}\t{state_str}\n')

# font style for plots
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 10
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 11

# pastel palette for plots
palette = {
    "pastel_blue"   : "#6AA7FF",
    "pastel_pink"   : "#FF82A9",
    "pastel_green"  : "#7ED6A6",
    "pastel_purple" : "#A894E8",
    "pastel_orange" : "#FFB27C",
    "pastel_teal"   : "#63D2C6"
}

# Retrieve current directory
current_directory = os.getcwd()

# ......................................................................................................................
#                                                   SIMULATION TIME
# ......................................................................................................................

# student number: 1244779 --> 1244ABC
A = 4
B = 2
C = 6

simulation_start_epoch = (
    35.4 * constants.JULIAN_YEAR
    + A * 7.0 * constants.JULIAN_DAY
    + B * constants.JULIAN_DAY
    + C * constants.JULIAN_DAY / 24.0
)
simulation_end_epoch = simulation_start_epoch + constants.JULIAN_DAY * 7

# ......................................................................................................................
#                                               CREATE ENVIRONMENT & BODIES
# ......................................................................................................................

# ..... Space Environment .....
# Load spice kernels.
spice.load_standard_kernels()
spice.load_kernel(current_directory + "/juice_mat_crema_5_1_150lb_v01.bsp")

# Create settings for celestial bodies
bodies_to_create = ["Jupiter", "Io", "Europa", "Ganymede", "Callisto","Saturn", "Sun"]
global_frame_origin = "Jupiter"
global_frame_orientation = "ECLIPJ2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# ..... Create Space Vehicle .....
body_settings.add_empty_settings("JUICE")
# note: this is used only for cases 5 - 8 hopefully doesn't affect the values of cases 1 - 4


# ......................................................................................................................
#                                               ACCELERATION MODELS
# ......................................................................................................................

bodies_to_propagate_moons = ["Io", "Europa", "Ganymede", "Callisto"]
central_bodies_for_moons = ["Jupiter", "Jupiter", "Jupiter", "Jupiter"]

bodies_to_propagate_juice = bodies_to_propagate_moons + ["JUICE"]
central_bodies_for_juice = central_bodies_for_moons + ["Ganymede"]

reference_area = 100.0                                  # m2
drag_coefficient = 1.2
radiation_pressure_coefficient = 1.2                    # kg
density_scale_height = 40e3                             # m
density_at_zero_alt = 2e-9                              # kg/m3

body_settings.add_empty_settings("JUICE")

body_settings.get("Ganymede").atmosphere_settings = environment_setup.atmosphere.exponential(density_at_zero_alt, density_scale_height)

# Create environment
bodies = environment_setup.create_system_of_bodies(body_settings)
bodies.get_body('JUICE').mass = 2000.0

aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(reference_area,[drag_coefficient,0.0,0.0])

environment_setup.add_aerodynamic_coefficient_interface(bodies, "JUICE", aero_coefficient_settings)

occulting_bodies = ["Ganymede"]
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball("Sun", reference_area, radiation_pressure_coefficient, occulting_bodies)

environment_setup.add_radiation_pressure_interface(bodies, "JUICE", radiation_pressure_settings)


# ..... Acceleration Settings on The Moons .....

def acceleration_settings_fn(case = 1, include_juice = False):

    acceleration_settings = {}
    case = case if case <= 4 else case - 4
    # case 1 - jupiter pm only
    for moon in bodies_to_propagate_moons:
        acceleration_settings[moon] = dict(Jupiter = [propagation_setup.acceleration.point_mass_gravity()])

    # case 2 - adding mutual point mass perturbations
    if case >= 2:
        for moon in bodies_to_propagate_moons:
            for other_moon in bodies_to_propagate_moons:
                if moon!=other_moon:
                    acceleration_settings[moon][other_moon] = [propagation_setup.acceleration.point_mass_gravity()]

    # case 3 - spherical harmonics (8,0) for jupiter
    if case >= 3:
        for moon in bodies_to_propagate_moons:
            acceleration_settings[moon]["Jupiter"] = [propagation_setup.acceleration.spherical_harmonic_gravity(8, 0)]


    # case 4 - spherical harmonics (8,0) for jupiter + (2,2) for moons
    if case >= 4:
        for moon in bodies_to_propagate_moons:
            for other_moon in bodies_to_propagate_moons:
                if moon!=other_moon:
                    acceleration_settings[moon][other_moon] = [propagation_setup.acceleration.spherical_harmonic_gravity(2,2)]


# CLARIFY WITH PROFESSOR
    if include_juice:
        acceleration_settings["JUICE"] = dict(
            Ganymede=
            [
                propagation_setup.acceleration.spherical_harmonic_gravity(2, 2),
                propagation_setup.acceleration.aerodynamic()
            ],

            Jupiter=
            [
                propagation_setup.acceleration.spherical_harmonic_gravity(4, 0)
            ],

            Sun=
            [
                propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.cannonball_radiation_pressure()
            ],

            Saturn=
            [
                propagation_setup.acceleration.point_mass_gravity()
            ],

            Europa=
            [
                propagation_setup.acceleration.point_mass_gravity()
            ],

            Io=
            [
                propagation_setup.acceleration.point_mass_gravity()
            ],

            Callisto=
            [
                propagation_setup.acceleration.point_mass_gravity()
            ]
        )



    return acceleration_settings

# ......................................................................................................................
#                                                 PROPAGATION
# ......................................................................................................................


propagation_results = {}
moon_states = {}

for case in range(1,9):

    dependent_variables = []
    system_initial_state = []

    bodies_to_propagate = bodies_to_propagate_juice if case>=5 else bodies_to_propagate_moons
    central_bodies = central_bodies_for_juice if case>=5 else central_bodies_for_moons
    include_juice = bool(case >= 5)

    for i, target_body in enumerate(bodies_to_propagate):
        observer =  "Ganymede" if target_body == "JUICE" else "Jupiter"
        dependent_variables.append(
            propagation_setup.dependent_variable.relative_position(target_body, observer))


        initial_state = spice.get_body_cartesian_state_at_epoch(
            target_body_name=target_body,
            observer_body_name= observer,
            reference_frame_name="ECLIPJ2000",
            aberration_corrections="NONE",
            ephemeris_time=simulation_start_epoch,
        )

        system_initial_state.append(initial_state)

    system_initial_state = np.concatenate(system_initial_state)

    fixed_step_size = 10.0
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )

    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

    acceleration_settings = acceleration_settings_fn(case, include_juice)
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        system_initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_settings,
        propagator = propagation_setup.propagator.TranslationalPropagatorType.cowell,
        output_variables = dependent_variables,
    )

    propagator_settings.print_settings.print_initial_and_final_conditions = True
    dynamics_simulator = simulator.create_dynamics_simulator(bodies, propagator_settings)

# ......................................................................................................................
#                                                 RESULTS
# ......................................................................................................................
    propagation_results[case] = dynamics_simulator.propagation_results
    state_history = propagation_results[case].state_history
    dependent_variables_to_save = propagation_results[case].dependent_variable_history

    state_array = np.vstack(list(state_history.values()))
    dep_var_array = np.vstack(list(dependent_variables_to_save.values()))

    moon_idx = 0
    moon_states[case] = {}
    for moon in bodies_to_propagate: # note moon includes juice asw
        moon_states[case][moon] = state_array[:, moon_idx: moon_idx + 6]
        moon_idx += 6

    save2txt(
        solution=state_history, filename=f"Q1_State_History_Case{case}.dat", directory="./Results/Q1",
    )

    save2txt(
        solution=dependent_variables_to_save, filename=f"Q1_Dependent_Variables_Case{case}.dat", directory="./Results/Q1/"
    )

    state_array = np.vstack(list(state_history.values()))
    print(state_array.shape)

    # note: case 3 and 4 tho prints sim vals they have a small difference in their values this value can be useful while
    #       writing the report - don't forget!

# ......................................................................................................................
#                                                 PLOTS
# ......................................................................................................................

def rsw(case, moon_name):
    if case <= 4:
        inertial_state_a = moon_states[case][moon_name]
        inertial_state_b = moon_states[case + 1][moon_name]

    else:
        inertial_state_a = moon_states[case]["JUICE"]
        inertial_state_b = moon_states[case + 1]["JUICE"]

    inertial_state_diff = inertial_state_b - inertial_state_a
    inertial_position_diff = inertial_state_diff[:, :3]
    rsw_diff = np.zeros_like(inertial_position_diff)
    for t in range(len( inertial_position_diff )):
        rot_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(inertial_state_a[t]) # WHY REF STATE NOT DIFF????
        rsw_diff[t] = rot_matrix @ inertial_position_diff[t]

    return rsw_diff


plot_labels = {
    1 : "I",
    2 : "II",
    3 : "III",
    4 : "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
}


for case in [1,2,3,4,5,6,7]:

    if case !=4:
        time_sec = np.array(list(propagation_results[case].state_history.keys()))
        time_days = [((t - simulation_start_epoch) / constants.JULIAN_DAY) for t in time_sec]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9)) if case<=4 else plt.subplots(1, 1, figsize=(10, 10))
        axes = axes.flatten() if case <= 4 else [axes]

        for i, moon in (enumerate(bodies_to_propagate_moons) if case <= 4 else enumerate(["JUICE"])):

            rsw_diff = rsw(case, moon)
            radial =  rsw_diff[:,0]
            along_track = rsw_diff[:,1]
            cross_track = rsw_diff[:,2]

            axes[i].plot(time_days, radial, label = "Radial Component (R) [m]", color = palette["pastel_pink"])
            axes[i].plot(time_days, along_track, label="Along Track Component (S) [m]", color = palette["pastel_purple"])
            axes[i].plot(time_days, cross_track, label="Cross Track Component (W) [m]", color = palette["pastel_teal"])
            if case>4 and case<7:
                axes[i].axhline(0.1, linewidth = 0.7, linestyle = "--", color = "red", label = "Radial Track Component Limit")
                axes[i].axhline(2, linewidth =0.7, linestyle = "--", color = "green", label = "Along Track Component Limit")
                axes[i].axhline(1, linewidth =0.7, linestyle = "--", color = "blue", label = "Cross Track Component Limit")
                axes[i].set_xlabel("Time [days]")
            axes[i].set_ylabel("Position Difference [m]")
            axes[i].set_title(moon)
            axes[i].grid(True, linestyle = "--", linewidth = 0.3, color = "gray")
            axes[i].legend() if case<4 else axes[i].legend(fontsize = "12")
            # axes[i].set_yscale("symlog")

        juice_inc = " JUICE Relative to Ganymede" if case>4  else ""
        fig.suptitle("RSW Plot for Case " + plot_labels[case] + " & " + plot_labels[case + 1] + juice_inc)
        plt.savefig(f"./Results/Q1/RSW_Plot_Case{case}.png", dpi=300, bbox_inches="tight")
        plt.tight_layout()
        plt.show()

