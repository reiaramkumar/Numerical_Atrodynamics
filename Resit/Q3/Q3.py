
# IMPORTS
import math
import numpy as np
from matplotlib import pyplot as plt
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, environment, propagation_setup, propagation, simulator
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.astro.time_representation import DateTime

from Q3_helper import *


## Aerodynamic guidance class

# Create a class for the aerodynamic guidance of the STS, inheriting from 'propagation.AerodynamicGuidance'
class STSAerodynamicGuidance:

    def __init__(self, bodies: environment.SystemOfBodies):

        # Extract the STS and Earth bodies
        self.vehicle = bodies.get_body("STS")
        self.earth = bodies.get_body("Earth")

        # Extract the STS flight conditions, angle calculator, and aerodynamic coefficient interface
        environment_setup.add_flight_conditions( bodies, 'STS', 'Earth' )
        self.vehicle_flight_conditions = bodies.get_body("STS").flight_conditions
        self.aerodynamic_angle_calculator = self.vehicle_flight_conditions.aerodynamic_angle_calculator
        self.aerodynamic_coefficient_interface = self.vehicle_flight_conditions.aerodynamic_coefficient_interface

        self.current_time = float("NaN")

    def getAerodynamicAngles(self, current_time: float):
        self.updateGuidance( current_time )
        return np.array([self.angle_of_attack, 0.0, self.bank_angle])

    # Function that is called at each simulation time step to update the ideal bank angle of the vehicle
    def updateGuidance(self, current_time: float):

        if( math.isnan( current_time ) ):
            self.current_time = float("NaN")
        elif( current_time != self.current_time ):
            # Get the (constant) angular velocity of the Earth body
            earth_angular_velocity = np.linalg.norm(self.earth.body_fixed_angular_velocity)
            # Get the distance between the vehicle and the Earth bodies
            earth_distance = np.linalg.norm(self.vehicle.position)
            # Get the (constant) mass of the vehicle body
            body_mass = self.vehicle.mass

            # Extract the current Mach number, airspeed, and air density from the flight conditions
            mach_number = self.vehicle_flight_conditions.mach_number
            airspeed = self.vehicle_flight_conditions.airspeed
            density = self.vehicle_flight_conditions.density

            # Set the current Angle of Attack (AoA). The following line enforces the followings:
            # * the AoA is constant at 40deg when the Mach number is above 12
            # * the AoA is constant at 10deg when the Mach number is below 6
            # * the AoA varies close to linearly when the Mach number is between 12 and 6
            # * a Logistic relation is used so that the transition in AoA between M=12 and M=6 is smoother
            self.angle_of_attack = np.deg2rad(30 / (1 + np.exp(-2*(mach_number-9))) + 10)

            # Update the variables on which the aerodynamic coefficients are based (AoA and Mach)
            current_aerodynamics_independent_variables = [self.angle_of_attack, mach_number]

            # Update the aerodynamic coefficients
            self.aerodynamic_coefficient_interface.update_coefficients(
                current_aerodynamics_independent_variables, current_time)

            # Extract the current force coefficients (in order: C_D, C_S, C_L)
            current_force_coefficients = self.aerodynamic_coefficient_interface.current_force_coefficients
            # Extract the (constant) reference area of the vehicle
            aerodynamic_reference_area = self.aerodynamic_coefficient_interface.reference_area

            # Get the heading, flight path, and latitude angles from the aerodynamic angle calculator
            heading = self.aerodynamic_angle_calculator.get_angle(environment_setup.aerodynamic_coefficients.AerodynamicsReferenceFrameAngles.heading_angle)
            flight_path_angle = self.aerodynamic_angle_calculator.get_angle(environment_setup.aerodynamic_coefficients.AerodynamicsReferenceFrameAngles.flight_path_angle)
            latitude = self.aerodynamic_angle_calculator.get_angle(environment_setup.aerodynamic_coefficients.AerodynamicsReferenceFrameAngles.latitude_angle)

            # Compute the acceleration caused by Lift
            lift_acceleration = 0.5 * density * airspeed ** 2 * aerodynamic_reference_area * current_force_coefficients[2] / body_mass
            # Compute the gravitational acceleration
            downward_gravitational_acceleration = self.earth.gravitational_parameter / (earth_distance ** 2)
            # Compute the centrifugal acceleration
            spacecraft_centrifugal_acceleration = airspeed ** 2 / earth_distance
            # Compute the Coriolis acceleration
            coriolis_acceleration = 2 * earth_angular_velocity * airspeed * np.cos(latitude) * np.sin(heading)
            # Compute the centrifugal acceleration from the Earth
            earth_centrifugal_acceleration =    earth_angular_velocity ** 2 * earth_distance * np.cos(latitude) * (
                                                np.cos(latitude) * np.cos(flight_path_angle) + np.sin(flight_path_angle) * np.sin(
                                                latitude) * np.cos(heading))

            # if math.isnan(lift_acceleration) or abs(lift_acceleration) < 1e-6:
            #     self.bank_angle = 0.0
            # else:
            #     # Compute the cosine of the ideal bank angle
            #     cosine_of_bank_angle = ((downward_gravitational_acceleration - spacecraft_centrifugal_acceleration) * np.cos(flight_path_angle) - coriolis_acceleration - earth_centrifugal_acceleration) / lift_acceleration
            #     # If the cosine lead to a value out of the [-1, 1] range, set to bank angle to 0deg or 180deg
            #     if (cosine_of_bank_angle < -1):
            #         self.bank_angle = np.pi
            #     elif (cosine_of_bank_angle > 1):
            #         self.bank_angle = 0.0
            #     else:
            #         # If the cos is in the correct range, return the computed bank angle
            #         self.bank_angle = np.arccos(cosine_of_bank_angle)
            # self.current_time = current_time

            numerator = (
                    (
                            downward_gravitational_acceleration
                            - spacecraft_centrifugal_acceleration
                    )
                    * np.cos(flight_path_angle)
                    - coriolis_acceleration
                    - earth_centrifugal_acceleration
            )

            if math.isnan(lift_acceleration) or abs(lift_acceleration) < 1e-12:
                cosine_of_bank_angle = np.nan
                self.bank_angle = 0.0
                saturation_reason = "negligible/invalid lift"

            else:
                cosine_of_bank_angle = numerator / lift_acceleration

                if cosine_of_bank_angle < -1.0:
                    self.bank_angle = np.pi
                    saturation_reason = "cos(bank) < -1"

                elif cosine_of_bank_angle > 1.0:
                    self.bank_angle = 0.0
                    saturation_reason = "cos(bank) > 1"

                else:
                    self.bank_angle = np.arccos(cosine_of_bank_angle)
                    saturation_reason = "valid bank angle"

            # print(
            #     f"t = {current_time - simulation_start_epoch:9.2f} s | "
            #     f"h = {(earth_distance - self.earth.shape_model.average_radius) / 1e3:8.2f} km | "
            #     f"rho = {density:.3e} kg/m³ | "
            #     f"M = {mach_number:6.2f} | "
            #     f"CL = {current_force_coefficients[2]:8.4f} | "
            #     f"L/m = {lift_acceleration:10.4e} m/s² | "
            #     f"numerator = {numerator:10.4e} m/s² | "
            #     f"cos(bank) = {cosine_of_bank_angle:10.4f} | "
            #     f"bank = {np.rad2deg(self.bank_angle):7.2f} deg | "
            #     f"{saturation_reason}"
            # )



# ..... Environment .....
spice.load_standard_kernels()
simulation_start_date = DateTime(2000, 1, 1, 1, 40)
simulation_start_epoch = DateTime(2000, 1, 1, 1, 40).to_epoch()
max_simulation_time = 3*constants.JULIAN_DAY # Set the maximum simulation time (avoid very long skipping re-entry)

# ..... Bodies ......
bodies_to_create = ["Earth"]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# ..... Vehicle .....
body_settings.add_empty_settings("STS")
body_settings.get("STS").constant_mass = 5000

# ..... Aerodynamic Coeff .....
aero_coefficients_files = {0: "input/STS_CD.dat", 2:"input/STS_CL.dat"}
# Setup the aerodynamic coefficients settings tabulated from the files
coefficient_settings = environment_setup.aerodynamic_coefficients.tabulated_force_only_from_files(
    force_coefficient_files=aero_coefficients_files,
    reference_area=2690.0*0.3048*0.3048,
    independent_variable_names=[environment_setup.aerodynamic_coefficients.AerodynamicCoefficientsIndependentVariables.angle_of_attack_dependent, environment_setup.aerodynamic_coefficients.AerodynamicCoefficientsIndependentVariables.mach_number_dependent],
)
# Add predefined aerodynamic coefficients database to the body
body_settings.get("STS").aerodynamic_coefficient_settings = coefficient_settings
bodies = environment_setup.create_system_of_bodies(body_settings)

# ### Add rotation model based on aerodynamic guidance
# Create the aerodynamic guidance object
aerodynamic_guidance_object = STSAerodynamicGuidance(bodies)
rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
    'Earth', '', 'STS_Fixed', aerodynamic_guidance_object.getAerodynamicAngles )
environment_setup.add_rotation_model( bodies, 'STS', rotation_model_settings )


bodies_to_propagate = ["STS"]
central_bodies = ["Earth"]

accelerations_settings_STS = dict(
    Earth=[
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.aerodynamic(),
    ]
)

acceleration_settings = {"STS": accelerations_settings_STS}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)



# Set the initial state of the STS as spherical elements, and convert them to a cartesian state
initial_radial_distance = bodies.get_body("Earth").shape_model.average_radius + 120e3

# Convert the initial state
initial_earth_fixed_state = element_conversion.spherical_to_cartesian_elementwise(
    radial_distance=initial_radial_distance,
    latitude=np.deg2rad(20),
    longitude=np.deg2rad(140),
    speed=7.5e3,
    flight_path_angle=np.deg2rad(-0.6),
    heading_angle=np.deg2rad(15),
)

# Convert the state from the Earth-fixed frame to the inertial frame
earth_rotation_model = bodies.get_body("Earth").rotation_model
initial_state = environment.transform_to_inertial_orientation(
    initial_earth_fixed_state, simulation_start_epoch, earth_rotation_model
)


# Define the list of dependent variables to save during the propagation
dependent_variables_to_save = [
    propagation_setup.dependent_variable.flight_path_angle("STS", "Earth"),
    propagation_setup.dependent_variable.altitude("STS", "Earth"),
    propagation_setup.dependent_variable.bank_angle("STS", "Earth"),
    propagation_setup.dependent_variable.angle_of_attack("STS", "Earth"),
    propagation_setup.dependent_variable.aerodynamic_force_coefficients("STS"),
    propagation_setup.dependent_variable.airspeed("STS", "Earth"),
    propagation_setup.dependent_variable.total_acceleration_norm("STS"),
    propagation_setup.dependent_variable.mach_number("STS", "Earth")
]




# Define a termination conditions to stop once altitude goes below 25 km
altitude_limit = 25.0e3
termination_altitude_settings = (
    propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.altitude(
            "STS", "Earth"
        ),
        limit_value=altitude_limit,
        use_as_lower_limit=True,
    )
)
# Define a termination condition to stop after a given time (to avoid an endless skipping re-entry)
termination_time_settings = propagation_setup.propagator.time_termination(simulation_start_epoch + max_simulation_time)
# Combine the termination settings to stop when one of them is fulfilled
termination_conditions = [termination_altitude_settings, termination_time_settings]
# Add string representations of the termination conditions
termination_conditions_repr = [
    f"Altitude Termination at {altitude_limit/1e3}km",
    "Time Termination",
]

combined_termination_settings = propagation_setup.propagator.hybrid_termination(
    termination_conditions, fulfill_single_condition=True
)

# ......................................................................................................................
#                                                       PROPAGATOR ROUTINES
# ......................................................................................................................


results = {}

benchmark_step_size = 0.03125 # not sure
step_sizes = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
variable_tolerances = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
fixed_integrator_config = [
    ("rk4", propagation_setup.integrator.rkf_45, propagation_setup.integrator.lower),
    ("rk6", propagation_setup.integrator.rkf_56, propagation_setup.integrator.higher),
    ("rk8", propagation_setup.integrator.rkf_78, propagation_setup.integrator.higher),
]
variable_integrator_config = [
    ("rkf45", propagation_setup.integrator.rkf_45),
    ("rkf56", propagation_setup.integrator.rkf_56),
    ("rkf78", propagation_setup.integrator.rkf_78)
]

no_of_fn_evals_per_step = {
    'rk4':             4,
    'rk6':             8,
    'rk8':            13,
    'rkf45':           6,
    'rkf56':           8,
    'rkf78':          13,}



#... HALF STEP INTEGRATOR ROUTINE ...
results_half = {}
for label, coeff_set, order in fixed_integrator_config:
    results_half.setdefault(label, {})
    for step_size in step_sizes:
        integrator_settings_fixed = propagation_setup.integrator.runge_kutta_fixed_step_size(float(step_size), coeff_set, order)
        propagator_settings_fixed = propagation_setup.propagator.translational(
            central_bodies=["Earth"],
            acceleration_models=acceleration_models,
            bodies_to_integrate=["STS"],
            initial_states=initial_state,
            initial_time=simulation_start_epoch,
            integrator_settings=integrator_settings_fixed,
            termination_settings=combined_termination_settings, )

        simulator_fixed = simulator.create_dynamics_simulator(bodies, propagator_settings_fixed)
        state_history_fixed = simulator_fixed.propagation_results.state_history
        results_half[label][step_size] = state_history_fixed


max_errors_step_half = {}
position_error_norm_step_half = {}
for label, coeff_set, order in fixed_integrator_config:
    max_errors_step_half[label] = {}
    position_error_norm_step_half[label] = {}
    for step_size in step_sizes[1:]:
        full_step_state = results_half[label][step_size]
        half_step_state = results_half[label][step_size/2]
        task_1_errors = {}
        for epoch in full_step_state.keys():
            if epoch in half_step_state:
                task_1_errors[epoch] = full_step_state[epoch] - half_step_state[epoch]

        task_1_epochs = np.array(list(task_1_errors.keys()))
        times_hours = (task_1_epochs - simulation_start_epoch) / 3600.0

        state_difference = np.vstack(list(task_1_errors.values()))
        position_error_norm = np.linalg.norm(state_difference[:, :3], axis=1)
        position_error_norm_step_half[label][step_size] = (times_hours, position_error_norm)
        max_errors_step_half[label][step_size] = np.max(position_error_norm)

for label in max_errors_step_half:
    plt.figure(figsize=(9, 5))
    for step_size, (times_hours, position_error_norm) in position_error_norm_step_half[label].items():
        plt.plot(times_hours, position_error_norm, linewidth=1, label=f"dt={step_size}s")

    plt.legend()
    plt.title(f"Position Error Norm vs Time for {label} Integrator")
    plt.yscale("log")
    plt.xlabel("Time [hours]"), plt.ylabel("Position Error Norm [m]")
    plt.grid(True, alpha=0.5, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    plt.savefig(f"Q3_position_error_norm_{label}.png", dpi=300, bbox_inches="tight")
    plt.show()

plt.figure(figsize=(9, 5))
for label in max_errors_step_half:
        steps = sorted(max_errors_step_half[label].keys())
        max_pos_error = [max_errors_step_half[label][s] for s in steps]
        plt.plot(steps, max_pos_error, marker='o', linewidth=1, label=label)
plt.title("Maximum Position Error vs Step Size")
plt.legend()
plt.xlabel("Δt [s]"), plt.ylabel("Max Position Error [m]")
plt.yscale("log"), plt.xscale("log")
plt.grid(True, alpha = 0.5, linestyle='--', linewidth=0.3)
plt.tight_layout()
plt.savefig("Q3_max_position_error_vs_step_size.png", dpi=300, bbox_inches="tight")
plt.show()


# ... BENCHMARK CASE ...................................................................................................

benchmark_integrator_settings = get_fixed_step_size_integrator_settings(benchmark_step_size)

perturbed_propagator_settings = propagation_setup.propagator.translational(
    central_bodies=["Earth"],
    acceleration_models=acceleration_models,
    bodies_to_integrate=['STS'],
    initial_states=initial_state,
    initial_time=simulation_start_epoch,
    integrator_settings=benchmark_integrator_settings,  #
    termination_settings= combined_termination_settings
)

benchmark_dynamics_simulator = simulator.create_dynamics_simulator(bodies, perturbed_propagator_settings)
benchmark_state_history = benchmark_dynamics_simulator.propagation_results.state_history

interpolator_settings = interpolators.lagrange_interpolation(8)
benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
    benchmark_state_history,
    interpolator_settings,
)



# ... FIXED INTEGRATOR ROUTINE .........................................................................................

results = {}
for label, coeff_set, order in fixed_integrator_config:
    results[label] = {}
    for dt in step_sizes:
        integrator_settings_fixed = propagation_setup.integrator.runge_kutta_fixed_step_size(float(dt), coeff_set, order)
        propagator_settings_fixed = propagation_setup. propagator.translational(
            central_bodies = ["Earth"],
            acceleration_models = acceleration_models,
            bodies_to_integrate = ["STS"],
            initial_states = initial_state,
            initial_time = simulation_start_epoch,
            integrator_settings = integrator_settings_fixed,
            termination_settings = combined_termination_settings)

        aerodynamic_guidance_object.current_time = float("NaN")


        fixed_sim = simulator.create_dynamics_simulator(bodies, propagator_settings_fixed)
        state_history = fixed_sim.propagation_results.state_history
        common_epoch_states = {epoch: state_history[epoch] for epoch in state_history if epoch in benchmark_state_history}
        benchmark_difference = get_difference_wrt_benchmarks(common_epoch_states, benchmark_interpolator)
        state_difference = np.vstack(list(benchmark_difference.values()))
        position_error = np.linalg.norm(state_difference[:, :3], axis=1)
        max_position_error_norm = np.max(position_error)

        n_steps = len(state_history) - 1  # why? as no of steps correspond to +1 states
        n_fn_evals = n_steps * no_of_fn_evals_per_step[label]

        results[label][dt] = (max_position_error_norm, n_fn_evals)



# ... VARIABLE INTEGRATOR ROUTINE ......................................................................................

for label, coeff_set in variable_integrator_config:
    results[label] = {}
    for tol in variable_tolerances:
        step_size_control_settings = propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(tol, tol)
        step_size_validation_settings = propagation_setup.integrator.step_size_validation(1.0e-12, np.inf)
        integrator_settings_var = propagation_setup.integrator.runge_kutta_variable_step(10.0,
                                                                                     coeff_set,
                                                                                     step_size_control_settings,
                                                                                      step_size_validation_settings)

        propagator_settings_variable = propagation_setup. propagator.translational(
            central_bodies = ["Earth"],
            acceleration_models = acceleration_models,
            bodies_to_integrate = ["STS"],
            initial_states = initial_state,
            initial_time = simulation_start_epoch,
        integrator_settings = integrator_settings_var,
        termination_settings = combined_termination_settings,)

        aerodynamic_guidance_object.current_time = float("NaN")
        var_sim = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_variable)

        state_history = var_sim.propagation_results.state_history
        common_epoch_states = {epoch: state_history[epoch] for epoch in state_history if
                               epoch in benchmark_state_history}
        benchmark_difference = get_difference_wrt_benchmarks(common_epoch_states, benchmark_interpolator)
        state_difference = np.vstack(list(benchmark_difference.values()))
        position_error = np.linalg.norm(state_difference[:, :3], axis=1)
        max_position_error_norm = np.max(position_error)

        n_steps = len(state_history) - 1
        n_fn_evals = n_steps * no_of_fn_evals_per_step[label]

        epochs = np.array(list(benchmark_difference.keys()))
        time_hours = (epochs - simulation_start_epoch) / 3600.0
        results[label][tol] = (max_position_error_norm, n_fn_evals)

plt.figure(figsize=(9, 5))
marker_shapes = ["o", "s", "^", "D", "v", "P", "*"]

for label in results:
    function_evals = []
    max_errors = []
    settings = []
    for setting, (max_pos_error, function_eval) in results[label].items():
        function_evals.append(function_eval)
        max_errors.append(max_pos_error)
        settings.append(setting)

    linestyle = "-" if label in ["rk4", "rk6", "rk8"] else "--"
    line, = plt.plot(function_evals, max_errors, linestyle = linestyle, linewidth = 1, marker = None, label = f"{label}")

    for i, (x,y, setting) in enumerate(zip(function_evals, max_errors, settings)):
        if y>0:
            plt.plot(x, y, markersize = 6, marker = marker_shapes[i], linestyle = "None", color = line.get_color())
            if label == "rkf45":
                plt.annotate("1e-14, 1e-12 below numerical resolution", (1e4, 1e-10), xytext=(5, 5),
                             textcoords="offset points", fontsize=7, color=line.get_color())
            if label == "rkf56":
                plt.annotate("1e-14 below numerical resolution", (1e3, 1e-10), xytext=(5, -15),
                             textcoords="offset points", fontsize=7, color=line.get_color())
    main_legend = plt.legend(loc="upper left", title="Integrator", fontsize = 8, title_fontsize = 9, bbox_to_anchor=(1.02, 1))
    plt.gca().add_artist(main_legend)


marker_handles = []
for i, marker in enumerate(marker_shapes):
    label_text = (
        f"dt = {step_sizes[i]:g} / "
        f"tol = {variable_tolerances[i]:.0e}")
    marker_handle, = plt.plot([], [], marker = marker, linestyle = "None", color = "black", markersize = 6, label = label_text)
    marker_handles.append(marker_handle)


plt.title("Maximum Position Error vs Function Evaluations")
plt.legend(handles = marker_handles, loc = "lower left", title = "Step Size/Tolerance", fontsize = 8, title_fontsize = 9, bbox_to_anchor=(1.02, 0))
plt.yscale("log"), plt.xscale("log")
plt.ylim(1e-11, 1.0e4)
plt.xlabel("Number of Function Evaluations [-]"), plt.ylabel("Maximum Position Error [m]")
plt.grid(True, alpha=0.5, linestyle='--', linewidth=0.3)
plt.tight_layout()
plt.savefig("Q3_max_position_error_vs_function_evaluations.png", dpi=300, bbox_inches="tight")
plt.show()

for label in ["rkf45", "rkf56"]:
    for setting, (error, fn_eval) in results[label].items():
        print(
            label,
            f"tol={setting:.0e}",
            f"error={error:.16e}",
            f"fn={fn_eval}"
        )
#
