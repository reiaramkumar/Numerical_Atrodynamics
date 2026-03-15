""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

import numpy as np
from tudatpy import constants, numerical_simulation
from tudatpy.astro import element_conversion, two_body_dynamics
from tudatpy.data import save2txt
from tudatpy.interface import spice
from tudatpy import plotting
from tudatpy.numerical_simulation import (
    environment,
    environment_setup,
    estimation,
    estimation_setup,
    propagation,
    propagation_setup,)

# Define departure/arrival epoch - in seconds since J2000
departure_epoch = 2132.212895 * constants.JULIAN_DAY
time_of_flight = 157.9635921 * constants.JULIAN_DAY
arrival_epoch = departure_epoch + time_of_flight
target_body = 'Mars'
global_frame_orientation = "ECLIPJ2000"
fixed_step_size = 3600.0

################ HELPER FUNCTIONS: DO NOT MODIFY ########################################


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def write_propagation_results_to_file(
    dynamics_simulator: numerical_simulation.SingleArcSimulator,
    lambert_arc_ephemeris: environment.Ephemeris,
    file_output_identifier: str,
    output_directory: str,
) -> None:
    """
    This function will write the results of a numerical propagation, as well as the Lambert arc states at the epochs of the
    numerical state history, to a set of files. Two files are always written when calling this function (numerical state history, a
    and Lambert arc state history). If any dependent variables are saved during the propagation, those are also saved to a file

    Parameters
    ----------
    dynamics_simulator : numerical_simulation.SingleArcSimulator
        Object that was used to propagate the dynamics, and which contains the numerical state and dependent variable results

    lambert_arc_ephemeris : environment.Ephemeris
        Lambert arc state model as returned by the get_lambert_problem_result() function

    file_output_identifier : str
        Name that will be used to correctly save the output data files

    output_directory : str
        Directory to which the files will be written

    Files written
    -------------

    <output_directory/file_output_identifier>_numerical_states.dat
    <output_directory/file_output_identifier>_dependent_variables.dat
    <output_directory/file_output_identifier>_lambert_states.dat


    Return
    ------
    None

    """

    propagation_results = dynamics_simulator.propagation_results

    # Save numerical states
    state_history = propagation_results.state_history
    save2txt(
        solution=state_history,
        filename=output_directory + file_output_identifier + "_numerical_states.dat",
        directory="./",
    )

    # Save dependent variables
    dependent_variables = propagation_results.dependent_variable_history
    if len(dependent_variables.keys()) > 0:
        save2txt(
            solution=dependent_variables,
            filename=output_directory
            + file_output_identifier
            + "_dependent_variables.dat",
            directory="./",
        )

    # Save Lambert arc states
    lambert_arc_states = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

    save2txt(
        solution=lambert_arc_states,
        filename=output_directory + file_output_identifier + "_lambert_states.dat",
        directory="./",
    )

    return


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_lambert_problem_result(
    bodies: environment.SystemOfBodies,
    target_body: str,
    departure_epoch: float,
    arrival_epoch: float,
) -> environment.Ephemeris:
    """
    This function solved Lambert's problem for a transfer from Earth (at departure epoch) to
    a target body (at arrival epoch), with the states of Earth and the target body defined
    by ephemerides stored inside the SystemOfBodies object (bodies). Note that this solver
    assumes that the transfer departs/arrives to/from the center of mass of Earth and the target body

    Parameters
    ----------
    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    target_body : str
        The name (string) of the body to which the Lambert arc is to be computed

    departure_epoch : float
        Epoch at which the departure from Earth's center of mass is to take place

    arrival_epoch : float
        Epoch at which the arrival at he target body's center of mass is to take place

    Return
    ------
    Ephemeris object defining a purely Keplerian trajectory. This Keplerian trajectory defines the transfer
    from Earth to the target body according to the inputs to this function. Note that this Ephemeris object
    is valid before the departure epoch, and after the arrival epoch, and simply continues (forwards and backwards)
    the unperturbed Sun-centered orbit, as fully defined by the unperturbed transfer arc
    """

    # Gravitational parameter of the Sun
    central_body_gravitational_parameter = bodies.get_body(
        "Sun"
    ).gravitational_parameter

    # Set initial and final positions for Lambert targeter
    initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="Earth",
        observer_body_name="Sun",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=departure_epoch,
    )

    final_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name=target_body,
        observer_body_name="Sun",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=arrival_epoch,
    )

    # Create Lambert targeter
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3],
        final_state[:3],
        arrival_epoch - departure_epoch,
        central_body_gravitational_parameter,
    )

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state = initial_state
    lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()

    # Compute Keplerian state of Lambert arc
    lambert_arc_keplerian_elements = element_conversion.cartesian_to_keplerian(
        lambert_arc_initial_state, central_body_gravitational_parameter
    )

    # Setup Keplerian ephemeris model that describes the Lambert arc
    kepler_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(
            lambert_arc_keplerian_elements,
            departure_epoch,
            central_body_gravitational_parameter,
        ),
        "",  # for keplerian ephemeris, this argument does not have an effect
    )

    return kepler_ephemeris


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_lambert_arc_history(
    lambert_arc_ephemeris: environment.Ephemeris, simulation_result: dict
) -> dict:
    """
    This function extracts the state history (as a dict with time as keys, and Cartesian states as values)
    from an Ephemeris object defined by a lambert solver. This function takes a dictionary of states (simulation_result)
    as input, iterates over the keys of this dict (which represent times) to ensure that the times
    at which this function returns the states of the lambert arcs are identical to those at which the
    simulation_result has (numerically calculated) states


    Parameters
    ----------
    lambert_arc_ephemeris : environment.Ephemeris
        Ephemeris object from which the states are to be extracted

    simulation_result : dict
        Dictionary of (numerically propagated) states, from which the keys
        are used to determine the times at which this function is to extract states
        from the lambert arc

    Return
    ------
    Dictionary of Cartesian states of the lambert arc, with the keys (epochs) being the same as those of the input
    simulation_result and the corresponding Cartesian states of the Lambert arc.
    """

    lambert_arc_states = dict()
    for epoch in simulation_result:
        lambert_arc_states[epoch] = lambert_arc_ephemeris.cartesian_state(epoch)

    return lambert_arc_states


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def propagate_trajectory(
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    bodies: environment.SystemOfBodies,
    lambert_arc_ephemeris: environment.Ephemeris,
    use_perturbations: bool,
    initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
) -> numerical_simulation.SingleArcSimulator:
    """
    This function will be repeatedly called throughout the assignment. Propagates the trajectory based
    on several input parameters

    Parameters
    ----------
    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    lambert_arc_ephemeris : environment.Ephemeris
        Lambert arc state model as returned by the get_lambert_problem_result() function

    use_perturbations : bool
        Boolean to indicate whether a perturbed (True) or unperturbed (False) trajectory
        is propagated

    initial_state_correction : np.ndarray, default=np.array([0, 0, 0, 0, 0, 0])
        Cartesian state which is added to the Lambert arc state when computing the numerical initial state

    Return
    ------
    Dynamics simulator object from which the state- and dependent variable history can be extracted

    """

    # Compute initial state along Lambert arc (and apply correction if needed)
    lambert_arc_initial_state = (
        lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction
    )

    # Get propagator settings for perturbed/unperturbed forwards/backwards arcs
    if use_perturbations:
        propagator_settings = get_perturbed_propagator_settings(
            bodies, lambert_arc_initial_state, initial_time, termination_condition
        )

    else:
        propagator_settings = get_unperturbed_propagator_settings(
            bodies, lambert_arc_initial_state, initial_time, termination_condition
        )

    # Propagate dynamics with required settings
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )

    return dynamics_simulator


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def propagate_variational_equations(
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    bodies: environment.SystemOfBodies,
    lambert_arc_ephemeris: environment.Ephemeris,
    initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
) -> numerical_simulation.SingleArcVariationalSimulator:
    """
    Propagates the variational equations for a given range of epochs for a perturbed trajectory.

    Parameters
    ----------
    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    lambert_arc_ephemeris : environment.Ephemeris
        Lambert arc state model as returned by the get_lambert_problem_result() function

    initial_state_correction : np.ndarray, default=np.array([0, 0, 0, 0, 0, 0])
        Cartesian state which is added to the Lambert arc state when computing the numerical initial state

    Return
    ------
    Variational equations solver object, from which the state-, state transition matrix-, and
    sensitivity matrix history can be extracted.
    """

    # Compute initial state along Lambert arc
    lambert_arc_initial_state = (
        lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction
    )

    # Get propagator settings
    propagator_settings = get_perturbed_propagator_settings(
        bodies,
        lambert_arc_initial_state,
        initial_time,
        termination_condition,
    )

    # Define parameters for variational equations
    sensitivity_parameters = get_sensitivity_parameter_set(propagator_settings, bodies)

    # Propagate variational equations
    variational_equations_solver = (
        numerical_simulation.create_variational_equations_solver(
            bodies, propagator_settings, sensitivity_parameters
        )
    )

    return variational_equations_solver


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_sensitivity_parameter_set(
    propagator_settings: propagation_setup.propagator.PropagatorSettings,
    bodies: environment.SystemOfBodies,
) -> numerical_simulation.estimation.EstimatableParameterSet:
    """
    Function creating the parameters for which the variational equations are to be solved.

    Parameters
    ----------
    propagator_settings : propagation_setup.propagator.PropagatorSettings
        Settings used for the propagation of the dynamics

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    Return
    ------
    Propagation settings of the unperturbed trajectory.
    """
    parameter_settings = estimation_setup.parameter.initial_states(
        propagator_settings, bodies
    )

    return estimation_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings
    )


################ HELPER FUNCTIONS: MODIFY ########################################


# STUDENT CODE TASK - full function (except signature and return)
def get_unperturbed_propagator_settings(
    bodies: environment.SystemOfBodies,
    initial_state: np.ndarray,
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """
    Creates the propagator settings for an unperturbed trajectory.

    Parameters
    ----------
    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    initial_state : np.ndarray
        Cartesian initial state of the vehicle in the simulation

    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    Return
    ------
    Propagation settings of the unperturbed trajectory.
    """

    # Create propagation settings.

    central_bodies = ['Sun']
    bodies_to_propagate = ['Spacecraft']
                                                                        # should mars/venus be added?
    acceleration_settings_on_vehicle = {'Sun' : [propagation_setup.acceleration.point_mass_gravity()]}

    acceleration_settings = {'Spacecraft': acceleration_settings_on_vehicle}

    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies
    )

    fixed_time_step = 10.0
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set = propagation_setup.integrator.CoefficientSets.rk_4)




    simulation_start_epoch = initial_time

    system_initial_state = initial_state

    simulation_end_epoch = termination_condition


    # termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

    dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('Spacecraft', 'Sun')]


    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        initial_time,
        integrator_settings,
        termination_condition,
        output_variables = dependent_variables_to_save
    )


    return propagator_settings


def create_simulation_bodies() -> environment.SystemOfBodies:
    """
    Creates the body objects required for the simulation, using the
    environment_setup.create_system_of_bodies for natural bodies,
    and manual definition for vehicles

    Parameters
    ----------
    none

    Return
    ------
    Body objects required for the simulation.

    """

    bodies_to_create = ['Sun', 'Earth', 'Venus']
    global_frame_origin = 'Sun'
    global_frame_orientation = 'ECLIPJ2000'
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation
    )
    body_settings.add_empty_settings('Spacecraft')
    bodies = environment_setup.create_system_of_bodies(body_settings)
    return bodies
