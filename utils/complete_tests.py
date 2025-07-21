from utils import graph_functions
import roadrunner
import pickle
import random
from copy import deepcopy
from collections import defaultdict
import numpy as np
import math

def get_initial_concentrations(filename, species_list):

    rr = roadrunner.RoadRunner(filename)
    initial_concentrations = {}
    for species in species_list:
        initial_concentrations[species] = rr[f"[{species}]"]

    return initial_concentrations


def simulate_until_convergence(sbml_path, species_list, threshold=1e-10, max_time=1e4, step_size=1.0):
    """
    Simulate the system until all species' concentrations change less than `threshold`
    between time steps of `step_size`.

    Parameters:
    - rr: roadrunner.RoadRunner object with loaded model
    - threshold: float, convergence threshold for species changes
    - max_time: float, maximum simulation time to avoid infinite loops
    - step_size: float, time step to evaluate convergence

    Returns:
    - result: dictionary where each key is a species having as value the concentration at the last step of the simulation
    """

    rr = roadrunner.RoadRunner(sbml_path)

    times = [0]
    values = []
    values.append([rr[f"[{species}]"] for species in species_list])
    t = 0

    while t < max_time:
        try:
            rr.simulate(t, t + step_size, 2)
        except Exception as e:
            return [None, None]

        t += step_size
        new_values = [rr[f"[{species}]"] for species in species_list]
        diff = np.abs(np.array(new_values) - np.array(values[-1]))

        if np.all(diff < threshold):
            print(f"Converged at time {t}")
            break

        values.append(new_values)
        times.append(t)
    else:
        print("Warning: Maximum time reached before convergence.")

    simulation_results = {}
    final_values = values[-1]

    # Map each species to the result coming from simulation
    for i, species in enumerate(species_list):
        simulation_results[species] = final_values[i]
    result = np.column_stack((times, np.array(values)))
    return simulation_results

def complete_tests2(
        sbml_path,
        altered_sbml_path,
        graph_description_path,
        weight_configuration):

    # Will store, for both normal and altered case, the real results (coming from simulation) and the
    # predicted ones (coming from propagation algorithm)
    results = {}
    results['normal'] = {'real': [], 'predicted': []}
    results['altered'] = {'real': [], 'predicted': []}

    # Start by monitoring the normal functioning case
    pickle_file = open(graph_description_path, 'rb')
    graph_description = pickle.load(pickle_file)
    for reaction in graph_description['reactions']:
        reaction.append(0)
    species_list = graph_description['species']
    altered_reaction = graph_description['altered']

    # Mark the altered reaction with a 2 as per the propagation algorithm rule
    altered_graph_description = deepcopy(graph_description)
    for reaction in altered_graph_description['reactions']:
        if reaction[0] == altered_reaction:
            reaction[1] = 2

    # Results coming from simulation: will serve as ground truth
    results['normal']['real'] = simulate_until_convergence(sbml_path, species_list)
    results['altered']['real'] = simulate_until_convergence(altered_sbml_path, species_list)

    initial_concentrations = get_initial_concentrations(sbml_path, species_list)
    results['initial_concentrations'] = initial_concentrations

    kinetic_constants = graph_description['reactions']
    #kinetic_constants

    # Configurations of weights:
    # 100 configurations with one value per reaction drawn uniformly at random in [-0.2,0.2]
    number_configurations = 0
    configurations = []
    while number_configurations < 100:
        configuration = []
        for reaction in kinetic_constants:
            configuration.append([reaction[0], (random.uniform(-0.2, 0.2))])
        configurations.append(configuration)
        number_configurations += 1

    # If a configuration is provided as argument use it
    if len(weight_configuration):
        configurations = weight_configuration

    # Test each weight configuration
    for configuration in configurations:
        species = []

        # Generate an instance of the pathway graph having such configuration of weights
        new_graph_description = graph_functions.graph_description_with_weights(altered_graph_description,                                                                           configuration)
        graph = graph_functions.generate_graph(new_graph_description)
        total_species = len(graph_description['species'])

        for i in range(100):
            cnt = 0
            species.append(graph_functions.update_graph(graph, 0, i + 1))

            # Stop if values do not change anymore
            if i > 0:
                for specie in graph_description['species']:
                    if math.fabs(species[i][specie] - species[i - 1][specie]) < 1e-5:
                        cnt += 1

            if cnt == total_species:
                #print('Stopped at iteration', i)
                break

        # Save the results coming from the last step of propagation
        results['altered']['predicted'].append([configuration, species[-1]])

    return results

# Each species will be assigned a list of values, compute the confidence interval
def compute_confidence_interval(values):

    scores = np.array(values)
    mean = np.mean(scores)
    std = np.std(scores)
    std_error = std / np.sqrt(len(scores))
    margin = 2 * std_error
    lower = mean - margin
    higher = mean + margin

    return [lower, higher]

# New way to compute interval based on percentile
def compute_confidence_interval_percentile(values, interval):
    scores = np.array(values)
    lower = np.percentile(scores, interval[0])
    higher = np.percentile(scores, interval[1])

    return [lower, higher]

def get_insights_confidence_intervals(alteration, benchmark_results, verbose, detailed_classification, threshold, ignore_uncertain):

    total_species = 0
    uncertain_species = 0

    # How much normal and uncertain species change due to alteration
    deltas_normal = []
    deltas_uncertain = []

    # Get the real values coming from the simulation
    normal_case = benchmark_results['normal']['real']
    altered_case = benchmark_results[alteration]['real']
    initial_concentrations = benchmark_results['initial_concentrations']
    expected_results = {}
    predictions = {}
    configurations_results = []
    # For each specie store the result coming from simulation and the one coming from the algorithm, the confidence intervals
    # and the deltas coming from simulation
    results = defaultdict(lambda: [0,0, [], []])
    # Will store, for each specie the lower and higher limit of the confidence
    propagation_results = defaultdict(lambda: [])

    # In benchmark_results['altered']['predicted'] we have 100 items (one per configuration)
    # each item is a tuple (configuration, results)
    for res in benchmark_results['altered']['predicted']:
        propagation_scores = res[-1]
        for specie in propagation_scores:
            propagation_results[specie].append(propagation_scores[specie])

    for specie in normal_case:

        total_species += 1

        # For each specie store two values: one indicating the number of experiments where
        # it increased and one indicating the number of experiments where it decreased
        predictions[specie] = [0, 0]

        # Differentiate between highly changed and the rest (unchanged)
        # Consider both the two deltas discussed if the initial concentration is not zero
        # Otherwise only focus on the ratio (at least 10 percent change)
        delta = 0
        lower_percentile = 2.5
        higher_percentile = 97.5
        lower, higher = compute_confidence_interval_percentile(propagation_results[specie], [lower_percentile, higher_percentile])

        while lower < 0.5 <= higher:
            lower_percentile += 2.5
            higher_percentile -= 2.5
            lower, higher = compute_confidence_interval_percentile(propagation_results[specie],
                                                                   [lower_percentile, higher_percentile])
            #print('out of interval, restricting confidence: ', lower_percentile, higher_percentile, lower, higher)
            if lower_percentile > 10:
                break

        if detailed_classification:
            if initial_concentrations[specie] != 0:
                delta1 = math.fabs((altered_case[specie] - normal_case[specie]) / initial_concentrations[specie])
                delta2 = math.fabs(1 - (altered_case[specie] / normal_case[specie]))
                delta = delta1 * delta2

            else:
                delta = math.fabs(1 - (altered_case[specie] / normal_case[specie])) / 100

            if delta >= threshold and altered_case[specie] > normal_case[
                specie]:  # Should be 0.1 but for some reason does not work
                expected_results[specie] = "increased"
                results[specie][0] = 1
            elif delta >= threshold and altered_case[specie] < normal_case[specie]:
                expected_results[specie] = "decreased"
                results[specie][0] = -1
            else:
                expected_results[specie] = "unchanged"
                results[specie][0] = 0
        else:
            if altered_case[specie] > normal_case[specie]:
                expected_results[specie] = "increased"
                results[specie][0] = 1
            else:
                expected_results[specie] = "decreased"
                results[specie][0] = -1

        results[specie][3] = [altered_case[specie] - normal_case[specie], delta]

        # The first field is the classification result
        # The second is the average of the extremes used in case it's uncertain
        if lower >= 0.5:
            results[specie][1] = [1, (lower + higher) / 2]
            deltas_normal.append(results[specie][3])
        if higher <= 0.5:
            results[specie][1] = [-1, (lower + higher) / 2]
            deltas_normal.append(results[specie][3])

        if lower == 0.5 and higher == 0.5:
            total_species -= 1
            results[specie][1] = [0, 0.5]
        if lower_percentile > 10:
            deltas_uncertain.append(results[specie][3])
            uncertain_species += 1
            avg_intervals = (lower + higher) / 2
            results[specie][1] = [0, avg_intervals]
            if not ignore_uncertain:
                if avg_intervals >= 0.5:
                    results[specie][1] = [1, avg_intervals]
                    #print(specie, 'uncertain set to increase')
                else:
                    #print(specie, 'uncertain set to decrease')
                    results[specie][1] = [-1, avg_intervals]
            else:
                # No prediction is made so remove from list of predicted species
                total_species -= 1

        # To investigate more on the crossing-boundary species
        #if lower < 0.5 <= higher:
            #print(specie, ' crossed boundary  ', 'normal: ', normal_case[specie], ' altered: ', altered_case[specie], ' delta: ', delta)

        results[specie][2] = [lower, higher]

    if verbose == 1:
        for specie in results.keys():
            print('Specie: ', specie, 'expected result: ', expected_results[specie], ' normal final: ', normal_case[specie], ' altered final: ', altered_case[specie])
            print('confidence interval: ', results[specie][2])
            print('deltas coming from simulation: ', results[specie][3])

    return results, total_species, uncertain_species, deltas_normal, deltas_uncertain

# Detailed_classification is a boolean value indicating whether to focus only on change (up or down)
# Or to also consider magnitude (so have also an unchanged class)
def get_insigths(alteration, benchmark_results, verbose, detailed_classification, threshold):

    # Get how the species should change
    normal_case = benchmark_results['normal']['real']
    altered_case = benchmark_results[alteration]['real']
    initial_concentrations = benchmark_results['initial_concentrations']
    expected_results = {}
    predictions = {}
    configurations_results = []

    # For each specie store in the first position the state after simulation (to better figure out deltas)
    # the state after propagation
    # the average, std, min and max of the values coming from propagation
    results = defaultdict(lambda: [0,0,[]])

    # Stores the results according to the new accuracy metric based on confidence intervals
    results_confidence_intervals = {}

    i = 0

    for specie in normal_case:

        # For each specie store two values: one indicating the number of experiments where
        # it increased and one indicating the number of experiments where it decreased
        predictions[specie] = [0, 0]

        # Differentiate between highly changed and the rest (unchanged)
        # Consider both the two deltas discussed if the initial concentration is not zero
        # Otherwise only focus on the ratio (at least 10 percent change)
        delta = 0

        if detailed_classification:
            if initial_concentrations[specie] != 0:
                delta1 = math.fabs((altered_case[specie] - normal_case[specie]) / initial_concentrations[specie])
                delta2 = math.fabs(1 - (altered_case[specie] / normal_case[specie]))
                delta = delta1 * delta2
            else:
                delta = math.fabs(1 - (altered_case[specie] / normal_case[specie])) / 100

            if delta >= threshold and altered_case[specie] > normal_case[
                specie]:  # Should be 0.1 but for some reason does not work
                expected_results[specie] = "increased"
                results[specie][0] = 1
            elif delta >= threshold and altered_case[specie] < normal_case[specie]:
                expected_results[specie] = "decreased"
                results[specie][0] = -1
            else:
                expected_results[specie] = "unchanged"
                results[specie][0] = 0
        else:
            if altered_case[specie] > normal_case[specie]:
                expected_results[specie] = "increased"
                results[specie][0] = 1
            else:
                expected_results[specie] = "decreased"
                results[specie][0] = -1

        #print('specie ', specie, ' delta: ', altered_case[specie] - normal_case[specie])

    # For every specie store every value coming from propagation (will be used to compute statistics)
    propagation_values = defaultdict(lambda: [])

    predicted_results = benchmark_results[alteration]['predicted']
    # One for each configuration of the weights
    for predicted_result in predicted_results:
        conf_res = [predicted_result[0], 0] # The second element stores the number of errors
        values = predicted_result[1]

        confidence_intervals = compute_confidence_interval(values)



        for specie in values:

            propagation_values[specie].append(values[specie])

            if values[specie] > 0.5:

                results[specie][1] += 1
                predictions[specie][1] += 1
                if expected_results[specie] == "decreased":
                    conf_res[1] += 1

            else:
                results[specie][1] -= 1
                predictions[specie][0] += 1
                if expected_results[specie] == "increased":
                    conf_res[1] += 1

        configurations_results.append(conf_res)

    highly_mutated = defaultdict(lambda: [])

    for specie in results.keys():
        if results[specie][0] != 0:
            highly_mutated[specie].append(results[specie][0])
            highly_mutated[specie].append(results[specie][1])

    if verbose:
        print('expected results: ', expected_results)
        for specie in predictions:
            print('the specie ', specie, 'increased ', predictions[specie][1], 'times')
            print('the specie ', specie, 'decreased ', predictions[specie][0], 'times')


    total_species = len(highly_mutated.keys())
    correct_species = 0

    propagation_statistic = defaultdict(lambda: [])

    for s in highly_mutated.keys():
        if highly_mutated[s][0] == 1 and highly_mutated[s][1] > 0:
            correct_species += 1
        if highly_mutated[s][0] == -1 and highly_mutated[s][1] <= 0:
            correct_species += 1


    for s in propagation_values.keys():

        if s in list(highly_mutated.keys()):
            real_res = highly_mutated[s][0]
        else:
            real_res = 0

        propagation_statistic[s] = [
            real_res,
            np.average(np.array(propagation_values[s])),
            np.std(np.array(propagation_values[s])),
            np.min(np.array(propagation_values[s])),
            np.max(np.array(propagation_values[s])),
            propagation_values[s]

        ]

    if total_species != 0:
        correct_ratio = correct_species / total_species
    else:
        correct_ratio = -1

    return [configurations_results, correct_ratio, propagation_statistic, highly_mutated]
