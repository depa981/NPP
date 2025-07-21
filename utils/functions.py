import random
import roadrunner
import libsbml
import networkx as nx
import matplotlib.pyplot as plt
import math
import pickle


# To compute specie nodes
def sigmoid(x):
    return 1 / (1 + math.exp(-(4*(x -0.5))))


def logit(x):
    return math.log(x / (1-x)) / 4 + 0.5

def sign(x):
    if x == 0:
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1


# Test use instead of exponential which is not symmetric
def tanh(x):
    return math.tanh(x) + 1


def load_sbml(filename):

    reader = libsbml.SBMLReader()
    document = reader.readSBML(filename)
    if document.getNumErrors() > 0:
        print("Errors occurred while loading the SBML file.")
        return None
    return document.getModel()


def get_list_parameters(model):
    parameters = model.getListOfParameters()[0]
    print('parameters', parameters)


def parse_formula(formula, model):
    #  print('formula original', formula)
    formula = formula.replace('(', '')
    formula = formula.replace(')', '')
    # formula = formula.replace('-', '')
    formula = formula.replace('*', '')
    formula = formula.split()
    formula.pop(0)
    # print('formula parsed', formula)

    rates = []

    for parameter in formula:
        if 'k' in parameter:
            rates.append(parameter)
            # print('rate', model.getParameter(parameter).getValue())
    return rates
def generate_pathway(input_filename, output_filename, alterations):

    #directory = 'pathways/synthetics/'
    reader = libsbml.SBMLReader()
    document = reader.readSBML(input_filename)
    model = document.getModel()

    for i in range(model.getNumParameters()):
        param = model.getParameter(i)
        for alteration in alterations:

            if param.getId() == alteration[0]:
                param.setValue(alteration[1])

    modified_sbml = libsbml.writeSBML(document, output_filename)


def run_simulation(filename, species_list):
    rr = roadrunner.RoadRunner(filename)
    res = rr.simulate(0, 10000, 1000)
    simulation_results = {}
    initial_concentrations = {}
    # print("Species concentrations:")
    for i, species in enumerate(species_list):
        # print(f"{species}: {res[-1, i + 1]}")
        simulation_results[species] = res[-1, i + 1]
        initial_concentrations[species] = res[0, i + 1]

    return [simulation_results, initial_concentrations]


def run_simulation_complete_results(filename, species_list):
    rr = roadrunner.RoadRunner(filename)
    res = rr.simulate(0, 10000, 1000)
    simulation_results = {}
    initial_concentrations = {}
    # print("Species concentrations:")

    for i, species in enumerate(species_list):
        initial_concentrations[species] = res[0, i + 1]

    for j in range(len(res)):
        for i, species in enumerate(species_list):
            simulation_results[species] = res[j, i + 1]

    return [simulation_results, initial_concentrations]
def plot_simulation_complete(filename):

    rr = roadrunner.RoadRunner(filename)
    res = rr.simulate(0, 10000, 1000)
    rr.plot(res)
    print(res)
    final_concentrations = [res[1::][1::], res[0][1::]]

    return final_concentrations


def steady_state(filename):
    rr = roadrunner.RoadRunner(filename)
    rr.steadyState()
    steady_state_concentrations = rr.getSteadyStateValues()

    # Print the results
    species_ids = rr.getFloatingSpeciesIds()
    for species, concentration in zip(species_ids, steady_state_concentrations):
        print(f"{species}: {concentration}")

def plot_graph(reaction_graph):
    pos = nx.spring_layout(reaction_graph)
    nx.draw(reaction_graph, pos, with_labels=True, node_color=[{'specie': 'blue', 'reaction': 'red'}[reaction_graph.nodes[n]['type']] for n in reaction_graph.nodes])
    #edge_labels = nx.get_edge_attributes(reaction_graph, 'weight')
    #nx.draw_networkx_edge_labels(reaction_graph, pos)
    plt.show()

def generate_connected_reaction_network(num_species, num_reactions, reversible_ratio):
    '''
    Generates the description of a connected reaction network whose number of species and reactions
    are given by num_species and num_reactions, the reversible ratio can be also provided (in our case all reactions
    are reversible though). The pathway generate may not conserve quantities
    :param num_species:
    :param num_reactions:
    :param reversible_ratio:
    :return:
    '''
    species = []
    reactions = []
    connections = []

    # Start with one initial species
    species.append("S0")
    species_counter = 1

    connected_species = {"S0"}
    connected_reactions = set()

    while len(reactions) < num_reactions:
        reaction_id = f"R{len(reactions)}"
        #reversible = random.random() < reversible_ratio
        reversible = True
        reactions.append([reaction_id, 0])

        # Pick 1-2 reactants from connected species
        num_reactants = random.randint(1, min(2, len(connected_species)))
        reactants = random.sample(list(connected_species), num_reactants)

        # Create 1-2 new product species
        num_new_products = min(random.randint(1, 2), num_species - len(species))
        new_products = []
        for _ in range(num_new_products):
            new_species_id = f"S{species_counter}"
            species.append(new_species_id)
            connected_species.add(new_species_id)
            new_products.append(new_species_id)
            species_counter += 1

        # If we still need more products and can't add new ones, use existing ones
        if len(new_products) == 0:
            num_existing_products = random.randint(1, 2)
            new_products = random.sample(list(connected_species), num_existing_products)

        # Add connections
        for r in reactants:
            connections.append([r, reaction_id, 0])
        for p in new_products:
            connections.append([reaction_id, p, 0])

        if reversible:

            reversible_reaction = [f"Rr{len(reactions) - 1}", 0]
            reactions.append(reversible_reaction)

            for p in new_products:
                connections.append([p, reversible_reaction[0], 0])
            for r in reactants:
                connections.append([reversible_reaction[0], r, 0])

            connected_reactions.add(reversible_reaction[0])

        connected_species.update(new_products)
        connected_reactions.add(reaction_id)

    return {
        "species": species,
        "reactions": reactions,
        "connections": connections
    }



def generate_random_pathway_LMA(min_num_reactions, max_num_reactions):

    '''
    Generates a reaction network whose number of reactions is in the interval [min_num_reactions, max_num_reactions]
    the generated pathway preserves quantities
    :param min_num_reactions:
    :param max_num_reactions:
    :return:
    '''

    weights = [0.25, 0.5, 0.75, 1]
    description = {
        'species': [],
        'reactions': [],
        'connections': []
    }

    num_reactions = random.randint(min_num_reactions, max_num_reactions)

    species = [['species' + str(i)] for i in range(20)]
    # Generate the list of available species with their random weight
    for i in range(len(species)):
        species[i].append(weights[random.randrange(0, len(weights))])


    for i in range(num_reactions):

        reactants = []

        description['reactions'].append(['kf' + str(i + 1), 0])
        description['reactions'].append(['kr' + str(i + 1), 0])

        # Choose number of products
        num_products = random.randrange(1, 3)

        # Shuffle list
        random.shuffle(species)
        products = species[0:num_products]

        products_weight = 0
        for product in products:
            products_weight += product[1]
            if product[0] not in description['species']:
                description['species'].append(product[0])
            description['connections'].append(['kf' + str(i + 1), product[0], product[1]])
            description['connections'].append([product[0], 'kr' + str(i + 1), product[1]])

        reactants_weight = 0
        reactants_counter = 0
        # Take reactant candidates and sort them according to weight
        candidates = species[num_products::]
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

        while reactants_weight < products_weight:
            reactant = candidates[reactants_counter][0]
            reactant_weight = candidates[reactants_counter][1]

            # To avoid reactants' weight to be greater than products' one
            while reactants_weight + reactant_weight > products_weight:
                reactants_counter += 1
                reactant = candidates[reactants_counter][0]
                reactant_weight = candidates[reactants_counter][1]

            reactants_weight += reactant_weight

            if reactant not in description['species']:
                description['species'].append(reactant)
            description['connections'].append([reactant, 'kf' + str(i + 1), reactant_weight])
            description['connections'].append(['kr' + str(i + 1), reactant, reactant_weight])
            reactants.append([reactant, reactant_weight])
            reactants_counter += 1

    return description

def save_graph_description(description, path):

    pickle_file = open(path, 'ab')
    pickle.dump(description, pickle_file)
    pickle_file.close()

def load_graph_description(path):

    pickle_file = open(path, 'rb')
    description = pickle.load(pickle_file)
    return description