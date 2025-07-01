import numpy as np
import math
import networkx as nx
from copy import deepcopy

def sigmoid(x):
    return 1 / (1 + math.exp(-(1*(x -0.5))))


def logit(x):
    if x == 1:
        x -= 1e-12
    if x == 0:
        x += 1e-12
    return math.log(x / (1-x)) / 1 + 0.5

def compute_new_specie_value(values):

    new_value = 0.5
    deltas = []

    for value in values:
        if value != 0.5:
            deltas.append(value - 0.5)

    if len(deltas) > 0:
        accumulator = 0
        divisor = 0
        for delta in deltas:
            accumulator += delta * math.fabs(delta)
            divisor += math.fabs(delta)

        return new_value + np.average(deltas)

    return new_value



# Takes as argument the reaction graph and perform the propagation for a given iteration
def update_graph(reaction_graph, verbose, iteration):

    starting_values = {}
    for node in reaction_graph.nodes():
        if reaction_graph.nodes[node]['type'] == 'specie':
            starting_values[node] = reaction_graph.nodes[node]['size']

    if verbose:
        print('---values before iteration---')
        print(starting_values)

    for node in reaction_graph.nodes():
        if reaction_graph.nodes[node]['type'] == 'reaction':

            #=== Fire the reaction: ===#
            if verbose:
                print('Firing reaction ', node)

            # Will store the deltas of reactants
            list_of_deltas = []
            reaction_strength = reaction_graph.nodes[node]['strength']

            reaction_weight = reaction_graph.nodes[node]['weight'] / math.sqrt(iteration)

            reactants = list(reaction_graph.predecessors(node))
            products = list(reaction_graph.successors(node))

            for reactant in reactants:
                current_value = reaction_graph.nodes[reactant]['size']
                # Compute the shift from the base case for each reactant
                delta = (current_value - 0.5)
                list_of_deltas.append(delta)

                if verbose:
                    print('reactant: ', reactant, ' delta: ', delta)

            if len(list_of_deltas) > 0:

                if np.sum(list_of_deltas) >= 0:
                    new_reaction_strength = (np.sum(list_of_deltas) + (np.sum(list_of_deltas) * reaction_weight))
                else:
                    # If the sum of deltas is lower than zero invert the sign of the weight, otherwise it has the opposite effect
                    new_reaction_strength = (np.sum(list_of_deltas) + (np.sum(list_of_deltas) * reaction_weight * -1))
                new_reaction_strength = math.tanh(new_reaction_strength)
            else:
                new_reaction_strength = 0

            if verbose:
                print(reaction_graph.nodes[node])
                print(
                    'new reaction strength: ', new_reaction_strength,
                    ' reaction weight: ', reaction_weight,
                    ' reaction strength ', reaction_strength
                )

            list_of_deltas = []

            # Change the reactant elements
            for reactant in reactants:

                current_value = logit(reaction_graph.nodes[reactant]['size'])
                # Compute the potential by adding to the alteration mark (if present) the contribution coming from reactants
                # scaled by the weight.Then apply potential decay
                s = (reaction_strength) + (new_reaction_strength)
                s /= (math.sqrt(iteration))


                new_value = current_value - s
                new_value = sigmoid(new_value)

                reaction_graph.nodes[reactant]['values'].append(new_value)
                delta = (new_value - reaction_graph.nodes[reactant]['previous_size'])
                list_of_deltas.append(delta)

                if verbose:
                    print(
                        'reactant: ', reactant,
                        ' new value: ', new_value,
                        ' delta: ', delta
                    )

            products = list(reaction_graph.successors(node))
            for product in products:

                current_prod_value = logit(reaction_graph.nodes[product]['size'])
                s = (reaction_strength) + (new_reaction_strength)
                s /= (math.sqrt(iteration))
                if verbose:
                    print('POTENTIAL: ', s)

                new_prod_value = current_prod_value + s
                new_prod_value = sigmoid(new_prod_value)

                if verbose:
                    print('product: ', product, 'new value: ', new_prod_value)

                reaction_graph.nodes[product]['values'].append(new_prod_value)

    # Finally update the nodes' values by averaging the temporary values computed above
    for node in reaction_graph.nodes():
        if reaction_graph.nodes[node]['type'] == 'specie':
            if len(reaction_graph.nodes[node]['values']) > 0:
                reaction_graph.nodes[node]['previous_size'] = reaction_graph.nodes[node]['size']
                reaction_graph.nodes[node]['size'] = compute_new_specie_value(reaction_graph.nodes[node]['values'])

            else:
                reaction_graph.nodes[node]['previous_size'] = reaction_graph.nodes[node]['size']

            if verbose:
                print('Updating specie ', node, ' values: ', reaction_graph.nodes[node]['values'])


            reaction_graph.nodes[node]['values'] = []
            starting_values[node] = reaction_graph.nodes[node]['size']

    return starting_values

def generate_graph(description):
    '''
    Generates a Networkx instance of the graph starting from its definition
    :param description: Definition of the network
    :return: The network instance
    '''

    reaction_graph = nx.DiGraph()

    for node in description['species']:
        reaction_graph.add_node(node, type='specie', size=0.5, previous_size=0.5, values=[])
    for node in description['reactions']:
        # The fourth field of a reaction node (if present) is the list of modifiers
        if len(node) > 3:
            modifiers = node[3]
        else:
            modifiers = []

        # The third field of a reaction node (if present) is its weight
        if node[2]:
            weight = node[2]
        else:
            weight = 1

        reaction_graph.add_node(node[0], type='reaction', strength=node[1], weight=weight, modifiers=modifiers)

    for connection in description['connections']:
        reaction_graph.add_edge(connection[0], connection[1])

    return reaction_graph


def graph_description_with_weights(graph_description, weights_configuration):
    '''
    This function returns an updated graph description where each reaction node is assigned a weight
    :param graph_description: The original graph description
    :param weights_configuration: The given configuration of weights: each item is a tuple (reaction, weight)
    :return: The updated graph description
    '''

    new_graph_description = deepcopy(graph_description)

    for configuration in weights_configuration:
        for reaction in new_graph_description['reactions']:
            if reaction[0] == configuration[0]:
                reaction[2] = configuration[1]

    return new_graph_description
