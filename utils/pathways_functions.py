import libsbml

def generate_sbml(description, fileName, kinetic_constants):

    #directory = 'pathways/syntethics/LMA/'

    document = libsbml.SBMLDocument()
    model = document.createModel()
    comp = model.createCompartment()
    comp.setId('cell')

    # Add species
    for specie in description['species']:
        s = model.createSpecies()
        s.setId(str(specie))
        s.setCompartment('cell')
        s.setInitialAmount(1)

    for reaction in description['reactions']:

        r = model.createReaction()
        kinetic_law = r.createKineticLaw()
        reactants = []
        r.setId('reaction' + reaction[0])

        parameter = model.createParameter()
        parameter.setId(str(reaction[0]))
        parameter.setValue(kinetic_constants[str(reaction[0])])

        for connection in description['connections']:
            if connection[1] == reaction[0]:
                react = r.createReactant()
                react.setSpecies(str(connection[0]))
                reactants.append(str(connection[0]))

            if connection[0] == reaction[0]:
                prod = r.createProduct()
                prod.setSpecies(str(connection[1]))

        formula = str(reaction[0])
        for reactant in reactants:
            formula += ' * ' + str(reactant)
        math_formula = libsbml.parseL3Formula(formula)
        kinetic_law.setMath(math_formula)

    res = libsbml.writeSBML(document, fileName)
