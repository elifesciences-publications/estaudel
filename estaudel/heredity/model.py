""" heredity/model.py -- Evolution of heredity thought experiment.
This file is part of the ecological scaffolding package/ heredity model subpackage.
Copyright 2018 Guilhem Doulcier, Licence GNU GPL3+
--

Phenotypes in this experiment have size 4:

[color, growth_rate, intra_type_interaction, inter_type_interaction]

- Color: c_i.
- Growth_rate: r_i .
- intra_type_interaction: a_i.
- inter_type_interaction: a'_i.

If N_i is the number of inidividuals of type i:

Birth rates are:
    b_i(N) = r_i*N_i
Death rate are:
    d_i(N) = r_i*N_i*[{sum_{j, st. c_i=c_j} N_j * a_j + {sum_{j, st. c_i!=c_j} N_j * a'_j]
"""
from functools import partial
import numpy as np
import scipy.integrate

COLOR_NAMES = ['red','blue']
POS_NAMES = ['color', 'growth_rate', 'interaction_intra', 'interaction_inter']

### Stochastic model

def bd_rates(phenotypes, state, K):
    """ Return the density dependent birth and death rates of the phenotypes.

    Used as argument for stochastic.bdm_process.

    phenotypes (array): Description of the types
    state (array of int): Number of individual of each type
    K (float): Carrying capacity
    """
    rates = np.zeros((len(state), 2))

    # Birth rates
    rates[:, 0] = np.multiply(state, phenotypes[:, 1])

    # Death rates
    competition = np.array([np.sum([N/K * phenotypes[j, 2+int(phenotypes[j, 0] != phenotypes[i, 0])]
                                    for j, N in enumerate(state)])
                            for i in range(len(state))])
    rates[:, 1] = np.multiply(np.multiply(state, competition), phenotypes[:,1])
    return rates

def mutation_r_inter_reflective(old, effect):
    """Mutation function. Only the growth rate and the inter-type
    competition are able to mutate.

    Used as argument for stochastic.bdm_process
    """
    new = old.copy()

    # Choose which trait will mutate.
    position = 1 if np.random.random() > 0.5 else 3
    direction = 1 if np.random.random() > 0.5 else -1

    # Apply the mutation.
    # take absolute value (mutation is reflected on 0)
    new[position] = abs(old[position]+np.random.random()*effect[position]*direction)
    return new

def mutation_inter(old, effect):
    """Mutation function. Only the inter-type competition are able to
    mutate. The mutation is normally distributed with average 0 and
    standard deviation `effect`*old_value.

    The new trait is reflected around 0 (absolute value)

    Used as argument for stochastic.bdm_process

    """
    new = old.copy()
    new[3] = abs(np.random.normal(old[3], effect))
    return new

def mutation_r_inter(old, effect):
    """Mutation function. Only the growth rate and the inter-type
    competition are able to mutate. The mutation is normally
    distributed with average 0 and standard deviation
    `effect`*old_value. The new trait is clamped so all traits stays
    positive.

    Used as argument for stochastic.bdm_process
    """
    new = old.copy()

    # Choose which trait will mutate.
    position = 1 if np.random.random()from functools import partial
 > 0.5 else 3

    # Apply the mutation.
    new[position] = np.random.normal(old[position], effect)

    # Clamp to force positive trait.
    new[position] = max(new[position], 0)

    return new

def collective_fitness(phenotypes, state, var=1, goal=.5):
    """Return the collective fitness of a collective with `state[i]`
    individuals of type `phenotypes[i]`.

    Used in argument in escaffolding.collective_generation.
    """
    prop = np.sum([n for g, n in zip(phenotypes, state) if g[0] == 0]) / state.sum()
    return 1/np.sqrt(np.pi*2*var) * np.exp(-(goal-prop)**2 / 2*var)

def gen_collective(max_types, B, green=(0, 1, .8, .3), red=(1, 1, .9, .3), proportion=None):
    """Return a (phenotypes, sate) tuple encoding a collective.

    - maxTypes (int): Maximum number of different phenotypes in this collective.
    - B (int): Initial population size.
    - green (array) and red (array): Initial trait values for red and green types.
    - proportion (float): initial proportion of red and green individuals.
                          If none, it will be taken at random.
    """

    phenotypes = np.zeros((max_types, 4))
    for i in range(max_types):
        phenotypes[i, :] = green if i%2==0 else red

    state = np.zeros(max_types)
    if proportion is None:
        proportion = np.random.random()
    state[0] = int(B*proportion)
    state[1] = B - state[0]

    return phenotypes, state
