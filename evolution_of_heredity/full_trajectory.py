""" full_trajectory.py -- main entrypoint to start a ecological scaffolding simulation.
Copyright 2018 Guilhem Doulcier, Licence GNU GPL3+
"""

from functools import partial
import argparse
import multiprocessing
import datetime
import platform
import sys
import pickle

import estaudel.stochastic as stochastic
import estaudel.escaffolding as escaffolding
import estaudel.heredity.stochastic as model
import estaudel.heredity.process as processing

PARAMETERS = {
    'D':(1000, 'Number of collectives'),
    'B':(15, 'Bottleneck size'),
    'N':(10000, 'Number of generations,'),
    'NPROC':(2, 'Number of CPU to use.'),
    'T':(1, 'Length of growth phase,'),
    'steps':(100, 'Number of discrete steps for bdm process.'),
    'skip':(100, ' Frequency at which growth trajectory are saved.'),
    'mutation_rate':(1, 'Probability that a birth event will also be a mutation event.'),
    'mutation_effect': ({1:0.1, 3:0.1}, 'Amplitude of mutational effects {trait:amplitude}'),
    'carrying_capacity': (1500, 'Max number of particle in a collective (will divide the a_intra, a_inter).'),
    'collectiveSelectionStrength':(1, ' Localisation parameter for the collective fitness function.'),
    'max_types':(4, 'Maximum number of types per collective'),
    'goal':(.5, ' Optimal proportion of types.'),
    'initial_type0':((0, 6, .8, .15), 'intial traits (c, r, a_intra, a_inter)'),
    'initial_type1':((1, 4, .3, .15), 'initial traits (c, r, a_intra, a_inter)'),
    'percentile':(20, ' Percentile of collective that go extinct each generation'),
    'selection':('rank', 'Rank or neutral'),
    'continue':(None, 'Path to an output file from which extract the initial population'),
    'name':(None, 'Filename')
}

def main(p,pool,filename):

    if len(p['mutation_effect']):
        mutation_function = partial(stochastic.normal_mutation_abs, effect=p['mutation_effect'])
    else:
        print('No mutation !')
        mutation_function = lambda x: None
        p['mutation_rate'] = 0


    growth = partial(stochastic.discrete_bdm_process,
                     T=p['T'],
                     steps=p['steps'],
                     skip=p['skip'],
                     mutation_rate=p['mutation_rate'],
                     mutation_function=mutation_function,
                     rate_function=partial(model.bd_rates, K=p['carrying_capacity']))

    collective_fitness_func = partial(model.collective_fitness,
                                      var=p['collectiveSelectionStrength'],
                                      goal=p['goal'])

    dil_and_growth =  partial(escaffolding.dilution_and_growth,
                              bottleneck_size=p['B'],
                              growth_function=growth)

    if p['continue'] is not None:
        out = escaffolding.Output(0,0,0,0)
        out.load(p['continue'])
        N = out.current_gen
        print('Loading gen {} from {}'.format(N,p['continue']))
        pop = [(out.phenotype[:,:,N,d], out.state[:,N,d])
               for d in range(out.state.shape[2])]
    else:
        pop = [model.gen_collective(p['max_types'], p['B'], p['initial_type0'], p['initial_type1'])
               for i in range(p['D'])]

    output = escaffolding.Output(p['N'], p['D'], pop[0][0].shape[0], pop[0][0].shape[1])
    output.parameters = p

    if p['selection']=='neutral':
        collective_birth_death_func = escaffolding.collective_birth_death_neutral
    elif p['selection']=='rank':
        collective_birth_death_func = escaffolding.collective_birth_death_process
    collective_birth_death_func = partial(collective_birth_death_func,
                                          percentile=p['percentile'])

    output,pop = escaffolding.collective_generations(p['N'], pop, output,
                                                     collective_fitness_func,
                                                     collective_birth_death_func,
                                                     dil_and_growth, filename=filename, pool=pool)

    sys.stdout.flush()
    processing.extract(output, pool)

    sys.stdout.flush()
    output.save(filename+'.pkle')
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    for name, (default, man) in PARAMETERS.items():
        parser.add_argument('--'+name, help=man,  nargs="?")
    args = parser.parse_args()

    ### UNSAFE USE OF EVAL ! Be Careful :-)
    PARAMETERS = ({k:eval(v) if v is not None else PARAMETERS[k][0] for k,v in vars(args).items()})
    tstart = datetime.datetime.now()
    TIME = tstart.strftime('%Y%m%d_%H%M%S')
    PATH = (((PARAMETERS['name'] + "_") if PARAMETERS['name'] is not None else '')
            + TIME + "_"
            + "_".join(["{}{}".format(k, v.replace("{","").replace("}","").replace(":","").replace('"',"").replace(",","_").replace("(","").replace(")","")) for k, v in vars(args).items() if (k != 'name' and v is not None)]))
    POOL = multiprocessing.Pool(PARAMETERS['NPROC'])

    print('='*80)
    print("{: ^80}".format("Exact Stochastic Simulation of the Evolution Machine"))
    print('='*80)
    print('Started at {}'.format(TIME))
    print('On {} | Python {}'.format(platform.platform(), platform.python_version()))
    print('Using {} / {} CPU'.format(PARAMETERS['NPROC'], multiprocessing.cpu_count()))
    print("{:=^80}".format(' Parameters '))
    print("\n".join(sorted(["{}: {}".format(k, v) for k, v in PARAMETERS.items()])))
    print("Will save in {}".format(PATH))
    main(PARAMETERS, POOL,PATH)
    POOL.close()
    POOL.join()

    print('='*80)
    tend = datetime.datetime.now()
    print('Ran in {}'.format((tend-tstart).total_seconds()))
