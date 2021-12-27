import random
import numpy as np

bits_per_genotype = 15


"""
population_size = N genotypes forming the population

Output:

inizial population
"""


def initialize_population(population_size):
    initial_population = []
    for genotype in range(population_size):
        temp_genotype = []
        for bit in range(bits_per_genotype):
            temp_genotype.append(random.randint(0, 1))
        initial_population.append(temp_genotype)
    return initial_population


def binary_tournament_selection(population, fitness):
    # choose two random indexes smaller than length of population
    parents_idx = random.choices(range(len(population)), k=2)
    print('in tournametn selection')
    print('Population:')
    print(population)
    print('fitness')
    print(fitness)
    # extract these parents respective fitness
    p_1_fitness = fitness[parents_idx[0]]
    p_2_fitness = fitness[parents_idx[1]]
    print('contrahent fitnesses:')
    print(p_1_fitness)
    print(p_2_fitness)
    # find stronger parent and return it
    if p_1_fitness > p_2_fitness:
        stronger_parent = population[parents_idx[0]]
        stronger_parent_fitness = p_1_fitness
    else:
        stronger_parent = population[parents_idx[1]]
        stronger_parent_fitness = p_2_fitness
    print('stronger parent')
    print(stronger_parent)
    print(stronger_parent_fitness)
    return stronger_parent, stronger_parent_fitness


"""
population = list containing all N genotypes
p_c = crossover probability
mu = difference threshold of crossover operation

Output:

two offspring
"""


def crossover(population, fitness, p_c, mu):
    # loop over population 10 times
    for j in range(10):
        # select two parents p1, p2 by binary tournament selection
        winner_parent_1, w_p_1_f = binary_tournament_selection(population, fitness)
        winner_parent_2, w_p_2_f = binary_tournament_selection(population, fitness)
        print('Winner Parent 1')
        print(type(winner_parent_1))
        print(winner_parent_1)
        print(w_p_1_f)
        # compute difference between p1, p2 (hamming distance normalized to 0-1)
        p_1 = np.asarray(winner_parent_1)
        p_2 = np.asarray(winner_parent_2)

        print('After converstion to numpy')
        print(type(p_1))
        print(p_1)

        differences = np.zeros(p_1.shape)
        differences[p_1 != p_2] = 1
        unique, counts = np.unique(differences, return_counts=True)
        temp_dict = dict(zip(unique, counts))
        print('Differences dict:')
        print(temp_dict)
        if len(temp_dict) == 1:
            # if only 0 in dictionary, they are the same
            normalized_hamming = 0
        else:
            hamming_distance = temp_dict[1]
            print('hamming distance:')
            print(hamming_distance)
            normalized_hamming = hamming_distance / (temp_dict[0] + temp_dict[1])
        print('normalized hamming:')
        print(normalized_hamming)

        if normalized_hamming > mu:
            # use these two parents for mating
            break
    # else go on until good parents found. If not, use last pair to mate

    # randomly generate r (0,1)
    r = random.uniform(0, 1)
    if r < p_c:
        # mate
        # reshape genotype of parents
        # p_1_resh = p_1.reshape(-1)
        # p_2_resh = p_2.reshape(-1)
        print('Mating:')
        print('p1')
        print(p_1)
        print('p2')
        print(p_2)
        # compute length of p1 and p2
        parent_length = len(p_1)
        # randomly choose 10 different integers from [0, len) and sort
        random_int = random.sample(range(parent_length), 4)
        random_int_np = np.asarray(random_int)
        sorted_random_int = np.sort(random_int_np)
        sorted_random_int = sorted_random_int.reshape(2, 2)
        print('start_stop indexes:')
        print(sorted_random_int)
        # crossover implementation: exchange genes at specified intervals between genotypes
        for k in range(len(sorted_random_int)):
            temp_start_idx = sorted_random_int[k, 0]
            temp_stop_idx = sorted_random_int[k, 1]

            p_1_gene = p_1[temp_start_idx:temp_stop_idx]
            p_1_gene_copy = p_1_gene.copy()
            p_2_gene = p_2[temp_start_idx:temp_stop_idx]

            p_1[temp_start_idx:temp_stop_idx] = p_2_gene
            p_2[temp_start_idx:temp_stop_idx] = p_1_gene_copy

        # p_1 = p_1_resh.reshape(block_gene_per_architecture, bits_per_block_gene)
        # p_2 = p_2_resh.reshape(block_gene_per_architecture, bits_per_block_gene)
        print('After Crossover')
        print(type(p_1))
        print(p_1)
        print(type(p_2))
        print(p_2)
        # Assign offspring
        o_1 = p_1.tolist()
        o_2 = p_2.tolist()

    else:
        o_1 = winner_parent_1
        o_2 = winner_parent_2

    return o_1, o_2


"""
parent_population: list of parent architectures, list of lists of length N
offspring_population: list of offspring architectures, list of lists of length N
fitness_parents: list containing parent fitness of length N
fitness_offspring: list containing offspring fitness of length N

output

next_generation: new generation comprising individuals from parent and offspring generation
"""


def environmental_selection(parent_population, offspring_population, fitness_parents, fitness_offspring):
    next_generation = []
    next_generation_fitness = []

    combined_population = parent_population + offspring_population
    fitness_parents_np = np.asarray(fitness_parents)
    fitness_offspring_np = np.asarray(fitness_offspring)

    # combine the fitness arrays
    combined_fitness = np.concatenate((fitness_parents_np, fitness_offspring_np), axis=0)
    # find the indexes of the five fittest individuals
    max_idx = np.argpartition(combined_fitness, -5)[-5:]

    # extract the fittest individuals from respective population
    for idx in range(len(max_idx)):
        temp_idx = max_idx[idx]
        next_generation.append(combined_population[temp_idx])
        next_generation_fitness.append(combined_fitness[temp_idx])

    # create new populations without the five fittest in them
    reduced_combined_population = []
    reduced_combined_fitness = []
    for idx in range(len(combined_fitness)):
        if idx not in max_idx:
            reduced_combined_population.append(combined_population[idx])
            reduced_combined_fitness.append(combined_fitness[idx])

    # fill up the next generation with individuals choosen by binary tournament selection
    while len(parent_population) > len(next_generation):
        winner_parent, winner_parent_fitness = binary_tournament_selection(combined_population, combined_fitness)
        next_generation.append(winner_parent)
        next_generation_fitness.append(winner_parent_fitness)

    return next_generation, next_generation_fitness


"""
architecture = list describing the architecture with its block genes
p_m = mutation probability
p_b = probability for bit to flip
"""


def mutation(genotype, p_m, p_b):
    print('in mutation')
    print(type(genotype))
    # genotype_np = np.asarray(genotype).reshape(-1)
    genotype_np = np.asarray(genotype)
    print('genotype before mutation')
    print(genotype_np)
    # check if mutation will occur
    r = random.uniform(0, 1)
    if r < p_m:
        print('mutation happens')
        # check flipping probability for every bit in the genotype
        for gene in range(len(genotype_np)):
            r = random.uniform(0, 1)
            if r < p_b:
                print('gene flipped')
                genotype_np[gene] = 1 if genotype_np[gene] == 0 else 0
    print('Genotype after mutation')
    print(genotype_np)
    # architecture_np_resh = architecture_np.reshape(block_gene_per_architecture, bits_per_block_gene)

    mutated_genotype = genotype_np.tolist()

    return mutated_genotype
