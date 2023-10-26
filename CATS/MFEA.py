import numpy as np
import pandas as pd
from operator import itemgetter
import random


def genotype(n_estimators, min_samples_leaf, max_features, factorial_cost, factorial_rank, factorial_skill, scalar_fitness, model_size):
    """
    Create the individual genotype

    :param n_estimators: The number of trees in the forest.
    :param max_depth: The maximum depth of the tree. 
    :param min_samples_split: The minimum number of samples required to split an internal node.
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
    :param bootstrap: Whether bootstrap samples are used when building trees.
    :param f1: accuracy fitness value
    :param f2: parsimony fitness value
    :return: the genotype, a dictionary with all hyperparameters
    """

    ind = dict(n_estimators=n_estimators, min_samples_leaf = min_samples_leaf, max_features = max_features,
               factorial_cost = factorial_cost, 
               factorial_rank = factorial_rank,
               factorial_skill = factorial_skill, 
               scalar_fitness = scalar_fitness,
               model_size = model_size
               )
    
    return ind


def random_genotype(var_names):
    """
    Create random genotype

    :return: the genotype, a dictionary with all hyperparameters
    """

    return genotype(
        random.randint(5, 1000), #n_estimators
        random.randint(1, 30), #min_samples_leaf
        random.choice(['sqrt', 'log2', 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), #max_features
        dict.fromkeys(list(var_names), None), #factorial_cost
        dict.fromkeys(list(var_names), None), #factorial_rank
        None, #factorial_skill
        None,  #scalar_fitness
        dict.fromkeys(list(var_names), None)   #model_size
    )



def initial_population(n, var_names):
    """
    Create a random population of size n

    :param n: the size of the population
    :return: a list with n random individuals
    """
    
    pop = []
    for i in range(n):
        pop.append(random_genotype(var_names))
        
    return pop


def phenotype(individual, X_train, y_train):
    """
    Instantiate the genotype, creating a fitted model with the genotype hyperparameters

    :param individual: a genotype
    :param X_train, y_train: the training dataset
    :param fts_method: the FTS method 
    :param parameters: dict with model specific arguments for fit method.
    :return: a fitted FTS model
    """
    
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(n_estimators=individual['n_estimators'], 
                                  min_samples_leaf=individual['min_samples_leaf'],
                                  max_features=individual['max_features'],
                                  bootstrap=True, n_jobs=-1, random_state=0)
    
    model.fit(X_train, y_train)

    return model


def evaluate(dataset, individual, **kwargs):
    """
    Evaluate an individual using a sliding window cross validation over the dataset.

    :param dataset: Evaluation dataset
    :param individual: genotype to be tested
    :param window_size: The length of scrolling window for train/test on dataset
    :param train_rate: The train/test split ([0,1])
    :param increment_rate: The increment of the scrolling window, relative to the window_size ([0,1])
    :param parameters: dict with model specific arguments for fit method.
    :return: a tuple (len_lags, rmse) with the parsimony fitness value and the accuracy fitness value
    """
    import measures

    errors = []
    size = []
    
    params = {
        'size_train': 1000,
        'size_test': 100
        }
    
    window = np.arange(0, (params['size_train']*3)+params['size_test'], params['size_train'])
    window = np.delete(window, -1)

    for w in window:
        
        X_train = dataset['X_train'].loc[w:w+params['size_train']]
        X_test = dataset['X_train'].loc[w+params['size_train']:w+params['size_train']+params['size_test']-1]
                
        y_train = dataset['y_train'].loc[w:w+params['size_train']]
        y_test = dataset['y_train'].loc[w+params['size_train']:w+params['size_train']+params['size_test']-1]
        
        
        model = phenotype(individual, X_train, y_train)
        
        forecasts = model.predict(X_test)
        
        nrmse = measures.nrmse(y_test, forecasts)

        errors.append(nrmse)

    
    nrmse = np.mean(errors)
    size = individual['n_estimators'] * sum([tree.tree_.max_depth for tree in model.estimators_])
    
     
    return nrmse, size


def tournament(population, objective, **kwargs):
    """
    Simple tournament selection strategy.

    :param population: the population
    :param objective: the objective to be considered on tournament
    :return:
    """
    n = len(population) - 1

    r1 = random.randint(0, n) if n > 2 else 0
    r2 = random.randint(0, n) if n > 2 else 1
    
    if objective == 'scalar_fitness':
        ix = r1 if population[r1]['scalar_fitness'] > population[r2]['scalar_fitness'] else r2
    else:
        ix = r1 if sum(population[r1]['model_size'].values()) < sum(population[r2]['model_size'].values()) else r2

        
    return population[ix]


def double_tournament(population, **kwargs):
    """
    Double tournament selection strategy.

    :param population:
    :return:
    """

    ancestor1 = tournament(population, 'scalar_fitness')
    ancestor2 = tournament(population, 'scalar_fitness')

    selected = tournament([ancestor1, ancestor2], 'model_size')

    return selected




def crossover(population, divergence_matrix, max_divergence, var_names):
    """
    Crossover operation between two parents

    :param population: the original population
    :return: a genotype
    """
    import random

    n = len(population) - 1

    r1, r2 = 0, 0
    while r1 == r2:
        r1 = random.randint(0, n)
        r2 = random.randint(0, n)
        
    divergence = divergence_matrix.loc[population[r1]['factorial_skill']][population[r2]['factorial_skill']]
   
    pmut = 1 - sum(((divergence_matrix >= 0) & (divergence_matrix <= max_divergence)).sum())/((len(var_names)**2) - len(var_names))

   
    if divergence < max_divergence:
        
        if population[r1]['scalar_fitness'] > population[r2]['scalar_fitness'] :
            best = population[r1]
            worst = population[r2]
        else:
            best = population[r2]
            worst = population[r1]
                   
    
        n_estimators = int(.7 * best['n_estimators'] + .3 * worst['n_estimators'])
        min_samples_leaf = int(.7 * best['min_samples_leaf'] + .3 * worst['min_samples_leaf'])
        max_features = best['max_features']
        
        if random.uniform(0,1) > 0.5:
            skill = best['factorial_skill']
        else:
            skill = worst['factorial_skill']
            
        descendent = genotype(n_estimators, min_samples_leaf, max_features,
                               dict.fromkeys(list(var_names), None),
                               dict.fromkeys(list(var_names), None),
                               skill, None, dict.fromkeys(list(var_names)))
        
        if pmut > random.uniform(0,1):
            descendent = mutation(descendent, var_names)
            
    else:
        
        descendent = mutation(population[r1], var_names)               

    return descendent



def mutation(individual, var_names):
    """
    Mutation operator

    :param individual: an individual genotype
    :param pmut: individual probability o
    :return:
    """

    n_estimators = min(1000, max(5,int(individual['n_estimators'] + (np.random.normal(0, 10)*np.random.choice([-1,1],1)[0]))))
    min_samples_leaf = min(30, max(1,int(individual['min_samples_leaf'] + (np.random.normal(0, 2)*np.random.choice([-1,1],1)[0]))))
    max_features = random.choice(['sqrt', 'log2', 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    descendent = genotype(n_estimators, min_samples_leaf, max_features,
                           dict.fromkeys(list(var_names), None),
                           dict.fromkeys(list(var_names), None),
                           individual['factorial_skill'], None, dict.fromkeys(list(var_names)))

    return descendent


def elitism(population, new_population):
    """
    Elitism operation, always select the best individual of the population and discard the worst

    :param population:
    :param new_population:
    :return:
    """
    population = sorted(population, key=itemgetter('scalar_fitness'))
    best = population[0]

    new_population = sorted(new_population, key=itemgetter('scalar_fitness'))
    
    if best["scalar_fitness"] > new_population[0]["scalar_fitness"]:
        new_population.insert(0,best)

    return new_population

def divergence(dataset, var_names):
    from scipy.spatial import distance

    divergence_matrix = pd.DataFrame(columns=var_names, index=var_names)

    for var1 in var_names:
        for var2 in var_names:
            if var1 != var2:
                divergence_matrix.loc[var1][var2] = distance.jensenshannon(dataset[var1], dataset[var2])
            else:
                divergence_matrix.loc[var1][var2] = -1
       
    max_divergence = np.round(max(divergence_matrix.max())/2,1)
    
    return divergence_matrix, max_divergence



def generate_factorial_rank(population, var_names):
    
    rank = pd.DataFrame(columns=var_names, index=range(len(population)))
    for variable in var_names:
        cost = []
        for individual in population:
            cost.append(individual['factorial_cost'][variable])
        rank_series = pd.Series(cost)
        
        rank[variable] = rank_series.rank(method = 'min') 
        rank_dict = rank.T.to_dict()

    for individual in range(len(population)):
        population[individual]['factorial_rank'] = rank_dict[individual]    
    
    return population

def get_skill(population):

    for individual in population:
        l = [k for k, v in individual['factorial_rank'].items() if v == min(individual['factorial_rank'].values())]
        individual['factorial_skill'] = random.choice(l)
        
    return population

def get_scalar_fitness(population):
    
    for individual in population:
        m = min(individual['factorial_rank'], key=individual['factorial_rank'].get)
        individual['scalar_fitness'] = 1/individual['factorial_rank'][m]
    
    return population

def GeneticAlgorithm(dataset, series):
    
    ngen = 100
    npop = 30
    psel = 0.5
    pcross = 0.5
    mgen = 7
    no_improvement_count = 0
        
    new_population = []
    
    var_names = list(dataset.keys())
    
    divergence_matrix, max_divergence = divergence(series, var_names)
    
    population = initial_population(npop, var_names)

    for individual in population:
        for variable in var_names:
            individual['factorial_cost'][variable], individual['model_size'][variable] = evaluate(dataset[variable], individual)
                
    population = generate_factorial_rank(population, var_names)
    population = get_skill(population)
    population = get_scalar_fitness(population)
    
    population = sorted(population, key=itemgetter('scalar_fitness'))
    
    best_list = [population[0]]
    
    for i in range(ngen):
        print("GENERATION {}".format(i))

        # Selection
        new_population = []
        for j in range(int(npop * psel)):
            new_population.append(double_tournament(population))        

        
        # Crossover
        new = []
        for z in range(int(npop * pcross)):
            child1 = crossover(new_population, divergence_matrix, max_divergence, var_names)
            for variable in var_names:
                child1['factorial_cost'][variable], child1['model_size'][variable] = evaluate(dataset[variable], child1)
            new.append(child1)
            
        new_population.extend(new)
        
        new_population = generate_factorial_rank(new_population, var_names)
        new_population = get_scalar_fitness(new_population)
        
        population = elitism(population, new_population)

        population = sorted(population, key=itemgetter('scalar_fitness'))

        population = population[:npop]
        
        print(population[0])

        best_list.append(population[0])
        
        if best_list[-2]['scalar_fitness'] >= best_list[-1]['scalar_fitness']:
            no_improvement_count +=1
            print("WITHOUT IMPROVEMENT {}".format(no_improvement_count))
            pcross += 0.05
        else:
            no_improvement_count = 0
            pcross = 0.5
        
        if no_improvement_count == mgen:
            break
        
    return best_list
