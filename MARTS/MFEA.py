# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:00:31 2023

@author: Patricia
"""

import numpy as np
import pandas as pd
from operator import itemgetter
import random
import ray
#from sklearn.neighbors import KernelDensity



def genotype(n_estimators, min_samples_leaf, max_features, factorial_cost, factorial_rank, factorial_skill, scalar_fitness, model_size, size):
    """

    """

    ind = dict(n_estimators=n_estimators, min_samples_leaf = min_samples_leaf, max_features = max_features,
               factorial_cost = factorial_cost, 
               factorial_rank = factorial_rank,
               factorial_skill = factorial_skill, 
               scalar_fitness = scalar_fitness,
               model_size = model_size,
               size = size
               )
    
    return ind


def random_genotype(var_names):
    """

    """

    return genotype(
        random.randint(5, 1000), #n_estimators
        random.randint(1, 30), #min_samples_leaf
        random.choice(['sqrt', 'log2', 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), #max_features
        dict.fromkeys(list(var_names), None), #factorial_cost
        dict.fromkeys(list(var_names), None), #factorial_rank
        None, #factorial_skill
        None,  #scalar_fitness
        dict.fromkeys(list(var_names), None),  #model_size
        None
    )



def initial_population(n, var_names):
    """

    """
    
    pop = []
    for i in range(n):
        pop.append(random_genotype(var_names))
        
    return pop


def phenotype(individual, X_train, y_train):
    """

    """
    
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(n_estimators=individual['n_estimators'], 
                                  min_samples_leaf=individual['min_samples_leaf'],
                                  max_features=individual['max_features'],
                                  bootstrap=True, n_jobs=-1, random_state=0)
    
    model.fit(X_train, y_train)

    return model


@ray.remote
def evaluate_parallel(dataset, individual, params):
    return evaluate(dataset, individual, params)


def evaluate(dataset, individual, params):
    """

    """
    import measures
    from sklearn.ensemble import RandomForestRegressor

    errors = []
    size = []
    
    window = np.arange(0, (params['size_train']*3)+params['size_test'], params['size_train'])
    window = np.delete(window, -1)

    for w in window:
        
        X_train = dataset['X_train'].loc[w:w+params['size_train']]
        X_test = dataset['X_train'].loc[w+params['size_train']:w+params['size_train']+params['size_test']-1]
                
        y_train = dataset['y_train'].loc[w:w+params['size_train']]
        y_test = dataset['y_train'].loc[w+params['size_train']:w+params['size_train']+params['size_test']-1]
        
        model = RandomForestRegressor(n_estimators=individual['n_estimators'], 
                                      min_samples_leaf=individual['min_samples_leaf'],
                                      max_features=individual['max_features'],
                                      bootstrap=True, n_jobs=-1, random_state=0)
        X_train_copy = X_train.copy()
        y_train_copy = y_train.copy()
        model.fit(X_train_copy, y_train_copy)
        
        del X_train
        del y_train
        
        X_test_copy = X_test.copy()
        y_test_copy = y_test.copy()
        forecasts = model.predict(X_test_copy)
        
        del X_test
        del y_test
        
        mea = measures.Measures(model)
        nrmse = mea.nrmse(y_test_copy, forecasts)

        errors.append(nrmse)

    
    nrmse = np.mean(errors)
    size = individual['n_estimators'] * sum([tree.tree_.max_depth for tree in model.estimators_])
    
     
    return nrmse, size


def tournament(population, objective):
    """

    """
    n = len(population) - 1

    r1 = random.randint(0, n) if n > 2 else 0
    r2 = random.randint(0, n) if n > 2 else 1
    
    if objective == 'scalar_fitness':
        ix = r1 if population[r1]['scalar_fitness'] > population[r2]['scalar_fitness'] else r2
    elif objective == 'size':
        ix = r1 if population[r1]['size'] < population[r2]['size'] else r2
    else:
        ix = r1 if sum(population[r1]['model_size'].values()) < sum(population[r2]['model_size'].values()) else r2

        
    return population[ix]


def double_tournament(population, i):
    """

    """

    ancestor1 = tournament(population, 'scalar_fitness')
    ancestor2 = tournament(population, 'scalar_fitness')
    
    selected = tournament([ancestor1, ancestor2], 'model_size')
    
    # if i == 0:
    #     selected = tournament([ancestor1, ancestor2], 'model_size')
    # else:
    #     selected = tournament([ancestor1, ancestor2], 'size')

    return selected




def crossover(population, divergence_matrix, max_divergence, var_names):
    """

    """
    import random

    n = len(population) - 1

    r1, r2 = 0, 0
    while r1 == r2:
        r1 = random.randint(0, n)
        r2 = random.randint(0, n)
        
    divergence = divergence_matrix.loc[population[r1]['factorial_skill']][population[r2]['factorial_skill']]
   
    pmut = 0.3
   
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
                               skill, None, dict.fromkeys(list(var_names)), None)
        
        if pmut > random.uniform(0,1):
            descendent = mutation(descendent, var_names)
            
        return descendent, skill
            
    else:
        
        descendent = mutation(population[r1], var_names)               

        return descendent, population[r1]['factorial_skill']



def mutation(individual, var_names):
    """

    """

    n_estimators = min(1000, max(5,int(individual['n_estimators'] + (np.random.normal(0, 10)*np.random.choice([-1,1],1)[0]))))
    min_samples_leaf = min(30, max(1,int(individual['min_samples_leaf'] + (np.random.normal(0, 2)*np.random.choice([-1,1],1)[0]))))
    max_features = random.choice(['sqrt', 'log2', 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    descendent = genotype(n_estimators, min_samples_leaf, max_features,
                           dict.fromkeys(list(var_names), None),
                           dict.fromkeys(list(var_names), None),
                           individual['factorial_skill'], None, dict.fromkeys(list(var_names)), None)

    return descendent


def elitism(population, new_population):
    """

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
       
    max_divergence = np.round(max(divergence_matrix.max()/2),1)
    
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

def get_size(population):
    
    for individual in population:
        m = individual['factorial_skill']
        individual['size'] = individual['model_size'][m]
        
    return population

def GeneticAlgorithm(dataset, series, params, distributive_version):
    
    print("HPO started...")
    
    #no_improvement_count = 0
        
    new_population = []
    
    var_names = list(dataset.keys())
    
    divergence_matrix, max_divergence = divergence(series, var_names)
    
    population = initial_population(params['npop'], var_names)

            
    if distributive_version:
        n = 0
        for individual in population:
            print(n)
            n = n + 1
            results = []
            for variable in var_names:
                    results.append(evaluate_parallel.remote(dataset[variable], individual, params))
            r = 0
            parallel_results = ray.get(results)
            for variable in var_names:
                individual['factorial_cost'][variable], individual['model_size'][variable] = parallel_results[r][0], parallel_results[r][1]
                r = r + 1
    else:
        for individual in population:
            for variable in var_names:
                individual['factorial_cost'][variable], individual['model_size'][variable] = evaluate(dataset[variable], individual, params)
                
            
           
    population = generate_factorial_rank(population, var_names)
    population = get_skill(population)
    population = get_scalar_fitness(population)
    population = get_size(population)
    
    population = sorted(population, key=itemgetter('scalar_fitness'))
    
    
    best_list = []
    df_best = pd.DataFrame(columns=population[0]['factorial_cost'].keys())
    df_best_list = []
    for ind in population:
        df_best.loc[len(df_best)] = np.array(list(ind['factorial_cost'].values())).reshape(1,-1)[0]
    
    for var in var_names:
        best_list.append(population[df_best[[var]].idxmin()[0]])
        
    df_best_list.append(best_list)
        
    pop = [population]
    best_zero = [population[0]]
    
    for i in range(params['ngen']):
        print("GENERATION {}".format(i))

        # Selection
        new_population = []
        for j in range(int(params['npop'] * params['psel'])):
            new_population.append(double_tournament(population, i))  


        
        # Crossover
        new = []
        for z in range(int(params['npop'])):
            print(z)
            child, variable = crossover(new_population, divergence_matrix, max_divergence, var_names)
            
            if distributive_version:
                for variable in var_names:                
                    results = []
                    for variable in var_names:
                        results.append(evaluate_parallel.remote(dataset[variable], child, params))
                    
                    r = 0
                    parallel_results = ray.get(results)
                    for variable in var_names:
                        child['factorial_cost'][variable], child['model_size'][variable] = parallel_results[r][0], parallel_results[r][1]
                        child['size'] = child['model_size'][variable]
                        r = r + 1
            else:
                for variable in var_names:
                    child['factorial_cost'][variable], child['model_size'][variable] = evaluate(dataset[variable], child, params)
                    child['size'] = child['model_size'][variable]
                    new.append(child)
                        
            
        new_population.extend(new)
        
        new_population = generate_factorial_rank(new_population, var_names)
        new_population = get_scalar_fitness(new_population)
        new_population = get_size(new_population)
        
        population = elitism(population, new_population)

        population = sorted(population, key=itemgetter('scalar_fitness'))

        population = population[:params['npop']]
        
        best_zero.append(population[0])
        
        
        best_list = []
        df_best = pd.DataFrame(columns=population[0]['factorial_cost'].keys())

        for ind in population:
            df_best.loc[len(df_best)] = np.array(list(ind['factorial_cost'].values())).reshape(1,-1)[0]
        
        for var in var_names:
            best_list.append(population[df_best[[var]].idxmin()[0]])
            
        df_best_list.append(best_list)
        
        # if best_zero[-2]['scalar_fitness'] >= best_zero[-1]['scalar_fitness']:
        #     no_improvement_count +=1
        #     print("WITHOUT IMPROVEMENT {}".format(no_improvement_count))
        # else:
        #     no_improvement_count = 0
        
        # if no_improvement_count == params['mgen']:
        #     break
        
        pop.append(population)
        
    return df_best_list[-1]


    







