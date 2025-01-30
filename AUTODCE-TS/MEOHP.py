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
from itertools import product
import warnings
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



def genotype(model,n_estimators, min_samples_leaf, max_features, factorial_cost, factorial_rank, factorial_skill, scalar_fitness, model_size, size):
    """

    """

    ind = dict(model=model, n_estimators=n_estimators, min_samples_leaf = min_samples_leaf, max_features = max_features,
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
        random.choice(['RandomForest', 'LGBoost', 'XGBoost']), #estimators
        random.randint(2, 500), #n_estimators
        random.randint(2, 30), #min_samples_leaf
        random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), #max_features
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




def LGBoost_depth(tree):
    if 'leaf_index' in tree:
        return 1
    else:
        try:
            left_depth = LGBoost_depth(tree['left_child'])
        except:
            left_depth = 1
        try:
            right_depth = LGBoost_depth(tree['right_child'])
        except:
            right_depth = 1
        return max(left_depth, right_depth) + 1

def XGBoost_depth(tree_dump):
    lines = tree_dump.split('\n')
    depths = [0]  # List to store the depth of each node
    max_depth = 0
    
    for line in lines:
        if line.strip() == '':
            continue
        depth = line.count('\t')
        depths.append(depth)
        if depth > max_depth:
            max_depth = depth
    
    return max_depth

@ray.remote
def evaluate_parallel(dataset, individual, params):
    return evaluate(dataset, individual, params)


def evaluate(dataset, individual, params):
    """

    """
    
    from AUTOTSF import measures
    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    warnings.filterwarnings("ignore", category=UserWarning)

    errors = []
    size = []
    
    window = np.arange(0, (params['size_train']*2)+params['size_test'], params['size_train'])
    window = np.delete(window, -1)
    

    for w in window:
        
        X_train = dataset['X_train'].loc[w:w+params['size_train']]
        X_test = dataset['X_train'].loc[w+params['size_train']:w+params['size_train']+params['size_test']-1]
                
        y_train = dataset['y_train'].loc[w:w+params['size_train']]
        y_test = dataset['y_train'].loc[w+params['size_train']:w+params['size_train']+params['size_test']-1]
        
        
        if individual['model'] == 'RandomForest':
            model = RandomForestRegressor(n_estimators=individual['n_estimators'], 
                                          min_samples_leaf=individual['min_samples_leaf'],
                                          max_features=individual['max_features'],
                                          bootstrap=True, n_jobs=-1)
        elif individual['model'] == 'LGBoost':
        
            model = LGBMRegressor(n_estimators = individual['n_estimators'], 
                                      colsample_bytree = individual['max_features'],
                                      min_child_samples = individual['min_samples_leaf'], 
                                      n_jobs = -1, verbosity = -1)
        else:
            model = XGBRegressor(n_estimators = individual['n_estimators'], 
                                          colsample_bytree = individual['max_features'],
                                          min_child_weight = individual['min_samples_leaf'],
                                          subsample = 0.5)
        
        
        X_train_copy = X_train.copy()
        y_train_copy = y_train.copy()
        model.fit(X_train_copy.values, y_train_copy.values)
        
        del X_train
        del y_train
        
        X_test_copy = X_test.copy()
        y_test_copy = y_test.copy()
        forecasts = model.predict(X_test_copy.values)
        
        del X_test
        del y_test
        
        mea = measures.Measures(model)
        nrmse = mea.nrmse(y_test_copy, forecasts)

        errors.append(nrmse)

    
    nrmse = np.mean(errors)
    #size = individual['n_estimators'] * sum([tree.tree_.max_depth for tree in model.estimators_])
    #size = individual['n_estimators'] * individual['min_samples_leaf']
    
    if individual['model'] == 'RandomForest':
        size = individual['n_estimators'] * sum([tree.tree_.max_depth for tree in model.estimators_])
    elif individual['model'] == 'LGBoost':
        booster = model.booster_
        trees = booster.dump_model()['tree_info']
        size = individual['n_estimators'] * sum([LGBoost_depth(tree['tree_structure']) for tree in trees])
    else:
        booster = model.get_booster()
        trees_dump = booster.get_dump(with_stats=True)
        size = individual['n_estimators'] * sum([XGBoost_depth(tree) for tree in trees_dump])
        
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
        model = best['model']
        
        if random.uniform(0,1) > 0.5:
            skill = best['factorial_skill']
        else:
            skill = worst['factorial_skill']
            
        descendent = genotype(model, n_estimators, min_samples_leaf, max_features,
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
    n_estimators = min(500, max(5,int(individual['n_estimators'] + (np.random.normal(0, 10)*np.random.choice([-1,1],1)[0]))))
    min_samples_leaf = min(30, max(1,int(individual['min_samples_leaf'] + (np.random.normal(0, 2)*np.random.choice([-1,1],1)[0]))))
    #max_features = random.choice(['sqrt', 'log2'])
    max_features = random.choice([0.2, 0.3, 0.4,0.5])
    model = random.choice(['RandomForest', 'LGBoost', 'XGBoost'])
    
    descendent = genotype(model, n_estimators, min_samples_leaf, max_features,
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

    pairs = list(product(var_names, var_names))
    for par in pairs:
        if par[0] != par[1]:
            divergence_matrix.loc[par[0]][par[1]] = distance.jensenshannon(dataset[par[0]], dataset[par[1]])
        else:
            divergence_matrix.loc[par[0]][par[1]] = -1
       
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
    
    len_population = len(population)
    for individual in range(len_population):
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
        
    new_population = []
    
    var_names = list(dataset.keys())
    
    population = initial_population(params['npop'], var_names)

    print("HPO started...")  

    divergence_matrix, max_divergence = divergence(series, var_names)
     
    if distributive_version:
        n = 0
        for individual in population:
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
    #best_zero = [population[0]]
    
    for i in tqdm(range(params['ngen']), desc="Processing"):

        #print("GENERATION {}".format(i))

        # Selection
        new_population = []
        for j in range(int(params['npop'] * 0.5)):
            new_population.append(double_tournament(population, i))  


        
        # Crossover
        new = []
        for z in range(int(params['npop'])):

            child, variable = crossover(new_population, divergence_matrix, max_divergence, var_names)
            
            if distributive_version:               
                results = []
                for variable in var_names:
                    results.append(evaluate_parallel.remote(dataset[variable], child, params))
                
                r = 0
                parallel_results = ray.get(results)
                for variable in var_names:
                    child['factorial_cost'][variable], child['model_size'][variable] = parallel_results[r][0], parallel_results[r][1]
                    child['size'] = child['model_size'][variable]
                    r = r + 1
                new.append(child)
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
                
        #best_zero.append(population[0])
        
        best_list = []
        df_best = pd.DataFrame(columns=population[0]['factorial_cost'].keys())

        for ind in population:
            df_best.loc[len(df_best)] = np.array(list(ind['factorial_cost'].values())).reshape(1,-1)[0]
            
        for var in var_names:
            best_list.append(population[df_best[[var]].idxmin()[0]])
            
        df_best_list.append(best_list)
        
        pop.append(population)
        
    return df_best_list, pop


    







