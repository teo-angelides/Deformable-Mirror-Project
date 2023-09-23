# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 22:07:43 2023

@author: teoan
"""

'''This Is the numerical genetic algorithm in which the parameters can be changed 
and after a number of runs we can see the average error that we should expect to 
have in our measurement. In other words what is the average deviation from getting 
the perfect results. Here the perfect result is the variable called answer. and the 
loop range expresses the times we repeat the GA, the higher the number the more accurate 
the result will be. This code also includes a random error assosiated to make the results 
closer to real results that have assosiated error too. '''

import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math

start_time=time.time()


fit=[]
for qu in range(100):  #how many times the GA is repeated. the more runs the better statistics we get
    # Constants
    POPULATION_SIZE =45
    NUM_ACTUATORS = 5
    COMMAND_RANGE = 4096
    MUTATION_RATE = 0.3
    TOP_IND = 20
    NUM_GENERATIONS = 25
    min_count = 57
    answer=[0,0,0,0,0]   #The Ideal Answer we are looking for  
    
    '''Here we calculate the rms error from the individuals created in the GA and the ideal answer.
    Its un artificial way to evaluate the GAs performance without the need to use the lab.'''
    def calculate_error(goal,current):
        return np.sqrt(((goal[0] - current[0])/4095) ** 2 + ((goal[1] - current[1])/4095) ** 2 + ((goal[2] - current[2])/4095) ** 2 + ((goal[3] - current[3])/4095) ** 2 + ((goal[4] - current[4])/4095) ** 2)
    
    
    '''The 2 definitions below are used for stopping the GA if it converges'''
    def count_identical_lists(main_list, target_list, min_count):
        count = 0
        for sublist in main_list:
            if sublist == target_list:
                count += 1
                if count >= min_count:
                    return True
        return False
    
    def check_identical_lists(main_list, min_count):
        for sublist in main_list:
            if count_identical_lists(main_list, sublist, min_count):
                return True
        return False
    
    
    '''This definition is used in the mutation to randomly add or substract a random value. It also includes the edge requirements to 
    not go above or below the PWM range allowed'''
    def add_or_subtract(a, b):
        operation = random.choice([0, 1])
        if operation == 1:
            if a+b>4095:
                 return 4095
            else:
                return a+b
        else:
            if a-b<0:
                return 0
            else:
                return a-b
            
    '''This definition is used to add the random error in each run. In this case is between 0-20% for each run'''
    def add_or_subtract2(a):
        operation = random.choice([0, 1])
        b=random.random()*0.2
        if operation == 1:
            return a*b
        else:
            return -a*b
    
    def fitness_function(individual):
        return (calculate_error(answer,individual)+add_or_subtract2(calculate_error(answer,individual)))
    
    # Generate a random individual (solution)
    def generate_random_individual():
        return [random.randint(0, COMMAND_RANGE-1) for _ in range(NUM_ACTUATORS)]
    
    # Create the initial population
    population = [generate_random_individual() for _ in range(POPULATION_SIZE)]
    
    # List to store individuals and their corresponding generations
    generation_data = []
    elite_data=[]
    elite_q=[]
    qualities2=[]
    generationx2=[]
    gen=[]
    # Genetic Algorithm loop
    for generation in range(NUM_GENERATIONS):
        
        # Evaluate the fitness of each individual in the population
        fitness_scores = [(individual, fitness_function(individual)) for individual in population]
    
        # Sort the population based on fitness (minimization problem)
        fitness_scores.sort(key=lambda x: x[1])
    
        # Extract the most elite individual (best one) for elitism
        elite_individual = fitness_scores[0][0]
        elite_data.append(elite_individual) 
        elite_q.append(fitness_scores[0][1])
        
        # Select the best top individuals for creating new children
        top_individuals = [individual for individual, _ in fitness_scores[:TOP_IND]]
        
        # Store individuals, fitness scores, and their generations in the generation_data list
        generation_data.extend([(individual, fitness, generation) for individual, fitness in fitness_scores])
        qualities= [item[1] for item in fitness_scores] 
        qualities2.append(qualities)
        generationx=np.array([generation] * POPULATION_SIZE)
        generationx2.append(generationx)
        gen.append(generation)
        
    
        if check_identical_lists(population, min_count): #if converge stop 
            break
        
        # Create the next generation through crossover and mutation
        new_generation = [elite_individual]
    
        # Perform crossover and mutation to create the rest of the population
        while len(new_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(top_individuals, 2)
    
            # Perform crossover (you can use different techniques like single-point, multi-point, etc.)
            crossover_point = random.randint(1, NUM_ACTUATORS - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            # child = parent1[:3] + parent2[3:]
    
            # Perform mutation 
            if random.random() < MUTATION_RATE:
                mutated_gene = random.sample(range(5), 2)
                child[mutated_gene[0]] = add_or_subtract(child[mutated_gene[0]], random.randint(0, 50))
                child[mutated_gene[1]] = add_or_subtract(child[mutated_gene[1]], random.randint(0, 50))
        
    
            new_generation.append(child)
    
        # Update the population with the new generation
        population = new_generation
    
    # After the loop, select the best individual as the final solution
    best_individual, best_fitness = min(fitness_scores, key=lambda x: x[1])
    
    negated_list_of_lists = [[-value for value in sublist] for sublist in qualities2]
    negated_values = [-value for value in elite_q]
    
    
    '''Unhash the plotting for visual inspection but if speed is needed keeping them hashed can make the process quicker'''
    # fig, ax1=plt.subplots()
    # ax1.scatter(generationx2,negated_list_of_lists,s=1)    #plots all the individuals' qualities in one plot
    # ax1.plot(gen,negated_values,'r-') 
    # plt.show()
    
    # fig, ax2=plt.subplots()
    # ax2.plot(gen,negated_values)
    # ax2.scatter(gen,negated_values,c='red')    #plots the elite individual data from each generation
    # ax2.set_ylim(-0.5, 0)
    # plt.show()
    
    
    # Your final solution
    print("Goal Individual:", answer)
    print("Best Individual:", best_individual)
    print("Best Fitness:", best_fitness)
    
    end_time=time.time()
    time_taken=end_time-start_time
    # Convert time taken to hours, minutes, and seconds
    hours, remainder = divmod(time_taken, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Time taken: {int(hours):02d} hours {int(minutes):02d} minutes {seconds:.2f} seconds") #prints the time needed to complete the code
    
    fit.append(best_fitness)
   
squared_values2 = [(t) ** 2 for t in fit]
mean_squared2 = sum(squared_values2) / len(squared_values2)
rms2=math.sqrt(mean_squared2)
print('THE RMS ERROR IS:',rms2)   