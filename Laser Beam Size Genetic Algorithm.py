# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:05:24 2023

@author: teoan
"""

import numpy as np
import random
import pylablib as pll
pll.par["devices/dlls/thorlabs_tlcam"] = "path/to/dlls"
from pylablib.devices import Thorlabs
import matplotlib.pyplot as plt
import time
from scipy.ndimage.measurements import center_of_mass
import socket
import pickle
import imageio
import laserbeamsize as lbs
import os
import math
import warnings
import tifffile


# -unhash if their is a Runtime warning that needs to be addressed because computations being slow due to low spec computer ( same with line 318)
# warnings.filterwarnings("ignore", category=RuntimeWarning) 


start_time=time.time()

'''Creates the PWM value range. If the quantisation is too much accuracy and is not 
needed due to noise a step (like this case being 2) can be added to help with convergence'''
values_list = list(range(0, 4095, 2))

'''The below lines are the ones that are responsible for the control box/ PC communication cteating a 
server/client relationship. In the serverAddress add your raspbery Pi Address instead (ex.111.222.333.444)'''
serverAddress=('111.222.333.444',2222)
bufferSize=1024
UDPClient=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

# Constants
POPULATION_SIZE = 60
NUM_ACTUATORS = 5
COMMAND_RANGE = 4096
MUTATION_RATE = 0.3
TOP_IND = 20
NUM_GENERATIONS = 80
start_cam_expo=0.000298 #put your appropriate camera exposition
min_count = 42 #this parameter is the value that if this many individuals have the same exact command stop the GA due to convergence

'''The below 8 lines are responsible for the code/camera contro to grab the images nessesary. They are based of the pylablib module.'''
Thorlabs.list_cameras_tlcam()
cam1 = Thorlabs.ThorlabsTLCamera(serial="14308")      
cam1.set_exposure(start_cam_expo)
cam1.setup_acquisition(nframes=24)  
cam1.start_acquisition()
cam1.set_roi(0,1440,0,1080)
cam1.wait_for_frame()  # wait for the next available frame
capture = cam1.read_oldest_image().astype(np.uint8)  # get the oldest image which hasn't been read yet

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


def fitness_function(individual):
    if generation>=0:
        cmd= f'{individual}'  #this lines send the individual's PWM values in a string form to the control box in order to change the DM surface
        cmd=cmd.encode('utf-8')
        UDPClient.sendto(cmd,serverAddress)
        Q2=[]
        v3=[]
        for _ in range(5):  #This loop determines the averaging for every individual to reduce error due to noise, here it is 5
            
            cam_expo=start_cam_expo
            raw_image_array=cam1.grab(1)
            image_array=np.array(raw_image_array)
            adjusted_image=image_array.squeeze()
            
            v2=np.nanmax(adjusted_image) #Measure the max intensity in the image 
            adjusted_image = np.where(adjusted_image < 0.08*v2, 0, adjusted_image)  #application of a filter. In this case its 0.08, so 8% filter of the max intensity
            
            '''Here we save the image that we took because the laserberamsize module that measures the diameters only opens
            grayscale images. So wee save the image , open it for the measurements and then later delete it.'''
            temp_path = "temp_plot.tif"
            tifffile.imsave(temp_path, adjusted_image) #save the image
            beam = imageio.imread(temp_path) #open the image
            x, y, dx, dy, phi = lbs.beam_size(beam) #make the measurement
            
            '''Here we say that if the center of mass (so the x,y) is too close the the image's adges we add to the quality an 
            additional 450 value as a 'penalty'. This is because if the spot is close to the edge the measurements of the 
            laserbeamsize module are wrong due to insufficient integration area and this individual will be favoured unfairly.'''
            if x<200 or x>1240 or y<180 or y>950:  
                Q=dx+dy+450
                Q1= np.abs(dx-dy)
            else:
                Q=dx+dy
                Q1= np.abs(dx-dy)
                
            '''Q3 is the total quality value while Q is the diameter term, Q1 is the symmetry components and the last rem is the
            intensity dependent term to avoid convergence to a local spot'''   
            Q3=Q+(2*Q1)+(500000/((v2/10)**2)) 
            Q2.append(Q3)
            v3.append(v2)
            os.remove(temp_path) #here we remove the image that we saved 
            
        squared_values = [t ** 2 for t in Q2]
        mean_squared = sum(squared_values) / len(squared_values)
        rms=math.sqrt(mean_squared)
        
        squared_values2 = [(t2-rms) ** 2 for t2 in Q2]
        mean_squared2 = sum(squared_values2) / len(squared_values2)
        rms2=math.sqrt(mean_squared2)
        
        squared_values3 = [t3 ** 2 for t3 in v3]
        mean_squared3 = sum(squared_values3) / len(squared_values3)
        rms3=math.sqrt(mean_squared3)
        
        squared_values4 = [(t4-rms3) ** 2 for t4 in v3]
        mean_squared4 = sum(squared_values4) / len(squared_values4)
        rms4=math.sqrt(mean_squared4)
        
        print('Generation',generation, 'Quality:',rms,"+-",rms2)
    
        # Create a figure and axes
        fig, ax = plt.subplots()
    
        # Display the original image
        ax.imshow(adjusted_image, cmap='gray')
    
        ax.set_title(f'Generation {generation},Quality:{rms:.4f},individual:{individual}')
        
        cmd= '[0,0,0,0,0]'  #makes all the actuators go to 0 before the next command to reduce the hysteresis as much as possible
        cmd=cmd.encode('utf-8')
        UDPClient.sendto(cmd,serverAddress)
        return (rms,rms3,rms2,rms4)


# Generate a random individual (solution)
def generate_random_individual():     #in the fixed list some 'forced' configuration can be put if we need to. otherwise we can leave empty
    # Create a list of fixed lists
    fixed_lists = [
        [1000,1000,1000,1000,1000],
        [200, 200, 200, 200, 200],
        [333,0,0,0,0],
        [666,0,0,0,0],
        [2000,0,0,0,0],
        [1000,0,0,0,0],
        [0,666,666,666,666],
        [0,333,333,333,333],
        [0,2000,2000,2000,2000],
        [0,1000,1000,1000,1000],
        [400,1500,0,1500,0],
        [850,2000,0,2000,0],
        [400,0,1500,0,1500],
        [850,0,2000,0,2000],
        [0,1000,0,1000,0],
        [0,2000,0,2000,0],
        [0,333,0,333,0],
        [0,666,0,666,0],
        [0,0,666,0,666],
        [0,0,333,0,333],
        [0,0,1000,0,1000],
        [0,0,2000,0,2000]        
    ]
    
    # Calculate how many random individuals are needed to reach the desired population size
    random_individuals_needed = POPULATION_SIZE - len(fixed_lists)
    
    # Generate random individuals and add them to the population
    random_individuals = [
        [random.choice(values_list) for _ in range(NUM_ACTUATORS)]
        for _ in range(random_individuals_needed)
    ]
    
    # Combine fixed and random individuals to create the initial population
    initial_population = fixed_lists + random_individuals
    
    return initial_population


# Create the initial population
# population = [generate_random_individual() for _ in range(POPULATION_SIZE)]
population = generate_random_individual()

# List to store individuals and their corresponding generations
generation_data = []
elite_data=[]
elite_q=[]
qualities2=[]
generationx2=[]
gen=[]
stable_ind=[]

# Genetic Algorithm loop
for generation in range(NUM_GENERATIONS):
    
    # Evaluate the fitness of each individual in the population
    fitness_scores = [(individual, fitness_function(individual)) for individual in population]
    
    '''Has the same individual in all runs to check for stability. idealy if no noise or hysteresis exist this individual 
    should always have the same quality value. Of course that never is the case. instead of 200s whatever value we want can be the input.'''
    fitness_scores2 = fitness_function([200,200,200,200,200])
    
    
    # Sort the population based on fitness (minimization problem)
    fitness_scores.sort(key=lambda x: x[1][0])

    # Extract the most elite individual (best one) for elitism 
    elite_individual = fitness_scores[0][0]
    elite_data.append(elite_individual)  #holds the info on the best individuals each generation (PWM values of teh actuators)
    
    
    elite_q.append((fitness_scores[0][1][0],fitness_scores[0][1][1],fitness_scores[0][1][2],fitness_scores[0][1][3])) #holds all the info about the best individuals of each generation. Thats what we see at the end.
    
    stable_ind.append((fitness_scores2[0],fitness_scores2[2])) #holds the data for the test/stable individual to test the stability
    
    # Select the best 10 individuals for creating new children
    top_individuals = [individual for individual, _ in fitness_scores[:TOP_IND]]
    
    # Store individuals, fitness scores, and their generations in the generation_data list
    generation_data.extend([(individual, fitness, generation) for individual, fitness in fitness_scores])
    qualities= [item[1][0] for item in fitness_scores] 
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

        # Perform mutation 
        if random.random() < 2*MUTATION_RATE:
            mutated_gene = random.sample(range(5), 2)
            child[mutated_gene[0]] = add_or_subtract(child[mutated_gene[0]], random.randint(0, 50))
            child[mutated_gene[1]] = add_or_subtract(child[mutated_gene[1]], random.randint(0, 50))

        new_generation.append(child)

    # Update the population with the new generation
    population = new_generation
    


negated_list_of_lists = [[-value for value in sublist] for sublist in qualities2]
negated_values = [-value for value,_,_,_ in elite_q]

fig, ax1=plt.subplots()
ax1.scatter(generationx2,negated_list_of_lists,s=1)
ax1.plot(gen,negated_values,'r-')  #plots all the individuals' qualities in one plot
plt.show()


fig, ax2=plt.subplots()
ax2.plot(gen,negated_values)
ax2.scatter(gen,negated_values,c='red')  #plots the elite individual data from each generation
plt.show()

best_fitness = max(negated_values)

# Your final solution
for u in range(NUM_GENERATIONS):
    if -elite_q[u][0]==max(negated_values):
        z=elite_data[u]
        print("Best Individual:",z )
# print("Best Individual:", best_individual)
print("Best Fitness:", best_fitness)

end_time=time.time()
time_taken=end_time-start_time
# Convert time taken to hours, minutes, and seconds
hours, remainder = divmod(time_taken, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Time taken: {int(hours):02d} hours {int(minutes):02d} minutes {seconds:.2f} seconds")  #prints the time needed to complete the code

#warnings.resetwarnings()
    
    
cam1.close()