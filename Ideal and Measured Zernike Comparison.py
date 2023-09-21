# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 22:07:42 2023

@author: teoan
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from scipy.ndimage.measurements import center_of_mass
import math



'''The type of file that was used in this project was in .xyz form and was opened with notepad .
 The processing shown here is specific to the output that present and changes my be needed depending 
 on how the xyz data are loaded for the surfaces in question'''
 
file_path = '/Users/name/Desktop/folder/image_name.xyz'

# Lists to store the values of each column
column3_values = []

# Open the file in read mode
with open(file_path, 'r') as file:
    for line in islice(file, 14, None):     #this basically says to the program to ignore the 1st 14 lines since in our file they have the configuration info and no actual data
        # Split the line into columns using space as the separator
        columns = line.split()
        
        '''The file had NO DATA written for the points that didnt have any value assigned to them.
        The below lines are to make the code ignore this points. Also at the end a # was used so that why when ==# break.'''
        # Check if the column has "NO DATA" or is empty
        # If so, set a default value (e.g., None) or skip the line
        if columns[0]=='#':
            break
        elif columns[2] == "No" or columns[2] == "":
            column3_value = np.nan  # or any other default value
        else:
            column3_value = float(columns[2])

        # Append the values to their respective lists
        column3_values.append(column3_value)  #load the z data only since we can create ourselfs the meshgrid of the x,y using .reshape

column3_array = np.array(column3_values).reshape(1200, 1200)

'''Before finding the center off mas we have to turn np.nan values to 0 otherwise it wont work.
 After we find the cenetr of mass we turn 0 values back to np.nan for uninterapted plotting'''
column3_array[np.isnan(column3_array)] = 0
center = center_of_mass(column3_array)
print(f"Center of mass: ({center[0]}, {center[1]})")
column3_array[column3_array==0] = np.nan
height, width= column3_array.shape


# Create a meshgrid for the entire image
x2, y2 = np.meshgrid(np.arange(width), np.arange(height))

# Calculate distances from the center
distances = np.sqrt((x2 - center[1])**2 + (y2 - center[0])**2)

outer_radius=375 #here we can set an outer range for the surface to only use for example the 80% for the measurements
circular_mask2 = distances <= outer_radius
column3_array[~circular_mask2] = np.nan  

# Create a mask for the circular section
circular_mask = distances <= outer_radius
circular_section = column3_array.copy()

new_image_center_x = width// 2
new_image_center_y = height// 2
new_image = np.zeros((height,width), dtype=float)
new_image[new_image==0] = np.nan

# Calculate the bounding box of the circular region within the new image
x_min = int(center[1] - outer_radius)    #if we want to align the measured surface and the ideal zernyke ploted we can add or subtract some value here
x_max = int(center[1] + outer_radius)    #if we want to align the measured surface and the ideal zernyke ploted we can add or subtract some value here
y_min =  int(center[0] - outer_radius)   #if we want to align the measured surface and the ideal zernyke ploted we can add or subtract some value here
y_max = int(center[0] + outer_radius)    #if we want to align the measured surface and the ideal zernyke ploted we can add or subtract some value here

new_x_min = int(new_image_center_x - (x_max - x_min) // 2)
new_x_max = int(new_image_center_x + (x_max - x_min) // 2)
new_y_min = int(new_image_center_y - (y_max - y_min) // 2)
new_y_max = int(new_image_center_y + (y_max - y_min) // 2)

# Copy the circular region from the original image to the new image
new_image[new_y_min:new_y_max, new_x_min:new_x_max] = circular_section[y_min:y_max, x_min:x_max] 

'''Before finding the center off mas we have to turn np.nan values to 0 otherwise it wont work.
 After we find the cenetr of mass we turn 0 values back to np.nan for uninterapted plotting'''
new_image[np.isnan(new_image)] = 0
center3 = center_of_mass(new_image)
print(f"Center of mass: ({center3[0]}, {center3[1]})")
new_image[new_image==0] = np.nan

x = np.linspace(-1, 1, 1200)
y = np.linspace(-1, 1, 1200)
X, Y = np.meshgrid(x, y)
R = np.sqrt((X)**2 + (Y)**2)
Theta = np.arctan2(Y, X)

'''This definition can create the ideal zernike surfaces for the comparison'''
def zernike_cartesian(r, theta, n, m):
    # Zernike radial polynomial
    R = 0.0
    for s in range((n - abs(m)) // 2 + 1):
        c = (-1) ** s * np.math.factorial(n - s)
        c /= np.math.factorial(s) * np.math.factorial((n + abs(m)) // 2 - s) * np.math.factorial((n - abs(m)) // 2 - s)
        R += c * r ** (n - 2 * s)
    
    # Zernike polynomial
    if m >= 0:
        return R * np.cos(m * theta)
    else:
        return R * np.sin(-m * theta)

# Parameters for the Zernike polynomial
n = 2
m = 2

shape = (1200, 1200)
R2 = np.full(shape, np.nan)
column3_array2=np.full(shape, np.nan)

for xval in range(0,1200):
    for yval in range(0,1200):
        if R[xval,yval]<=0.63:     #MAKE THE IDEAL ZERNYKE SMALLER
            R2[xval,yval]= R[xval,yval]


Z = zernike_cartesian(R2, Theta, n, m) #By adding a constant value in theta we can match the angle of ideal and measured surface

# Find the maximum and minimum values of the measured surface data in micrometers
measured_max = np.nanmax(column3_array)
measured_min = np.nanmin(column3_array)

# Find the maximum and minimum values of the Zernike polynomial surface
zernike_max = np.nanmax(Z)
zernike_min = np.nanmin(Z)

# Calculate the scaling factors for the Zernike polynomial
scale_factor_max = (measured_max - measured_min) / (zernike_max - zernike_min)
scale_factor_min = measured_min - (zernike_min * scale_factor_max)                 #Scale the zernike adeal to match the hight of the surface measured

# Scale the Zernike polynomial surface
Z_scaled = Z * scale_factor_max + scale_factor_min

'''Before finding the center off mas we have to turn np.nan values to 0 otherwise it wont work.
 After we find the cenetr of mass we turn 0 values back to np.nan for uninterapted plotting'''
Z_scaled[np.isnan(Z_scaled)] = 0
center2 = center_of_mass(Z_scaled)
print(f"Center of mass: ({center2[0]}, {center2[1]})")
Z_scaled[Z_scaled==0] = np.nan


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x2 = np.linspace(-1, 1, 1200)
y2 = np.linspace(-1, 1, 1200)
X2, Y2 = np.meshgrid(x2, y2)

# Plot the Zernike polynomial
surf=ax.plot_surface((X2), (Y2), new_image, cmap='plasma', alpha=0.9) #X-0.2 to shift the center and overlap
ax.plot_surface(R2 * np.cos(Theta), R2 * np.sin(Theta),Z_scaled, cmap='viridis', alpha=0.5)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Zernike Polynomial (n={n}, m={m})')
# fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a colorbar

ax.view_init(elev=0, azim=70)
plt.show()


'''In circular_section 3 we subtract the two surfaces (ideal and measured) and find the absolute difference between them in each point'''
circular_section3= np.abs(new_image-Z_scaled)  
center23 = center_of_mass(circular_section3)
height_f = circular_section3.shape[0]
width_f=circular_section3.shape[1]
circular_section3[np.isnan(circular_section3)] = 0
rms=[]


for i in range(width_f):
    for j in range(height_f):
        rms.append(circular_section3[j,i])  
        
result_list = [((value)**2) for value in rms]
a=sum(result_list)
mean_squared = (a / len(result_list))
RMS=math.sqrt(mean_squared)                #RMS is the rms error assosiated with the compariosn of the two surfaces
print(RMS)

circular_section3[circular_section3==0] = np.nan



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

err_max = np.nanmax(circular_section3)
print('the pv error is',err_max)

surf2=ax.plot_surface(X, Y, circular_section3, cmap='plasma', alpha=0.9)
plt.tight_layout()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.view_init(elev=90, azim=90) #control the 3d plot elevation viewing angle and the azimuthial viewing angle
fig.colorbar(surf2, shrink=0.5, aspect=5)  # Add a colorbar
plt.show()        

np.savetxt('data.txt', column3_array, delimiter=',')
np.savetxt('ideal.txt', Z_scaled, delimiter=',')                  #save the results
np.savetxt('rms error.txt', circular_section3, delimiter=',')