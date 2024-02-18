# Script to create csv data for machine learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create data
rows = 40
samples = 100 # columns

# random array
random_arr = (np.random.rand(rows, samples) - 0.5) * 0.1

# offset array
constant_values = np.linspace(-1, 1, rows)
offset_arr = np.tile(constant_values, (samples, 1)).T
np.random.shuffle(offset_arr)

# shapes array
shapes_arr = np.zeros((rows, samples))
subdivision = int(0.25 * rows)

# linear
for i in range(subdivision):
    random_slope = np.random.uniform(-1, 1)
    random_bias = np.random.uniform(-1, 1)
    
    line = random_slope * np.linspace(0, 1, samples) + random_bias
    
    shapes_arr[i, :] = line

# sinusoidal
for i in range(subdivision, subdivision + subdivision):
    random_shift = np.random.uniform(0, 2*np.pi)
    random_bias = np.random.uniform(-1, 1)
    random_y_scale = np.random.uniform(0.5, 1.5)
    random_x_scale = np.random.uniform(-1, 1) * 2*np.pi
    random_flip = np.random.choice([-1, 1])
    
    time = np.linspace(0 + random_shift, random_x_scale + random_shift, samples)
    
    shapes_arr[i, :] = random_flip * random_y_scale * np.sin(time) + random_bias

# nonlinear (inertia)
control_points = np.array([0, 0, 0, 0, 0.1, 0.15, 0.2, 0.22, 0.29, 0.4, 0.5, 0.7, 0.8, 0.9, 0.94, 0.96, 0.98, 0.99, 1, 1, 1, 1])
nonlinear_values = np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(control_points)), control_points)
for i in range(subdivision + subdivision, subdivision + subdivision + subdivision):
    scale_factor = np.random.uniform(0.5, 1.5)
    bias_shift = np.random.uniform(-0.2, 0.2)
    x_scale_factor = np.random.uniform(0.5, 1.5)
    random_flip = np.random.choice([-1, 1])
    
    nonlinear_values = random_flip * np.interp(x_scale_factor * np.linspace(0, 1, samples), np.linspace(0, 1, len(control_points)), control_points) * scale_factor + bias_shift
    
    shapes_arr[i, :] = nonlinear_values

# conllision shape
for i in range(3 * subdivision, 4 * subdivision):
    random_shift = np.random.uniform(-np.pi, np.pi)
    random_jump_location = np.random.randint(samples - 100, samples)
    random_jump_size = np.random.uniform(1, 5)
    
    time = np.linspace(0 + random_shift, 2*np.pi + random_shift, samples)
    
    shapes_arr[i, :] = np.sin(time)
    shapes_arr[i, random_jump_location:] += random_jump_size

# Sum all arrays and shuffle them
data = random_arr + offset_arr + shapes_arr
np.random.shuffle(data)
#data[0, :] += shapes_arr

# Plot data
plt.plot(data[0, :], label='Random plot 0')
plt.plot(data[1, :], label='Random plot 1')
plt.plot(data[2, :], label='Random plot 2')
plt.plot(data[3, :], label='Random plot 3')
plt.plot(data[4, :], label='Random plot 3')
plt.plot(data[5, :], label='Random plot 3')
plt.plot(data[6, :], label='Random plot 3')
plt.plot(data[7, :], label='Random plot 3')
plt.plot(data[8, :], label='Random plot 3')
plt.plot(data[9, :], label='Random plot 3')
plt.title('Example training data')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()

# Save to CSV file
#df.to_csv('data.csv', index=False, header=False)
np.savetxt('data.csv', data, delimiter=',', fmt='%.6f')