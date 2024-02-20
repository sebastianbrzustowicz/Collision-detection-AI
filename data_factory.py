# Script to create csv data for machine learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Data size (fully adjustable)
rows = 10000
samples = 500 # columns

mode = "train" # "train", "test", "custom"

if len(sys.argv) > 1 and (sys.argv[1] == "train" or sys.argv[1] == "test"):
    mode = str(sys.argv[1])

rows = 10000 if mode == "train" else rows
rows = 100 if mode == "test" else rows

if(len(sys.argv) > 1 and str(sys.argv[1]) == "custom"):
    mode = str(sys.argv[1])
    rows = int(sys.argv[2])
    samples = int(sys.argv[3])

print(f"Creating data for {mode} mode: {rows} rows, {samples} samples")

# create output training data
y_train = np.zeros((rows, 1))

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

# collision shape
for i in range(3 * subdivision, 4 * subdivision):
    random_shift = np.random.uniform(-np.pi, np.pi)
    random_jump_location = np.random.randint(0, samples)
    random_jump_size = np.random.uniform(0.3, 1.5)
    random_flip = np.random.choice([-1, 1])
    random_jump_size *= random_flip
    random_shape = np.random.choice([1, 2])

    if (random_shape == 1):
        time = np.linspace(0 + random_shift, 2*np.pi + random_shift, samples)
        shapes_arr[i, :] = np.sin(time)
        shapes_arr[i, random_jump_location:] += random_jump_size
    elif (random_shape == 2):
        random_slope = np.random.uniform(-1, 1)
        random_bias = np.random.uniform(-1, 1)
        shapes_arr[i, :] = random_slope * np.linspace(0, 1, samples) + random_bias
        shapes_arr[i, random_jump_location:] += random_jump_size

    # marking y_train as collision
    y_train[i] = 1

# Sum all X_train arrays and shuffle it in the same way like y_train
X_train = random_arr + offset_arr + shapes_arr
permutation_indices = np.random.permutation(rows)
X_train = X_train[permutation_indices, :]
y_train = y_train[permutation_indices, :]

#print(y_train[:10])

# Plot head of X_train
for i in range(10):
    plt.plot(X_train[i, :], label=f"X_{mode}[:,{i}] {'collision' if y_train[i] == 1 else ' '}")
plt.legend()
plt.title(f'Example X_{mode} data')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()

# Save to CSV file

np.savetxt(f'X_{mode}.csv', X_train, delimiter=',', fmt='%.6f')
np.savetxt(f'y_{mode}.csv', y_train, delimiter=',', fmt='%d')
if mode == "train": del X_train, y_train
elif mode == "test": del X_train, y_train

print(f"Data saved to X_{mode}.csv and y_{mode}.csv")
exit(0)