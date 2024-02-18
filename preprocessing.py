import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_data(normalized_data, num_rows_to_plot=5):
    plt.figure(figsize=(10, 6))

    for i in range(min(num_rows_to_plot, normalized_data.shape[0])):
        plt.plot(normalized_data[i, :], label=f'Row {i+1}')

    plt.title('Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def remove_mean_value(data):
    zero_mean_data = np.zeros_like(data)

    for i in range(data.shape[0]):
        row_mean = np.mean(data[i, :])
        zero_mean_data[i, :] = data[i, :] - row_mean
    return zero_mean_data

def normalize_data(data):
    normalized_data = np.zeros_like(data, dtype=float)
    for i in range(zero_mean_data.shape[0]):
        row = zero_mean_data[i, :]
        min_val = np.min(row)
        max_val = np.max(row)

        if max_val != min_val:
            normalized_data[i, :] = (row - min_val) / (max_val - min_val)
        else:
            normalized_data[i, :] = row
    return normalized_data

file = "X_train.csv"
if len(sys.argv) > 1 and (sys.argv[1] == "X_train.csv" or sys.argv[1] == "X_test.csv"):
    file = str(sys.argv[1])

data = np.genfromtxt(file, delimiter=',')

#print(data[:5, :])
#plot_data(data)

# Remove mean value
zero_mean_data = remove_mean_value(data)

#print(zero_mean_data[:5, :])
#plot_data(zero_mean_data)

# Normalize data
normalized_data = normalize_data(zero_mean_data)

#print(normalized_data[:5, :])
#plot_data(normalized_data)

np.savetxt(f'{file.replace(".csv","")}_preprocessed.csv', normalized_data, delimiter=',', fmt='%.6f')