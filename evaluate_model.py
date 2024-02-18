import numpy as np
import matplotlib.pyplot as plt

def plot_data(data, indices_of_max_values, y_test_val, y_hat_val, num_rows_to_plot=5):
    print("Printing 5 most incorrect data test cases")
    print(f"1. True value: {y_test_val[indices_of_max_values[0]]}, neural network guess: {y_hat_val[indices_of_max_values[0]]}, index: {indices_of_max_values[0]+1}")
    print(f"2. True value: {y_test_val[indices_of_max_values[1]]}, neural network guess: {y_hat_val[indices_of_max_values[1]]}, index: {indices_of_max_values[1]+1}")
    print(f"3. True value: {y_test_val[indices_of_max_values[2]]}, neural network guess: {y_hat_val[indices_of_max_values[2]]}, index: {indices_of_max_values[2]+1}")
    print(f"4. True value: {y_test_val[indices_of_max_values[3]]}, neural network guess: {y_hat_val[indices_of_max_values[3]]}, index: {indices_of_max_values[3]+1}")
    print(f"5. True value: {y_test_val[indices_of_max_values[4]]}, neural network guess: {y_hat_val[indices_of_max_values[4]]}, index: {indices_of_max_values[4]+1}")

    plt.figure(figsize=(10, 6))

    for i in range(len(indices_of_max_values)):
        plt.plot(data[indices_of_max_values[i], :], label=f'Error: {abs(y_test_val[indices_of_max_values[i]]-y_hat_val[indices_of_max_values[i]])}')

    plt.title('Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def single_plot(data, index, y_test_val, y_hat_val, num_rows_to_plot=10):
    print(f"Most incorrect data test case\nTrue value: {y_test_val}, neural network guess: {y_hat_val}, error: {abs(y_test_val-y_hat_val)}")
    plt.figure(figsize=(10, 6))

    plt.plot(data[index, :], label=f'Worst case')

    plt.title(f'Most incorrect data test case\nTrue value: {y_test_val}, neural network guess: {y_hat_val}, error: {abs(y_test_val-y_hat_val)}, index: {index+1}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

#file = "X_train_preprocessed.csv"
#X_train = np.genfromtxt(file, delimiter=',')
file = "X_test_preprocessed.csv"
X_test = np.genfromtxt(file, delimiter=',')
#file = "y_train.csv"
#y_train = np.genfromtxt(file, delimiter=',')
file = "y_test.csv"
y_test = np.genfromtxt(file, delimiter=',')
file = "y_hat.csv"
y_hat = np.genfromtxt(file, delimiter=',')

test_error = abs(y_test - y_hat)

index_of_max_value = np.argmax(test_error)

single_plot(X_test, index_of_max_value, y_test[index_of_max_value], y_hat[index_of_max_value])

indices_of_max_values = np.argsort(test_error)[-5:][::-1]

plot_data(X_test, indices_of_max_values, y_test, y_hat)