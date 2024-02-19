from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import warnings

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=FutureWarning)

#y = df["X_train"].apply(lambda x: 1 if x == "Yes" else 0)

file = "X_train_preprocessed.csv"
X_train = np.genfromtxt(file, delimiter=',')
file = "X_test_preprocessed.csv"
X_test = np.genfromtxt(file, delimiter=',')
file = "y_train.csv"
y_train = np.genfromtxt(file, delimiter=',')
file = "y_test.csv"
y_test = np.genfromtxt(file, delimiter=',')

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Show data frames
#pd.set_option('display.max_columns', None)
#print(X_train.columns)
#print(X_test.iloc[:, :10])
#print(y_train.head(10))
#print(len(X_test[0])) # actual input dim

# 1. Import dependencies
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# 2. Build and compile model
model = Sequential()
model.add(Dense(units=len(X_test[0]), activation="relu", input_dim=len(X_test[0])))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# 3. Fit, predict and evaluate
model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=2)

y_hat = model.predict(X_test)
np.savetxt('y_hat.csv', y_hat, delimiter=',', fmt='%.6f')
y_hat = [0 if val <0.5 else 1 for val in y_hat]

print("\nModel accuracy: " + str(accuracy_score(y_test, y_hat)))
accuracy = accuracy_score(y_test, y_hat)*100
if accuracy >= 90:
    print("Excellent results!")
elif 80 <= accuracy < 90:
    print("Very good results.")
elif 70 <= accuracy < 80:
    print("Good results.")
elif 60 <= accuracy < 70:
    print("Fair results.")
else:
    print("Poor results, need improvement.")

# 4. Saving and reloading model
model.save("tfmodel")
del model
model = load_model("tfmodel")

exit(0)