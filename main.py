import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Churn.csv")

X = pd.get_dummies(df.drop(["Churn", "Customer ID"], axis=1))
y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Show X_train data frame
#print(X_train.head())

# Show y_train data frame
#print(y_train.head())

# 1. Import dependencies
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# 2. Build and compile model
model = Sequential()
model.add(Dense(units=32, activation="relu", input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# 3. Fir, predict and evaluate
model.fit(X_train, y_train, epochs=100, batch_size=32)

y_hat = model.predict(X_test)
y_hat = [0 if val <0.5 else 1 for val in y_hat]

print("\nAccuracy: " + accuracy_score(y_test, y_hat))

# 4. Saving and reloading
model.save("tfmodel")
del model
model = load_model("tfmodel")