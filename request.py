import requests
import json
import numpy as np

file = "X_test_preprocessed.csv"
X_test = np.genfromtxt(file, delimiter=',')
X_test = X_test[0].reshape(1,-1)

X_test = X_test.tolist()
#print(X_test)
#exit(0)

url = 'http://localhost:8501/v1/models/tfmodel:predict'
data = {"instances": X_test}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)

print(response.json())

file = "y_test.csv"
y_test = np.genfromtxt(file, delimiter=',')
print(f"Real value: {y_test[0]}")
