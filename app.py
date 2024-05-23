from flask import Flask, jsonify
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Utworzenie modelu perceptron
model = Perceptron()

iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

X_train = df.iloc[:100, [0, 2]].values
y_train = df.iloc[:100, 4].values
y_train = np.where(y_train == 0, -1, 1) 

# Trenowanie modelu
model.fit(X_train, y_train)

# Utworzenie aplikacji Flask
app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict():
    data = np.array([3, 7]).reshape(1, -1)  
    prediction = model.predict(data)
    return jsonify({'prediction': int(prediction[0])}) 

if __name__ == '__main__':
    app.run(debug=True)
