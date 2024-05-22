import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from flask import Flask, request, jsonify

app = Flask(__name__)

# Your Perceptron class definition
class Perceptron:
    def __init__(self, n_iter=10, eta=0.01):
        self.n_iter = n_iter
        self.eta = eta

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
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Initialize and train the Perceptron model
iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])

X = df.iloc[:100,[0,2]].values
y = df.iloc[0:100,4].values
y = np.where(y == 0, -1, 1)

ppn = Perceptron(n_iter=10, eta=0.01)
ppn.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X_new = np.array(data['X'])
    predictions = ppn.predict(X_new).tolist()
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)
