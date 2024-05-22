from flask import Flask, request
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

app = Flask(__name__)

class Perceptron():
    def __init__(self, n_iter=10, eta=0.01):
        self.n_iter = n_iter
        self.eta = eta
        
    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
        return self

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

def read_data():
    iris = load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
    X = df.iloc[:100,[0,2]].values
    y = df.iloc[0:100,4].values
    y = np.where(y == 0, -1, 1)
    return X, y

@app.route('/', methods=['POST'])
def fit_perceptron():
    X, y = read_data()
    
    perceptron = Perceptron()
    perceptron.fit(X, y)

    weights = perceptron.w_
    errors = perceptron.errors_

    return weights

if __name__ == '__main__':
    app.run(debug=True)
