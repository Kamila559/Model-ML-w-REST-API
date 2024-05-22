from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

app = Flask(__name__)

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
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

@app.route('/', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        iris = load_iris()
        df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target'])
        X = df.iloc[:100,[0,2]].values
        y = df.iloc[0:100,4].values
        y = np.where(y == 0, -1, 1)
        model = Perceptron()
        model.fit(X, y)
        return jsonify({'success': 'Model trained successfully'})

if __name__ == '__main__':
    app.run(debug=True)
