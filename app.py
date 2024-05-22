from flask import Flask, jsonify
import numpy as np

app = Flask(__name__)

class Perceptron:
    def __init__(self, n_iter=10, eta=0.01):
        self.n_iter = n_iter
        self.eta = eta
        self.errors_ = []
        
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
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

# Stworzenie modelu perceptron
model = Perceptron()
# Przykładowe dane treningowe
X_train = np.array([[2, 2], [3, 3], [4, 4]])
y_train = np.array([1, -1, 1])
# Trenowanie modelu
model.fit(X_train, y_train)

@app.route('/', methods=['GET'])
def predict():
    # Przykładowe dane wejściowe
    data = np.array([3, 3])
    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
