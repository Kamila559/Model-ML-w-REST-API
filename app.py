import pickle
import numpy as np
from flask import Flask, request, jsonify

class Perceptron():
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta*(target-self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Create a flask
app = Flask(__name__)

@app.route('/', methods=['POST'])
def post_predict():
    # sepal length
    sepal_length = 6.3
    petal_length = 2.6
    
    features = np.array([[sepal_length, petal_length]])

    # Load pickled model file
    with open('model.pkl',"rb") as picklefile:
        model = pickle.load(picklefile)
        
    # Predict the class using the model
    predicted_class = int(model.predict(features))
    output = dict(features=features.tolist(), predicted_class=predicted_class)
    # Return a json object containing the features and prediction
    return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
