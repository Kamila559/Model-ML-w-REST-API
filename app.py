from flask import Flask, jsonify
import numpy as np
from sklearn.linear_model import Perceptron

app = Flask(__name__)

model = Perceptron()
model.coef_ = np.array([[0.1, 0.2, 0.3, 0.4]])
model.intercept_ = np.array([0.5])
model.classes_ = np.array([0, 1])

@app.route('/', methods=['GET'])
def predict():
    # Stałe dane wejściowe
    data = [5.1, 3.5, 1.4, 0.2]
    prediction = model.predict(np.array(data).reshape(1, -1))
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
