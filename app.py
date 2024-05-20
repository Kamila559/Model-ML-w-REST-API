import pickle
from flask import Flask, request
from sklearn.linear_model import Perceptron

app = Flask(__name__)

# Create an API end point
@app.route('/predict_get')
def get_prediction():
    per_clf = Perceptron()
    per_clf.fit(X,y)

    y_pred = per_clf.predict([[2, 0.5],[4,5.5]])
    
    # Load pickled model file
    with open('model.pkl',"rb") as picklefile:
        model = pickle.load(picklefile)
    
    return y_pred

if __name__ == '__main__':
    app.run()
