
# model.py
from sklearn.linear_model import LinearRegression

class MLModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# app.py
from flask import Flask, request, jsonify
from model import MLModel

app = Flask(__name__)
model = MLModel()

@app.route('/predict', methods=['POST'])
def predict():
   data = request.json['data']
   prediction = model.predict([data])
   return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
   app.run(debug=True, port=5000)

