import pickle
import numpy as np

from flask import Flask, request, jsonify

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


with open("dv.bin", 'rb') as f:
    dv= pickle.load(f)
with open("model1.bin", 'rb') as f:
    model=pickle.load(f)


app = Flask('credit card')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    credit = prediction >= 0.5
    
    result = {
        'credit_probability': float(prediction),
        'credit': bool(credit),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
