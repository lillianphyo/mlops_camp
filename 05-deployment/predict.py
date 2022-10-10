import pickle
import numpy as np

with open("dv.bin", 'rb') as f:
    dv= pickle.load(f)
with open("model1.bin", 'rb') as f:
    model=pickle.load(f)
def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]
customer={"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

print(predict_single(customer,dv,model))
