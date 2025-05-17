import joblib

# Load trained model
model = joblib.load("models/model.pkl")

def predict_disease(data):
    prediction = model.predict(data)
    prob = model.predict_proba(data)
    return prediction[0], prob[0][1]
