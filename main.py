from fastapi import FastAPI
import tensorflow as tf
import joblib
import numpy as np
import h5py
from pydantic import BaseModel


model = tf.keras.models.load_model("model.h5")
tfidf_vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()
 
class CaseData(BaseModel):
    text: str
    case_type: str

@app.get("/")
async def home():
    return {"messege":"Welcome to the Differentiated Court Cases Scheduling Algorithm"}

@app.post("/predict")
async def predict_priority(data: CaseData):
    text_vectorized = tfidf_vectorizer.transform([data.text]).toarray()
    case_type_encoded = scaler.transform([[data.case_type]])
    X_input = np.hstack((text_vectorized, case_type_encoded))
    
    prediction = model.predict(X_input)
    priority_score = float(prediction[0][0] * 100)

    confidence = model.predict_proba(X_input) if hasattr(model, "predict_proba") else None
    if confidence is not None:
        confidence_level = float(confidence[0].max() * 100)
    else:
        confidence_level = 100 - abs(priority_score - 50) 

    return {"priority": priority_score,"confidence":confidence_level}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# uvicorn main:app --reload