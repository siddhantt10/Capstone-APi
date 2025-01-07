from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from capstoneScript import scaler, best_model, classes

app = FastAPI()

class CancerData(BaseModel):
    features: list[float] 

@app.post("/predict")
def predict_cancer(data: CancerData):
    if len(data.features) != 30:
        raise HTTPException(status_code=400, detail="Features must contain exactly 30 values.")
    
    try:
        sample = np.array(data.features).reshape(1, -1)
        sample = scaler.transform(sample)  
        prediction = best_model.predict(sample) 
        result = classes[prediction[0]]
        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
