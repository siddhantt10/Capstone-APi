from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Import your scaler, model, and classes from capstoneScript
from capstoneScript import scaler, best_model, classes

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost:3000",  # Frontend in development
    "http://127.0.0.1:3000",  # Alternative local frontend URL
    "https://your-production-domain.com",  # Replace with your production domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define the request body structure
class CancerData(BaseModel):
    features: list[float]  # List of 30 float values representing cancer data features


# Prediction route
@app.post("/predict")
def predict_cancer(data: CancerData):
    if len(data.features) != 30:
        raise HTTPException(status_code=400, detail="Features must contain exactly 30 values.")
    
    try:
        # Preprocess the input features
        sample = np.array(data.features).reshape(1, -1)
        sample = scaler.transform(sample)  # Scale the features

        # Make a prediction
        prediction = best_model.predict(sample) 
        result = classes[prediction[0]]

        # Return the prediction result
        return {"prediction": result}

    except Exception as e:
        # Handle any errors during prediction
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Fun and testing route
@app.get("/tester-boi")
def tester_boi():
    song = """
    Cee don't be on the road too tough
    But I still cut through with the local thugs (thugs)
    Could've ran off the plug
    But I kept it real and I showed him love (showed him love)
    Lil' bro got the poker tucked
    Where we grow up, it's so corrupt
    Fans see me on the block, go nuts
    I'm not in my bag, I'm loadin' loadin'
    Back then it was hand to hand
    Nowadays I don't have no involvement
    In the trap, I stay on my own
    I'm home alone, Macaulay Culkin
    WhyJay no commented it
    It's complicated 'cause the case still open
    The boys in blue tryna find them clues
    In the station, problem solvin' (solvin')
    Bro just jumped out the ride
    With a mask on face like he dodgin' COVID (bap)
    Jumped off the porch and went my own way
    No way, I don't owe no olders (no way)
    Free all the guys and rest in peace
    To all of the fallen soldiers
    The world full up of impactive distractions
    So we all lose focus (haha)
    I think out loud, what comes out my mouth
    I can't control it
    "Live Yours" on the chain, I'm so lit
    Soon come out with the custom clothin'
    Fuck sake, you must be jokin'?
    CIDs outside of the cut patrollin'
    How did they know bout the ins and outs?
    Somebody must have told them
    """
    return {"song": song.strip()}
