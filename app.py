from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load model and data at startup
model = joblib.load("model/model.pkl")
feature_order = ["age", "income", "zipcode", "gender", "education"]  # Example feature order
demographics = pd.read_csv("data/zipcode_demographics.csv")  # Must have 'zipcode' column

class InputData(BaseModel):
    age: int
    income: float
    zipcode: str
    gender: str
    education: str

class MinimalInput(BaseModel):
    age: int
    income: float
    zipcode: str

def prepare_input(data: dict):
    # Join demographic data
    demo_row = demographics[demographics["zipcode"] == data["zipcode"]]
    if demo_row.empty:
        raise HTTPException(status_code=400, detail="Zipcode not found in demographics")
    demo_row = demo_row.iloc[0].to_dict()
    # Merge input and demographics
    merged = {**data, **demo_row}
    # Prepare input for model
    X = [merged[feat] for feat in feature_order]
    return X, merged

@app.post("/predict")
def predict(input: InputData):
    X, meta = prepare_input(input.dict())
    pred = model.predict([X])[0]
    return {"prediction": pred, "metadata": meta}

@app.post("/predict-minimal")
def predict_minimal(input: MinimalInput):
    # Fill missing features with defaults or demographic data
    data = input.dict()
    demo_row = demographics[demographics["zipcode"] == data["zipcode"]]
    if demo_row.empty:
        raise HTTPException(status_code=400, detail="Zipcode not found in demographics")
    demo_row = demo_row.iloc[0].to_dict()
    # Example: fill gender and education from demographics
    data["gender"] = demo_row.get("gender", "unknown")
    data["education"] = demo_row.get("education", "unknown")
    X, meta = prepare_input(data)
    pred = model.predict([X])[0]
    return {"prediction": pred, "metadata": meta}