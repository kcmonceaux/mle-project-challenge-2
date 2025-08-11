from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib

app = FastAPI()

# Load model and data at startup
model = joblib.load("model/model.pkl")
feature_order = [
    "age",
    "income",
    "zipcode",
    "gender",
    "education",
]  # Example feature order
demographics = pd.read_csv(
    "data/zipcode_demographics.csv"
)  # Must have 'zipcode' column


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

def get_demographics(zipcode: str) -> dict:
    """Return demographics for a zipcode or an empty dict if unknown."""
    demo_row = demographics[demographics["zipcode"] == zipcode]
    return demo_row.iloc[0].to_dict() if not demo_row.empty else {}


def prepare_input(data: dict, demo_row: Optional[dict] = None):
    """Prepare a feature vector for prediction.

    Demographic information is looked up by zipcode. If the zipcode does not
    exist in the demographics table, the input data is used as-is so that
    predictions can still be made.
    """
    demo_row = demo_row or get_demographics(data["zipcode"])
    # Merge, letting explicit input values take precedence
    merged = {**demo_row, **data}
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
    demo_row = get_demographics(data["zipcode"])
    # Fill optional features from demographics when available, otherwise default
    data["gender"] = demo_row.get("gender", "unknown")
    data["education"] = demo_row.get("education", "unknown")
    X, meta = prepare_input(data, demo_row)
    pred = model.predict([X])[0]
    return {"prediction": pred, "metadata": meta}
