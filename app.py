from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    "data/zipcode_demographics.csv",
    dtype=str
)
# Strip whitespace from all column names
demographics.columns = demographics.columns.str.strip()
# Strip whitespace from all zipcodes
demographics["zipcode"] = demographics["zipcode"].astype(str).str.strip()



class InputData(BaseModel):
    age: int
    income: float
    zipcode: str
    gender: str
    education: str



def prepare_input(data: dict):
    """Prepare a feature vector for prediction.

    Demographic information is looked up by zipcode. If the zipcode does not
    exist in the demographics table, the input data is used as-is so that
    predictions can still be made.
    """
    # Debug: print loaded zipcodes and incoming zipcode
    loaded_zipcodes = demographics["zipcode"].astype(str).str.strip().unique()
    print(f"Loaded zipcodes: {loaded_zipcodes}")
    input_zip = str(data["zipcode"]).strip()
    print(f"Incoming zipcode: {input_zip}")
    demo_row = demographics[demographics["zipcode"].astype(str).str.strip() == input_zip]
    if demo_row.empty:
        print(f"Zipcode {input_zip} not found!")
    demo_row = demo_row.iloc[0].to_dict() if not demo_row.empty else {}
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


