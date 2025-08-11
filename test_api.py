import pandas as pd
import requests
import json

API_URL = "http://localhost:8000/predict"  # Change if your API runs elsewhere

# Load examples
examples = pd.read_csv("data/future_unseen_examples.csv")

for idx, row in examples.iterrows():
    payload = row.dropna().to_dict()
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad responses
        try:
            result = response.json()
        except json.JSONDecodeError:
            print(f"Example {idx+1}: Invalid JSON response")
            result = response.text
    except requests.exceptions.RequestException as e:
        print(f"Example {idx+1}: Request failed - {e}")
        result = None

    print(f"Example {idx+1}:")
    print("Input:", payload)
    print("Response:", result)
    print("-" * 40)