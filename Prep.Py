import pandas as pd
df = pd.read_csv('data/future_unseen_examples.csv')
df.to_json('data/future_unseen_examples.json', orient='records', lines=False)python app.py