import requests
import json

from src.train.ml.data import load_data

data = load_data("src/data/census_cleaned.csv")

X = data.iloc[0]
y = X.pop('salary')

r = requests.post('https://ml-deployed.onrender.com/api', data=json.dumps(X.to_dict()))

print(f"status code: {r.status_code}")
print(f"prediction: {r.json()}, \t ground truth: {y}")