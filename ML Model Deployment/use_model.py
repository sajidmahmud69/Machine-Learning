# Import joblib
from joblib import load

text = ["Virat Kohli, AB de Villiers set to auction their 'Green Day' kits from 2016 IPL match to raise funds"]

# Load the model from a save file
pipeline = load ("text_classification.joblib")

print (pipeline.predict (text))