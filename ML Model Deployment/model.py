import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer  # Stop words don't contribute significant meaning to a sentence
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Read the dataset

data = pd.read_csv ("dataset/twitter_sentiments.csv")

"""
data:
    id: unique number for each row
    label: non racist tweet is 0 and racist is 1
    tweet: tweet posted on twitter
"""

# Train Test Split
train, test = train_test_split (data, test_size = 0.2, stratify = data['label'], random_state = 42)

"""
Create TF-ID vector of the tweet column using TfidVectorizer

params:
    lowercase: true; it will convert text to lowercase
    max_features: 1000
    stop_words: predefined list of stop words in scikit-learn
"""

# Create a Tf-ID vectorizer object
tfidf_vectorizer = TfidfVectorizer (lowercase = True, max_features = 1000, stop_words = ENGLISH_STOP_WORDS)

# Fit the object with the training data tweets
tfidf_vectorizer.fit (train.tweet)

# Transform train and test data tweets
train_idf = tfidf_vectorizer.transform (train.tweet)
test_idf = tfidf_vectorizer.transform (test.tweet)

# Create an object of LogisticRegression Model
model = LogisticRegression ()

# Fit the model with training data
model.fit (train_idf, train.label)

# Predict the label on training data
predict_train = model.predict (train_idf)

# Predict the label on test data
predict_test = model.predict (test_idf)

# F1_score on training data
train_score = f1_score (y_true = train.label, y_pred = predict_train)
print ("The f1_score on training data: {}".format (train_score))

# F1_score on test data
test_score = f1_score (y_true = test.label, y_pred = predict_test)
print ("The f1_score on test data: {}".format (test_score))


# Creata a pipeline
"""
This way the steps from creating an object of Tf-id vectorizer and fitting the data with the Logistic model will be done by the pipeline
"""
pipeline = Pipeline (steps = [('tfidf', TfidfVectorizer(lowercase = True,
                                                    max_features = 1000,
                                                    stop_words = ENGLISH_STOP_WORDS)),
                          ('model', LogisticRegression())])

pipeline.fit (train.tweet, train.label)


# Test the pipeline with a sample tweet
text = ["Virat Kohli, AB de Villiers set to auction their 'Green Day' kits from 2016 IPL match to raise funds"]

# predict the label using the pipeline
pipeline.predict(text)
print (pipeline.predict(text))
# Save the model using joblib function

from joblib import dump
dump (pipeline, filename = "text_classification.joblib")