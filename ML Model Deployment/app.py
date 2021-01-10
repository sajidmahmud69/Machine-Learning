from flask import Flask, render_template, request, redirect, url_for
from joblib import load
from tweets import get_related_tweets

# Load the ML model
pipeline = load ("text_classification.joblib")

 # Get results of a particular query
def requestResults (name):
    # Get the tweets text
    tweets = get_related_tweets (name)

    # get the prediction
    tweets['prediction'] = pipeline.predict (tweets['tweet_text'])

    # get the value counts of different labels predicted
    data = str (tweets.prediction.value_counts()) + '\n\n'

    return data + str (tweets)




# Start Flask 
app = Flask (__name__)
app.config ["DEBUG"] = True


# render default page
@app.route ('/')
def home ():
    return render_template ('index.html')




# if method is post redirect to success function
@app.route ('/', methods = ['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form ['search']
        return redirect (url_for ('success', name=user))



# get the data for requested query
@app.route ('/success/<name>')
def success (name):
    return "<xmp>" + str (requestResults (name)) + " </xmp>"



if __name__ == '__main__':
    app.run (host = 'localhost', port = 5000)