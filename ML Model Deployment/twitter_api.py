# import required libraries
import tweepy
import pandas as pd
pd.set_option('display.max_colwidth', 1000)

# api key
api_key = "Enter API Key Here"
# api secret key
api_secret_key = "Enter API Secret Key Here."
# access token
access_token = "Enter Access Token Here"
# access token secret
access_token_secret = "Enter Access Token Secret Here."

# authorize the API Key
authentication = tweepy.OAuthHandler(api_key, api_secret_key)

# authorization to user's access token and access token secret
authentication.set_access_token(access_token, access_token_secret)

# call the api
api = tweepy.API(authentication, wait_on_rate_limit=True)
