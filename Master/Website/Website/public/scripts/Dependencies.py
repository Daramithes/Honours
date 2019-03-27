#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Nick
#
# Created:     26/03/2019
# Copyright:   (c) Nick 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------



print("Loading Dependencys")

try:
    from pip._internal import main as pipmain

    print("Pip loaded successfully")
except:
    print("Error loading pip")
    exit()

try:
    import ploty
    print("ploty loaded succesfully")

except:
    pipmain(["install", "plotly"])
    print("Plotly installed correctly.")

try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    print("nltk loaded successfully")
except:
    pipmain(["install", "nltk"])
    import nltk
    nltk.download('stopwords')
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download('stopwords')
    print("Failed, Running pip installation of nltk")
try:
    import re
    print("re loaded successfully")
except:
    pipmain(["install", "re"])
    import re
    print("Failed, Running pip installation of re")
try:
    import string
    print("string loaded successfully")
except:
    pipmain(["install", "string"])
    import string
    print("Failed, Running pip installation of string")
try:
    import unicodedata
    print("unicodedata loaded successfully")
except:
    pipmain(["install", "unicodedata"])
    import unicodedata
    print("Failed, Running pip installation of unicodedata")
try:
    import pandas
    print("pandas loaded successfully")
except:
    pipmain(["install", "pandas"])
    import pandas
    print("Failed, Running pip installation of pandas")
try:
    import os
    print("os loaded successfully")
except:
    pipmain(["install", "os"])
    import os
    print("Failed, Running pip installation of os")
try:
    import collections
    print("collections loaded successfully")
except:
    pipmain(["install", "collections"])
    import collections
    print("Failed, Running pip installation of collections")
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    print("sklearn loaded successfully")
except:

    pipmain(["install", "sklearn"])
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    print("Failed, Running pip installation of sklearn")
try:
    import tweepy
    print("tweepy loaded successfully")
except:
    pipmain(["install", "tweepy"])
    import tweepy
    print("Failed, Running pip installation of tweepy")
try:
    import csv
    print("csv loaded successfully")
except:
    pipmain(["install", "csv"])
    import csv
    print("Failed, Running pip installation of csv")
try:
    import time
    print("time loaded successfully")
except:
    pipmain(["install", "time"])
    import time
    print("Failed, Running pip installation of time")
try:
    import pickle
    print("pickle loaded successfully")
except:
    pipmain(["install", "pickle"])
    import pickle
    print("Failed, Running pip installation of pickle")
