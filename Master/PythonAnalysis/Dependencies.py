#Import checks
from pip._internal import main as pipmain

try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except:
    pipmain(["install", "nltk"])
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
try:
    import re
except:
    pipmain(["install", "re"])
    import re
try:
    import string
except:
    pipmain(["install", "string"])
    import string
try:
    import unicodedata
except:
    pipmain(["install", "unicodedata"])
    import unicodedata
try:
    import pandas
except:
    pipmain(["install", "pandas"])
    import pandas
try:
    import os
except:
    pipmain(["install", "os"])
    import os
try:
    import collections
except:
    pipmain(["install", "collections"])
    import collections
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
except:
    pipmain(["install", "sklearn"])
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
try:
    import tweepy
except:
    pipmain(["install", "tweepy"])
    import tweepy
try:
    import csv
except:
    pipmain(["install", "csv"])
    import csv
try:
    import time
except:
    pipmain(["install", "time"])
    import time