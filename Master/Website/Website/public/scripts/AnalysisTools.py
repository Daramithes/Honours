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
import nltk


import re
import string
import unicodedata
import pandas
import os
import pickle
import tweepy
import time
import csv
import collections
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
import unicodedata
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english') # or any other list of your choice


def TweetbScrub(text, ):
    if text[0] == "b":
        text = text[1:]
    return text

def clean_text(text, ):

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text, characters = string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def stem_text(text, stemmer=default_stemmer):
        tokens = tokenize_text(text)
        return ' '.join([stemmer.stem(t) for t in tokens])

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)

    text = TweetbScrub(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'[^^a-zA-Z ]', '', text)
    text = text.strip(' ') # strip whitespaces
    text = text.lower() # lowercase
    text = stem_text(text) # stemming
    try:
        text = remove_special_characters(text) # remove punctuation and symbols
    except:
        print("Error")
    text = remove_stopwords(text) # remove stopwords
    #text.strip(' ') # strip whitespaces again?

    return text

PartyColours = {"Labour": "#dc241f",
              "Conservative": "#0087dc",
              "SNP": "#fff95d",
              "Green": "#6ab023",
              "UKIP": "#70147a",
              "DUP":"#d46a4c",
              "Non-Aligned": "#ffffff",
              "LibDem": "#fdbb30"}

#Twitter API credentials
consumer_key = "TC98w89JxQK2v4vPEqLLxJLx0"
consumer_secret = "le4t2JCgoT3CBZwToaKdOJx5LFYDDkGL5e3Pjl2ZtfTqYV46Fs"
access_key = "4459846396-tU9aYf4E5r9eHhJnniU7OsyrLNJhzEd4cpVeFFe"
access_secret = "UaY6kpdXbdV7cAsAxrKLzFTkKSLtW8dcNTe1CYniUl6xM"



def get_all_tweets(screen_name):

    #Twitter only allows access to a users most recent 3240 tweets with this method

    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    #initialize a list to hold all the tweepy Tweets
    alltweets = []

    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)

    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print("getting tweets before %s" % (oldest))

        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)

        #save most recent tweets
        alltweets.extend(new_tweets)

        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print("...%s tweets downloaded so far" % (len(alltweets)))

    #transform the tweepy tweets into a 2D array that will populate the csv



    outtweets = []
    for tweet in alltweets:
        media_count = 0
        url_count =0
        if "media" in tweet.entities:
            media_count =1
        if "urls" in tweet.entities:
            url_count =len(tweet.entities["urls"])
        outtweets.append([tweet.id_str, tweet.created_at, tweet.text.encode("utf-8"),tweet.retweet_count,tweet.favorite_count,media_count,url_count])

    #write the csv
    with open(screen_name + '-Tweets.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text","retweet_count","favorite_count","media_count","url_count"])
        writer.writerows(outtweets)

    pass

def get_all_favourites(screen_name):

    #Twitter only allows access to a users most recent 3240 tweets with this method

    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    #initialize a list to hold all the tweepy Tweets
    alltweets = []

    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.favorites(screen_name = screen_name,count=200)

    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print("getting tweets before %s" % (oldest))

        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.favorites(screen_name = screen_name,count=200,max_id=oldest)

        #save most recent tweets
        alltweets.extend(new_tweets)

        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print("...%s tweets downloaded so far" % (len(alltweets)))

    #transform the tweepy tweets into a 2D array that will populate the csv



    outtweets = []
    for tweet in alltweets:
        media_count = 0
        url_count =0
        if "media" in tweet.entities:
            media_count =1
        if "urls" in tweet.entities:
            url_count =len(tweet.entities["urls"])
        outtweets.append([tweet.id_str, tweet.created_at, tweet.text.encode("utf-8"),tweet.retweet_count,tweet.favorite_count,media_count,url_count])

    #write the csv
    with open(screen_name + '-Favourites.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text","retweet_count","favorite_count","media_count","url_count"])
        writer.writerows(outtweets)

    pass
def GeneratePie(collection, name):
    return
    import plotly
    plotly.tools.set_credentials_file(username='itjallingii', api_key='hiFqLEhfwSwdDlsj9lT8')

    import plotly.plotly as py
    import plotly.graph_objs as go


    labels = list(collection.keys())
    values = list(collection.values())
    try:
        colors = [PartyColoursB[i] for i in labels]
    except:
        colors = [PartyColoursA[i] for i in labels]



    trace = go.Pie(labels=labels, values=values,
                   hoverinfo='label+percent', textinfo='value',
                   textfont=dict(size=10),
                   marker=dict(colors = colors,
                               line=dict(color='#000000', width=2)))

    data = [trace]
    layout = go.Layout(title=name)
    fig = go.Figure(data=data,layout=layout)

    # Save the figure as a png image:
    py.image.save_as(fig, name + '.png')

def GeneratePieS(collection, name):
    return
    import plotly
    plotly.tools.set_credentials_file(username='itjallingii', api_key='hiFqLEhfwSwdDlsj9lT8')

    import plotly.plotly as py
    import plotly.graph_objs as go


    labels = list(collection.keys())
    values = list(collection.values())
    #colors = [PartyColours[i] for i in labels]


    trace = go.Pie(labels=labels, values=values,
                   hoverinfo='label+percent', textinfo='value',
                   textfont=dict(size=10),
                   marker=dict(
                               line=dict(color='#000000', width=2)))

    data = [trace]
    layout = go.Layout(title=name)
    fig = go.Figure(data=data,layout=layout)

    # Save the figure as a png image:
    py.image.save_as(fig, name + '.png')
#Remove Retweets
def Smallclean(text, ):
    text = TweetbScrub(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'[^^a-zA-Z@1-9 ]', '', text)
    return text

def FindTop(UserData):

    UserData = UserData[~UserData.text.str.contains("RT ")]
    favourite = UserData.loc[UserData['favorite_count'].idxmax()]
    retweet = UserData.loc[UserData['retweet_count'].idxmax()]
    file = open("Favourite.txt","w")
    file.writelines(favourite['text'] + "\n")
    file.writelines(str(favourite['favorite_count']))
    file.close()

    file = open("Retweet.txt","w")
    file.writelines(retweet['text'] + "\n")
    file.writelines(str(retweet['retweet_count']))
    file.close()

    return favourite['text'], retweet['text']

def AnalyseText(Text):
    BritishClassification, BritishClassificationSentiment = AnalyseSentences(Text, "British", BritishClassifer, BritishVectorizer)
    AmericanClassification, AmericanClassificationSentiment = AnalyseSentences(Text, "American", AmericanClassifer, AmericanVectorizer)

    Results = [[BritishClassification, BritishClassificationSentiment], [AmericanClassification, AmericanClassificationSentiment]]

    return Results

def AnalyseSentences(Text, Mode, classifier, vectorizer):

    Text = Text.replace("\n", "")
    Text = Text.split(".")
    for sentence in Text:
        if sentence == " ":
            Text.remove(sentence)
    Text = list(filter(None, Text))

    Text = pandas.DataFrame(Text)
    Text["Classification"] = ""
    Text.columns = ['Text', "Classification"]
    Text['CleanText'] = Text['Text'].apply(lambda row: clean_text(row))
    Sentence = vectorizer.transform(Text["CleanText"])
    Classification = classifier.predict(Sentence)
    Text["Classification"] = Classification
    TextS = Text
    if Mode == "British":
        TextS = Text.apply(SentimentclassiferBritish,axis=1)
    else:
        TextS = Text.apply(SentimentclassiferAmerican,axis=1)
    Text.drop('CleanText', axis=1, inplace=True)
    TextS.drop('CleanText', axis=1, inplace=True)
    Textcollection = collections.Counter(Text["Classification"])
    TextcollectionSentiment = collections.Counter(TextS["Classification"])

    return Textcollection, TextcollectionSentiment


def AnalyseSpeech(Filename, Mode, classifier, vectorizer):
    os.chdir("I:\Honours\Master\Speeches")
    Filename = Filename.replace(".txt", "")
    Speech = open(Filename + ".txt").read()
    Sentence = vectorizer.transform([Speech])
    classifier.predict(Sentence)
    Speech = Speech.replace("\n", "")
    Speech = Speech.split(".")
    for sentence in Speech:
        if sentence == " ":
            Speech.remove(sentence)
    Speech = list(filter(None, Speech))

    Speech = pandas.DataFrame(Speech)
    Speech["Classification"] = ""
    Speech.columns = ['Text', "Classification"]
    Speech['CleanText'] = Speech['Text'].apply(lambda row: clean_text(row))
    Sentence = vectorizer.transform(Speech["CleanText"])
    Classification = classifier.predict(Sentence)
    Speech["Classification"] = Classification
    SpeechS = Speech
    if Mode == "British":
        SpeechS = Speech.apply(SentimentclassiferBritish,axis=1)
    else:
        SpeechS = Speech.apply(SentimentclassiferAmerican,axis=1)
    Speech.drop('CleanText', axis=1, inplace=True)
    SpeechS.drop('CleanText', axis=1, inplace=True)
    Speechcollection = collections.Counter(Speech["Classification"])
    SpeechScollection = collections.Counter(SpeechS["Classification"])
    #Change directory to the classifications directory for saving
    if Mode == "British":
        os.chdir("I:\Honours\Master\Speeches\Classifications\British")
    else:
        os.chdir("I:\Honours\Master\Speeches\Classifications\American")
    #Save Classifications to classificaiton directory
    try:
        os.mkdir(Filename)
    except:
        pass
    os.chdir(Filename)
    Speech.to_csv(Filename + "-Classification.csv", encoding='utf-8')
    #GeneratePie(Speechcollection, Filename)
    Speech.to_csv(Filename + "-Sentiment-Classification.csv", encoding='utf-8')
    if Mode == "British":
        GeneratePieS(SpeechScollection, Filename + "-Sentiment")
    else:
        GeneratePie(SpeechScollection, Filename + "-Sentiment")

def AnalyseTweets(Username, UserData, Mode, classifier, vectorizer):
    try:
        os.mkdir(Mode)
    except:
        pass

    os.chdir(Mode)

    FindTop(UserData)
    #Vectorize the text
    UserData['CleanText'] = UserData['text'].apply(lambda row: clean_text(row))
    Sentences = vectorizer.transform(UserData['CleanText'])
    #Perform prediction of text inputted
    Classification = classifier.predict(Sentences)
    #Output a basic overview of the predictions
    #Append original data with model classification
    UserData["Classification"] = Classification
    UserDataSentiment = UserData

    Usercollection = collections.Counter(UserData["Classification"])
    w = csv.writer(open("output.csv", "w"))
    for key, val in Usercollection.items():
            w.writerow([key, val])
    UserData.to_csv(Username + "-Tweets-Classification.csv", encoding='utf-8')
    #GeneratePie(Usercollection, Username + "-Tweets")

    if Mode == "British":
        UserDataSentiment = UserDataSentiment.apply(SentimentclassiferBritish,axis=1)
    else:
        UserDataSentiment = UserDataSentiment.apply(SentimentclassiferAmerican,axis=1)
    UserData.drop('CleanText', axis=1, inplace=True)
    UserDataSentiment.drop('CleanText', axis=1, inplace=True)
    UsercollectionSentiment = collections.Counter(UserDataSentiment["Classification"])
    #Change directory to the classifications directory for saving
    #Save Classifications to classificaiton directory

    UserData.to_csv(Username + "-Tweets-Classification.csv", encoding='utf-8')
    #GeneratePie(Usercollection, Username + "-Tweets")
    UserDataSentiment.to_csv(Username + "-Tweets-Classification-S.csv", encoding='utf-8')
    #if Mode == "British":
        #GeneratePieS(UsercollectionSentiment, Username + "-Tweets-S")
    #else:
        #GeneratePie(UsercollectionSentiment, Username + "-Tweets-S")

    return Usercollection, UsercollectionSentiment


def AnalyseFavourites(Username, UserData, Mode, classifier, vectorizer):
    try:
        os.mkdir(Mode)
    except:
        pass

    os.chdir(Mode)

    #Vectorize the text
    UserData['CleanText'] = UserData['text'].apply(lambda row: clean_text(row))
    Sentences = vectorizer.transform(UserData['CleanText'])
    #Perform prediction of text inputted
    Classification = classifier.predict(Sentences)
    #Output a basic overview of the predictions
    #Append original data with model classification
    UserData["Classification"] = Classification
    UserDataSentiment = UserData


    Usercollection = collections.Counter(UserData["Classification"])
    #Change directory to the classifications directory for saving
    #Save Classifications to classificaiton directory

    if Mode == "British":
        UserDataSentiment = UserDataSentiment.apply(SentimentclassiferBritish,axis=1)
    else:
        UserDataSentiment = UserDataSentiment.apply(SentimentclassiferAmerican,axis=1)

    UserData.drop('CleanText', axis=1, inplace=True)
    UserDataSentiment.drop('CleanText', axis=1, inplace=True)
    UsercollectionSentiment = collections.Counter(UserDataSentiment["Classification"])
    #Change directory to the classifications directory for saving
    #Save Classifications to classificaiton directory
    UserDataSentiment.to_csv(Username + "-Favourites-Classification-S.csv", encoding='utf-8')
    #if Mode == "British":
        #GeneratePieS(UsercollectionSentiment, Username + "-Favourites-S")
    #else:
        #GeneratePie(UsercollectionSentiment, Username + "-Favourites-S")
    UserData.to_csv(Username + "-Favourites-Classification.csv", encoding='utf-8')
    #GeneratePie(Usercollection, Username + "-Favourites")

    return Usercollection, UsercollectionSentiment


def AnalyseUser(Username):
    os.chdir("I:\Honours\Master\Twitter-User-Data\Classifications")
    try:
        os.mkdir(Username)
    except:
        pass
    os.chdir(Username)
    print("Fetching " + Username + " Tweets")
    get_all_tweets(Username)
    print("Fetching " + Username + " Favourites")
    get_all_favourites(Username)

    Username = Username.replace(".csv", "")

    Tweets = pandas.DataFrame.from_csv(Username + "-Tweets.csv")
    Favourites = pandas.DataFrame.from_csv(Username + "-Favourites.csv")

    Tweets['text'] = Tweets['text'].apply(lambda row: TweetbScrub(row))
    Favourites['text'] = Favourites['text'].apply(lambda row: TweetbScrub(row))


    print("Analysing " + Username + " Tweets for British Classification")
    BritishTweets, SBritishTweets = AnalyseTweets(Username, Tweets, "British", BritishClassifer, BritishVectorizer)
    os.chdir("I:\Honours\Master\Twitter-User-Data\Classifications\\" + Username)
    print("Analysing " + Username + " Tweets for American Classification")
    AmericanTweets, SAmericanTweets = AnalyseTweets(Username, Tweets, "American",AmericanClassifer, AmericanVectorizer)


    os.chdir("I:\Honours\Master\Twitter-User-Data\Classifications\\" + Username)
    print("Analysing " + Username + " Favourites for British Classification")
    BritishFavourites, SBritishFavourites = AnalyseFavourites(Username,Favourites, "British", BritishClassifer, BritishVectorizer)
    os.chdir("I:\Honours\Master\Twitter-User-Data\Classifications\\" + Username)
    print("Analysing " + Username + " Favourites for American Classification")
    AmericanFavourites, SAmericanFavourites = AnalyseFavourites(Username,Favourites, "American",AmericanClassifer, AmericanVectorizer)

    Favourite, Retweet = FindTop(Tweets)

    BritishResults = [BritishTweets, BritishFavourites, SBritishTweets, SBritishFavourites]
    AmericanResults = [AmericanTweets, AmericanFavourites, SAmericanTweets, SAmericanFavourites]
    TopInformation = [Favourite, Retweet]


    CombinedResult = [BritishResults, AmericanResults, TopInformation]



    return(CombinedResult)

def SentimentclassiferBritish(sentence, ):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(sentence['CleanText'])
    sentence['Sentiment'] = sentiment['compound']
    if (sentence['Sentiment'] < -0) & (sentence['Classification'] != "Non-Aligned"):
        sentence['Classification'] = "Anti-" + sentence['Classification']
    return sentence
def SentimentclassiferAmerican(sentence, ):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(sentence['CleanText'])
    sentence['Sentiment'] = sentiment['compound']
    if (sentence['Sentiment'] < -0) & (sentence['Classification'] == "Republican"):
        sentence['Classification'] = "Democratic"
    elif (sentence['Sentiment'] < -0) & (sentence['Classification'] == "Democratic"):
        sentence['Classification'] = "Republican"
    return sentence

#Load British
def LoadBritish():
    global Mode
    global BritishVectorizer
    global BritishClassifer
    global PartyColoursB
    Mode = "British"
    print("Loading Mode: " + Mode)
    PartyColoursB = {"Labour": "#dc241f",
                  "Conservative": "#0087dc",
                  "SNP": "#fff95d",
                  "Green": "#6ab023",
                  "UKIP": "#70147a",
                  "DUP":"#d46a4c",
                  "Non-Aligned": "#ffffff",
                  "LibDem": "#fdbb30"}
    try:
        os.chdir("I:\Honours\Master\Models")
        print("File location found")
        BritishClassifer = pickle.load(open("ModelB", 'rb'))
        print("Pickle Model Loaded")
        BritishDF = pandas.DataFrame.from_csv("BritishCleaned.csv")
        print("CSV loaded")
        BritishVectorizer = CountVectorizer(min_df=0, lowercase=False)
        print("Vector built")
        BritishDF['text'] = BritishDF['text'].values.astype('U')
        BritishVectorizer.fit(BritishDF['text'].values.astype('U'))
        print("Data fit")
        sentences = BritishDF['text'].values
        y = BritishDF['Alignment'].values

        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.05, random_state=1000)

        BritishVectorizer = CountVectorizer()
        BritishVectorizer.fit(sentences_train)

        X_train = BritishVectorizer.transform(sentences_train)
        X_test  = BritishVectorizer.transform(sentences_test)

        score = BritishClassifer.score(X_test, y_test)

    except:
        print("Error")
        os.chdir("I:\Honours\Master\Twitter-Learning-Data\Collections")

        BritishDF = pandas.DataFrame.from_csv("AmericanCollection.csv")

        BritishDF['text'] = BritishDF['text'].apply(lambda row: clean_text(row))

        BritishVectorizer = CountBritishVectorizer(min_df=0, lowercase=False)

        BritishVectorizer.fit(BritishDF['text'])

        sentences = BritishDF['text'].values
        y = BritishDF['Alignment'].values

        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.05, random_state=1000)

        BritishVectorizer = CountVectorizer()
        BritishVectorizer.fit(sentences_train)

        X_train = BritishVectorizer.transform(sentences_train)
        X_test  = BritishVectorizer.transform(sentences_test)

        BritishClassifer = LogisticRegression()
        BritishClassifer.fit(X_train, y_train)
        score = BritishClassifer.score(X_test, y_test)

        print("Accuracy:", score)

        os.chdir("I:\Honours\Master\Models")
        pickle.dump(BritishClassifer, open("ModelB", 'wb'))
        BritishDF.to_csv("BritishCleaned.csv", encoding='utf-8')
#Load American
def LoadAmerican():
    global Mode
    global AmericanVectorizer
    global AmericanClassifer
    global PartyColoursA
    Mode = "American"
    print("Loading Mode: " + Mode)
    PartyColoursA = {"Republican": "#ff0000",
                  "Democratic": "#0015BC",
                  "Non-Aligned": "#ffffff"}
    try:
        os.chdir("I:\Honours\Master\Models")
        print("File location found")
        AmericanClassifer = pickle.load(open("ModelA", 'rb'))
        print("Pickle Model Loaded")
        AmericanDF = pandas.DataFrame.from_csv("AmericanCleaned.csv")
        print("CSV loaded")
        AmericanVectorizer = CountVectorizer(min_df=0, lowercase=False)
        print("Vector built")
        AmericanDF['text'] = AmericanDF['text'].values.astype('U')
        AmericanVectorizer.fit(AmericanDF['text'].values.astype('U'))
        print("Data fit")
        sentences = AmericanDF['text'].values
        y = AmericanDF['Alignment'].values

        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.05, random_state=1000)

        AmericanVectorizer = CountVectorizer()
        AmericanVectorizer.fit(sentences_train)

        X_train = AmericanVectorizer.transform(sentences_train)
        X_test  = AmericanVectorizer.transform(sentences_test)

        score = AmericanClassifer.score(X_test, y_test)

    except:
        print("Error")
        os.chdir("I:\Honours\Master\Twitter-Learning-Data\Collections")

        AmericanDF = pandas.DataFrame.from_csv("AmericanCollection.csv")

        AmericanDF['text'] = AmericanDF['text'].apply(lambda row: clean_text(row))

        AmericanVectorizer = CountVectorizer(min_df=0, lowercase=False)

        AmericanVectorizer.fit(AmericanDF['text'])

        sentences = AmericanDF['text'].values
        y = AmericanDF['Alignment'].values

        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.05, random_state=1000)

        AmericanVectorizer = CountVectorizer()
        AmericanVectorizer.fit(sentences_train)

        X_train = AmericanVectorizer.transform(sentences_train)
        X_test  = AmericanVectorizer.transform(sentences_test)

        AmericanClassifer = LogisticRegression()
        AmericanClassifer.fit(X_train, y_train)
        score = AmericanClassifer.score(X_test, y_test)

        print("Accuracy:", score)

        os.chdir("I:\Honours\Master\Models")
        pickle.dump(AmericanClassifer, open("ModelA", 'wb'))
        AmericanDF.to_csv("AmericanCleaned.csv", encoding='utf-8')
