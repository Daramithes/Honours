# @Author: John Isaacs <john>
# @Date:   08-Aug-172017
# @Filename: tweets.py
# @Last modified by:   john
# @Last modified time: 08-Aug-172017



#!/usr/bin/env python
import tweepy #https://github.com/tweepy/tweepy
import csv
import time
import html



#Twitter API credentials
consumer_key = "TC98w89JxQK2v4vPEqLLxJLx0"
consumer_secret = "le4t2JCgoT3CBZwToaKdOJx5LFYDDkGL5e3Pjl2ZtfTqYV46Fs"
access_key = "4459846396-tU9aYf4E5r9eHhJnniU7OsyrLNJhzEd4cpVeFFe"
access_secret = "UaY6kpdXbdV7cAsAxrKLzFTkKSLtW8dcNTe1CYniUl6xM"


def get_all_tweets(search_term):
    #Twitter only allows access to a users most recent 3240 tweets with this method

    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    #initialize a list to hold all the tweepy Tweets
    alltweets = []

    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.search(search_term,count=200)

    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(alltweets) < 1000:
        print("getting tweets before %s" % (oldest))

        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.search(search_term,count=200,max_id=oldest)

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
        tweet.text = tweet.text.encode('ascii', errors='ignore')
        try:
            tweet.text = html.unescape(tweet.text)
        except:
            print("Nothing found.")
        outtweets.append([tweet.id_str, tweet.created_at, tweet.text,tweet.retweet_count,tweet.favorite_count,media_count,url_count])

    #write the csv
    with open(search_term + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text","retweet_count","favorite_count","media_count","url_count"])
        writer.writerows(outtweets)

    pass


get_all_tweets("#brexit")