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

from flask import Flask, request

from AnalysisTools import *

import collections
import csv
import pandas
import json

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
#app.debug = True
LoadBritish()
LoadAmerican()

@app.route('/GetUser/<text>')
def TwitterAnalyser(text):

    print("Analysing " + text)

    Results = AnalyseUser(text)
    print("Stingifying Results")
    Results = json.dumps(Results)

    return(Results)

@app.route('/Hashtag/<text>/<amount>')
def Hashtag(text, amount):
    amount = int(amount)
    if amount < 2000:
        print("hello")

    print("Analysing " + text + " and fetching at many as ", amount)

    Results = AnalyseHashtag(text, amount)
    print("Stingifying Results")
    Results = json.dumps(Results)

    return(Results)

@app.route('/Text/<text>')
def CheckSentences(text):

    Results = AnalyseText(text)
    Results = json.dumps(Results)

    return(Results)


@app.route('/test/<text>')
def testpackage(text):
    print(text)
    return(text)

app.run()
