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

from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'