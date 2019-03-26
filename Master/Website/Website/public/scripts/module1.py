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
app.debug = True


@app.route('/ExecuteTest/<text>')
def index(text):
    f = open("test.txt", "w")

    f.write(text)

    f.close()

    return text


if __name__ == '__main__':
    app.run()