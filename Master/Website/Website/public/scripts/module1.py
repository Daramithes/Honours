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


import collections
import csv
import pandas
import json


import csv
with open('Peterfile.csv') as f:
    test = dict(filter(None, csv.reader(f)))

array = [test,test,test,test]

array2 = [test,test,test,test]

array3 = [array,array2]

print(array3)
