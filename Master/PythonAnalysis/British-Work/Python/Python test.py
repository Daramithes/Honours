#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      nickw
#
# Created:     03/03/2019
# Copyright:   (c) nickw 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer
import pandas

data = pandas.DataFrame.from_csv("Data.csv")

vectorizer = CountVectorizer(min_df=0, lowercase=False)


vectorizer.fit(data.text)#
vectorizer.transform(data.text).toarray()