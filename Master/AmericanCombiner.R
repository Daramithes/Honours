setwd("E:\\Honours\\Master\\R-Scripts\\American")

source("Democratic.r")
source("Republican.r")

setwd("E:\\Honours\\Master\\R-Scripts")
source("Non-Aligned\\Non-Aligned.R")

AmericanCollection <- rbind(Democratic, Republican, Generic)

AmericanCollection <- AmericanCollection[,c(8,3)]

setwd("E:\\Honours\\Master\\Twitter-Learning-Data\\Collections")

write.csv(AmericanCollection, "AmericanCollection.csv")

Data <- rbind(Democratic, Republican)

text <- Data[,3]

textS <- data.frame(text) 

textS$sentiment <- sentiment_by(as.character(textS$text))

highlight(textS$sentiment)

textS$polarity[textS$sentiment$ave_sentiment > 0] <- "Positive"
textS$polarity[textS$sentiment$ave_sentiment < 0] <- "Negative"
textS$polarity[textS$sentiment$ave_sentiment == 0] <- "Inconclusive"

table <- table(textS$polarity)

barplot(table)

sentimentTerms <- extract_sentiment_terms(as.character(textS$text))