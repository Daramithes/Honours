setwd("E:\\Honours\\Master\\Twitter-Learning-Data\\UKIP")

file_list <- list.files(pattern="*.csv") 

UKIP <- do.call(rbind,lapply(file_list,read.csv))

UKIP['Alignment'] <- "UKIP"

setwd("E:\\Honours\\Master\\R-Scripts\\British")