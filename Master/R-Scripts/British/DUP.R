setwd("E:\\Honours\\Master\\Twitter-Learning-Data\\DUP")

file_list <- list.files(pattern="*.csv") 

DUP <- do.call(rbind,lapply(file_list,read.csv))

DUP['Alignment'] <- "DUP"

setwd("E:\\Honours\\Master\\R-Scripts\\British")