setwd("E:\\Honours\\Master\\Twitter-Learning-Data\\Green")

file_list <- list.files(pattern="*.csv") 

Green <- do.call(rbind,lapply(file_list,read.csv))

Green['Alignment'] <- "Green"

setwd("E:\\Honours\\Master\\R-Scripts\\British")