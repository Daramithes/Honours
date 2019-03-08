setwd("E:\\Honours\\Master\\Twitter-Learning-Data\\Conservative")

file_list <- list.files(pattern="*.csv") 

Conservative <- do.call(rbind,lapply(file_list,read.csv))

Conservative['Alignment'] <- "Conservative"

setwd("E:\\Honours\\Master\\R-Scripts\\British")