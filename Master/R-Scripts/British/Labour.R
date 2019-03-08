setwd("E:\\Honours\\Master\\Twitter-Learning-Data\\Labour")

file_list <- list.files(pattern="*.csv") 

Labour <- do.call(rbind,lapply(file_list,read.csv))

Labour['Alignment'] <- "Labour"

setwd("E:\\Honours\\Master\\R-Scripts\\British")