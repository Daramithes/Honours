setwd("E:\\Honours\\Master\\Twitter-Learning-Data\\Democratic")

file_list <- list.files(pattern="*.csv") 

Democratic <- do.call(rbind,lapply(file_list,read.csv))

Democratic['Alignment'] <- "Democratic"

setwd("E:\\Honours\\Master\\R-Scripts\\American")
 