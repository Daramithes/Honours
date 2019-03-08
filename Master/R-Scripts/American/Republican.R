setwd("E:\\Honours\\Master\\Twitter-Learning-Data\\Republican")

file_list <- list.files(pattern="*.csv") 

Republican <- do.call(rbind,lapply(file_list,read.csv))

Republican['Alignment'] <- "Republican"

setwd("E:\\Honours\\Master\\R-Scripts\\American")