setwd("E:\\Honours\\Master\\Twitter-Learning-Data\\LibDem")


file_list <- list.files(pattern="*.csv") 

LibDem <- do.call(rbind,lapply(file_list,read.csv))

LibDem['Alignment'] <- "LibDem"

setwd("E:\\Honours\\Master\\R-Scripts\\British")