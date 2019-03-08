setwd("E:\\Honours\\Master\\Twitter-Learning-Data\\Non-Aligned")

file_list <- list.files(pattern="*.csv") 

Generic <- do.call(rbind,lapply(file_list,read.csv))

Generic['Alignment'] <- "Non-Aligned"

setwd("E:\\Honours\\Master")