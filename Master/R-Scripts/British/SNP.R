setwd("E:\\Honours\\Master\\Twitter-Learning-Data\\SNP")

file_list <- list.files(pattern="*.csv") 

SNP <- do.call(rbind,lapply(file_list,read.csv))

SNP['Alignment'] <- "SNP"

setwd("E:\\Honours\\Master\\R-Scripts\\British")