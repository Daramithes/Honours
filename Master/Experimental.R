#Run external scripts
setwd("E:\\Honours\\Master\\R-Scripts\\British")

source("Conservative.R")
source("DUP.R")
source("Green.R")
source("Labour.R")
source("LibDem.R")
source("SNP.R")
source("UKIP.R")

setwd("E:\\Honours\\Master\\R-Scripts\\American")

source("Democratic.r")
source("Republican.r")

setwd("E:\\Honours\\Master\\R-Scripts")
source("Non-Aligned\\Non-Aligned.R")

Experimental <- rbind(SNP, UKIP, Labour, LibDem, Conservative, DUP, Green, Generic, Democratic, Republican)

Experimental <- Experimental[,c(8,3)]

setwd("E:\\Honours\\Master\\Twitter-Learning-Data\\Collections")

write.csv(Experimental, "Experimental.csv")




