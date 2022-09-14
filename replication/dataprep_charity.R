# charity data preparation for python

# Gabriel Okasa and Kenneth A. Younge

# install/load haven package
packages <- c("haven")
# install not available packages
install.packages(setdiff(packages, rownames(installed.packages()))) 
# load packages
lapply(packages, require, character.only = TRUE)

# load dta file
data <- read_dta(file = "/Users/okasag/Desktop/EPFL/Projects/RMC/code/github/samplefit/replication/data/AER merged.dta") 

# donated amount outcome
data_amount <- as.data.frame(cbind(data$amount, data$treatment))
colnames(data_amount) <- c("amount", "treatment")
data_amount <- data_amount[complete.cases(data_amount), ]
success <- try(write.csv(data_amount,
                         file = "/Users/okasag/Desktop/EPFL/Projects/RMC/code/github/samplefit/replication/data/data_charity.csv",
                         row.names = F))

# print message
if (is.null(success)) {
  print('Charity data preparation was succesful.')
} else {
  print('Charity data preparation was NOT succesful.')
}

# dataprep done