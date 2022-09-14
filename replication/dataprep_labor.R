# lalonda data preparation for python

# Gabriel Okasa and Kenneth A. Younge

# install/load Matching package
packages <- c("Matching")
# install not available packages
install.packages(setdiff(packages, rownames(installed.packages()))) 
# load packages
lapply(packages, require, character.only = TRUE)

# load lalonde data
data(lalonde)

# re78 outcome, treat
data_lalonde <- as.data.frame(cbind(lalonde$re78, lalonde$treat))
colnames(data_lalonde) <- c("re78", "treatment")
data_lalonde <- data_lalonde[complete.cases(data_lalonde), ]
success <- try(write.csv(data_lalonde,
                         file = "/Users/okasag/Desktop/EPFL/Projects/RMC/code/github/samplefit/replication/data/data_lalonde.csv",
                         row.names = F))

# print message
if (is.null(success)) {
  print('Labor data preparation was succesful.')
} else {
  print('Labor data preparation was NOT succesful.')
}

# dataprep done