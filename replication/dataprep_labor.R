# lalonda data preparation for python

# Gabriel Okasa and Kenneth A. Younge

# install/load Matching and this.path packages
packages <- c("Matching", "this.path")
# install not available packages
install.packages(setdiff(packages, rownames(installed.packages()))) 
# load packages
lapply(packages, require, character.only = TRUE)

# set the directory to the one of the source file
path <- dirname(this.path())

# load lalonde data
data(lalonde)

# re78 outcome, treat
data_lalonde <- as.data.frame(cbind(lalonde$re78, lalonde$treat))
colnames(data_lalonde) <- c("re78", "treatment")
data_lalonde <- data_lalonde[complete.cases(data_lalonde), ]
success <- try(write.csv(data_lalonde,
                         file = paste0(path, "/data/data_lalonde.csv"),
                         row.names = F))

# print message
if (is.null(success)) {
  print('Labor data preparation was succesful.')
} else {
  print('Labor data preparation was NOT succesful.')
}

# dataprep done