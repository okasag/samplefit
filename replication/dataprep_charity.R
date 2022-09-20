# charity data preparation for python

# Gabriel Okasa and Kenneth A. Younge

# install/load haven and this.path packages
packages <- c("haven", "this.path")
# install not available packages
install.packages(setdiff(packages, rownames(installed.packages()))) 
# load packages
lapply(packages, require, character.only = TRUE)

# set the directory to the one of the source file
path <- dirname(this.path())

# load dta file
data <- read_dta(file = paste0(path, "/data/AER merged.dta")) 

# donated amount outcome
data_amount <- as.data.frame(cbind(data$amount, data$treatment))
colnames(data_amount) <- c("amount", "treatment")
data_amount <- data_amount[complete.cases(data_amount), ]
success <- try(write.csv(data_amount,
                         file = paste0(path, "/data/data_charity.csv"),
                         row.names = F))

# print message
if (is.null(success)) {
  print('Charity data preparation was succesful.')
} else {
  print('Charity data preparation was NOT succesful.')
}

# dataprep done