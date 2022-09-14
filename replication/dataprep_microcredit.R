# microcredit data preparation for python

# Gabriel Okasa and Kenneth A. Younge

# set path
load("/Users/okasag/Desktop/EPFL/Projects/RMC/code/github/samplefit/replication/data/microcredit_project_data.RData")

# save corresponding datasets

# profit outcome
data_profit <- as.data.frame(cbind(angelucci_profit, angelucci_treatment))
colnames(data_profit) <- c("profit", "treatment")
data_profit <- data_profit[complete.cases(data_profit), ]
write.csv(data_profit,
          file = "/Users/okasag/Desktop/EPFL/Projects/RMC/code/github/samplefit/replication/data/data_profit.csv", row.names = F)
# profit PPP standardizer
data_profit_ppp <- the_profit_standardiser_USD_PPP_per_fortnight[1]
success <- try(write.csv(data_profit_ppp,
                         file = "/Users/okasag/Desktop/EPFL/Projects/RMC/code/github/samplefit/replication/data/data_profit_ppp.csv",
                         row.names = F))

# print message
if (is.null(success)) {
  print('Microcredit data preparation was succesful.')
} else {
  print('Microcredit data preparation was NOT succesful.')
}

# dataprep done