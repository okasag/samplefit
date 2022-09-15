# microcredit data preparation for python

# Gabriel Okasa and Kenneth A. Younge

# set the directory to the one of the source file (requires Rstudio)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
path <- getwd()

# load data
load(paste0(path, "/data/microcredit_project_data.RData"))

# save corresponding datasets

# profit outcome
data_profit <- as.data.frame(cbind(angelucci_profit, angelucci_treatment))
colnames(data_profit) <- c("profit", "treatment")
data_profit <- data_profit[complete.cases(data_profit), ]
success_profit <- try(write.csv(data_profit,
                                file = paste0(path, "/data/data_profit.csv"),
                                row.names = F))
# profit PPP standardizer
data_profit_ppp <- the_profit_standardiser_USD_PPP_per_fortnight[1]
success_profit_ppp <- try(write.csv(data_profit_ppp,
                                    file = paste0(path, "/data/data_profit_ppp.csv"),
                                    row.names = F))

# print message
if (is.null(success_profit) && is.null(success_profit_ppp)) {
  print('Microcredit data preparation was succesful.')
} else {
  print('Microcredit data preparation was NOT succesful.')
}

# dataprep done