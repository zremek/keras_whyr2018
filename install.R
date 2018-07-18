install.packages("devtools")
library(devtools)
devtools::install_github("rstudio/keras")
library(keras)
keras::install_keras() #cpu
# keras::install_keras(tensorflow = "gpu")

use_condaenv("r-tensorflow")
