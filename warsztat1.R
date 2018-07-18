library(tidyverse)
library(gridExtra)

## tensor is a kind of multidimensional array:
# 2D vector - standard frame: samples with features
# 3D timeseries
# 4D images
# 5D video

# keras is an API for TensorFlow, another is Estimator API for ml models on gpu (not neural networks)

### in keras we build models in 3 ways:
# sequentail model - most common
# functional API - special functions 
# using pre-trained models

## hidden layvers are interpreted as new variables - vars are created 

# define model architecture

boston_model <- keras_model_sequential()
summary(boston_model)

# batches are smaller parts of dataset, created for faster counting of derivatives
# epoch - 'epoka' 
# 1 epoch = 3 baches