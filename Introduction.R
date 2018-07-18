library(keras)
library(tidyverse)

# EXAMPLE
# Load train and test sets
load("WhyR2018_keras-master/data/boston.RData")

# Check shape of the data
boston_train_X %>% dim()
boston_train_Y %>% dim()

# Initialize sequential model: keras_model_sequential()
boston_model <- keras_model_sequential()
summary(boston_model)

# Add hidden layer_dense() with 16 units and tanh function as activation
boston_model %>%
  layer_dense(units = 16, activation = "tanh", input_shape = c(13))
summary(boston_model)

# Explain why do we have 224 params ?
13 * 16 + 16

# Add output layer_dense() with 1 units and linear function as activation
boston_model %>%
  layer_dense(units = 1, activation = "linear")
summary(boston_model)

# Configure model for training. Use SGD as optimizer, MSE as loss function and add MAE as additional metric.
boston_model %>% compile(
  optimizer = "sgd",
  loss = "mse",
  metrics = c("mae")
)

# Fit the model
history_boston <- boston_model %>%
  fit(x = boston_train_X,
      y = boston_train_Y,
      validation_split = 0.2,
      epochs = 100,
      batch_size = 30)

# Evaluate on test set
boston_model %>%
  evaluate(boston_test_X, boston_test_Y)

# Get predictions
boston_predictions <- boston_model %>% predict(boston_test_X)

# Save the model
save_model_hdf5(boston_model, "boston_model.hdf5")

# Ex.1 - Build a MLP for 10-class classification problem.
load("WhyR2018_keras-master/data/fashion_mnist.RData")

# 1. Change labels vectors to one-hot-encoding matrix using to_categorical() function
fashion_mnist_train_Y <- fashion_mnist_train_Y %>% to_categorical(., 10)
fashion_mnist_test_Y <- fashion_mnist_test_Y %>% to_categorical(., 10)

# 2. Scale pixel values to [0, 1] interval
fashion_mnist_train_X <- fashion_mnist_train_X / 255
fashion_mnist_test_X <- fashion_mnist_test_X / 255

# 3. Model architecture:
# Dense layer with 512 units and "relu" activation
# Dropout layer with 20% drop rate
# Dense layer with 512 units and "relu" activation
# Dropout layer with 20% drop rate
# Output dense layer (how many units and what activation should You use?)
fashion_model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = 784) %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 10, activation = "softmax")

# 4. Set SGD as optimizer and use categorical crossentropy as loss function. Use accuracy as additional metric.
fashion_model %>% compile(
  optimizer = "sgd",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# 5. Fit the model. Use 20% of the data for validation, 20 epochs and 128 samples for batch size.
history <- fashion_model %>%
  fit(x = fashion_mnist_train_X,
      y = fashion_mnist_train_Y,
      validation_split = 0.2,
      epochs = 20,
      batch_size = 128)

# 6. Evaluate model on test set
fashion_model %>% 
  evaluate(fashion_mnist_test_X, fashion_mnist_test_Y)

# 7. Calculate predictions for the test set
fashion_predictions <- fashion_model %>% predict(fashion_mnist_test_X)
