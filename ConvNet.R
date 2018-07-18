library(keras)
library(tidyverse)
library(gridExtra)
source("WhyR2018_keras-master/utils.R") # tu są funkcje napisane przez prowadzącego

# Load train and test sets
load("WhyR2018_keras-master/data/ships.RData")

# Check shape of the data
ships_train$data %>% dim() # 4D because we have images
ships_train$labels %>% dim()

# Sample image
plot_sample_image(ships_train$data, ships_train$labels, show_layers = TRUE, row_nr = 7)

# Ex. 2 - Build a simple ConvNet for binary classification
# 1. Change labels vectors to one-hot-encoding matrix
ships_train$labels <- ships_train$labels %>% to_categorical(., 2)
ships_test$labels <- ships_test$labels %>% to_categorical(., 2)
  
# 2. Initialize sequential model and add 2d convolutional layer - 2D is for images
# with 32 filters, 3x3 kernel, 1x1 stride, "relu" activation
ships_model <- keras_model_sequential() %>% 
  layer_conv_2d(input_shape = c(80, 80, 3),
                filters = 32,
                kernel_size = c(3, 3),
                strides = c(1, 1),
                activation = "relu") # relu is a simple, fast function used for such tasks

summary(ships_model)

# 3. Explain output shape and nr of params
32 * 3 * 3 * 3 + 32

# 4. Add more layers
# max pooling with 2x2 pool (kernel), 2x2 strides
# convolution with 64 filters, 3x3 kernel, 1x1 strides, "relu" activation
# max pooling with 2x2 pool (kernel), 2x2 strides
# flattening layer
# Dense layer as output

ships_model %>% 
  layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), strides = c(1, 1), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(2, activation = "softmax") # softmax is for binary clasification

summary(ships_model)

# 5. Compile the model using binary crossentropy as loss function and
# SGD as optimizer with learning rate equal to 0.0001 and learning rate decay equal to 1e-6
ships_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_sgd(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

# 6. Fit the model. Use 20% of the data for validation, 20 epochs and 32 samples for batch size.
ships_fit <- ships_model %>% fit(ships_train$data, ships_train$labels,
                                              epochs = 20, batch_size = 32,
                                              validation_split = 0.2)
  
# 7. Evaluate model on test set
ships_model %>% evaluate(ships_test$data, ships_test$labels)
  
# 8. Save model in hdf5 format
save_model_hdf5(ships_model, "ships_model.hdf5")  
  
# Ex 3. Build second model using batch normalization. Use early stopping and checkpoints. Save logs to Tensorboard
# 1. Model architecture:
# convolution with 64 filters, 3x3 kernel, 1x1 strides, "linear" activation, "same" padding
# batch normalization
# "relu" activation
# max pooling with 2x2 pool (kernel), 2x2 strides
# dropout layer with 25% drop rate
# convolution with 64 filters, 3x3 kernel, 1x1 strides, "linear" activation, "same" padding
# batch normalization
# "relu" activation
# max pooling with 2x2 pool (kernel), 2x2 strides
# dropout layer with 25% drop rate
# flattening layer
# dense layer with 512 units and "relu" activation
# dropout layer with 25% drop rate
# Dense layer as output
ships_model2 <- keras_model_sequential() %>% 
  layer_conv_2d(input_shape = c(80, 80, 3),
                filters = 64,
                kernel_size = c(3, 3),
                strides = c(1, 1),
                activation = "linear",
                padding = "same") %>% 
  layer_batch_normalization() %>% 
  layer_activation(activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2),
                       strides = c(2, 2)) %>% 
  layer_dropout(rate = .25) %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                strides = c(1, 1),
                activation = "linear",
                padding = "same") %>% 
  layer_batch_normalization() %>% 
  layer_activation(activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2),
                       strides = c(2, 2)) %>% 
  layer_dropout(rate = .25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dropout(rate = .25) %>% 
  layer_dense(2, activation = "softmax")

# 2. Compile the model using binary crossentropy as loss function and
# Adamax as optimizer with learning rate equal to 0.0001 and learning rate decay equal to 1e-6
ships_model2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adamax(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

# 3. Fit the model. Use 20% of the data for validation, 20 epochs and 32 samples for batch size.
# Use early stopping with respect to validation loss, set patience to 5.
# Save best model (create checkpoint) with respect to validation loss every 2 epochs.
# Save logs for Tensorboard (to the "logs" folder)
ships_fit2 <- ships_model2 %>% fit(ships_test$data, ships_test$labels,
                                   epochs = 20, batch_size = 32,
                                   validation_split = 0.2,
                                   callbacks = c(callback_early_stopping(monitor = "val_loss", patience = 5),
                                                 callback_model_checkpoint(monitor = "val_loss", period = 2,
                                                                           save_best_only = TRUE,
                                                                           filepath = "ships_best.hdf5"),
                                                 callback_tensorboard(log_dir = "logs")))

dir.create("logs")

tensorboard("logs")



# 7. Evaluate model on test set

ships_model2 %>% evaluate(ships_test$data, ships_test$labels)