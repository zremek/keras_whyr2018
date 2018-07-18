library(keras)
library(tidyverse)

# 1 idea polega na uzyciu sieci wcześniej wytrenowanej
# 2 usuwamy głębokie warstwy 
# pozostawiamy i zamrażamy pierwsze warstwy konwulsyjne - 
  # one rozpoznają podstawowe kształty
# 3 dodajemy swoje warstwy głebokie
# 4 trenujemy na swoich danych

train_dir <- "WhyR2018_keras-master/data/cats-vs-dogs/train"
validation_dir <- "WhyR2018_keras-master/data/cats-vs-dogs/validation"
test_dir <- "WhyR2018_keras-master/data/cats-vs-dogs/test"

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 2, activation = "sigmoid")

summary(model)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical"
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

test_datagen <- image_data_generator(rescale = 1/255)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

cats_dogs_predictions <- model %>%
  predict_generator(
    test_generator,
    steps = 50
  )

# Ex. 4 - Data augumentation
# 1. Perform data augumentation:
# Rescale pixel values to [0, 1] interval
# Add 35 degree rotation
# Add 30% width and height shift
# Add 20% zoom
# Add horizontal flip
train_datagen <- 

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "categorical"
)

# 2. Retrieve batch from the generator using generator_next()
batch <- 

for (i in 1:4) {
  plot(as.raster(batch[[1]][i,,,]))
}

# Ex. 5 - Fine-tuning (You have to compile the model to see difference!)
# 1. Retrive convolutional base of VGG16 model pretreined on IMAGENET. Set input shape to 150x150x3
conv_base <- 
summary(conv_base)

# 2. Freeze first 6 layers of VGG16 model (from "block1_conv1" to "block2_pool")


# 3. Create new model based on VGG16. Add:
# flattening layer
# dense layer with 256 units and "relu" activation
# dense layer as output
model <- 

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("accuracy"))
summary(model)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)
