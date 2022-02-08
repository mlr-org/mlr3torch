# Experimenting with image tasks
mlr_reflections$task_feature_types[["img"]] <- "imageuri"
# add class "imageuri" to column

# torchvision dataset format, stores binary files on disk
cifar10_bin <- torchvision::cifar10_dataset(root = "/opt/example-data", download = TRUE)

# Keras version: Not sure where files are stored
cifar10_keras <- keras::dataset_cifar10()
