library(mlr3torch)
library(torch)
library(torchdatasets)


# Define the path to the Cats vs. Dogs dataset
# Ensure that the dataset is organized in subdirectories for each class
# For example:
# data/
# ├── cats/
# │   ├── cat001.jpg
# │   ├── cat002.jpg
# │   └── ...
# └── dogs/
#     ├── dog001.jpg
#     ├── dog002.jpg
#     └── ...

data_dir <- here::here("data")

dogs_vs_cats_dataset(data_dir, train = TRUE, download = TRUE)


# Create a Torch dataset using ImageFolder structure
torchdatasets::dogs_vs_cats_dataset(
  data_dir = data_dir,
  train = FALSE
)

dataset_cv <- torchvision::dataset_image_folder(

  path = data_dir,
  transform = transform_to_tensor()
)


# the files in ./data/dogs-vs-cats/train are like dog.100.jpg, cat.100.jpg,
# create a data.table with the paths and whether they are a dog or cat
pathd = list.files(file.path(data_dir, "dogs-vs-cats/train"), pattern = "*.jpg", recursive = TRUE, full.names = TRUE)
# regex that matches everything that ends with dog.<id>.jpg
labels = ifelse(grepl("dog\\.\\d+\\.jpg", paths), "dog", "cat")


# Wrap the dataset in a lazy_tensor
# This allows for efficient on-the-fly data loading without storing all tensors in memory
lazy_cv <- lazy_tensor(dataset_cv)
