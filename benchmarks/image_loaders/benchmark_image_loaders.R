library(torch)
library(torchvision)
library(mlr3torch)
library(here)

library(data.table)
setDTthreads(threads = 1)

training_metadata = fread(here::here("cache", "ISIC_2020_Training_GroundTruth.csv"))

# hard-coded cache directory that I use locally
cache_dir = here("cache")

ds_base_loader = torch::dataset(
  initialize = function(n_images) {
    self$.metadata = fread(here(cache_dir, "ISIC_2020_Training_GroundTruth.csv"))[1:n_images, ]
    self$.path = file.path(here(cache_dir), "train")
  },
  .getitem = function(idx) {
    force(idx)

    x = torchvision::base_loader(file.path(self$.path, paste0(self$.metadata[idx, ]$image_name, ".jpg")))
    x = torchvision::transform_to_tensor(x)

    return(list(x = x))
  },
  .length = function() {
    nrow(self$.metadata)
  }
)

ds_magick_loader = torch::dataset(
  initialize = function(n_images) {
    self$.metadata = fread(here(cache_dir, "ISIC_2020_Training_GroundTruth.csv"))[1:n_images, ]
    self$.path = file.path(here(cache_dir), "train")
  },
  .getitem = function(idx) {
    force(idx)

    image_name = self$.metadata[idx, ]$image_name

    x = torchvision::magick_loader(file.path(self$.path, paste0(image_name, ".jpg")))
    x = torchvision::transform_to_tensor(x)

    return(list(x = x, image_name = image_name))
  },
  .length = function() {
    nrow(self$.metadata)
  }
)

n_images = 100

ds_base = ds_base_loader(n_images)
ds_magick = ds_magick_loader(n_images)

bench::mark(
  for (i in 1:n_images) ds_base$.getitem(i),
  for (i in 1:n_images) ds_magick$.getitem(i)
)
