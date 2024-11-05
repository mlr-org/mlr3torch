library(torch)
library(torchvision)
library(mlr3torch)
library(here)

library(data.table)
library(tidytable)

# TODO: figure out whether we want the v2 file
training_metadata_v2 = fread(here::here("cache", "ISIC_2020_Training_GroundTruth_v2.csv"))

cache_dir = here("cache")
# construct a torch dataset
ds = torch::dataset(
  initialize = function() {
    self$.metadata = fread(here(cache_dir, "ISIC_2020_Training_GroundTruth.csv"))
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

melanoma_ds = ds()

dd = as_data_descriptor(melanoma_ds, list(x = NULL))
lt = lazy_tensor(dd)
dt_train = cbind(training_metadata_v2, data.table(x = lt))
# as_task_regr(dt_train, target = "corr", id = "guess_the_correlation")

training_duplicates = fread(here(cache_dir, "ISIC_2020_Training_Duplicates.csv"))
