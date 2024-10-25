library(torch)
library(torchvision)
library(mlr3torch)
library(here)

library(data.table)
library(tidytable)

# TODO: figure out whether we want the v2 file
# I think no, since I don't really see a "use" for the lesion ID
training_metadata = fread(here::here("cache", "ISIC_2020_Training_GroundTruth.csv"))
# training_metadata_v2 = fread(here::here("cache", "ISIC_2020_Training_GroundTruth_v2.csv"))

cache_dir = here("cache")
# construct a torch dataset
ds = torch::dataset(
  initialize = function() {
    self$.metadata = fread(here("cache", "ISIC_2020_Training_GroundTruth.csv"))
    self$.path = file.path(here(cache_dir), "train")
  },
  .getitem = function(idx) {
    force(idx)

    x = torchvision::base_loader(file.path(self$.path, paste0(self$.metadata[idx, ]$image_name, ".jpg")))
    # TODO: decide on these transformations
    x = torchvision::transform_to_tensor(x) %>% torchvision::transform_rgb_to_grayscale()

    # TODO: should we only return the images here?
    return(list(x = x))
  },
  .length = function() {
    nrow(self$.metadata)
  }
)

melanoma_ds = ds()



# Construct lazy tensor for each image (e.g. a data table with a single ltnsr column)

# Join with the metadata file

# TODO: figure out the dimensions. The images each have a different dimension.
# dd_dims = c(NA, )
# dd = as_data_descriptor(ds, list(x = dd_dims))

lt = lazy_tensor(dd)

dt_train = inner_join(training_metadata, data.table(..., x = lt), by = image_name)

# as_task_regr(dt_train, target = "corr", id = "guess_the_correlation")