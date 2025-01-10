library(mlr3torch)
library(mlr3verse)


# chunk 1
library(torchdatasets)
data_dir = here::here("data")
dogs_vs_cats_dataset(data_dir, download = TRUE)

# chunk 2
ds = torch::dataset("dogs_vs_cats",
  initialize = function(pths) {
    self$pths = pths
  },
  .getitem = function(i) {
    list(image = torchvision::base_loader(self$pths[i]))
  },
  .length = function() {
    length(self$pths)
  }
)

# chunk 3
paths = list.files(file.path(data_dir, "dogs-vs-cats/train"), full.names = TRUE)
dogs_vs_cats = ds(paths)

# chunk 4
lt = as_lazy_tensor(ds, list(image = NULL))
head(lt)

# chunk 5
labels = ifelse(grepl("dog\\.\\d+\\.jpg", paths), "dog", "cat")
table(labels)

# chunk 6
tbl = data.table(image = lt, class = label)
task = as_task_classif(tbl, target = "class", id = "dogs_vs_cats")
task

# chunk 7
augment = po("augment_random_vertical_flip", p = 0.5)

# chunk 8
preprocess = po("trafo_reshape", shape = c(NA, 3, 224, 224))

# chunk 9
resnet = lrn("classif.resnet18",
  pretrained = TRUE,
  epochs = 10,
  validate = 1 / 3,
  measures_valid = msrs(c("classif.logloss", "classif.acc"))
)

# chunk 10
unfreezer = t_clbk("unfreeze",
  starting_weights = c("fc.weights", "fc.bias"),
  unfreeze = data.table(
    epochs = 3, weights = select_all()
  )
)
learner$set_values(
  callbacks = list(unfreezer, t_clbk("history"))
)


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
