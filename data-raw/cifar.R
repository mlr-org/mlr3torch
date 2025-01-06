devtools::load_all()

library(mlr3misc)
library(data.table)
library(torchvision)

# cached
constructor_cifar10 = function(path) {
  require_namespaces("torchvision")

  tv_ds_train = torchvision::cifar10_dataset(root = path, download = TRUE)
  tv_ds_test = torchvision::cifar10_dataset(root = path, train = fALSE, download = FALSE)

  tv_data_train = tv_ds$.getitem(1:50000)
  tv_data$class_names = readLines(file.path(path, "cifar-10-batches-bin", "batches.meta.txt"))

  tv_data

  # return: data.table
  # nested list
  # train
    # x, y, class_names
  # test
    # x, y
}

withr::local_options(mlr3torch.cache = TRUE)
path <- file.path(get_cache_dir(), "datasets", "cifar10", "raw")

# begin CIFAR-10
data <- constructor_cifar10(path)

cifar10_ds_generator = torch::dataset(
  initialize = function() {
    self$.data = data
  },
  .getitem = function(idx) {
    force(idx)

    x = torch_tensor(read_cifar_image(self$.data$file[idx], self$.data$idx_in_file[idx]))

    return(list(x = x))
  },
  .length = function() {
    nrow(self$.data)
  }
)

cifar10_ds = cifar10_ds_generator()

dd = as_data_descriptor(cifar10_ds, list(x = c(NA, 32, 32, 3)))
lt = lazy_tensor(dd)

tsk_dt = cbind(data, data.table(image = lt))

tsk_cifar10 = as_task_classif(tsk_dt, target = "class", id = "cifar10")
tsk_cifar10$col_roles$feature = "image"

ci = col_info(tsk_cifar10$backend)

saveRDS(ci, here::here("inst/col_info/cifar10.rds"))
# end CIFAR-10

path <- file.path(get_cache_dir(), "datasets", "cifar100", "raw")

# begin CIFAR-100
constructor_cifar100 = function(path) {
  require_namespaces("torchvision")

  tv_ds = torchvision::cifar100_dataset(root = path, download = TRUE)
  class_names = readLines(file.path(path, "cifar-100-binary", "fine_label_names.txt"))

  tv_data = tv_ds$.getitem(1:60000)

  tv_data$class_names = class_names

  tv_data
}

data = constructor_cifar100(path)

cifar100_ds_generator = torch::dataset(
  initialize = function() {
    self$.img_arr = data$x
  },
  .getitem = function(idx) {
    force(idx)

    x = torch_tensor(self$.img_arr[i, , , ])

    return(list(x = x))
  },
  .length = function() {
    dim(self$.img_arr)[1]
  }
)

cifar100_ds = cifar100_ds_generator()

dd = as_data_descriptor(cifar100_ds, list(x = c(NA, 32, 32, 3)))
lt = lazy_tensor(dd)

dt = data.table(
  class = factor(data$y, labels = data$class_names),
  image = lt,
  split = factor(rep(c("train", "test"), c(50000, 10000))),
  ..row_id = seq_len(60000)
)

task = as_task_classif(dt, target = "class")

task$col_roles$feature = "image"

ci = col_info(task$backend)

saveRDS(ci, here::here("inst/col_info/cifar100.rds"))

