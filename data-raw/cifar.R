devtools::load_all()

library(mlr3misc)
library(data.table)
library(torchvision)

# cached
constructor_cifar10 = function(path) {
  require_namespaces("torchvision")

  torchvision::cifar10_dataset(root = path, download = TRUE)

  train_files = file.path(path, "cifar-10-batches-bin", sprintf("data_batch_%d.bin", 1:5))
  test_file = file.path(path, "cifar-10-batches-bin", "test_batch.bin")

  # TODO: convert these to the meaningful names
  train_labels_ints = unlist(map(train_files, read_cifar_labels_batch, type = 10))
  class_names = readLines(file.path(path, "cifar-10-batches-bin", "batches.meta.txt"))
  class_names = class_names[class_names != ""]

  data.table(
    class = factor(c(train_labels_ints, rep(NA, times = 10000)), labels = class_names),
    file = c(rep(train_files, each = 10000),
             rep(test_file, 10000)),
    idx_in_file = c(rep(1:10000, 5),
             1:10000),
    split = factor(rep(c("train", "test"), c(50000, 10000))),
    ..row_id = seq_len(60000)
  )
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

  # torchvision::cifar100_dataset(root = path, download = TRUE)

  # train_file = file.path(path, "cifar-100-binary", "train.bin")
  # test_file = file.path(path, "cifar-100-binary", "test.bin")

  # train_labels_ints = read_cifar_labels_batch(train_file, type = 100)
  # class_names = readLines(file.path(path, "cifar-100-binary", "fine_label_names.txt"))

  # class = factor(c(train_labels_ints, rep(NA, times = 10000)), labels = class_names)

  # data.table(
  #   class = factor(c(train_labels_ints, rep(NA, times = 10000)), labels = class_names),
  #   file = c(rep(train_file, 50000),
  #            rep(test_file, 10000)),
  #   idx_in_file = c(1:50000, 1:10000),
  #   split = factor(rep(c("train", "test"), c(50000, 10000))),
  #   ..row_id = seq_len(60000)
  # )
  tv_ds = torchvision::cifar100_dataset(root = path, download = TRUE)
  class_names = readLines(file.path(path, "cifar-100-binary", "fine_label_names.txt"))
  
  # TODO: use $.getitem() instead of $x
  # data.table(
  #   class = factor(as.integer(tv_ds$y), labels = class_names),
  #   image = self$x,
  #   split = factor(rep(c("train", "test"), c(50000, 10000))),
  #   ..row_id = seq_len(60000)
  # )
}

data = constructor_cifar100(path)

cifar100_ds_generator = torch::dataset(
  initialize = function() {
    self$.data = data
  },
  .getitem = function(idx) {
    force(idx)

    x = torch_tensor(read_cifar_image(self$.data$file[idx], self$.data$idx_in_file[idx], type = 100))

    return(list(x = x))
  },
  .length = function() {
    nrow(self$.data)
  }
)

cifar100_ds = cifar100_ds_generator()

dd = as_data_descriptor(cifar100_ds, list(x = c(NA, 32, 32, 3)))
lt = lazy_tensor(dd)

dt = cbind(data, data.table(image = lt))

task = as_task_classif(dt, target = "class")

task$col_roles$feature = "image"

ci = col_info(task$backend)

saveRDS(ci, here::here("inst/col_info/cifar100.rds"))

