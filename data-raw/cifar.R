devtools::load_all()

library(mlr3misc)
library(data.table)
library(torchvision)

# cached
cifar_ds_generator = torch::dataset(
  initialize = function(images) {
    self$images = images
  },
  .getitem = function(idx) {
    force(idx)

    x = torch_tensor(self$images[idx, , , ])

    return(list(x = x))
  },
  .length = function() {
    dim(self$images)[1L]
  }
)

constructor_cifar = function(path, type = 10) {
  if (type == 10) {
    d_train = torchvision::cifar10_dataset(root = path, train = TRUE, download = TRUE)
    d_test = torchvision::cifar10_dataset(root = path, train = FALSE, download = FALSE)
    class_names = readLines(file.path(path, "cifar-10-batches-bin", "batches.meta.txt"))
    class_names = class_names[class_names != ""]
  } else if (type == 100) {
    d_train = torchvision::cifar100_dataset(root = path, train = TRUE, download = TRUE)
    d_test = torchvision::cifar100_dataset(root = path, train = FALSE, download = FALSE)
    class_names = readLines(file.path(path, "cifar-100-binary", "fine_label_names.txt"))
  }

  classes = c(d_train$y, d_test$y)
  images = array(NA, dim = c(60000, 3, 32, 32))
  # original data has channel dimension at the end
  perm_idx = c(1, 4, 2, 3)
  images[1:50000, , , ] = aperm(d_train$x, perm_idx, resize = TRUE)
  images[50001:60000, , , ] = aperm(d_test$x, perm_idx, resize = TRUE)

  return(list(class = factor(classes, labels = class_names), images = images))
}

constructor_cifar10 = function(path) {
  require_namespaces("torchvision")

  return(constructor_cifar(path, type = 10))
}

withr::local_options(mlr3torch.cache = TRUE)
path = file.path(get_cache_dir(), "datasets", "cifar10", "raw")

# begin CIFAR-10
data <- constructor_cifar10(path)

cifar10_ds = cifar_ds_generator(data$images)

dd = as_data_descriptor(cifar10_ds, list(x = c(NA, 3, 32, 32)))
lt = lazy_tensor(dd)

tsk_dt = data.table(
  class = data$class,
  image = lt,
  split = factor(rep(c("train", "test"), c(50000, 10000))),
  ..row_id = seq_len(60000)
)

# tsk_dt = cbind(data, data.table(image = lt))

tsk_cifar10 = as_task_classif(tsk_dt, target = "class", id = "cifar10")
tsk_cifar10$col_roles$feature = "image"

ci = col_info(tsk_cifar10$backend)

saveRDS(ci, here::here("inst/col_info/cifar10.rds"))
# end CIFAR-10

path = file.path(get_cache_dir(), "datasets", "cifar100", "raw")

# begin CIFAR-100
constructor_cifar100 = function(path) {
  require_namespaces("torchvision")

  return(constructor_cifar(path, type = 100))
}

data = constructor_cifar100(path)

cifar100_ds = cifar_ds_generator(data$images)

dd = as_data_descriptor(cifar100_ds, list(x = c(NA, 3, 32, 32)))
lt = lazy_tensor(dd)

dt = data.table(
  class = data$class,
  image = lt,
  split = factor(rep(c("train", "test"), c(50000, 10000))),
  ..row_id = seq_len(60000)
)

task = as_task_classif(dt, target = "class")

task$col_roles$feature = "image"

ci = col_info(task$backend)

saveRDS(ci, here::here("inst/col_info/cifar100.rds"))

