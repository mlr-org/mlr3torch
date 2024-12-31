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
  train_labels = unlist(map(train_files, read_cifar_labels_batch, type = 10))

  data.table(
    class = factor(c(train_labels, rep(NA, times = 10000))),
    file = c(rep(train_files, each = 10000),
             rep(test_file, 10000)),
    idx_in_file = c(rep(1:10000, 5),
             1:10000),
    split = factor(rep(c("train", "test"), c(50000, 10000))),
    ..row_id = seq_len(60000)
  )
}

path <- here::here("cache")

withr::local_options(mlr3torch.cache = TRUE)
path <- file.path(get_cache_dir(), "datasets", "cifar100")

fs::dir_create(path, recurse = TRUE)

# begin CIFAR-10
# data <- constructor_cifar10(path)

# cifar10_ds_generator = torch::dataset(
#   initialize = function() {
#     self$.data = data
#   },
#   .getitem = function(idx) {
#     force(idx)

#     x = torch_tensor(read_cifar_image(self$.data$file[idx], self$.data$idx_in_file[idx]))

#     return(list(x = x))
#   },
#   .length = function() {
#     nrow(self$.data)
#   }
# )

# cifar10_ds = cifar10_ds_generator()

# dd = as_data_descriptor(cifar10_ds, list(x = c(NA, 32, 32, 3)))
# lt = lazy_tensor(dd)

# tsk_dt = cbind(data, data.table(image = lt))

# tsk_cifar10 = as_task_classif(tsk_dt, target = "class", id = "cifar10")
# tsk_cifar10$col_roles$feature = "image"

# ci = col_info(tsk_cifar10$backend)

# saveRDS(ci, here::here("inst/col_info/cifar10.rds"))
# end CIFAR-10

# test interactively: look at torchvision version and this version for a few images, they should look the same
# img = cifar10_ds$.getitem(2)$x
# img_uint8 = (img * 255)$to(dtype = torch::torch_uint8())
# torchvision::tensor_image_browse(img_uint8)

# img_arr = as.array(img)

# torchvision direct dataset

# tv_cifar10_ds = cifar10_dataset(root = path, download = FALSE)
# tv_img = tv_cifar10_ds$.getitem(2)$x

# all.equal(img_arr, tv_img)

# test_same_at_idx = function(idx, ds_mlr3torch, ds_torch) {
#   all.equal(as.array(ds_mlr3torch$.getitem(idx)$x), ds_torch$.getitem(idx)$x)
# }

# trn_idx = 1:50000
# int_mlr3torch_responses = as.integer(tsk_dt$class[trn_idx])
# get_response = function(idx, ds) {
#   ds$.getitem(idx)$y
# }
# int_tv_responses = map_int(trn_idx, get_response, ds = tv_cifar10_ds)

# all.equal(int_mlr3torch_responses, int_tv_responses)

# idx_to_test = c(1, 2, 27, 9999,
#   10000, 10001, 10901, 19999,
#   20000, 20001, 29999,
#   30000, 30001, 39999,
#   40000, 40001, 49999,
#   50000)

# all(map_lgl(.x = idx_to_test, .f = test_same_at_idx, ds_mlr3torch = cifar10_ds, ds_torch = tv_cifar10_ds))

# test_same_at_idx(10001, cifar10_ds, tv_cifar10_ds)

# begin CIFAR-100
constructor_cifar100 = function(path) {
  require_namespaces("torchvision")

  torchvision::cifar100_dataset(root = path, download = TRUE)

  train_file = file.path(path, "cifar-100-binary", "train.bin")
  test_file = file.path(path, "cifar-100-binary", "test.bin")

  train_labels = read_cifar_labels_batch(train_file, type = 100)

  data.table(
    class = factor(c(train_labels, rep(NA, times = 10000))),
    file = c(rep(train_file, 50000),
             rep(test_file, 10000)),
    idx_in_file = c(1:50000, 1:10000),
    split = factor(rep(c("train", "test"), c(50000, 10000))),
    ..row_id = seq_len(60000)
  )
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

