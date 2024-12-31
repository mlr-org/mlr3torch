devtools::load_all()

# ci = col_info(get_private(tsk("cifar10")$backend)$.constructor())

# saveRDS(ci, here::here("inst/col_info/cifar10.rds"))

library(mlr3misc)
library(data.table)
library(torchvision)

# for a specific batch file
read_cifar_labels_batch = function(file_path) {
  con = file(file_path, "rb")
  on.exit({close(con)}, add = TRUE)

  labels = integer(length = 10000)
  for (i in 1:10000) {
    labels[i] = readBin(con, integer(), n = 1, size = 1, endian="big")
    seek(con, 32 * 32 * 3, origin = "current")
  }

  labels
}

# for a specific batch file
read_cifar_image = function(file_path, i, type = 10) {
  record_size = 1 + (32 * 32 * 3)

  con = file(file_path, "rb")
  on.exit({close(con)}, add = TRUE)

  seek(con, (i - 1) * record_size, origin = "start") # previous labels and images
  seek(con, 1, origin = "current") # label

  r = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))
  g = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))
  b = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))

  img = array(dim = c(32, 32, 3))
  img[,,1] = matrix(r, ncol = 32, byrow = TRUE)
  img[,,2] = matrix(g, ncol = 32, byrow = TRUE)
  img[,,3] = matrix(b, ncol = 32, byrow = TRUE)

  img
}

# cached
constructor_cifar10 = function(path) {
  require_namespaces("torchvision")

  torchvision::cifar10_dataset(root = path, download = TRUE)

  train_files = file.path(path, "cifar-10-batches-bin", sprintf("data_batch_%d.bin", 1:5))
  test_file = file.path(path, "cifar-10-batches-bin", "test_batch.bin")

  train_labels = unlist(map(train_files, read_cifar_labels_batch))

  # TODO: ensure this is all correct, Claude-generated
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
fs::dir_create(path, recurse = TRUE)

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

# begin 100

withr::local_options(mlr3torch.cache = TRUE)
devtools::load_all()
ci = col_info(get_private(tsk("cifar100")$backend)$.constructor())
saveRDS(ci, here::here("inst/col_info/cifar100.rds"))

path = file.path(get_cache_dir(), "datasets", "cifar100")

torchvision::cifar100_dataset(root = path, download = TRUE)

train_file = file.path(path, "cifar-100-binary", "train.bin")
test_file = file.path(path, "cifar-100-binary", "test.bin")

train_labels = read_cifar_labels_batch(train_file, type = 100)

# TODO: ensure this is all correct, Claude-generated
data = data.table(
  class = factor(c(train_labels, rep(NA, times = 10000))),
  file = c(rep(train_file, 50000),
           rep(test_file, 10000)),
  idx_in_file = c(1:50000, 1:10000),
  split = factor(rep(c("train", "test"), c(50000, 10000))),
  ..row_id = seq_len(60000)
)

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

task$filter(1:50000)

ci = col_info(task$backend)

saveRDS(ci, here::here("inst/col_info/cifar100.rds"))

task_new = TaskClassif$new(
  id = "cifar100",
  backend = task$backend,
  target = "class"
)

task_new$data()


