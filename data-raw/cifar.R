devtools::load_all()
library(mlr3misc)
library(data.table)

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

  seek(con, (i - 1) * record_size, origin = "current") # previous labels and images
  seek(1) # label

  r = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))
  g = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))
  b = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))

  img[,,1] = matrix(r, ncol = 32, byrow = TRUE)
  img[,,2] = matrix(g, ncol = 32, byrow = TRUE)
  img[,,3] = matrix(b, ncol = 32, byrow = TRUE)

  img
}

# cached
constructor_cifar10 = function(path) {
  require_namespaces("torchvision")

  torchvision::cifar10_dataset(root = path, download = FALSE)
  browser()
  train_files = file.path(path, "cifar-10-batches-bin", sprintf("data_batch_%d.bin", 1:5))
  test_file = file.path(path, "cifar-10-batches-bin", "test_batch.bin")

  train_labels = unlist(map(train_files, read_cifar_labels_batch))

  # TODO: ensure this is all correct, Claude-generated
  data.table(
    class = factor(c(train_labels, rep(NA, times = 10000))),
    file = c(rep(train_files, each = 10000),
             rep(test_file, 10000)),
    index = c(rep(1:10000, 5),
             1:10000),
    split = factor(rep(c("train", "test"), c(50000, 10000))),
    ..row_id = seq_len(60000)
  )
}

path <- here::here("cache")
fs::dir_create(path, recurse = TRUE)

lazy_dt <- constructor_cifar10(path)

dd = as_data_descriptor(cifar_ds, list(x = c(NA, 3, 32, 32)))
