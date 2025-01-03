devtools::load_all()
withr::local_options(mlr3torch.cache = TRUE)

constructor_cifar10 = function(path) {
    require_namespaces("torchvision")

    ds_train = torchvision::cifar10_dataset(root = file.path(path), download = TRUE, train = TRUE)

    browser()


}

constructor_cifar10(path <- file.path(get_cache_dir(), "datasets", "cifar10"))

devtools::load_all()
library(here)

path <- here("cache")

cifar_ds_train <- torchvision::cifar10_dataset(root = file.path(path), train = TRUE)
cifar_ds_test <- torchvision::cifar10_dataset(root = file.path(path), download = FALSE, train = TRUE)

# path: the full path to the batch binary file
# i: the "global" index (1 to 60k) of the image
# so, the lazy version of this data needs to store
  # file name
  # offset
read_img_from_batch = function(path, i) {
  img = array(dim = c(32, 32, 3))
  
  con = file(path, open = "rb")

  on.exit({close(con)}, add = TRUE)

  label = readBin(con, integer(), size = 1, n = 1, endian = "big")

  r = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))
  g = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))
  b = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))

  img[,,1] = matrix(r, ncol = 32, byrow = TRUE)
  img[,,2] = matrix(g, ncol = 32, byrow = TRUE)
  img[,,3] = matrix(b, ncol = 32, byrow = TRUE)

  list(img = img, label = label)
}

# first: remove the response (handled separately)
dd = as_data_descriptor(cifar_ds_train, list(x = c(NA, 3, 32, 32)))

img10 = read_img_from_batch(file.path(path, "cifar-10-batches-bin", "data_batch_2.bin"), 1)
