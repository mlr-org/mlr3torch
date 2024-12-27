devtools::load_all()

path <- here::here("cache")
fs::dir_create(path, recurse = TRUE)

cifar_ds_train <- torchvision::cifar10_dataset(root = file.path(path), train = TRUE)
cifar_ds_test <- torchvision::cifar10_dataset(root = file.path(path), download = FALSE, train = TRUE)

dd = as_data_descriptor(cifar_ds, list(x = c(NA, 3, 32, 32)))

read_batch <- function(path, type = 10) {
  if (type == 10)
    n <- 10000
  else if (type == 100 && grepl("test", path))
    n <- 10000
  else
    n <- 50000

  imgs <- array(dim = c(n, 32, 32, 3))
  labels <- integer(length = n)
  if (type == 100)
    fine_labels <- integer(length = n)

  con <- file(path, open = "rb")
  on.exit({close(con)}, add = TRUE)

  for (i in seq_len(n)) {

  labels[i] <- readBin(con, integer(), size=1, n=1, endian="big")

    if (type == 100) {
      fine_labels[i] <- readBin(con, integer(), size=1, n=1, endian="big")
    }

    r <- as.integer(readBin(con, raw(), size=1, n=1024, endian="big"))
    g <- as.integer(readBin(con, raw(), size=1, n=1024, endian="big"))
    b <- as.integer(readBin(con, raw(), size=1, n=1024, endian="big"))

    imgs[i,,,1] <- matrix(r, ncol = 32, byrow = TRUE)
    imgs[i,,,2] <- matrix(g, ncol = 32, byrow = TRUE)
    imgs[i,,,3] <- matrix(b, ncol = 32, byrow = TRUE)
  }

  if (type == 100)
    list(imgs = imgs, labels = fine_labels)
  else
    list(imgs = imgs, labels = labels)
}

# path: the full path to the batch binary file
# i: the "global" index (1 to 60k) of the image
read_img_from_batch = function(path, i) {
  n = 10000

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

batch = read_batch(file.path(path, "cifar-10-batches-bin", "data_batch_2.bin"))
readBin(file.path(path, "cifar-10-batches-bin", "data_batch_2.bin"), integer(), size=1, )