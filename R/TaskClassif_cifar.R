# #' @title CIFAR-10 Classification Task
# #' 
# #' @name mlr_tasks_cifar_10
# #' 
# #' @description 
# #' The CIFAR-10 subset of the 80 million tiny images dataset.
# #' The data is obtained from [`torchvision::cifar10_dataset()`].
# NULL

# # TODO: implement both CIFAR-10 and CIFAR-100 in the same file
# # the torchvision implementation and the PipeOpAdaptiveAvgPool implementations are probably helpful here
# constructor_cifar10 = function(path) {
#   require_namespaces("torchvision")

#   torchvision::cifar10_dataset(root = path, download = TRUE)
  
#   train_files = file.path(path, "cifar-10-batches-bin", 
#     sprintf("data_batch_%d.bin", 1:5))
#   test_file = file.path(path, "cifar-10-batches-bin", "test_batch.bin")

#   train_labels = map_int(train_files, read_cifar_labels)

#   data.table(
#     class = factor(c(train_labels, rep(NA, times = 10000))),
#     file = c(rep(train_files, each = 10000)
#              rep(test_file, 10000)),
#     index = c(rep(1:10000, 5),
#              1:10000),
#     split = factor(rep(c("train", "test"), c(50000, 10000))),
#     ..row_id = seq_len(60000)
#   )
# }

# load_task_cifar10 = function(id = "cifar10") {
#   cached_constructor = function(backend) {
#     data = cached(constructor_cifar10, "datasets", "cifar10")$data

#     cifar10_ds_generator = torch::dataset(
#       initialize = function(data, cache_dir) {
#         self$.data = data
#         self$.cache_dir = cache_dir
#       },
#       .getitem = function(idx) {
#         list(img = read_cifar_image(file.path(cache_dir, self$.data[, "file"]), idx))
#       },
#       .length = function() {
#         nrow(self$.data)
#       }
#     )

#     cifar10_ds = cifar10_ds_generator(data, file.path(get_cache_dir(), "datasets", "cifar10"))
#   }

#   backend = DataBackendLazy$new(
#     constructor = cached_constructor,
#     rownames = seq_len(60000),
#     col_info = load_col_info("cifar10"),
#     primary_key = "..row_id"
#   )

#   task = TaskClassif$new(
#     backend = backend,
#     id = "cifar10",
#     target = "class",
#     label = "CIFAR-10 Classification"
#   )

#   backend$hash = task$man = "mlr3torch::mlr_tasks_cifar10"

#   task$filter(1:50000)

#   return(task)
# }

# read_cifar_labels = function(file) {
#   con = file(file, "rb")
#   on.exit({close(con)}, add = TRUE)
  
#   labels = vector("integer", 10000)
#   for (i in 1:10000) {
#     labels[i] = readBin(con, "integer", n = 1, size = 1)
#     seek(con, 32 * 32 * 3)
#   }
  
#   labels
# }

# read_cifar_image = function(file, i, type = 10) {
#   record_size = 1 + (32 * 32 * 3)

#   con = file(file, "rb")
#   on.exit({close(con)}, add = TRUE)

#   seek(con, (i - 1) * record_size) # previous labels and images
#   seek(1) # label

#   r = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))
#   g = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))
#   b = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))

#   img[,,1] = matrix(r, ncol = 32, byrow = TRUE)
#   img[,,2] = matrix(g, ncol = 32, byrow = TRUE)
#   img[,,3] = matrix(b, ncol = 32, byrow = TRUE)

#   img
# }

# register_task("cifar10", load_task_cifar10)