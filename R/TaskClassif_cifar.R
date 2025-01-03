#' @title CIFAR Classification Tasks
#'
#' @name mlr_tasks_cifar10
#'
#' @format [R6::R6Class] inheriting from [mlr3::TaskClassif].
#' @include aaa.R
#'
#' @description
#' The CIFAR-10 and CIFAR-100 subsets of the 80 million tiny images dataset.
#' The data is obtained from [`torchvision::cifar10_dataset()`].
#'
#' @section Construction:
#' ```
#' tsk("cifar10")
#' tsk("cifar100")
#' ```
#'
#' @template task_download
#'
#' @section Properties:
#' `r rd_info_task_torch("cifar10", missings = FALSE)`
#'
#' @references
#' `r format_bib("cifar2009")`
#' @examplesIf torch::torch_is_installed()
#' task_cifar10 = tsk("cifar10")
#' task_cifar100 = tsk("cifar100")
#' print(task_cifar10)
#' print(task_cifar100)
NULL

# for a specific batch file
read_cifar_labels_batch = function(file_path, type = 10) {
  con = file(file_path, "rb")
  on.exit({close(con)}, add = TRUE)

  if (type == 10) {
    batch_size <- 10000
  } else if (type == 100 && grepl("test", file_path)) {
    batch_size <- 10000
  } else {
    batch_size <- 50000
  }

  labels = integer(length = batch_size)
  if (type == 100) {
    fine_labels = integer(length = batch_size)
  }

  for (i in 1:batch_size) {
    labels[i] = readBin(con, integer(), n = 1, size = 1, endian = "big")
    if (type == 100) {
      fine_labels[i] = readBin(con, integer(), n = 1, size = 1, endian = "big")
    }
    seek(con, 32 * 32 * 3, origin = "current")
  }

  if (type == 100) fine_labels else labels
}

# for a specific batch file
read_cifar_image = function(file_path, i, type = 10) {
  fine_label = as.integer(type == 100)
  record_size = 1 + fine_label + (32 * 32 * 3)

  con = file(file_path, "rb")
  on.exit({close(con)}, add = TRUE)

  seek(con, (i - 1) * record_size, origin = "start") # previous labels and images
  seek(con, 1 + fine_label, origin = "current") # seek past the current label(s)

  r = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))
  g = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))
  b = as.integer(readBin(con, raw(), size = 1, n = 1024, endian = "big"))

  img = array(dim = c(32, 32, 3))
  img[,,1] = matrix(r, ncol = 32, byrow = TRUE)
  img[,,2] = matrix(g, ncol = 32, byrow = TRUE)
  img[,,3] = matrix(b, ncol = 32, byrow = TRUE)

  img
}

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

load_task_cifar10 = function(id = "cifar10") {
  cached_constructor = function(backend) {
    data = cached(constructor_cifar10, "datasets", "cifar10")$data

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

    dt = cbind(data, data.table(image = lt))

    DataBackendDataTable$new(data = dt, primary_key = "..row_id")
  }

  backend = DataBackendLazy$new(
    constructor = cached_constructor,
    rownames = seq_len(60000),
    col_info = load_col_info("cifar10"),
    primary_key = "..row_id"
  )

  task = TaskClassif$new(
    backend = backend,
    id = "cifar10",
    target = "class",
    label = "CIFAR-10 Classification"
  )

  task$col_roles$feature = "image"

  # TODO: different hash, same manual
  backend$hash = task$man = "mlr3torch::mlr_tasks_cifar"

  task$filter(1:50000)

  return(task)
}

register_task("cifar10", load_task_cifar10)

#' @title CIFAR-100 Classification Task
#'
#' @name mlr_tasks_cifar100
#'
#' @format [R6::R6Class] inheriting from [mlr3::TaskClassif].
#' @include aaa.R
#'
#' @description
#' The 0CIFAR-100 subsets0 of the 80 million tiny images dataset.
#' The data is obtained from [`torchvision::cifar100_dataset()`].
#'
#' @section Construction:
#' ```
#' tsk("cifar100")
#' ```
#'
#' @template task_download
#'
#' @section Properties:
#' `r rd_info_task_torch("cifar100", missings = FALSE)`
#'
#' @references
#' `r format_bib("cifar2009")`
#' @examplesIf torch::torch_is_installed()
#' task_cifar100 = tsk("cifar100")
#' print(task_cifar100)
NULL

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

load_task_cifar100 = function(id = "cifar100") {
  cached_constructor = function(backend) {
    data = cached(constructor_cifar100, "datasets", "cifar100")$data

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

    DataBackendDataTable$new(data = dt, primary_key = "..row_id")
  }

  backend = DataBackendLazy$new(
    constructor = cached_constructor,
    rownames = seq_len(60000),
    col_info = load_col_info("cifar100"),
    primary_key = "..row_id"
  )

  task = TaskClassif$new(
    backend = backend,
    id = "cifar100",
    target = "class",
    label = "CIFAR-100 Classification"
  )

  task$col_roles$feature = "image"

  backend$hash = task$man = "mlr3torch::mlr_tasks_cifar"

  task$filter(1:50000)

  return(task)
}

register_task("cifar100", load_task_cifar100)
