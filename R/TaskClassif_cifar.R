#' @title CIFAR Classification Tasks
#'
#' @name mlr_tasks_cifar
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

constructor_cifar10 = function(path) {
  require_namespaces("torchvision")

  tv_ds_train = torchvision::cifar10_dataset(root = path, download = TRUE)
  tv_data_train = tv_ds_train$.getitem(1:50000)

  tv_ds_test = torchvision::cifar10_dataset(root = path, train = FALSE, download = FALSE)
  tv_data_test = tv_ds_test$.getitem(1:10000)

  labels = c(tv_data_train$y, tv_data_test$y)
  images = array(c(tv_data_train$x, tv_data_test$x), dim = c(60000, 32, 32, 3))

  class_names = readLines(file.path(path, "cifar-10-batches-bin", "batches.meta.txt"))
  class_names = class_names[class_names != ""]

  return(list(labels = labels, images = images, class_names = class_names))
}

load_task_cifar10 = function(id = "cifar10") {
  cached_constructor = function(backend) {
    data <- cached(constructor_cifar10, "datasets", "cifar10")$data

    cifar10_ds_generator = torch::dataset(
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

    cifar10_ds = cifar10_ds_generator(data$images)

    dd = as_data_descriptor(cifar10_ds, list(x = c(NA, 32, 32, 3)))
    lt = lazy_tensor(dd)

    dt = data.table(
      class = factor(data$labels, labels = data$class_names),
      image = lt,
      split = factor(rep(c("train", "test"), c(50000, 10000))),
      ..row_id = seq_len(60000)
    )

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

  backend$hash = "mlr3torch::mlr_tasks_cifar10"
  task$man = "mlr3torch::mlr_tasks_cifar"

  return(task)
}

register_task("cifar10", load_task_cifar10)

constructor_cifar100 = function(path) {
  require_namespaces("torchvision")

  tv_ds_train = torchvision::cifar100_dataset(root = path, download = TRUE)
  tv_data_train = tv_ds_train$.getitem(1:50000)

  tv_ds_test = torchvision::cifar100_dataset(root = path, train = FALSE, download = FALSE)
  tv_data_test = tv_ds_test$.getitem(1:10000)

  labels = c(tv_data_train$y, tv_data_test$y)
  images = array(c(tv_data_train$x, tv_data_test$x), dim = c(60000, 32, 32, 3))

  class_names = readLines(file.path(path, "cifar-100-binary", "fine_label_names.txt"))

  return(list(labels = labels, images = images, class_names = class_names))
}

load_task_cifar100 = function(id = "cifar100") {
  cached_constructor = function(backend) {
    data = cached(constructor_cifar100, "datasets", "cifar100")$data

    cifar100_ds_generator = torch::dataset(
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

    cifar100_ds = cifar100_ds_generator(data$images)

    dd = as_data_descriptor(cifar100_ds, list(x = c(NA, 32, 32, 3)))
    lt = lazy_tensor(dd)

    dt = data.table(
      class = factor(data$labels, labels = data$class_names),
      image = lt,
      split = factor(rep(c("train", "test"), c(50000, 10000))),
      ..row_id = seq_len(60000)
    )

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

  backend$hash = "mlr3torch::mlr_tasks_cifar100"
  task$man = "mlr3torch::mlr_tasks_cifar"

  return(task)
}

register_task("cifar100", load_task_cifar100)
