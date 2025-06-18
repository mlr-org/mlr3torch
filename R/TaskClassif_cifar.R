#' @title CIFAR Classification Tasks
#'
#' @name mlr_tasks_cifar
#' @aliases mlr_tasks_cifar10 mlr_tasks_cifar100
#'
#' @format [R6::R6Class] inheriting from [mlr3::TaskClassif].
#' @include aaa.R
#'
#' @description
#' The CIFAR-10 and CIFAR-100 datasets. A subset of the 80 million tiny images dataset
#' with noisy labels was supplied to student labelers, who were asked to filter out
#' incorrectly labeled images.
#' The images are have datatype `torch_long()`.
#'
#' CIFAR-10 contains 10 classes. CIFAR-100 contains 100 classes, which may be partitioned into 20 superclasses of 5 classes each.
#' The CIFAR-10 and CIFAR-100 classes are mutually exclusive.
#' See Chapter 3.1 of [the technical report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) for more details.
#'
#' The data is obtained from [`torchvision::cifar10_dataset()`] (or `torchvision::cifar100_dataset()`).
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

load_task_cifar10 = function(id = "cifar10") {
  cached_constructor = function(backend) {
    data <- cached(constructor_cifar10, "datasets", "cifar10")$data

    cifar10_ds = cifar_ds_generator(data$images)

    dd = as_data_descriptor(cifar10_ds, list(x = c(NA, 3, 32, 32)))
    lt = lazy_tensor(dd)

    dt = data.table(
      class = data$class,
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

  return(constructor_cifar(path, type = 100))
}

load_task_cifar100 = function(id = "cifar100") {
  cached_constructor = function(backend) {
    data = cached(constructor_cifar100, "datasets", "cifar100")$data

    cifar100_ds = cifar_ds_generator(data$images)

    dd = as_data_descriptor(cifar100_ds, list(x = c(NA, 3, 32, 32)))
    lt = lazy_tensor(dd)

    dt = data.table(
      class = data$class,
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
