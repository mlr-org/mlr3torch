#' @title MNIST Image classification
#' @name mlr_tasks_mnist
#' @description
#' Classic MNIST image classification.
#'
#' The underlying [`DataBackend`][mlr3::DataBackend] contains columns `"label"`, `"image"`, `"row_id"`, `"split"`, where the last column
#' indicates whether the row belongs to the train or test set.
#'
#' The first 60000 rows belong to the training set, the last 10000 rows to the test set.
#'
#' @section Construction:
#' ```
#' tsk("mnist")
#' ```
#'
#' @template task_download
#'
#' @source
#' \url{https://torchvision.mlverse.org/reference/mnist_dataset.html}
#'
#' @section Properties:
#' `r rd_info_task_torch("mnist", missings = FALSE)`
#'
#' @references
#' `r format_bib("mnist")`
#' @examplesIf torch::torch_is_installed()
#' task = tsk("mnist")
#' task
NULL

# @param path (`character(1)`)\cr
#   The cache_dir/datasets/tiny_imagenet folder.
constructor_mnist = function(path) {
  require_namespaces("torchvision")
  # path points to {cache_dir, tempfile}/data/mnist
  d_train = torchvision::mnist_dataset(root = file.path(path), train = TRUE, download = TRUE)
  # test set is already downloaded with the training set above
  d_test = torchvision::mnist_dataset(root = file.path(path), train = FALSE)

  images_train = array(d_train$data, dim = c(60000, 1, 28, 28))
  images_test = array(d_test$data, dim = c(10000, 1, 28, 28))

  images = array(NA, dim = c(70000, 1, 28, 28))
  images[1:60000, , , ] = images_train
  images[60001:70000, , , ] = images_test

  list(label = c(d_train$targets, d_test$targets), image = images)
}

load_task_mnist = function(id = "mnist") {
  cached_constructor = function(backend) {
    data = cached(constructor_mnist, "datasets", "mnist")$data
    labels = factor(data$label, levels = 1:10, labels = as.character(0:9))
    ds = dataset(
      initialize = function(images) {
        self$images = images
      },
      .getbatch = function(idx) {
        list(image = torch_tensor(self$images[idx, , , , drop = FALSE], dtype = torch_float32()))
      },
      .length = function() dim(self$images)[1L]
    )(data$image)

    data_descriptor = DataDescriptor$new(dataset = ds, list(image = c(NA, 1, 28, 28)))
    dt = data.table(
      image = lazy_tensor(data_descriptor),
      label = labels,
      ..row_id = seq_along(labels),
      split = factor(rep(c("train", "test"), times = c(60000, 10000)))
    )
    DataBackendDataTable$new(data = dt, primary_key = "..row_id")
  }

  backend = DataBackendLazy$new(
    constructor = cached_constructor,
    rownames = seq_len(70000),
    col_info = load_col_info("mnist"),
    primary_key = "..row_id",
    data_formats = "data.table"
  )

  task = TaskClassif$new(
    backend = backend,
    id = "mnist",
    target = "label",
    label = "MNIST Digit Classification",
  )

  backend$hash = task$man = "mlr3torch::mlr_tasks_mnist"
  task$col_roles$feature = "image"

  return(task)
}

register_task("mnist", load_task_mnist)
