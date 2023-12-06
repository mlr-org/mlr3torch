#' @title MNIST Image classification
#' @name mlr_tasks_mnist
#' @description
#' Classical MNIST image classification.
#'
#' The underlying [`DataBackend`] contains columns `"label"`, `"image"`, `"row_id"`, `"split"`, where the last column
#' indicates whether the row belongs to the train or test set.
#'
#' The first 60000 rows belong to the training set, the last 10000 rows to the test set.
#'
#' @template task_download
#' @section Construction:
#' ```
#' tsk("mnist")
#' ```
#'
#' @source
#' \url{https://torchvision.mlverse.org/reference/mnist_dataset.html}
#'
#' @section Meta Information:
#' `r rd_info_task_torch("mnist", missings = FALSE)`
#'
#' @references
#' `r format_bib("mnist")`
#' @examples
#' task = tsk("mnist")
#' task
NULL

# @param path (`character(1)`)\cr
#   The cache_dir/datasets/tiny_imagenet folder.
constructor_mnist = function(path) {
  # path points to {cache_dir, tempfile}/data/mnist
  d_train = torchvision::mnist_dataset(root = file.path(path), train = TRUE, download = TRUE)
  # test set is already downloaded with the training set above
  d_test = torchvision::mnist_dataset(root = file.path(path), train = FALSE)

  images_train = array(d_train$data, dim = c(60000, 1, 28, 28))
  images_test = array(d_test$data, dim = c(10000, 1, 28, 28))

  images = array(NA, dim = c(70000, 1, 28, 28))
  images[1:60000, , , ] = images_train
  images[60001:70000, , , ] = images_test
  labels = factor(c(d_train$targets, d_test$targets), levels = 1:10, labels = as.character(0:9))

  list(label = labels, image = images)
}

load_task_mnist = function(id = "mnist") {
  cached_constructor = function() {
    # factor level ordering can depend on locale
    # in this case, nothing should go wrong but we keep it here as a reminder (is e.g. needed in tiny imagenet)
    withr::with_locale(c(LC_COLLATE = "C"), {
      data = cached(constructor_mnist, "datasets", "mnist")$data
    })

    ds = dataset(
      initialize = function(images) {
        self$images = torch_tensor(images, dtype = torch_float32())
      },
      .getbatch = function(idx) {
        list(image = self$images[idx, , , drop = FALSE])
      },
      .length = function() dim(self$images)[1L]
    )(data$image)

    data_descriptor = DataDescriptor(dataset = ds, list(image = c(NA, 1, 28, 28)))

    dt = data.table(
      image = lazy_tensor(data_descriptor),
      label = data$label,
      ..row_id = seq_along(data$label),
      split = factor(c(rep("train", 60000), rep("test", 10000)))
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

  task$row_roles$use = seq_len(60000)
  task$row_roles$test = seq(from = 60001, 70000)
  task$col_roles$feature = "image"

  return(task)
}

register_task("mnist", load_task_mnist)