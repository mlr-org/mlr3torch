#' @title Tiny ImageNet Classification Task
#'
#' @name mlr_tasks_tiny_imagenet
#'
#' @description
#' Subset of the famous ImageNet dataset.
#' The data is obtained from [`torchvision::tiny_imagenet_dataset()`].
#' It contains the train, validation and test data.
#' The row role `use` is set to the training data, the row role `test` to the valdiation data and the row role
#' `holdout` is set to the test data.
#' There are no labels available for the test data.
#'
#' The underlying [`DataBackend`] contains columns `"class"`, `"image"`, `"row_id"`, `"split"`, where the last column
#' indicates whether the row belongs to the train, validation or test set.
#'
#'
#' @section Construction:
#' ```
#' tsk("tiny_imagenet")
#' ```
#'
#' @section Meta Information:
#' `r rd_info_task_torch("tiny_imagenet", missings = FALSE)`
#'
#' @references
#' `r format_bib("imagenet2009")`
#' @examples
#' task = tsk("tiny_imagenet")
#' task
NULL

# @param path (`character(1)`)\cr
#   The cache_dir/datasets/tiny_imagenet folder.
constructor_tiny_imagenet = function(path) {
  require_namespaces("torchvision")
  # path points to {cache_dir, tempfile}/data/tiny_imagenet
  torchvision::tiny_imagenet_dataset(root = file.path(path), download = TRUE)
  download_folder = file.path(path, "tiny-imagenet-200")

  lookup = fread(sprintf("%s/words.txt", download_folder), header = FALSE)

  colnames(lookup) = c("id", "label")

  get_uris = function(dir, set) {
    folder_names = list.files(file.path(dir, set))
    folder_names = folder_names[folder_names != "val_annotations.txt"]
    res = map(folder_names, function(folder_name) {
      if (set == "train") {
        uris = list.files(file.path(dir, set, folder_name, "images"), full.names = TRUE)
      } else {
        uris = list.files(file.path(dir, set, folder_name), full.names = TRUE)
      }
      label = lookup[folder_name, "label", on = "id"][[1L]]
      list(uris = uris, label = label)
    })
    uris = map(res, "uris")
    labels = map_chr(res, "label")
    uri_vector = vector("character", length = sum(lengths(uris)))
    i = 1
    for (j in seq_along(uris)) {
      uri_vector[i:(i + length(uris[[j]]) - 1)] = uris[[j]]
      i = i + length(uris[[j]])
    }

    label_vector = rep(labels, times = lengths(uris))
    list(labels = label_vector, uris = uri_vector)
  }

  train_res = get_uris(download_folder, "train")
  valid_res = get_uris(download_folder, "val")
  test_uris = list.files(file.path(download_folder, "test", "images"), full.names = TRUE)

  ci = load_col_info("tiny_imagenet")
  classes = c(train_res$labels, valid_res$labels, rep(NA_character_, length(test_uris)))
  uris = c(train_res$uris, valid_res$uris, test_uris)
  splits = rep(c("train", "valid", "test"), times = map_int(list(train_res$labels, valid_res$labels, test_uris), length))

  data.table(class = factor(classes, levels = ci[id == "image", "levelsl"][[1L]]), image = uris, split = factor(splits))
}

#' @include utils.R
load_task_tiny_imagenet = function(id = "tiny_imagenet") {
  cached_constructor = crate(function() {

    dt$image = as_lazy_tensor(dataset_image(dt$image), dataset_shapes = list(x = c(NA, 3, 64, 64)))
    dt$..row_id = seq_len(nrow(dt))
    DataBackendDataTable$new(data = dt, primary_key = "..row_id")
  }, .parent = topenv())

  backend = DataBackendLazy$new(
    constructor = cached_constructor,
    rownames = seq_len(120000),
    col_info = load_col_info("tiny_imagenet"),
    primary_key = "..row_id",
    data_formats = "data.table"
  )

  task = TaskClassif$new(
    backend = backend,
    id = "tiny_imagenet",
    target = "class",
    label = "ImageNet Subset"
  )

  backend$hash = task$man = "mlr3torch::mlr_tasks_tiny_imagenet"

  task$row_roles$use = seq_len(100000)
  task$row_roles$test = seq(from = 100001, 110000)
  task$row_roles$holdout = seq(from = 110001, 120000)
  task$col_roles$feature = "image"

  return(task)
}

register_task("tiny_imagenet", load_task_tiny_imagenet)
