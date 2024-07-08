#' @title Tiny ImageNet Classification Task
#'
#' @name mlr_tasks_tiny_imagenet
#'
#' @description
#' Subset of the famous ImageNet dataset.
#' The data is obtained from [`torchvision::tiny_imagenet_dataset()`].
#'
#' The underlying [`DataBackend`][mlr3::DataBackend] contains columns `"class"`, `"image"`, `"..row_id"`, `"split"`, where the last column
#' indicates whether the row belongs to the train, validation or test set that defined provided in torchvision.
#'
#' There are no labels for the test rows, so by default, these observations are inactive, which means that the task
#' uses only 110000 of the 120000 observations that are defined in the underlying data backend.
#'
#' @section Construction:
#' ```
#' tsk("tiny_imagenet")
#' ```
#'
#' @template task_download
#'
#' @section Properties:
#' `r rd_info_task_torch("tiny_imagenet", missings = FALSE)`
#'
#' @references
#' `r format_bib("imagenet2009")`
#' @examplesIf torch::torch_is_installed()
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

  classes = c(train_res$labels, valid_res$labels, rep(NA_character_, length(test_uris)))
  uris = c(train_res$uris, valid_res$uris, test_uris)

  data.table(
    class = classes,
    image = uris,
    split = factor(rep(c("train", "valid", "test"), times = c(100000, 10000, 10000)))
  )
}

#' @include utils.R
load_task_tiny_imagenet = function(id = "tiny_imagenet") {
  cached_constructor = crate(function(backend) {
    dt = cached(constructor_tiny_imagenet, "datasets", "tiny_imagenet")$data
    setDT(dt)

    ci = col_info(backend)
    set(dt, j = "class", value = factor(dt$class, levels = ci[list("class"), "levels", on = "id"][[1L]][[1L]]))
    set(dt, j = "image", value = as_lazy_tensor(dataset_image(dt$image), dataset_shapes = list(x = c(NA, 3, 64, 64))))
    set(dt, j = "..row_id", value = seq_len(nrow(dt)))
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
  task$col_roles$feature = "image"

  # NAs in the target make the task annoying to work with
  task$filter(1:110000)

  return(task)
}

register_task("tiny_imagenet", load_task_tiny_imagenet)
