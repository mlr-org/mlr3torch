#' @title Melanoma Image classification
#' @name mlr_tasks_melanoma
#' @description
#' Classification of melanoma tumor images.
#' 
#' More descriptive text.
#' 
#' @section Construction:
#' ```
#' tsk("melanoma")
#' ```
#' 
#' @template task_download
#' 
#' @source 
#' \url{https://www.kaggle.com/c/siim-isic-melanoma-classification/data}
#' 
#' @section Properties:
#' `r rd_info_task_torch("mnmelanoma", missings = FALSE)`
#' 
#' @references 
#' `r format_bib("melanoma")`
#' @examplesIf torch::torch_is_installed()
#' task = tsk("melanoma")
#' task
NULL

# @param path (`character(1)`)\cr
#   The cache_dir/datasets/melanoma folder
constructor_melanoma = function(path) {
  # download data
  # TODO: look at the similar code from the `torchdatasets` package and decide what you want to include

  data.table(
    # image: ltsnr
    # metadata cols
  )
}

load_task_melanoma = function(id = "melanoma") {
  # construct a DataBackendLazy for this large dataset
  backend = DataBackendLazy$new(
    constructor = cached_constructor,
    rownames = seq_len(n_rows), # TODO: compute
    col_info = load_col_info("melanoma")
    primary_key = "..row_id" # TODO: explain
  )

  # the DataBackendLazy implements the logic for downloading, processing, caching the dataset. 
  # in this case, we only need to implement the download and processing because the private `cached()` function implements caching
  # TODO: find this private `cached()` function

  # the DataBackendLazy also hardcodes some metadata that will be available even before the data is downloaded.
  # this metadata will be stored in `.inst/col_info`
  # and can be loaded using `load_column_info()`
  # the code that generates this hardcoded metadata should be in `./data-raw`

  # create a TaskClassif from this DataBackendLazy
  task = TaskClassif$new(
    backend = backend,
    id = "melanoma",
    target = "class",
    label = "Melanoma classification"
  )

  return(task)
}

register_task("melanoma", load_task_melanoma)