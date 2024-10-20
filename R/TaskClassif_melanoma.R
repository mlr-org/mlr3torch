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

load_task_melanoma = function(id = "melanoma") {
  # construct a DataBackendLazy for this large dataset

  # the DataBackendLazy implements the logic for downloading, processing, caching the dataset. 
  # in this caes, we only need to implement the download and processing becuase the private `cached()` function implements caching

  # the DataBackendLazy also hardcodes some metadata that will be available even before the data is downloaded.
  # this metadata will be stored in `.inst/col_info`
  # and can be loaded using `load_column_info()`
  # the code that generates this hardcoded metadata should be in `./data-raw`

  # create a TaskClassif from this DataBackendLazy
  return(task)
}

register_task("melanoma", load_task_melanoma)