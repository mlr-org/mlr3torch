#' @title Melanoma Image classification
#' @name mlr_tasks_melanoma
#' @description
#' Classification of melanoma tumor images.
#'
#' The data comes from the 2020 SIIM-ISIC challenge.
#'
#' @section Construction:
#' ```
#' tsk("melanoma")
#' ```
#'
#' @template task_download
#'
#' @source
#' \url{https://challenge2020.isic-archive.com/}
#'
#' @section Properties:
#' `r rd_info_task_torch("melanoma", missings = FALSE)`
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
  # TODO: change to Hugging Face URLs
  # TODO: use the code from the attic
  training_jpeg_images_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip"
  training_metadata_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv"
  training_metadata_v2_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv"
  training_duplicate_image_list_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Duplicates.csv"

  test_metadata_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_Metadata.csv"

  urls = c(
    training_jpeg_images_url, training_metadata_url, training_metadata_v2_url, training_duplicate_image_list_url,
    test_jpeg_images_url, test_metadata_url
  )

  download_melanoma_file = function(url) {
    prev_options = options(timeout = 36000)
    on.exit(options(prev_options))

    download.file(url, path)
  }

  mlr3misc::walk(urls, download_melanoma_file)

  unzip(here(path, basename(training_jpeg_images_url)), exdir = path)
  unzip(here(cache_dir, basename(test_jpeg_images_url)), exdir = path)

  training_metadata = fread(here(path, basename(training_metadata_v2_url)))

  # if you want some operation to be cached (e.g. if it's expensive) do it here
  ds_train = torch::dataset(
    initialize = function() {
      self$.metadata = fread(here(path, "ISIC_2020_Training_GroundTruth_v2.csv"))
      self$.path = file.path(here(path), "train")
    },
    .getitem = function(idx) {
      force(idx)

      x = torchvision::base_loader(file.path(self$.path, paste0(self$.metadata[idx, ]$image_name, ".jpg")))
      x = torchvision::transform_to_tensor(x)

      return(list(x = x))
    },
    .length = function() {
      nrow(self$.metadata)
    }
  )

  ds_test = torch::dataset(
    initialize = function() {
      self$.metadata = fread(here(path, "ISIC_2020_Test_Metadata.csv"))
      self$.path = file.path(here(path), "ISIC_2020_Test_Input")
    }
    .getitem = function(idx) {
      force(idx)

      x = torchvision::base_loader(file.path(self$.path, paste0(self$.metadata[idx, ]$image_name, ".jpg")))
      x = torchvision::transform_to_tensor(x)

      return(list(x = x))
    },
    .length = function() {
      nrow(self$.metadata)
    }
  )

  dd_train = as_data_descriptor(melanoma_ds, list(x = NULL))
  lt_train = lazy_tensor(dd)

  return(cbind(training_metadata, data.table(x = lt)))
}

load_task_melanoma = function(id = "melanoma") {
  cached_constructor = function(backend) {
    data = cached(constructor_melanoma, "datasets", "melanoma")$data

    ds = dataset(
      initialize = function() {
        self$.metadata = fread(here(cache_dir, "ISIC_2020_Training_GroundTruth.csv"))
        self$.path = file.path(here(cache_dir), "train")
      },
      .getitem = function(idx) {
        force(idx)

        x = torchvision::base_loader(file.path(self$.path, paste0(self$.metadata[idx, ]$image_name, ".jpg")))
        x = torchvision::transform_to_tensor(x)

        return(list(x = x))
      },
      .length = function() {
        nrow(self$.metadata)
      }
    )(data$image)

    # TODO: some preprocessing

    dd = as_data_descriptor(melanoma_ds, list(x = NULL))
    lt = lazy_tensor(dd)
    dt = cbind(training_metadata, data.table(x = lt))

    # set ..row_id = 

    DataBackendDataTable$new(data = dt, primary_key = "..row_id")
  }

  # construct a DataBackendLazy for this large dataset
  backend = DataBackendLazy$new(
    constructor = cached_constructor,
    rownames = seq_len(32701), # TODO: is it weird to have rownames different from primary_key?
    # hard-coded info about the task (nrows, ncols)
    col_info = load_col_info("melanoma"),
    primary_key = "..row_id"
  )

  # the DataBackendLazy implements the logic for downloading, processing, caching the dataset.
  # in this case, we only need to implement the download and processing because the private `cached()` function implements caching

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
  task$set_col_roles("patient_id", roles = "group")

  return(task)
}

register_task("melanoma", load_task_melanoma)
