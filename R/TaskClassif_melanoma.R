#' @title Melanoma Image classification
#' @name mlr_tasks_melanoma
#' @description
#' Classification of melanoma tumor images.
#'
#' The data comes from the 2020 ISIC challenge.
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
  training_jpeg_images_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip"
  training_metadata_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv"
  training_metadata_v2_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv"
  training_duplicate_image_list_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Duplicates.csv"

  test_jpeg_images_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip"
  test_metadata_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_Metadata.csv"

  urls = c(
    training_jpeg_images_url, training_metadata_url, training_metadata_v2_url, training_duplicate_image_list_url,
    test_jpeg_images_url, test_metadata_url
  )

  download_melanoma_file = function(url) {
    # change if necessary
    prev_options = options(timeout = 3600)
    on.exit(options(prev_options))

    download.file(url, here(cache_dir, basename(url)))
  }

  mlr3misc::walk(urls, download_melanoma_file)
  
  unzip(here(cache_dir, basename(training_jpeg_images_url)), exdir = here(cache_dir))
  unzip(here(cache_dir, basename(test_jpeg_images_url)), exdir = here(cache_dir))

  train_metadata = fread(here(path, basename(test_jpeg_images_url)))
  # train_images = 

  # TODO: decide whether to delete these, since there are no ground truth labels
  # test_metadata = fread(here(path, basename(test_metadata_url)))
  # test_images = fread(here())

  data.table(
    # image: ltsnr
    # metadata cols
  )
}

load_task_melanoma = function(id = "melanoma") {
  cached_constructor = function(backend) {
    data = cached(constructor_melanoma, "datasets", "melanoma")$data
    labels = ...

    ds = dataset(

    )(data$image)

    # some preprocessing

    # TODO: determine the end dimensionality
    data_descriptor = DataDescriptor$new(dataset = ds, list(image = c(NA, channel, spatial_dims)))
  }

  # construct a DataBackendLazy for this large dataset
  backend = DataBackendLazy$new(
    constructor = cached_constructor,
    rownames = seq_len(n_rows), # TODO: compute
    # hard-coded info about the task (nrows, ncols)
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

download_melanoma = function() {
  download.file("https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv",
    dest = "~/Downloads/metadata.csv"
  )
}
