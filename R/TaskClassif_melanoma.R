#' @title Melanoma Image classification
#' @name mlr_tasks_melanoma
#' @description
#' Classification of melanoma tumor images.
#' The data is a preprocessed version of the 2020 SIIM-ISIC challenge where
#' the images have been reshaped to size $(3, 128, 128)$.
#'
#' By default only the training rows are active in the task,
#' but the test data (that has no targets) is also included.
#' Whether an observation is part of the train or test set is indicated by the column `"test"`.
#'
#' There are no labels for the test rows, so by default, these observations are inactive,
#' which means that the task uses only 32701 of the 43683 observations that are defined in the underlying data backend.
#'
#' The data backend also contains a more detailed `diagnosis` of the specific type of tumor.
#'
#' Columns:
#' * `outcome` (factor): the target variable. Whether the tumor is benign or malignant (the positive class)
#' * `anatom_site_general_challenge` (factor): the location of the tumor on the patient's body
#' * `sex` (factor): the sex of the patient
#' * `age_approx` (int): approximate age of the patient at the time of imaging
#' * `image` (lazy_tensor): The image (shape $(3, 128, 128)$) of the tumor.
#'ee `split` (character): Whether the observation os part of the train or test set.
#'
#' @section Construction:
#' ```
#' tsk("melanoma")
#' ```
#'
#' @template task_download
#'
#' @source
#' \url{https://huggingface.co/datasets/carsonzhang/ISIC_2020_small}
#'
#' @section Properties:
#' `r rd_info_task_torch("melanoma", missings = FALSE)`
#'
#' @references
#' `r format_bib("melanoma2021")`
#' @examples
#' task = tsk("melanoma")
NULL

# @param path (`character(1)`)\cr
#   The cache_dir/datasets/melanoma folder
constructor_melanoma = function(path) {
  require_namespaces("curl")

  lg$info("Downloading the dataset")

  # should happen automatically, but this is needed for curl to work
  if (!dir.exists(path)) dir.create(path)

  base_url = "https://huggingface.co/datasets/carsonzhang/ISIC_2020_small/resolve/main/"

  compressed_tarball_file_name = "hf_ISIC_2020_small.tar.gz"
  compressed_tarball_path = file.path(path, compressed_tarball_file_name)
  on.exit({file.remove(compressed_tarball_path)}, add = TRUE)
  curl::curl_download(paste0(base_url, compressed_tarball_file_name), compressed_tarball_path)
  if (Sys.info()["sysname"] == "Linux") {
    utils::untar(compressed_tarball_path, exdir = path, extras = "--warning=no-unknown-keyword")
  } else {
    utils::untar(compressed_tarball_path, exdir = path)
  }

  training_metadata_file_name = "ISIC_2020_Training_GroundTruth_v2.csv"
  training_metadata = fread(file.path(path, training_metadata_file_name))

  test_metadata_file_name = "ISIC_2020_Test_Metadata.csv"
  test_metadata = fread(file.path(path, test_metadata_file_name))

  training_metadata = training_metadata[, split := "train"]
  test_metadata = setnames(test_metadata,
    old = c("image", "patient", "anatom_site_general"),
    new = c("image_name", "patient_id", "anatom_site_general_challenge")
  )[, split := "test"]
  # response column needs to be filled for the test data
  metadata = rbind(training_metadata, test_metadata, fill = TRUE)
  metadata[, "image_name" := NULL]
  metadata[, "target" := NULL]
  setnames(metadata, old = "benign_malignant", new = "outcome")

  metadata
}

load_task_melanoma = function(id = "melanoma") {
  cached_constructor = crate(function(backend) {
    res = cached(constructor_melanoma, "datasets", "melanoma", requires_disk = TRUE)
    metadata = res$data
    path = res$path

    melanoma_ds_generator = torch::dataset(
      initialize = function(metadata, cache_dir) {
        self$.metadata = metadata
        self$.cache_dir = cache_dir
      },
      .getitem = function(idx) {
        force(idx)

        x = torchvision::base_loader(file.path(self$.cache_dir, "raw", paste0(self$.metadata[idx, ]$file_name)))
        x = torchvision::transform_to_tensor(x)

        return(list(x = x))
      },
      .length = function() {
        nrow(self$.metadata)
      }
    )

    melanoma_ds = melanoma_ds_generator(metadata, path)

    dd = as_data_descriptor(melanoma_ds, list(x = c(NA, 3, 128, 128)))
    lt = lazy_tensor(dd)

    data = cbind(metadata, data.table(image = lt))

    char_vars = c("outcome", "sex", "anatom_site_general_challenge")
    data[, (char_vars) := lapply(.SD, factor), .SDcols = char_vars]

    dt = cbind(
      data,
      data.table(
        ..row_id = seq_len(nrow(data))
      )
    )

    DataBackendDataTable$new(data = dt, primary_key = "..row_id")
  }, .parent = topenv())

  backend = DataBackendLazy$new(
    constructor = cached_constructor,
    rownames = seq_len(32701 + 10982),
    col_info = load_col_info("melanoma"),
    primary_key = "..row_id"
  )

  task = TaskClassif$new(
    backend = backend,
    id = "melanoma",
    target = "outcome",
    positive = "malignant",
    label = "Melanoma Classification"
  )

  task$set_col_roles("patient_id", "group")
  task$col_roles$feature = c("sex", "anatom_site_general_challenge", "age_approx", "image")

  backend$hash = task$man = "mlr3torch::mlr_tasks_melanoma"

  task$filter(1:32701)

  return(task)
}

register_task("melanoma", load_task_melanoma)
