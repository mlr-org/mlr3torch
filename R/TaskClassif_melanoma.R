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
#' `r format_bib("melanoma2021")`
#' @examplesIf torch::torch_is_installed()
#' task = tsk("melanoma")
#' task
NULL

# @param path (`character(1)`)\cr
#   The cache_dir/datasets/melanoma folder
constructor_melanoma = function(path) {
  # file_names = c(
  #   "ISIC_2020_Training_GroundTruth_v2.csv", "train1", "train2", "train3", "train4",
  #   "ISIC_2020_Test_Metadata.csv", "ISIC_2020_Test_Input1", "ISIC_2020_Test_Input2"
  # )
  withr::local_options(mlr3torch.cache = TRUE)
  path = file.path(get_cache_dir(), "datasets", "melanoma")
  base_url = "https://huggingface.co/datasets/carsonzhang/ISIC_2020_small/resolve/main/"

  training_metadata_file_name = "ISIC_2020_Training_GroundTruth_v2.csv"
  curl::curl_download(paste0(base_url, training_metadata_file_name), file.path(path, training_metadata_file_name))
  training_metadata = fread(here::here(path, training_metadata_file_name))

  train_dir_names = c("train1", "train2", "train3", "train4")
  for (dir in train_dir_names) {
    if (!dir.exists(file.path(path, dir))) dir.create(file.path(path, dir))
  }

  pmap(
    list(paste(base_url, training_metadata$file_name, sep = ""), paste(path, "/", training_metadata$file_name, sep = "")),
    curl::curl_download
  )

  test_metadata_file_name = "ISIC_2020_Test_Metadata.csv"
  curl::curl_download(paste0(base_url, test_metadata_file_name), file.path(path, test_metadata_file_name))
  test_metadata = fread(here::here(path, test_metadata_file_name))

  test_dir_names = c("ISIC_2020_Test_Input1", "ISIC_2020_Test_Input2")
  for (dir in train_dir_names) {
    if (!dir.exists(file.path(path, dir))) dir.create(file.path(path, dir))
  }

  pmap(
    list(paste(base_url, test_metadata$file_name, sep = ""), paste(path, "/", test_metadata$file_name, sep = "")),
    curl_download
  )

  training_metadata = training_metadata[, split := "train"]
  test_metadata = setnames(test_metadata,
    old = c("image", "patient", "anatom_site_general"),
    new = c("image_name", "patient_id", "anatom_site_general_challenge")
  )[, split := "test"]
  metadata = rbind(training_metadata, test_metadata, fill = TRUE)

  melanoma_ds_generator = torch::dataset(
    initialize = function() {
      self$.metadata = metadata
      self$.path = hf_dataset_path
    },
    .getitem = function(idx) {
      force(idx)

      x = torchvision::base_loader(file.path(self$.path, paste0(self$.metadata[idx, ]$file_name)))
      x = torchvision::transform_to_tensor(x)

      return(list(x = x))
    },
    .length = function() {
      nrow(self$.metadata)
    }
  )

  melanoma_ds = melanoma_ds_generator()

  dd = as_data_descriptor(melanoma_ds, list(x = c(NA, 3, 128, 128)))
  lt = lazy_tensor(dd)

  return(cbind(metadata, data.table(image = lt)))
}

load_task_melanoma = function(id = "melanoma") {
  cached_constructor = function(backend) {
    data = cached(constructor_melanoma, "datasets", "melanoma")$data

    # remove irrelevant cols: image_name, target
    print(names(data))
    # if ("image_name" %in% names(data)) data[, image_name := NULL]
    data[, image_name := NULL]
    data[, target := NULL]

    # change the encodings of variables: diagnosis, benign_malignant
    data[, benign_malignant := factor(benign_malignant, levels = c("benign", "malignant"))]

    char_features = c("sex", "anatom_site_general_challenge")
    data[, (char_features) := lapply(.SD, factor), .SDcols = char_features]

    dt = cbind(
      data,
      data.table(
        ..row_id = seq_len(nrow(data))
      )
    )

    DataBackendDataTable$new(data = dt, primary_key = "..row_id")
  }

  backend = DataBackendLazy$new(
    constructor = cached_constructor,
    rownames = seq_len(32701 + 10982),
    col_info = load_col_info("melanoma"),
    primary_key = "..row_id"
  )

  task = TaskClassif$new(
    backend = backend,
    id = "melanoma",
    target = "benign_malignant",
    label = "Melanoma classification"
  )

  task$set_col_roles("patient_id", "group")
  task$col_roles$feature = c("sex", "anatom_site_general_challenge", "age_approx", "image")

  backend$hash = task$man = "mlr3torch::mlr_tasks_melanoma"

  task$filter(1:32701)

  return(task)
}

register_task("melanoma", load_task_melanoma)
