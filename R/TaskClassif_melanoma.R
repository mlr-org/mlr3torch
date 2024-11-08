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
  file_names = c(
    "ISIC_2020_Training_GroundTruth_v2.csv", "train1", "train2", "train3", "train4",
    "ISIC_2020_Test_Metadata.csv", "ISIC_2020_Test_Input1", "ISIC_2020_Test_Input2"
  )

  withr::with_envvar(c(HUGGINGFACE_HUB_CACHE = path), {
    hfhub::hub_snapshot("carsonzhang/ISIC_2020_small", repo_type = "dataset")
  })

  hf_dataset_path = here(path, "datasets--carsonzhang--ISIC_2020_small", "snapshots", "2737ff07cc2ef8bd44d692d3323472fce272fca3")

  training_metadata = fread(here(hf_dataset_path, "ISIC_2020_Training_GroundTruth_v2.csv"))[, split := "train"]
  test_metadata = setnames(fread(here(hf_dataset_path, "ISIC_2020_Test_Metadata.csv")), 
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
    data[, image_name := NULL]
    data[, target := NULL]

    # change the encodings of variables: diagnosis, benign_malignant
    data[, benign_malignant := factor(benign_malignant, levels = c("benign", "malignant"))]

    char_features = c("sex", "anatom_site_general_challenge", "diagnosis")
    data[, lapply(.SD, factor), .SDcols = char_features]

    dt = cbind(data,
      data.table(
        ..row_id = seq_along(data$benign_malignant)
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
    target = "target",
    label = "Melanoma classification"
  )

  backend$hash = task$man = "mlr3torch::mlr_tasks_melanoma"
  
  task$set_col_roles("patient_id", roles = "group")

  task$filter(1:32701)

  return(task)
}

register_task("melanoma", load_task_melanoma)
