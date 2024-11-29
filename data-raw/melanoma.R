devtools::load_all()

# manually construct the task once
# library(here)
# library(data.table)
library(data.table)
withr::local_options(mlr3torch.cache = TRUE)

unzip2 <- function(path, exdir) {
  if (grepl("linux", R.version$os)) {
    utils::unzip(path, exdir = exdir)
  } else {
    zip::unzip(path, exdir = exdir)
  }
}

constructor_melanoma = function(path) {
  require_namespaces("curl")

  base_url = "https://huggingface.co/datasets/carsonzhang/ISIC_2020_small/resolve/main/"

  compressed_tarball_file_name = "hf_ISIC_2020_small.tar.gz"
  compressed_tarball_path = file.path(path, compressed_tarball_file_name)
  curl::curl_download(paste0(base_url, compressed_tarball_file_name), compressed_tarball_path)
  utils::untar(compressed_tarball_path, exdir = path)
  on.exit({file.remove(compressed_tarball_path)}, add = TRUE)

  training_metadata_file_name = "ISIC_2020_Training_GroundTruth_v2.csv"
  training_metadata = data.table::fread(file.path(path, training_metadata_file_name))

  test_metadata_file_name = "ISIC_2020_Test_Metadata.csv"
  test_metadata = file.path(path, test_metadata_file_name)

  training_metadata = training_metadata[, split := "train"]
  test_metadata = setnames(test_metadata,
    old = c("image", "patient", "anatom_site_general"),
    new = c("image_name", "patient_id", "anatom_site_general_challenge")
  )[, split := "test"]
  metadata = rbind(training_metadata, test_metadata)
  metadata[, image_name := NULL]
  metadata[, target := NULL]
  metadata = setnames(metadata, old = "benign_malignant", new = "outcome")

  melanoma_ds_generator = torch::dataset(
    initialize = function() {
      self$.metadata = metadata
      self$.path = path
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

bench::system_time(melanoma_dt <- constructor_melanoma(file.path(get_cache_dir(), "datasets", "melanoma")))
# melanoma_dt = constructor_melanoma(file.path(get_cache_dir(), "datasets", "melanoma"))

# change the encodings of variables: diagnosis, benign_malignant
melanoma_dt[, benign_malignant := factor(benign_malignant, levels = c("benign", "malignant"))]

char_features = c("sex", "anatom_site_general_challenge")
melanoma_dt[, (char_features) := lapply(.SD, factor), .SDcols = char_features]

tsk_melanoma = as_task_classif(melanoma_dt, target = "benign_malignant", id = "melanoma")
tsk_melanoma$set_col_roles("patient_id", "group")
tsk_melanoma$col_roles$feature = c(char_features, "age_approx", "image")

tsk_melanoma$label = "Melanoma Classification"

ci = col_info(tsk_melanoma$backend)

saveRDS(ci, here::here("inst/col_info/melanoma.rds"))
