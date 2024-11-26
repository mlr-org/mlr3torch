devtools::load_all()

# manually construct the task once
# library(here)
# library(data.table)

withr::local_options(mlr3torch.cache = TRUE)

constructor_melanoma = function(path) {
  # path = file.path(get_cache_dir(), "datasets", "melanoma")
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
    curl::curl_download
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

bench::system_time(melanoma_dt <- constructor_melanoma(file.path(get_cache_dir(), "datasets", "melanoma")))
melanoma_dt = constructor_melanoma(file.path(get_cache_dir(), "datasets", "melanoma"))

melanoma_dt[, image_name := NULL]
melanoma_dt[, target := NULL]

# change the encodings of variables: diagnosis, benign_malignant
melanoma_dt[, benign_malignant := factor(benign_malignant, levels = c("benign", "malignant"))]

char_features = c("sex", "anatom_site_general_challenge")
melanoma_dt[, (char_features) := lapply(.SD, factor), .SDcols = char_features]

tsk_melanoma = as_task_classif(melanoma_dt, target = "benign_malignant", id = "melanoma")
tsk_melanoma$set_col_roles("patient_id", "group")
tsk_melanoma$col_roles$feature = c(char_features, "age_approx", "image")

tsk_melanoma$label = "Melanoma classification"

ci = col_info(tsk_melanoma$backend)

saveRDS(ci, here::here("inst/col_info/melanoma.rds"))
saveRDS(melanoma_)
