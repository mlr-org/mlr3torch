devtools::load_all()

# manually construct the task once
library(here)
library(data.table)

withr::local_options(mlr3torch.cache = TRUE)

# hf_cache_dir = here::here("cache2")

# hf_dataset_parent_path = here::here(hf_cache_dir, "raw", "datasets--carsonzhang--ISIC_2020_extrasmall", "snapshots")
# there should only be a single directory whose name is a hash value, this avoids hard-coding it
# hf_dataset_path = here::here(hf_dataset_parent_path, list.files(hf_dataset_parent_path))

constructor_melanoma = function(path) {
  file_names = c(
    "ISIC_2020_Training_GroundTruth_v2.csv", "train1", "train2", "train3", "train4",
    "ISIC_2020_Test_Metadata.csv", "ISIC_2020_Test_Input1", "ISIC_2020_Test_Input2"
  )

  withr::with_envvar(c(HUGGINGFACE_HUB_CACHE = path), {
    hfhub::hub_snapshot("carsonzhang/ISIC_2020_extrasmall", repo_type = "dataset")
  })
  hf_dataset_parent_path = here::here(path, "raw", "datasets--carsonzhang--ISIC_2020_extrasmall", "snapshots")
  # there should only be a single directory whose name is a hash value, this avoids hard-coding it
  hf_dataset_path = here::here(hf_dataset_parent_path, list.files(hf_dataset_parent_path))

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

melanoma_dt = constructor_melanoma(get_cache_dir())

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
