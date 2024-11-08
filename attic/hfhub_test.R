library(here)
library(data.table)

devtools::load_all()

file_names = c(
  "ISIC_2020_Training_GroundTruth_v2.csv", "train1", "train2", "train3", "train4",
  "ISIC_2020_Test_Metadata.csv", "ISIC_2020_Test_Input1", "ISIC_2020_Test_Input2"
)

hf_cache_dir = here::here("cache", "hf_downloaded")

# withr::with_envvar(c(HUGGINGFACE_HUB_CACHE = hf_cache_dir), {
#   path <- hfhub::hub_snapshot("carsonzhang/ISIC_2020_small", repo_type = "dataset")
# })

# print(paths)


hf_dataset_path = here(hf_cache_dir, "datasets--carsonzhang--ISIC_2020_small", "snapshots", "2737ff07cc2ef8bd44d692d3323472fce272fca3")

constructor_melanoma = function(path) {
  file_names = c(
    "ISIC_2020_Training_GroundTruth_v2.csv", "train1", "train2", "train3", "train4",
    "ISIC_2020_Test_Metadata.csv", "ISIC_2020_Test_Input1", "ISIC_2020_Test_Input2"
  )

  # withr::with_envvar(c(HUGGINGFACE_HUB_CACHE = path), {
  #   hfhub::hub_snapshot("carsonzhang/ISIC_2020_small", repo_type = "dataset")
  # })

  hf_dataset_path = here(path, "datasets--carsonzhang--ISIC_2020_small", "snapshots", "2737ff07cc2ef8bd44d692d3323472fce272fca3")

  training_metadata = fread(here(hf_dataset_path, "ISIC_2020_Training_GroundTruth_v2.csv"))[, split := "train"]
  test_metadata = setnames(fread(here(hf_dataset_path, "ISIC_2020_Test_Metadata.csv")), 
    old = c("image", "patient", "anatom_site_general"), 
    new = c("image_name", "patient_id", "anatom_site_general_challenge")
  )[, split := "test"]
  metadata = rbind(training_metadata, test_metadata, fill = TRUE)

  # write to disk?

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

melanoma_ds = constructor_melanoma(hf_cache_dir)
