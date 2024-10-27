library(data.table)
library(purrr)

cache_dir = here("cache")

duplicates = fread(here(cache_dir, "ISIC_2020_Training_Duplicates.csv"))

metadata_file_paths = c(
  here(cache_dir, "ISIC_2020_Training_GroundTruth.csv"),
  here(cache_dir, "ISIC_2020_Training_GroundTruth_v2.csv"),
  here(cache_dir, "ISIC_2020_Test_Metadata.csv")
)

metadata_dt_list = map(metadata_file_paths, fread)

add_hf_file_name_col = function(metadata_dt, image_relative_dir) {
  metadata_dt[, (file_name) := file.path(image_relative_dir, metadata_dt$image_name)]
}

image_relative_paths = c("train", "train", "ISIC_2020_Test_Input")

walk2(metadata_dt_list, image_relative_paths, add_hf_file_name_col)
