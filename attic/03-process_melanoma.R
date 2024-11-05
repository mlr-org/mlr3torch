library(data.table)
library(tidytable)
library(purrr)

library(here)

library(fs)

# this script changes the data into the format expected by Hugging Face
# It expects that you have downloaded and extracted the original data by running the download_melanoma.R script
# and that you have already resized it with PyTorch

cache_dir = here("cache")

duplicates = fread(here(cache_dir, "ISIC_2020_Training_Duplicates.csv"))

metadata_file_paths = c(
  here(cache_dir, "ISIC_2020_Training_GroundTruth.csv"),
  here(cache_dir, "ISIC_2020_Training_GroundTruth_v2.csv"),
  here(cache_dir, "ISIC_2020_Test_Metadata.csv")
)
metadata_dt_list = map(metadata_file_paths, fread)
metadata_dt_list[[3]] = rename(metadata_dt_list[[3]], image_name = image)

# deduplicate the metadata
dedup = function(metadata_dt, duplicate_file_names) {
  metadata_dt[!(image_name %in% duplicate_file_names), ]
}

training_metadata = dedup(metadata_dt_list[[1]], duplicates$image_name_2)
training_metadata_v2 = dedup(metadata_dt_list[[2]], duplicates$image_name_2)
test_metadata = metadata_dt_list[[3]]

hf_dataset_dir = here(cache_dir, "hf_dataset")
hf_train_dir = here(hf_dataset_dir, "train")
hf_test_dir = here(hf_dataset_dir, "ISIC_2020_Test_Input")

train_dirnames_for_each_img = paste0("train", (training_metadata_v2[, .I] %% 4) + 1)
test_dirnames_for_each_img = paste0("ISIC_2020_Test_Input", (test_metadata[, .I] %% 2) + 1) 

# add a column that Hugging Face wants
add_hf_file_name_col = function(metadata_dt, image_relative_dirnames) {
  metadata_dt[, file_name := paste0(file.path(image_relative_dirnames, metadata_dt$image_name), ".jpg")]
}

# image_relative_paths = c("train", "train", "ISIC_2020_Test_Input")

add_hf_file_name_col(training_metadata, train_dirnames_for_each_img)
add_hf_file_name_col(training_metadata_v2, train_dirnames_for_each_img)
add_hf_file_name_col(metadata_dt_list[[3]], test_dirnames_for_each_img)

# delete the duplicated images
list.files(hf_train_dir) |> length()
file.remove(here(hf_train_dir, paste0(duplicates$image_name_2, ".jpg")))
list.files(hf_train_dir) |> length()

old_names = function(metadata_dt, dir) {
  paste0(file.path(dir, metadata_dt$image_name), ".jpg")
}

create_if_necessary = function(dirname) {
  if (!dir.exists(dirname)) {
    dir.create(dirname)
  }
}

walk(here(hf_dataset_dir, unique(train_dirnames_for_each_img)), create_if_necessary)
walk(here(hf_dataset_dir, unique(test_dirnames_for_each_img)), create_if_necessary)

# file_move(old_names(training_metadata), here(hf_dataset_dir, train_dirnames_for_each_img, paste0(training_metadata$image_name, ".jpg")))
file_move(old_names(training_metadata_v2, hf_train_dir), here(hf_dataset_dir, train_dirnames_for_each_img, paste0(training_metadata_v2$image_name, ".jpg")))
file_move(old_names(test_metadata, hf_test_dir), here(hf_dataset_dir, test_dirnames_for_each_img, paste0(test_metadata$image_name, ".jpg")))

test_metadata = rename(test_metadata, image = image_name)

fwrite(training_metadata, here(hf_dataset_dir, "ISIC_2020_Training_GroundTruth.csv"))
fwrite(training_metadata_v2, here(hf_dataset_dir, "ISIC_2020_Training_GroundTruth_v2.csv"))
fwrite(test_metadata, here(hf_dataset_dir, "ISIC_2020_Test_Metadata.csv"))

# test1 = list.files(here(hf_dataset_dir, "ISIC_2020_Test_Input1"))
# test2 = list.files(here(hf_dataset_dir, "ISIC_2020_Test_Input2"))
# setdiff(test1, test2)

# test_metadata |> filter(image_name == "ISIC_9999302") |> pull(file_name)
