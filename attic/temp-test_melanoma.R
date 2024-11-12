library(data.table)
library(here)
library(tidytable)

withr::local_options(mlr3torch.cache = TRUE)

load_col_info("melanoma")

task = tsk("melanoma")
# this makes the test faster
task$row_roles$use = 1:10
expect_equal(task$id, "melanoma")
expect_equal(task$label, "Melanoma classification")
expect_equal(task$feature_names, c("sex", "anatom_site_general_challenge", "age_approx", "image"))
expect_equal(task$target_names, "benign_malignant")
expect_equal(task$man, "mlr3torch::mlr_tasks_melanoma")
expect_equal(task$properties, c("twoclass", "groups"))

x = materialize(task$data(task$row_ids[1:2], cols = "image")[[1L]], rbind = TRUE)
expect_equal(x$shape, c(2, 3, 128, 128))
expect_equal(x$dtype, torch_float32())

training_metadata = fread(here::here("cache", "hf_dataset", "train", "ISIC_2020_Training_GroundTruth_v2.csv"))
training_metadata_extrasmall = training_metadata |>
  filter(file_name %in% list.files(here("cache", "hf_dataset", "train"), pattern = ".jpg$", recursive = TRUE))
fwrite(training_metadata_extrasmall, here("cache", "hf_dataset", "train", "ISIC_2020_Training_GroundTruth_v2.csv"))

test_metadata = fread(here::here("cache", "hf_dataset", "ISIC_2020_Test_Input", "ISIC_2020_Test_Metadata.csv"))
test_metadata_extrasmall = test_metadata |>
  filter(file_name %in% list.files(here("cache", "hf_dataset", "ISIC_2020_Test_Input"), pattern = ".jpg$", recursive = TRUE))
fwrite(test_metadata_extrasmall, here("cache", "hf_dataset", "ISIC_2020_Test_Input", "ISIC_2020_Test_Metadata.csv"))
