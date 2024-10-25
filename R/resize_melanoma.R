library(torch)
library(torchvision)

library(purrr)

library(here)

# change to wherever your files live
cache_dir = here("cache")

path_to_melanoma_train = here(cache_dir, "train")
path_to_melanoma_test = here(cache_dir, "ISIC_2020_Test_Input")
path_to_output_train = here(cache_dir, "train_small")
path_to_output_test = here(cache_dir, "ISIC_2020_Test_Input_small")

resize_to_dims = c(128, 128)

resize_and_write = function(image_file_name, path_to_input_train, path_to_output_dir, dims) {
  image = base_loader(file.path(path_to_input_train, image_file_name))
  small_image = torchvision::transform_resize(image, dims)

  output_file_name = file.path(path_to_output_dir, basename(image_file_name))

  torch::torch_save(small_image, path_to_output_dir)
}

walk(.x = list.files(path_to_melanoma_train), .f = resize_and_write(path_to_melanoma_train, path_to_output_train, resize_to_dims), .progress = TRUE)
walk(.x = list.files(path_to_melanoma_test), .f = resize_and_write(path_to_melanoma_test, path_to_output_test), .progress = TRUE)
