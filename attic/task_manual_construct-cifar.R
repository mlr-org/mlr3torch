devtools::load_all()
withr::local_options(mlr3torch.cache = TRUE)

constructor_cifar10 = function(path) {
    require_namespaces("torchvision")

    ds_train = torchvision::cifar10_dataset(root = file.path(path), download = TRUE, train = TRUE)

    browser()


}

constructor_cifar10(path <- file.path(get_cache_dir(), "datasets", "cifar10"))

library(here)

path <- here("cache")

cifar_ds_train <- torchvision::cifar10_dataset(root = file.path(path), train = TRUE)
cifar_ds_test <- torchvision::cifar10_dataset(root = file.path(path), download = FALSE, train = TRUE)

# first: remove the response (handled separately)
dd = as_data_descriptor(cifar_ds, list(x = c(NA, 3, 32, 32)))

