devtools::load_all()
withr::local_options(mlr3torch.cache = TRUE)

constructor_cifar10 = function(path) {
    require_namespaces("torchvision")

    torchvision::cifar10_dataset(root = file.path(path), download = TRUE)

    browser()

}

constructor_cifar10(path <- file.path(get_cache_dir(), "datasets", "cifar10"))

library(here)

path <- here("cache")
fs::dir_create(path, recurse = TRUE)

cifar_ds <- torchvision::cifar10_dataset(root = file.path(path), download = TRUE)
