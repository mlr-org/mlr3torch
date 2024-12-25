#' @title CIFAR-10 Classification Task
#' 
#' @name mlr_tasks_cifar_10
#' 
#' @description 
#' The CIFAR-10 subset of the 80 million tiny images dataset.
#' The data is obtained from [`torchvision::cifar10_dataset()`].
NULL

# TODO: implement both CIFAR-10 and CIFAR-100 in the same file
# the torchvision implementation and the PipeOpAdaptiveAvgPool implementations are probably helpful here
constructor_cifar10 = function(path) {
    require_namespaces("torchvision")

    # download
    d_train = torchvision::cifar10_dataset(root = file.path(path), train = TRUE, download = TRUE)
    d_test = torchvision::cifar10_dataset(root = file.path(path), train = FALSE)

    # look at the directory structure
    # create a data.table with columns class, image (uri of the image in the downloaded dir)
}

load_task_cifar10 = function(id = "cifar10") {
    # consruct lazy tensors from the cached img uris


}

