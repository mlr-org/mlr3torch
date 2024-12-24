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
}

load_task_cifar10 = function(id = "cifar10") {
    
}

