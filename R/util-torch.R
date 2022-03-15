#' Get available devices for torch
#'
#' If no GPU is available
#' @return Character of length `torch::cuda_device_count() + 1`.
#' "cpu" if no GPU is available, otherwise "cpu" and 0-indexed GPUs prefixed
#' with "cuda:".
#' @export
#'
#' @examples
#' # Returns at least "cpu"
#' get_available_device()
get_available_device <- function() {
  devices <- NULL

  if (torch::cuda_is_available()) {
    devices <- paste0("cuda:", seq(0, torch::cuda_device_count() - 1))
  }

  c("cpu", devices)
}

#' Choose the correct device for torch
#'
#' @param gpu_index `[0]` Preferred index (starting from 0) of GPU to use.
#' Will be checked against `get_available_device()`
#'
#' @return `character(1)` with a valid torch device.
#' @export
#'
#' @examples
#' # GPU 0 or CPU
#' choose_device()
#'
#' # Choose second GPU if available
#' # choose_device(1)
choose_device <- function(gpu_index = 0) {
  checkmate::assert_int(gpu_index)

  if (!torch::cuda_is_available()) return("cpu")

  device <- paste0("cuda:", gpu_index)

  stopifnot(device %in% get_available_device())

  device
}


# Retrieving torch objects ------------------------------------------------

#' Retrieve a torch loss function by name
#'
#' The following loss functions are currently supported:
#'
#' - `"adaptive_log_softmax_with"`: [`torch::nn_adaptive_log_softmax_with_loss()`]
#' - `"bce"`: [`torch::nn_bce_loss()`]
#' - `"bce_with_logits"` : [`torch::nn_bce_with_logits_loss()`]
#' - `"cosine_embedding"`: [`torch::nn_cosine_embedding_loss()`]
#' - `"cross_entropy"`: [`torch::nn_cross_entropy_loss()`]
#' - `"ctc"`: [`torch::nn_ctc_loss()`]
#' - `"hinge_embedding"`: [`torch::nn_hinge_embedding_loss()`]
#' - `"kl_div"`: [`torch::nn_kl_div_loss()`]
#' - `"l1"`: [`torch::nn_l1_loss()`]
#' - `"margin_ranking"`: [`torch::nn_margin_ranking_loss()`]
#' - `"mse"`: [`torch::nn_mse_loss()`]
#' - `"multi_margin"`: [`torch::nn_multi_margin_loss()`]
#' - `"multilabel_margin"`: [`torch::nn_multilabel_margin_loss()`]
#' - `"multilabel_soft_margin"`: [`torch::nn_multilabel_soft_margin_loss()`]
#' - `"nll"`: [`torch::nn_nll_loss()`]
#' - `"poisson_nll"`: [`torch::nn_poisson_nll_loss()`]
#' - `"smooth_l1"`: [`torch::nn_smooth_l1_loss()`]
#' - `"soft_margin"`: [`torch::nn_soft_margin_loss()`]
#' - `"triplet_margin"`: [`torch::nn_triplet_margin_loss()`]
#' - `"triplet_margin_with_distance"`: [`torch::nn_triplet_margin_with_distance_loss()`]
#' @param name `[character]` Name of the loss function.
#' @param ... Optional named arguments passed to the loss function generator,
#' e.g. `weight` for [`torch::nn_bce_loss()`].
#'
#' @return An object inherting from `"nn_loss"` and `"nn_module"`.
#' @export
#' @family torch objects
#' @examples
#' bce <- get_torch_loss("bce")
#'
#' # Equivalent to calling:
#' torch_bce <- torch::nn_bce_loss()
#'
#' # With arguments
#' bce <- get_torch_loss("bce", reduction = "none")
#' # reduction is set in resulting module
#' bce$reduction
get_torch_loss <- function(name, ...) {

  if (inherits(name, "nn_loss") & inherits(name, "nn_module_generator")) {
    return(name(...))
  }

  switch(
    name,
    adaptive_log_softmax_with = torch::nn_adaptive_log_softmax_with_loss(...),
    bce = torch::nn_bce_loss(...),
    bce_with_logits = torch::nn_bce_with_logits_loss(...),
    cosine_embedding = torch::nn_cosine_embedding_loss(...),
    cross_entropy = torch::nn_cross_entropy_loss(...),
    ctc = torch::nn_ctc_loss(...),
    hinge_embedding = torch::nn_hinge_embedding_loss(...),
    kl_div = torch::nn_kl_div_loss(...),
    l1 = torch::nn_l1_loss(...),
    margin_ranking = torch::nn_margin_ranking_loss(...),
    mse = torch::nn_mse_loss(...),
    multi_margin = torch::nn_multi_margin_loss(...),
    multilabel_margin = torch::nn_multilabel_margin_loss(...),
    multilabel_soft_margin = torch::nn_multilabel_soft_margin_loss(...),
    nll = torch::nn_nll_loss(...),
    poisson_nll = torch::nn_poisson_nll_loss(...),
    smooth_l1 = torch::nn_smooth_l1_loss(...),
    soft_margin = torch::nn_soft_margin_loss(...),
    triplet_margin = torch::nn_triplet_margin_loss(...),
    triplet_margin_with_distance = torch::nn_triplet_margin_with_distance_loss(...),
    stop("Loss not supported")
  )
}

#' Retrieve a torch optimizer by name
#'
#' The following optimizers are currently supported:
#'
#' - `"adadelta"`: [`torch::optim_adadelta`][torch::optim_adadelta]
#' - `"adagrad"`: [`torch::optim_adagrad`][torch::optim_adagrad]
#' - `"adam"`: [`torch::optim_adam`][torch::optim_adam]
#' - `"asgd"`: [`torch::optim_asgd`][torch::optim_asgd]
#' - `"lbfgs"`: [`torch::optim_lbfgs`][torch::optim_lbfgs]
#' - `"rmsprop"`: [`torch::optim_rmsprop`][torch::optim_rmsprop]
#' - `"rprop"`: [`torch::optim_rprop`][torch::optim_rprop]
#' - `"sgd"`: [`torch::optim_sgd`][torch::optim_sgd]
#' - `"madgrad"`: [`madgrad::optim_madgrad`][madgrad::optim_madgrad]
#'
#' @param name `[character]` Name of the optimizer.
#'
#' @return A function equivalent to the exported function from \CRANpkg{torch}.
#' @export
#' @family torch objects
#' @examples
#' adam <- get_torch_optimizer("adam")
#'
#' # Same as torch::optim_adam
#' identical(adam, torch::optim_adam)
get_torch_optimizer <- function(name) {

  if (name == "madgrad" & !requireNamespace("madgrad", quietly = TRUE)) {
    stop("Please install 'madgrad' to use the madgrad optimizer")
  }

  switch(
    name,
    adadelta = torch::optim_adadelta,
    adagrad = torch::optim_adagrad,
    adam  = torch::optim_adam,
    asgd = torch::optim_asgd,
    lbfgs = torch::optim_lbfgs,
    rmsprop = torch::optim_rmsprop,
    rprop = torch::optim_rprop,
    sgd = torch::optim_sgd,
    madgrad = madgrad::optim_madgrad,
    stop("Optimizer not supported")
  )
}
