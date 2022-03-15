#' @title Reflections mechanism for torch
#'
#' @details
#' Used to store / extend available hyperparameter levels for options used throughout torch,
#' e.g. the available 'loss' for a given Learner.
#'
#' @format [environment].
#' @export
torch_reflections = new.env(parent = emptyenv())

register_mlr3 = function() {

  # Learners ----------------------------------------------------------------

  lrns = utils::getFromNamespace("mlr_learners", ns = "mlr3")
  tsks = utils::getFromNamespace("mlr_tasks", ns = "mlr3")

  # classification learners
  # lrns$add("classif.torch", LearnerClassifTorch)
  lrns$add("classif.torch.tabnet", LearnerClassifTorchTabnet)
  lrns$add("classif.torch.alexnet", LearnerClassifTorchAlexNet)
  lrns$add("classif.torch.resnet", LearnerClassifTorchResNet)

  # regression learners
  lrns$add("regr.torch.tabnet", LearnerRegrTorchTabnet)


  # Reflections -------------------------------------------------------------
  reflcts = utils::getFromNamespace("mlr_reflections", ns = "mlr3")

  # Image URI feature (e.g. file path to .jpg etc.) for image classif tasks
  reflcts$task_feature_types[["img"]] = "imageuri"
  reflcts$data_formats = c(reflcts$data_formats, "torch_tensor")

  local({
    torch_reflections$loss <- list(
      classif = c(
        "adaptive_log_softmax_with", "bce", "bce_with_logits", "cosine_embedding",
        "ctc", "cross_entropy", "hinge_embedding", "kl_div", "margin_ranking",
        "multi_margin", "multilabel_margin", "multilabel_soft_margin", "nll",
        "soft_margin", "triplet_margin", "triplet_margin_with_distance"
      ),
      regr = c("l1", "mse", "poisson_nll", "smooth_l1")
    )

    torch_reflections$optimizer <- c(
      "rprop", "rmsprop", "adagrad", "asgd", "adadelta", "lbfgs", "sgd", "adam"
      )
  })

  local({
    torch_reflections$loss <- list(
      classif = c(
        "adaptive_log_softmax_with", "bce", "bce_with_logits", "cosine_embedding",
        "ctc", "cross_entropy", "hinge_embedding", "kl_div", "margin_ranking",
        "multi_margin", "multilabel_margin", "multilabel_soft_margin", "nll",
        "soft_margin", "triplet_margin", "triplet_margin_with_distance"
      ),
      regr = c("l1", "mse", "poisson_nll", "smooth_l1")
    )

    torch_reflections$optimizer <- c(
      "rprop", "rmsprop", "adagrad", "asgd", "adadelta", "lbfgs", "sgd",
      "adam", "madgrad"
      )

    # Resnet configs included in torchvision, used to check config arg in ResNet
    # learner
    torch_reflections$models$resnet_configs <- c(
      "resnet101", "resnet152", "resnet18", "resnet34", "resnet50",
      "wide_resnet101_2", "wide_resnet50_2"
    )
  })

}

.onLoad = function(libname, pkgname) {
  register_namespace_callback(pkgname, "mlr3", register_mlr3)
}
