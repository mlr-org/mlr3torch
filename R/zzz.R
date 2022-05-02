#' @keywords internal
"_PACKAGE"

#' @import paradox
#' @import mlr3
#' @import checkmate
#' @import data.table
#' @import mlr3misc
#' @importFrom R6 R6Class
#' @import torch
#' @import luz
#' @import mlr3pipelines
#' @import mlr3misc
#' @import mlr3
#' @importFrom zeallot `%<-%`
#' @importFrom coro loop
NULL

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

  tsks$add("tiny_imagenet", load_task_tiny_imagenet)

  # classification learners
  lrns$add("classif.tabnet", LearnerClassifTabNet)
  lrns$add("classif.alexnet", LearnerClassifAlexNet)
  lrns$add("classif.torch", LearnerClassifTorch)

  # regression learners
  # lrns$add("regr.torch.tabnet", LearnerClassifTabNet)

  # PipeOps
  mlr_pipeops = mlr3pipelines::mlr_pipeops
  mlr_pipeops$add("imagetrafo", PipeOpImageTrafo)



  # Reflections -------------------------------------------------------------
  reflcts = utils::getFromNamespace("mlr_reflections", ns = "mlr3")

  # Image URI feature (e.g. file path to .jpg etc.) for image classif tasks
  reflcts$task_feature_types[["img"]] = "imageuri"
  reflcts$data_formats = c(reflcts$data_formats, "torch_tensor")

  local({
    torch_reflections$loss = list(
      classif = c(
        "adaptive_log_softmax_with", "bce", "bce_with_logits", "cosine_embedding",
        "ctc", "cross_entropy", "hinge_embedding", "kl_div", "margin_ranking",
        "multi_margin", "multilabel_margin", "multilabel_soft_margin", "nll",
        "soft_margin", "triplet_margin", "triplet_margin_with_distance"
      ),
      regr = c("l1", "mse", "poisson_nll", "smooth_l1")
    )

    torch_reflections$optimizer = c(
      "rprop", "rmsprop", "adagrad", "asgd", "adadelta", "lbfgs", "sgd", "adam"
    )

    torch_reflections$activation = c(
      "elu", "hardshrink", "hardsigmoid", "hardtanh", "hardswish", "leaky_relu", "log_sigmoid",
      "prelu", "relu", "relu6", "rrelu", "selu", "sigmoid",
      "softplus", "softshrink", "softsign", "tanh", "tanhshrink", "threshold", "glu"
    )

    torch_reflections$image_trafos = c(
      "random_crop",
      "center_crop",
      "hflip",
      "adjust_gamma",
      "random_order",
      "adjust_brightness",
      "pad",
      "random_affine",
      "affine",
      "random_rotation",
      "vflip",
      "random_resized_crop",
      "crop",
      "resized_crop",
      "random_choice",
      "resize",
      "rgb_to_grayscale",
      "adjust_saturation",
      "linear_transformation",
      "random_vertical_flip",
      "random_horizontal_flip",
      "color_jitter",
      "adjust_contrast",
      "rotate",
      "adjust_hue",
      "normalize",
      "random_apply",
      "to_tensor"
    )
  })

}

.onLoad = function(libname, pkgname) {
  # For caching directory
  backports::import(pkgname)
  backports::import(pkgname, "R_user_dir", force = TRUE)

  register_namespace_callback(pkgname, "mlr3", register_mlr3)
  assign("lg", lgr::get_logger(pkgname), envir = parent.env(environment()))
  if (Sys.getenv("IN_PKGDOWN") == "true") {
    lg$set_threshold("warn")
  }
}
