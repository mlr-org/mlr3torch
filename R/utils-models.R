#' Reset a classification models last layer
#'
#' Used for pretrained models where the last layer is set to e.g. 1000 classes
#' but training is intended to be applied to e.g. 10 classes.
#'
#' @note As of this, this also freezes the parameters of all but the last layer.
#'
#' @rdname reset-layer
#' @param model A pretrained model, e.g.
#' `torchvision::model_alexnet(pretrained = TRUE)`
#' @param num_classes Number of desired output classes.
#' @param bias `[TRUE]` Whether to use bias in the last layer.
#'
#' @return Same as input `model` with modified last layer.
#' @export
#'
#' @examples
#' \dontrun{
#' # AlexNet
#' model <- torchvision::model_alexnet(pretrained = TRUE)
#' model$classifier[[7]]$out_feature
#' model <- reset_last_layer(model, 10)
#' model$classifier[[7]]$out_feature
#'
#' # ResNet
#' model <- torchvision::model_resnet18(pretrained = TRUE)
#' model$fc$out_feature
#' model <- reset_last_layer(model, 10)
#' model$fc$out_feature
#' }
reset_last_layer <- function(model, num_classes, bias = TRUE) {
  UseMethod("reset_last_layer")
}

#' @rdname reset-layer
#' @export
reset_last_layer.AlexNet <- function(model, num_classes, bias = TRUE) {

  # Freeze weights
  # for (par in model$parameters) {
  #   par$requires_grad_(FALSE)
  # }

  model$classifier$`6` <- torch::nn_linear(
    in_features = model$classifier$`6`$in_features,
    out_features = num_classes,
    bias = bias
  )
  return(model)
}

#' @rdname reset-layer
#' @export
reset_last_layer.resnet <- function(model, num_classes, bias = TRUE) {
  # Freeze weights
  for (par in model$parameters) {
    par$requires_grad_(FALSE)
  }

  model$fc <- torch::nn_linear(
    in_features = model$fc$in_features,
    out_features = num_classes,
    bias = bias
  )
  return(model)
}
