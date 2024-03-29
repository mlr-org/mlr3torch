#' Reset a classification models last layer
#'
#' Used for pretrained models where the last layer is set to e.g. 1000 classes
#' but training is intended to be applied to e.g. 10 classes.
#'
#' @note As of this, this also freezes the parameters of all but the last layer.
#'
#' @rdname reset_layer
#' @param model A pretrained model, e.g.
#' `torchvision::model_alexnet(pretrained = TRUE)`
#' @param num_classes Number of desired output classes.
#' @param bias `[TRUE]` Whether to use bias in the last layer.
#' @param freeze (`logical(1)`)\cr
#'   Whether to freee all layers expect for the output layer.
#'
#'
#' @return Same as input `model` with modified last layer.
#' @export
#'
#' @examplesIf torch::torch_is_installed()
#' @examples
#' \dontrun{
#' # AlexNet
#' if (mlr3misc::requires_namespaces("torchvision") && torch::torch_is_installed(h)) {
#'  model = torchvision::model_alexnet(pretrained = TRUE)
#'  model$classifier[[7]]$out_feature
#'  model = reset_last_layer(model, 10)
#'  model$classifier[[7]]$out_feature
#'
#'  # ResNet
#'  model = torchvision::model_resnet18(pretrained = TRUE)
#'  model$fc$out_feature
#'  model = reset_last_layer(model, 10)
#'  model$fc$out_feature
#' }
#' }
reset_last_layer = function(model, num_classes, bias, freeze) {
  UseMethod("reset_last_layer")
}


#' @rdname reset_layer
#' @export
reset_last_layer.AlexNet = function(model, num_classes, bias = TRUE, freeze = FALSE) { # nolint
  model$classifier$`6` = torch::nn_linear(
    in_features = model$classifier$`6`$in_features,
    out_features = num_classes,
    bias = bias
  )
  return(model)
}

#' @rdname reset_layer
#' @export
reset_last_layer.resnet = function(model, num_classes, bias = TRUE, freeze = FALSE) { # nolint
  model$fc = torch::nn_linear(
    in_features = model$fc$in_features,
    out_features = num_classes,
    bias = bias
  )
  return(model)
}
