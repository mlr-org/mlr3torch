#' Reset a classification models last layer
#'
#' Used for pretrained models where the last layer is set to e.g. 1000 classes
#' but training is intended to be applied to e.g. 10 classes.
#'
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
#' model <- torchvision::model_alexnet(pretrained = TRUE)
#' model$classifier[[7]]$out_feature
#' model <- reset_last_layer(model, 10)
#' model$classifier[[7]]$out_feature
#' }
reset_last_layer <- function(model, num_classes, bias = TRUE) {
  # Number of modules total, assuming last module is the classifier
  module_count <- length(model$children)

  # 0-indexed last layer index
  last_layer_index <- length(model[[module_count]]) - 1

  # reset with linear layer based on input
  model[[module_count]][[last_layer_index]] <- torch::nn_linear(
    in_features = model[[module_count]][[last_layer_index - 1]]$out_feature,
    out_features = num_classes,
    bias = bias
  )

  model
}
