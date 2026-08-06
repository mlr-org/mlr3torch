#' @section Input Scaling:
#' Neural networks are sensitive to the scale of their inputs, so features on very different scales
#' can slow down or destabilize training.
#' Standardize them beforehand, e.g. by prepending `po("scale")` to the learner.
