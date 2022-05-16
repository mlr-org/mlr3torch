#' @section Architecture:
#' Calls [<%= pkg[1] %>::<%= model %>] from package \CRANpkg{<%= pkg %>} to load the
#' network architecture. If the parameter `pretrained == TRUE`, the learner is initialized
#' with pretrained weights and the final layer is replaced with an simple linear
#' output layer tailored to the task when the method `$train()` is called.
