#' @export
as_learner_torch = function(x, clone = FALSE) {
  output = x$pipeops[[x$output$op.id]]
  learner = if (inherits(output, "TorchOpModelClassif") || inherits(output, "TorchOpModelRegr")) {
    GraphLearnerTorch$new(x, clone_graph = clone)
  } else {
    stopf("Output must be TorchOpModel{Classif, Regr}")
  }
  learner$id = "torch.graph"
  return(learner)
}

