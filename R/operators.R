#' @title Adds two TorchOps or Graphs
#'
#' @description
#' A [`gunion()`][mlr3pipelines::gunion] is created and the outputs of a and b are
#' added using the [TorchOpMerge].
#'
#' @param a, b (`Graph` or `TorchOp`)\cr
#'   Values to be added.
#'
#' @return
#' Returns a [Graph][mlr3pipelines::Graph].
#' @export
`%+%` = function(a, b) {
  UseMethod("%+%")
}

#' @param a,b (`TorchOp` or `Graph`)\cr
#'   Graph / PipeOps whose outputs are to be add
#' @export
`%+%.Graph` = function(a, b) {
  assert_mergeable(a)
  assert_mergeable(b)
  a = as_graph(a)
  b = as_graph(b)
  g = gunion(list(a, b))

  add_id = uniqueify("add", c(a$ids(), b$ids()))
  add_op = top("add")
  add_op$id = add_id
  g$add_pipeop(add_op)
  g$add_edge(src_id = a$rhs, dst_id = add_id)
  g$add_edge(src_id = b$rhs, dst_id = add_id)
  return(g)
}


#' @export
`%+%.TorchOp` = function(a, b) {
  as_graph(a) %+% as_graph(b)
}

assert_mergeable = function(a) {
  if (is_graph(a)) {
    assert_true(a$pipeops[[a$rhs]]$outnum == 1L && a$pipeops[[a$rhs]]$output$train == "ModelArgs")
    return(TRUE)
  } else if (is_torchop(a)) {
    return(TRUE)
  }
  stopf(
    "Argument %s must be of class 'TorchOp' or 'Graph' but is '%s'.",
    deparse(substitute(a)), class(a)[[1L]]
  )
}
