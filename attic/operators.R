#' @title Adds two TorchOps or Graphs
#'
#' @description
#' A [`gunion()`][mlr3pipelines::gunion] is created and the outputs of a and b are
#' added using the [TorchOpMerge].
#'
#' @param a,b (`Graph` or `TorchOp`)\cr
#'   Values to be e.g. added, multiplied etc..
#'
#' @return
#' Returns a [Graph][mlr3pipelines::Graph].
#'
#' @name operator
NULL

#' @export
#' @rdname operator
`%++%` = function(a, b) { # nolint
  UseMethod("%++%")
}

#' @export
#' @rdname operator
`%**%` = function(a, b) { # nolint
  UseMethod("%**%")
}

#' @rdname operator
#' @export
`%++%.Graph` = function(a, b) { # nolint
  merge(a, b, "add")
}

#' @export
#' @rdname operator
`%**%.Graph` = function(a, b) { # nolint
  merge(a, b, "mul")
}

#' @export
#' @rdname operator
`%++%.TorchOp` = function(a, b) { # nolint
  as_graph(a) %++% as_graph(b)
}

#' @export
#' @rdname operator
`%**%.TorchOp` = function(a, b) { # nolint
  as_graph(a) %**% as_graph(b)
}

assert_mergeable = function(a) {
  if (is_graph(a)) {
    assert_true(a$pipeops[[a$rhs]]$outnum == 1L && a$pipeops[[a$rhs]]$output$train == "ModelConfig")
    return(TRUE)
  } else if (is_torchop(a)) {
    return(TRUE)
  }
  stopf(
    "Argument %s must be of class 'TorchOp' or 'Graph' but is '%s'.",
    deparse(substitute(a)), class(a)[[1L]]
  )
}

merge = function(a, b, op) {
  assert_mergeable(a)
  assert_mergeable(b)
  a = as_graph(a)
  b = as_graph(b)
  g = gunion(list(a, b))

  id = uniqueify(op, c(a$ids(), b$ids()))
  op = top(op)
  op$id = id
  g$add_pipeop(op)
  g$add_edge(src_id = a$rhs, dst_id = id)
  g$add_edge(src_id = b$rhs, dst_id = id)
  return(g)
}
