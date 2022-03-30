#' These functions here allow for simple construction of commonly used graphs


#' @title Para(llel) Graph
#' @param .paths List of graphs with on input and one output
#' @param .merge the method parameter of the TorchOpMerge
# TODO: take care of order of elements
paragraph = function(graphs, merge, block = FALSE) {
  names = names(graphs)
  assert_names(names, type = "strict") # unique?
  graph = top("fork", .outnum = length(graphs)) %>>%
    gunion(graphs) %>>%
    top("merge", method = merge, .innum = length(graphs))
  if (block) {
    graph = top("block", .block = graph)
  }
  return(graph)
}

if (FALSE) {
  g = paragraph(
    list(
      a = top("linear", out_features = 10L) %>>% top("relu"),
      b = top("linear", out_features = 10L)
    ),
    merge = "add"
  )

  g$plot()
  block = top("block", .block = g)

  graph = top("flatten") %>>%
    block

}
