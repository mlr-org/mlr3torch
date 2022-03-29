#' @param .paths List of graphs with on input and one output
#' @param param_vals The parameters passed to TorchOpMerge
parallel = function(.paths, ...) {
  names = names(.paths)
  assert_names(names, type = "strict") # unique?
  top("fork", .names = names) %>>%
    gunion(.paths) %>>%
    top("merge", ..., .innum = length(.paths))
}

if (FALSE) {
  g = paragraph(
    list(
      a = top("linear", out_features = 10L) %>>% top("relu"),
      b = top("linear")
    )
  )
  g$plot()
  block = top("block", .block = g)

  graph = top("flatten") %>>%
    block

}
