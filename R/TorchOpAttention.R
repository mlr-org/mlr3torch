TorchOpAttention = R6Class("TorchOpAttention",
  inherit = TorchOp,
  public = list(
    initialize = function(id = "attention", param_vals = list()) {
      param_set = ps(
        d_token = p_int(default = NO_DEF, lower = 1L, tags = "train"),
        n_heads = p_int(default = 1L, lower = 1L, tags = "train"),
        dropout = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train"),
        bias = p_lgl(default = TRUE, tags = "train"),
        # TODO: correct default?
        initialization = p_fct(default = "kaiming", levels = c("kaiming", "xavier"))
      )
      input = data.table(
        name = c("query", "value"),
        train = c("ModelArgs", "ModelArgs"),
        predict = c("*", "*")
      )
      super$initialize(
        id = id,
        param_vals = param_vals,
        input = input,
        param_set = param_set
      )
    }
  ),
  private = list(
    .train = function(inputs) {
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("attention", TorchOpAttention)

if (FALSE) {
  graph = top("linear0") %>>%
    top("relu") %>>%
    gunion(
      graphs = list(
        value = top("linear", out_features = 10L),
        query = top("linear", out_features = 10L)
      )
    ) %>>%
    top("selfattention")
  # top("linear") %>>%
  #   top("attention")
}
