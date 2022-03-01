#' @title [CLS] Token
#' @description Appends a trainable [CLS] Token along the second dimension.
#' @references
#' `r format_bib("devlin2018bert", "gorishniy2021revisiting")
#' @export
nn_cls = nn_module("nn_cls",
  initialize = function(d_token) {
    self$weight = nn_parameter(torch_empty(d_token))
    d_sqrt_inv = 1 / sqrt(d_token)
    nn_init_uniform_(self$weight, a = -d_sqrt_inv, b = d_sqrt_inv)
  },
  #' Appends a CLS token along the token dimension
  #' (n_batch, n_features, d_token) --> (n_batch, n_features + 1, d_token)
  #' @param x (`torch_tensor) The input tensor
  forward = function(x) {
    cls = self$weight$view(c(1L, -1L))$expand(c(nrow(x), 1L, -1L))
    x = torch_cat(list(x, cls), dim = 2L)
    return(x)
  }
)
