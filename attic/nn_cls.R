#' \[CLS\] Token
#'
#' Appends a trainable \[CLS\] Token along the second dimension.
#'
#' @param d_token (`integer(1)`)\cr
#'   The token dimensionality.
#'
#' `r format_bib("devlin2018bert", "gorishniy2021revisiting")`
nn_cls = nn_module("nn_cls",
  initialize = function(d_token) {
    self$weight = nn_parameter(torch_empty(d_token))
    d_sqrt_inv = 1 / sqrt(d_token)
    nn_init_uniform_(self$weight, a = -d_sqrt_inv, b = d_sqrt_inv)
  },
  forward = function(x) {
    cls = self$weight$view(c(1L, -1L))$expand(c(nrow(x), 1L, -1L))
    x = torch_cat(list(x, cls), dim = 2L)
    return(x)
  }
)
