# TODO: do we want a tabular tokenizer PipeOp? Or only the separate ones for num. and cat. features?
#' Tabular Tokenizers
#'
#' Tokenizes tabular data.
#'
#' @param n_features (`integer(1)`)\cr
#'   The number of numeric features.
#' @param cardinalities (`integer()`)\cr
#'   The cardinalities (levels) for the factor variables.
#' @param d_token (`integer(1)`)\cr
#'   The dimension of the tokens.
#' @param bias (`logical(1)`)\cr
#'   Whether to use a bias.
#' @param cls (`logical(1)`)\cr
#'   Whether to add a cls token.
#'
#' @references `r format_bib("gorishniy2021revisiting")`
PipeOpTorchTabTokenizer = R6Class("PipeOpTorchTabTokenizer",
  inherit = PipeOpTorch,
  public = list(
    initialize = function(id = "nn_tab_tokenizer", param_vals = list()) {
      param_set = ps(
        d_token = p_int(tags = "train"),
        bias = p_lgl(default = TRUE, tags = "train"),
        cls = p_lgl(default = FALSE, tags = "train")
      )
      
      private$.label = "Tabular Tokenizer"
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_tab_tokenizer
      )
    }
  ),
  private = list(
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      # TODO: determine whether we can compute this without the task
      # because otherwise we need to require that the task is passed in
      assert_false(test_null(task))

      types = task$feature_types[, type]
      n_numeric = sum(types %in% c("numeric", "integer"))
      n_categorical = sum(types %in% c("logical", "factor", "ordered"))
      
      # TODO: determine what exactly cardinalities should be... and then extract them from the task
      c(param_vals, list(n_features = n_numeric, cardinalities = ...))
    },
    # begin LLM code
    .shapes_out = function(shapes_in, param_vals, task) {
      n_num = if (length(shapes_in[[1L]]) > 1) shapes_in[[1L]][[2L]] else 0
      n_cat = if (length(shapes_in) > 1L) shapes_in[[2L]][[2L]] else 0
      n_total = n_num + n_cat
      
      if (param_vals$cls) {
        n_total = n_total + 1  # Add 1 for CLS token
      }
      
      # Output shape: [batch_size, n_features, d_token]
      list(c(NA, n_total, param_vals$d_token))
    }
    # end LLM code
  )
)

PipeOpTorchTokenizerCateg = R6Class("PipeOpTorchTokenizerCateg", 
  inherit = PipeOpTorch,
  public = list(
    param_set = ps(
      cardinaliaties = p_uty(tags = c("required", "train"), custom_check =  function(x) {
        assert_integerish(x,
                          lower = 1L, any.missing = FALSE,
                          min.len = 1L, coerce = TRUE)
      }),
      d_token = p_uty(tags = c("required", "train"), custom_check = function(x) {
        assert_integerish(x,
                          lower = 1L, any.missing = FALSE, len = 1,
                          coerce = TRUE)
      }),
      bias = p_lgl(tags = c("required", "train")),
      # TODO: determine the possible values for `initialization` (factor? levels: uniform, ...)
      initialization = p_fct(tags = c("required", "train"), levels = )
    )
  ),
  private = list(

  )
)

PipeOpTorchTokenizerNum = R6Class("PipeOpTorchTokenizerNum", 
  inherit = PipeOpTorch,
  public = list(

  ),
  private = list(
    
  )
)